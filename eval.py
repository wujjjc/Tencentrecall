"""全量候选口径的验证评测。

对齐 O_o/main.py:225-289，但有两处**有意偏离**（设计文档 §7.4）：
  1. 候选集是**全量 4,783,154**，不是批内候选 —— 分数与 O_o 的日志不可直接比较
     （候选集大两个数量级，全量口径的指标必然低得多）。
  2. 因此必须显式排除该用户历史已见 item，O_o 的批内口径不需要（批内负样本本就不含历史）。

位置口径（设计文档 §7.4）：只在 `next_token_type == 1 且 next_action_type == 1`
（点击）的位置算指标。**两种口径都支持**，默认用前者：
  - `all_click`（默认，对齐 O_o）：**每个点击位置**一个 query —— 逐字对应
    `O_o/main.py:258` 的 `mask = (next_token_type==1) & (next_action_type==1)`。
  - `last_click`（留一法）：每个用户只取**最后一个**点击位置 t*（按位置 gather，不是取
    `[:, -1]` —— t* 可能在序列中间），与 `baseline/` 可比。

两者共用同一遍前向，`evaluate_full(..., last_click_too=True)` 会在一次遍历里把两套
指标都算出来（多出来的只是每个 query 一次候选打分，前向不重复）。

**一处与计划的实现差异（内存）**：计划书里 `evaluate_full` 先分配 `scores[n, V]` 再
逐行 `masked_fill`。V=4,783,155、n 最大 256 时那是 4.9GB 的瞬时缓冲，而它唯一的用途
是「数出比目标分高的候选个数」—— 可以分块累加而不落盘。这里改成：
    1. 目标分单独算（`tgt_score`），它**永不**被历史掩蔽（设计 §7.4「不含 target 本身」）；
    2. 候选分块累加 `greater`；
    3. 把「本该被掩蔽的候选」里的**高分者**单独数出来减掉（历史 item 与 pad 行）。
第 3 步用严格大于做判据，所以即使 target 同时出现在历史里也自洽（它等于自己的分、
不会被计入），但代码仍然显式把它排除掉，好让实现与设计文档的说法逐字对应。
结果与「先物化再掩蔽」逐位等价 —— tests/test_eval.py 里有对暴力实现的差分测试钉住这点。
"""
import contextlib

import numpy as np
import torch

import config


def hr_ndcg_from_ranks(ranks, k=10):
    """ranks 是 0-based 的排名（目标排在它前面、严格大于它的候选有几个）。

    命中 = rank < k。命中项的 NDCG 增益 = 1/log2(rank+2)。
    与 O_o 的 1-based 写法等价：O_o 的 `ranks = 1 + greater_cnt`、
    `hits10 = ranks <= 10`（⟺ greater_cnt < 10）、增益 `1/log2(ranks+1)`
    = `1/log2(greater_cnt+2)`。
    """
    # rank 是「比目标分高的候选个数」，负数没有意义 —— 一旦出现（曾经因为掩蔽扣除
    # 用了另一套 kernel 而出现），下面 `rank + 2 = 1` 会让增益变成 1/log2(1) = inf，
    # 一个 query 就能把整轮 NDCG 打成 inf。这里夹一下，别让指标层再产出非有限值。
    ranks = ranks.float().clamp_min(0)
    if ranks.numel() == 0:
        return 0.0, 0.0
    hit = (ranks < k).float()
    hr = float(hit.mean().item())
    ndcg = float((hit / torch.log2(ranks + 2)).mean().item())
    return hr, ndcg


def score_of(hr10, ndcg10):
    """O_o/main.py:289 的 0.31/0.69 加权。"""
    return 0.31 * hr10 + 0.69 * ndcg10


def _ranks_to_metrics(rank_list, n_skipped, k=10):
    """一串排名字典 -> 指标 dict。rank_list 为空表示这一口径一个 query 都没有。"""
    if not rank_list:
        return {'hr10': 0.0, 'ndcg10': 0.0, 'score': 0.0, 'n_eval': 0,
                'mean_rank': 0.0, 'n_skipped': n_skipped, 'top1': 0.0}
    ranks = torch.cat(rank_list)
    hr10, ndcg10 = hr_ndcg_from_ranks(ranks, k)
    # top1 只给过拟合测试用：它比 HR@10 严格得多，是「管线是否真的接通」的判据
    # （列序错位 / mask 写反 / 标签错位都会让它停在随机水平 ~2e-7）。
    return {'hr10': hr10, 'ndcg10': ndcg10, 'score': score_of(hr10, ndcg10),
            'n_eval': int(ranks.numel()), 'mean_rank': float(ranks.float().mean().item()),
            'n_skipped': n_skipped, 'top1': float((ranks == 0).float().mean().item())}


def _autocast(amp_dtype):
    """amp_dtype=None -> 空上下文。不用 `torch.autocast(..., dtype=None)`：
    关掉时那个 dtype 就没有意义，写死成空上下文能避免「启用但 dtype 缺失」的歧义。"""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=amp_dtype)


@torch.no_grad()
def build_candidate_matrix(model, item_feat, item_cols, chunk=8192, device="cuda",
                           amp_dtype=torch.bfloat16):
    """全量候选 embedding -> [V, H]（fp32，已 L2 归一化）。

    行 0 是 pad，特征全 0 -> 输出零向量，评测时被排除，永远不会被召回。
    权重每 epoch 变，所以每 epoch 重算一次。

    注意 H = hidden_units = 512，fp32 下矩阵是 4.78M×512×4B ≈ **9.8GB**
    （设计文档 §7.4 写的 4.9GB 是 bf16 的估算）；80GB 卡上放得下，所以存 fp32
    以保排序精度。
    """
    model.eval()
    V = item_feat.shape[0]
    out = torch.empty(V, model.hidden_units, dtype=torch.float32)
    for lo in range(0, V, chunk):
        hi = min(lo + chunk, V)
        ids = torch.arange(lo, hi, device=device, dtype=torch.long).unsqueeze(1)   # [n,1]
        feat = torch.from_numpy(np.asarray(item_feat[lo:hi])[:, item_cols])
        feat = feat.to(device).long().unsqueeze(1)                                  # [n,1,C]
        with _autocast(amp_dtype):
            emb = model(ids, item_feat=feat)
        out[lo:hi] = emb.squeeze(1).float().cpu()
    return out


def _masked_pairs(seq, token_type, hist_row, t_sel, tgt, pad_id=0):
    """每个 query 要排除的候选 id -> (row, id) 两条 [P] 长整数张量。

    排除集 = {pad_id} ∪ {t* 之前（含 t*）的 item 位置上出现过的 id} \\ {target}。
    pad 行必须排除：它的候选向量恒为零向量，目标分一旦为负，pad 就会「赢过」目标。

    **两个索引空间不要混**：`hist_row` 是**批内行号**（用来从 seq/token_type 里取历史），
    返回的 `row` 是**query 序号**（0..n-1，用来索引 q/tgt/q[m_row]）。
    两者在 evaluate_full 跳过了 target 为 pad 的用户之后就不再相同 ——
    拿批内行号去索引 tgt 会越界（跳过的那行排在最后时又恰好不越界，所以这个 bug
    只在特定排布下才炸 / 才算错，必须靠下面的回归测试钉住）。
    """
    n, S = t_sel.numel(), seq.shape[1]
    sseq = seq[hist_row]                                # [n, S]
    stt = token_type[hist_row]
    grid = torch.arange(S, device=seq.device).view(1, S)
    m = (grid <= t_sel.view(-1, 1)) & (stt == 1) & (sseq > pad_id)   # [n, S]
    row = torch.arange(n, device=seq.device).view(-1, 1).expand(-1, S)[m]   # 位置索引
    ids = sseq[m]
    # 设计 §7.4：seen「不含 target 本身」—— 目标重复出现时仍必须留在候选集里
    keep = ids != tgt[row]
    row, ids = row[keep], ids[keep]
    # pad 每行都要排掉
    row = torch.cat([torch.arange(n, device=seq.device), row])
    ids = torch.cat([torch.zeros(n, dtype=torch.long, device=seq.device), ids])
    # 去重不在这里做：ranks_full_candidates 会统一处理（见那里的注释）。
    return row, ids


@torch.no_grad()
def ranks_full_candidates(q, tgt, cand, masked_row=None, masked_id=None,
                          cand_chunk=262144, pad_id=0):
    """全量候选下每个 query 的 0-based 排名（严格大于目标分的候选个数）。

    q     [n, H]  已归一化的 query
    tgt   [n]     目标候选 id（> pad_id）
    cand  [V, H]  已归一化的全量候选矩阵（fp32）
    masked_row/masked_id  [P]  需要从候选集里排除的 (行, 候选id) 对，见 _masked_pairs。
                    允许重复、也允许出现目标自身 —— 去重与「目标不排除」都在这里统一做，
                    调用方不需要先洗干净（重复的 id 若各减一次会把 rank 减多）。
    """
    n = q.shape[0]
    V = cand.shape[0]
    if n == 0:
        return torch.zeros(0, dtype=torch.long, device=q.device)

    # 目标分：**先单独算**，从而与「是否被掩蔽」无关。它必须用原始分参与比较，
    # 否则目标恰好也在历史里时会被自己的掩蔽规则打成 -inf。
    tgt_score = (q * cand[tgt]).sum(dim=-1)                     # [n]

    # 掩蔽扣除的分必须与计数**同源**（见分块循环里的注释），所以先去重、再在循环里
    # 把这块矩阵乘的结果取出来。去重是因为历史里同一个 item 会出现多次。
    m_row = m_id = hs = None
    if masked_id is not None and masked_id.numel():
        # 目标本身永不计入排除（与设计文档的「排除集 ... \ {target}」逐字对应）。
        keep = masked_id != tgt[masked_row]
        key = torch.unique(masked_row[keep] * V + masked_id[keep])
        m_row, m_id = key // V, key % V
        if m_id.numel():
            hs = torch.empty(m_id.numel(), dtype=q.dtype, device=q.device)

    greater = torch.zeros(n, dtype=torch.float32, device=q.device)
    for lo in range(0, V, cand_chunk):
        hi = min(lo + cand_chunk, V)
        s = q @ cand[lo:hi].t()                                 # [n, c]
        # 目标自己那一列必须排除在分子之外。`tgt_score` 是按行 gather 单独算的，
        # 与分块矩阵乘里同一列可能差 1 ulp；不处理的话「目标赢过自己」会被记一笔，
        # 约一半样本 rank 因此 +1 —— 正好卡在 HR@10 的边界上，静默地把指标压低。
        # 置成与阈值相等，严格大于恒不成立。
        in_chunk = (tgt >= lo) & (tgt < hi)
        if in_chunk.any():
            rows = in_chunk.nonzero(as_tuple=False).squeeze(1)
            s[rows, tgt[rows] - lo] = tgt_score[rows]
        if m_id is not None and m_id.numel():
            # 被掩蔽候选的分从**这一块矩阵乘**里取，而不是回头 gather 一次：
            # `q @ cand.T` 与 `(q*cand[id]).sum(-1)` 是两套 kernel，fp32 下累加顺序
            # 不同、结果就有舍入差，CUDA 上开了 TF32 差得更远（实测分歧率 8.03e-05，
            # 关掉 TF32 是 2.32e-07）。同一个候选的两套分若跨过目标分，扣除会比计数
            # 多减 1 -> rank = -1 -> NDCG 里 1/log2(rank+2) = 1/log2(1) = inf，
            # 早停分从此冻在 inf 上（实测让训练在 epoch 3 假性早停）。
            # 取同一份分则恒有「扣除 <= 计数」，rank 不可能为负。
            # 取的位置在目标列被改写**之后**：掩蔽对万一撞上目标 id，取出来就等于
            # 阈值、严格大于不成立，与上面的排除一起构成双保险。
            sel = (m_id >= lo) & (m_id < hi)
            if sel.any():
                hs[sel] = s[m_row[sel], m_id[sel] - lo]
        greater += (s > tgt_score.unsqueeze(1)).sum(dim=1)

    if m_id is not None and m_id.numel():
        greater -= torch.zeros(n, dtype=torch.float32, device=q.device).scatter_add_(
            0, m_row, (hs > tgt_score[m_row]).to(torch.float32))

    return greater.round().long()


def _click_queries(click, log_feats, pos, mode, pad_id=0):
    """一批样本 -> 这一口径下的 (hist_row, t_sel, q, tgt, n_skipped)。

    `hist_row` 是**批内行号**（从 seq/token_type 里取历史用），`t_sel` 是 query 所在的
    序列位置。两者都是「每个 query 一位」，与 query 序号不是一回事：
      - `all_click` 同一行会重复出现（一个用户多个点击位置），
      - `last_click` 过滤掉无点击的行之后行号也会错位。
    `n_skipped` 数的是**目标本身是 pad**（无特征行 / reid 为 0）的 query，不是用户数。
    """
    B, S = click.shape
    if mode == "all_click":
        # nonzero 按行主序返回 -> query 按用户分组，顺序无关紧要（最后取均值）
        hist_row, t_sel = click.nonzero(as_tuple=True)
    elif mode == "last_click":
        grid = torch.arange(S, device=click.device).expand(B, S)
        t_star = torch.where(click, grid, torch.full_like(grid, -1)).max(dim=1).values
        has = t_star >= 0
        hist_row = has.nonzero(as_tuple=False).squeeze(1)
        t_sel = t_star[hist_row]
    else:
        raise ValueError("未知的取位口径 %r（只支持 all_click / last_click）" % (mode,))

    q = log_feats[hist_row, t_sel]                             # [n_q, H]
    tgt = pos[hist_row, t_sel]                                 # [n_q]
    n_skipped = int((tgt <= pad_id).sum().item())
    ok = tgt > pad_id
    if not ok.all():
        hist_row, t_sel, q, tgt = hist_row[ok], t_sel[ok], q[ok], tgt[ok]
    return hist_row, t_sel, q, tgt, n_skipped


@torch.no_grad()
def evaluate_full(model, loader, cand_mat, device="cuda", k=10,
                  amp_dtype=torch.bfloat16, cand_chunk=262144, pad_id=0,
                  all_clicks=True, last_click_too=False):
    """返回 dict(hr10, ndcg10, score, n_eval, mean_rank, n_skipped, top1)。

    all_clicks=True（默认）: 每个点击位置一个 query（对齐 O_o/main.py:258）
    all_clicks=False      : 每用户只在最后一个点击位置 t* 算一次（留一法）
    last_click_too=True   : 除了主口径，再在同一遍前向里把 t* 口径也算出来，
                            以 `*_last_click` 后缀追加返回（早停只看主口径的 score）

    没有点击位置、或目标本身是 pad（该 item 没有特征行、reid 为 0）的 query 不计入
    —— 后者会让 `n_eval` 略小于验证用户数，属于数据侧的正常缺口，不是 bug。
    """
    V = cand_mat.shape[0]
    cand = cand_mat.to(device)
    model.eval()

    modes = ["all_click" if all_clicks else "last_click"]
    if last_click_too and "last_click" not in modes:
        modes.append("last_click")
    all_ranks = {m: [] for m in modes}
    n_skipped = {m: 0 for m in modes}
    for batch in loader:
        (seq, pos, neg, token_type, next_token_type, next_action_type,
         seq_feat, pos_feat, neg_feat, seq_ts,
         neg_feat_ssl1, neg_feat_ssl2, neg_ssl1, neg_ssl2, user_feat) = batch[:15]
        user_arrays = [t.to(device, non_blocking=True) for t in batch[15:]]
        seq = seq.to(device)
        token_type = token_type.to(device)
        seq_feat = seq_feat.to(device)
        seq_ts = seq_ts.to(device)
        user_feat = user_feat.to(device)
        pos = pos.to(device)
        next_token_type = next_token_type.to(device)
        next_action_type = next_action_type.to(device)

        with _autocast(amp_dtype):
            log_feats = model(seq, token_type, user_feat, seq_feat, seq_ts, user_arrays)
        log_feats = log_feats.float()

        # 点击位置 = 「下一个 token 是 item」且「那个 action 是点击」（设计 §7.4）。
        # 两个条件都写：next_action_type==1 在数据侧已经蕴含 next_token_type==1，
        # 但显式合取才不会在数据侧语义变动时静默放宽。
        click = (next_token_type == 1) & (next_action_type == 1)

        for mode in modes:
            hist_row, t_sel, q, tgt, nsk = _click_queries(click, log_feats, pos, mode, pad_id)
            n_skipped[mode] += nsk
            if q.shape[0] == 0:
                continue
            # 传进去的 `hist_row` 是**批内行号**（取历史用），返回的 m_row 是
            # **query 序号**（索引 q/tgt 用）—— `all_click` 下同一行可能出现多个
            # query、`ok` 过滤后行号还会错位，两者不可互用（见 _masked_pairs 的注释）。
            m_row, m_id = _masked_pairs(seq, token_type, hist_row, t_sel, tgt, pad_id=pad_id)
            ranks = ranks_full_candidates(q, tgt, cand, m_row, m_id,
                                          cand_chunk=cand_chunk, pad_id=pad_id)
            all_ranks[mode].append(ranks.cpu())

    out = _ranks_to_metrics(all_ranks[modes[0]], n_skipped[modes[0]], k)
    for mode in modes[1:]:
        sub = _ranks_to_metrics(all_ranks[mode], n_skipped[mode], k)
        out.update({key + "_" + mode: value for key, value in sub.items()})
    return out
