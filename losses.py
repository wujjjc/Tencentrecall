"""InfoNCE（+ -logQ 修正）与 SSL 对比损失，逐字对齐 O_o/main.py:100-225。

本模块只依赖 torch：不 import model / data / config，这样它能被单测直接驱动，
也不会把模型的构造依赖带进损失测试。

InfoNCE 的负样本构成（O_o/main.py:116 的注释）：
    分母 = 正样本 1 列 + 批内易负样本 M 列 + 全量难负样本 M 列（**不子采样**）
    难负样本 = 全 batch 内除本序列以外的所有正样本（按 seq_idx 屏蔽同序列列）
损失侧张量因此是 [M, 1+2M]，内存随 batch **平方**增长（设计文档 §7.6）——
batch_size 调到 512 以上时这里是第一处会 OOM 的地方。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce(pos_embs, neg_embs, log_feats, temperature, next_token_type,
             next_action_type, pos_ids, neg_ids, item_logQ,
             click_scale=1.0, exp_scale=0.1, filter_current_only=True):
    """对齐 O_o/main.py:100-207。参数与返回同它，额外返回 sample_weight 便于单测。

    pos_ids / neg_ids / item_logQ 都是**必需**的：-logQ 修正不是可选项，
    它把「批内均匀负采样」的分布修回「按流行度采样」的目标分布。缺了它
    InfoNCE 会系统性地低估热门 item —— 静默变差，不报错，所以用从 O_o 抄来的断言拦住。
    """
    assert temperature > 0.0, "temperature must be > 0"
    assert item_logQ is not None, "item_logQ is required for -logQ correction"
    assert pos_ids is not None and neg_ids is not None, \
        "pos_ids and neg_ids are required for -logQ correction"

    mask = (next_token_type == 1)
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    Qn = log_feats[mask]          # [M, D]
    Kpn = pos_embs[mask]          # [M, D]
    Knegn = neg_embs[mask]        # [M, D]
    act = next_action_type[mask]  # [M]

    device = Qn.device
    M = Qn.size(0)
    if M == 0:
        return torch.zeros((), device=device), {
            'mean_pos_sim': 0.0, 'mean_neg_sim': 0.0, 'mean_hard_neg_sim': 0.0,
            'mask_rate_easy': 0.0, 'mask_rate_hard': 0.0,
            'sample_weight': torch.zeros(0, device=device),
        }

    pos_ids_flat = pos_ids[mask].long()   # [M]
    neg_ids_pool = neg_ids[mask].long()   # [M]

    # 正样本 logits：sim/T - logQ[pos]
    pos_sim = (Qn * Kpn).sum(dim=-1, keepdim=True)                        # [M,1]
    logQ_pos = item_logQ[pos_ids_flat].to(device=device, dtype=pos_sim.dtype).unsqueeze(1)
    pos_logits = (pos_sim / temperature) - logQ_pos                       # [M,1]

    # 批内「易负样本」池 logits：sim/T - logQ[neg_column]
    neg_sim = Qn @ Knegn.t()                                              # [M,M]
    logQ_cols = item_logQ[neg_ids_pool].to(device=device, dtype=neg_sim.dtype)
    neg_logits = (neg_sim / temperature) - logQ_cols.unsqueeze(0)         # [M,M]

    # 点击/曝光样本级权重。**必须归一化到和为 1**：否则 click_scale 就是个全局
    # 学习率乘子，改它等于偷偷改 lr（O_o 在这点上踩过坑）。
    act_f = act.to(Qn.dtype)
    sample_weight = torch.where(
        act_f > 0.5, torch.full_like(act_f, click_scale), torch.full_like(act_f, exp_scale))
    sample_weight = sample_weight / sample_weight.sum().clamp_min(1e-12)

    # 屏蔽「易负样本」里与本位正样本同 id 的列
    neg_large = torch.tensor(-1e9, device=device, dtype=pos_logits.dtype)
    invalid_easy = torch.zeros((M, M), dtype=torch.bool, device=device)
    if filter_current_only:
        invalid_easy = pos_ids_flat.view(M, 1).eq(neg_ids_pool.view(1, M))
        if invalid_easy.any():
            neg_logits = neg_logits.masked_fill(invalid_easy, neg_large)

    # ==================== 全量「难负样本」= 其他序列的正样本 ====================
    # 找到每个被计算位置对应的 batch 序列索引（第 0 维）
    idxs = mask.nonzero(as_tuple=False)     # [M, 2] -> (b, l)
    seq_idx = idxs[:, 0]                    # [M]

    hard_neg_sim_full = Qn @ Kpn.t()                                    # [M,M]
    logQ_hard_cols = item_logQ[pos_ids_flat].to(device=device, dtype=hard_neg_sim_full.dtype)
    hard_logits = (hard_neg_sim_full / temperature) - logQ_hard_cols.unsqueeze(0)

    # 屏蔽「同一序列」的列（含自身）—— 同序列的其他正样本是「同一个用户的下一条
    # 历史」，拿它当负样本等于让模型把自己跟自己的未来对比。
    invalid_hard = seq_idx.view(M, 1).eq(seq_idx.view(1, M))            # True = 屏蔽
    if invalid_hard.any():
        hard_logits = hard_logits.masked_fill(invalid_hard, neg_large)
    mask_rate_hard = float(invalid_hard.float().mean().item())

    # 拼接分母：正样本 + 易负样本 + 难负样本（全量）
    logits = torch.cat([pos_logits, neg_logits, hard_logits], dim=1)     # [M, 1+M+M]
    labels = torch.zeros(M, dtype=torch.long, device=device)

    per_example_loss = F.cross_entropy(logits, labels, reduction='none')  # [M]
    loss = (per_example_loss * sample_weight).sum()

    if (~invalid_hard).any():
        mean_hard_neg_sim = float(hard_neg_sim_full[~invalid_hard].mean().item())
    else:
        mean_hard_neg_sim = 0.0

    stats = {
        'mean_pos_sim': float(pos_sim.mean().item()),
        'mean_neg_sim': float(neg_sim.mean().item()),
        'mean_hard_neg_sim': mean_hard_neg_sim,
        'mask_rate_easy': float(invalid_easy.float().mean().item()),
        'mask_rate_hard': mask_rate_hard,
        'sample_weight': sample_weight.detach(),
    }
    return loss, stats


def ssl_loss(z1, z2, temperature):
    """对齐 O_o/main.py:210-225。

    **明确 .float()** —— O_o 的 InfoNCE 不转、SSL 转，照抄。bf16 下 cross_entropy
    的 logits 若保持 bf16，归一化后的相似度只有 8 位有效位，softmax 会因精度不足
    而失真；InfoNCE 那边因为分母列多、量级大，反而不敢随便升精度（显存）。
    """
    assert temperature > 0.0
    if z1.numel() == 0 or z2.numel() == 0:
        return torch.zeros((), device=z1.device)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    M = z1.size(0)
    labels = torch.arange(M, device=z1.device)
    logits12 = (z1 @ z2.t()) / float(temperature)
    logits21 = (z2 @ z1.t()) / float(temperature)
    loss12 = F.cross_entropy(logits12.float(), labels, reduction='mean')
    loss21 = F.cross_entropy(logits21.float(), labels, reduction='mean')
    return 0.5 * (loss12 + loss21)


def init_weights(m):
    """对齐 O_o/main.py:83-95。

    padding 行只对**声明了 padding_idx 的表**清零：`hstu.time` 是一张
    nn.Embedding(129, 1) 但 padding_idx=None，它第 0 行是「对角线自连接」这个真实
    可学习参数（model.py:149 手工 normal_(std=0.02)），不能被清零。

    因为这里给 Embedding 的统一初始化就是 N(0, 0.02)，与那行手写初始化**完全一致**，
    所以 apply 之后不需要再单独排除 self.time；也不需要在外面额外做一遍
    「把 item_emb[0]/user_emb[0]/sparse_emb[k][0] 清零」——本函数的 padding_idx
    分支已经覆盖，且我们的每张表都声明了 padding_idx=0。
    """
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.padding_idx is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, getattr(nn, 'RMSNorm', ()))):
        if getattr(m, 'weight', None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
