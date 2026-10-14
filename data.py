"""序列构造：从一个用户的记录序列里产出逐位置的训练张量。

对齐 O_o/dataset.py:692 的 `__getitem__`。样本的索引空间是**用户**：
`__len__` = 用户数，一个用户一个样本（O_o/dataset.py:822）——
batch_size 是「用户数」，不是「(用户, 位置) 对数」。

**下标逻辑的正确性由 tests/test_data_layout.py 逐位断言**（金标准见 DIFF_vs_O_o.md §7）。
不要凭直觉改这里的填充顺序。
"""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import config

# 一个用户样本的字段顺序（collate_fn 和 train.py 都按这个顺序解包）。
# 前 10 项对齐 O_o 的 14 元组语义，后面 5 项是 user 侧特征与两条 SSL 视图。
SAMPLE_KEYS = (
    "seq", "pos", "neg", "token_type", "next_token_type", "next_action_type",
    "seq_feat", "pos_feat", "neg_feat", "seq_ts",
    "neg_feat_ssl1", "neg_feat_ssl2", "neg_ssl1", "neg_ssl2",
    "user_feat",
) + tuple("user_array%d" % i for i in range(len(config.USER_ARRAY)))


def build_user_sample(items, actions, times, maxlen, has_user, user_id):
    """单用户记录序列 -> 逐位置张量。逐位对齐 DIFF_vs_O_o.md §7 的四条记录走查。

    布局：右对齐、pad 在左（idx 0 是唯一的 pad 位）。
        S   = maxlen + 1 = 序列槽位数
        R   = 中间数组：R[k] 填到 idx = maxlen - k
        target 是最后一条记录的 item，只作标签、不写进 seq

    **R 的次序 = 最终序列的倒序**（填充从 idx = maxlen 往回走），这一点是下面
    `token_at` 全部下标的依据，绕在其中读「R 里谁是正序」一定会读反。

    O_o 的写法（`dataset.py:718-741`）是
        ext = reversed(user_tokens) + item_tokens    # item 块倒序，接 user 块正序
    再整条 reverse 一次写进 R —— 于是 item 块被反了两次（最终序列里时间正序）、
    user 块被反了一次（最终序列里**时间倒序**）。我们默认照抄这个结果
    （`USER_SEQ_ORDER == 'o_o'`）。

    `'chrono'` 下 user 块不再被那次多余的 reversed 影响：它在最终序列里按记录
    实际顺序（时间正序）排列，与 item 块方向一致。user 块占的槽位不变，
    受影响的只有各 user 槽位装的是哪条记录、以及 `seq_ts` 的取值 ——
    走查见 DIFF_vs_O_o.md §2.3。

    **user token 的 `seq` 装的是 `user_id`，不是 item id** —— DIFF §7 的走查里
    user 槽位写的是 `U7@400`，`seq` 是主 id 列、每个位置只装自己那一侧的实体。
    `user_id` 是 **user_embbeding 的行号**（= 用户下标 + 1），不是原始 user_id；
    item 位置上它被 `_side_emb` 的 is_item 掩码丢弃，所以取值只要不越界即可。

    「下一个 token」= idx+1 位置的 token（即 R[k-1]），k=0 的下一个是 target。

    `has_user=False`（该用户在 user_feat 里没有行，O_o 的 `if u and user_feat`）时
    一个 user token 都不产，user 槽位全留给 pad。
    """
    n = len(items)
    S = maxlen + 1
    out = {k: np.zeros(S, dtype=np.int64) for k in
           ("seq", "pos", "neg", "token_type", "next_token_type", "next_action_type",
            "seq_ts", "action900")}

    # R[k] 的语义 -> (是否 item, 记录下标)
    # 填充是「右对齐、idx 从 maxlen 递减」，所以 **R 的次序 = 最终序列的倒序**：
    # R 里正序的块，在序列里读出来是倒序，反之亦然。
    def token_at(k):
        if k < n - 1:
            return True, n - 2 - k          # item 块：R 倒序 -> 序列时间正序
        if not has_user:
            return None, None
        j = k - (n - 1)
        if config.USER_SEQ_ORDER == "o_o":
            return False, j                 # O_o：R 正序 -> 序列时间**倒序**
        return False, n - 1 - j             # chrono：R 倒序 -> 序列时间正序

    n_u = n if has_user else 0
    len_R = (n - 1) + n_u
    for k in range(min(S, len_R)):
        is_item, r = token_at(k)
        idx = maxlen - k
        out["seq"][idx] = items[r] if is_item else user_id
        out["token_type"][idx] = 1 if is_item else 2
        out["seq_ts"][idx] = times[r]
        if is_item:
            # 900：None->0, 曝光->1, 点击->2。只填 item token 位 —— 它对应的是「这条
            # 记录」的 action，user token 位上没有这个概念（恒 0）。
            out["action900"][idx] = config.action_to_900(actions[r])

        # 下一个 token：k == 0 时是被丢掉的 target，否则是刚填过的 R[k-1]
        if k == 0:
            nxt_item, nxt_r = True, n - 1
        else:
            nxt_item, nxt_r = token_at(k - 1)
        if nxt_item:
            out["next_token_type"][idx] = 1
            out["next_action_type"][idx] = config.action_to_next_action(actions[nxt_r])
            if items[nxt_r] != 0:
                out["pos"][idx] = items[nxt_r]
        elif nxt_item is False:
            out["next_token_type"][idx] = 2
    return out


# ---------------- 负采样 ----------------

def sample_neg(pool_ids, pool_cdf, hist_set, rng, tries=32, uniform_tries=256):
    """按流行度 CDF 采一个不在 hist_set 里的 item。

    复刻 O_o/dataset.py:257-277 的 `_random_neq`：先按流行度拒绝采样 32 次，
    再均匀随机试 256 次，都失败返回 (0, False)。
    返回 (item_id, ok)；ok=False 表示放弃、填 0。
    （注意：调用方把 0 当作「无负样本」，而 0 也是 pad，语义重合但两者都只意味着
    「这一位不参与」，与 O_o 一致。）
    """
    for _ in range(tries):
        r = rng.random()
        idx = int(np.searchsorted(pool_cdf, r, side="right"))
        idx = min(idx, pool_ids.size - 1)
        cand = int(pool_ids[idx])
        if cand not in hist_set:
            return cand, True
    for _ in range(uniform_tries):
        cand = int(pool_ids[rng.randint(0, pool_ids.size)])
        if cand not in hist_set:
            return cand, True
    return 0, False


# ---------------- SSL 两视图 ----------------

def make_ssl_views(feat_row, item_id, rng, mask_ratio=0.6, allow_id_mask=True):
    """对单个 item 的特征行做 rfm_no_compl 两视图增广。

    返回 (id1, id2, v1, v2, mask_id1, mask_id2)：两份**独立**抽掩蔽集合得到视图，
    以及「该视图的主 id 是否被掩」（被掩则该视图的主 id 置 0 = padding）。

    K = C 列 + __ID__ 哨兵；m = max(1, min(K, round(mask_ratio * K)))。
    两个视图的掩蔽集合各自独立抽一次 —— 共用集合会让对比任务退化成恒等映射。

    **与 O_o 的一处有意偏离**：O_o 的候选域是
    `feature_types['item_sparse'] + item_array + '__ID__'` = 17 + 0 + 1 = 18
    （item_sparse 里含 900 与两个交叉域），但那 3 个域在 `include_user=False` 的
    item 塔里**根本不参与计算**（O_o/model.py:439 只取 ITEM_SPARSE_FEAT，交叉只有
    `include_user` 时才拼），于是 O_o 的 0.6 实际只作用在 14/18 的列上、且每视图被掩的
    真实列数是个随机变量。我们直接在 C 个真实列 + __ID__ = K 个域上按同样的比例抽取
    （固定 m），语义更干净、效果等价。写进 DIFF。
    """
    C = feat_row.shape[0]
    K = C + (1 if allow_id_mask else 0)
    m = max(1, min(K, int(round(mask_ratio * K))))

    def _draw():
        who = rng.choice(np.arange(K), size=m, replace=False)
        id_sel = bool(allow_id_mask and (K - 1) in who.tolist())
        v = feat_row.copy()
        v[who[who < C]] = 0                 # 只在 < C 的域上是真实列，K-1 是 __ID__ 哨兵
        return (0 if id_sel else item_id), v, id_sel

    id1, v1, m1 = _draw()
    id2, v2, m2 = _draw()
    return id1, id2, v1, v2, m1, m2


# ---------------- Dataset ----------------

class TowerDataset(Dataset):
    """索引空间 = 用户。每个用户一个样本，内部逐位给出目标。

    只读 mmap，不复制全量数据进内存：seq 三个数组 + user_off 用 mmap_mode='r'，
    item_feat 同理（4.78M×16 int32 = 306MB，每个 worker 一份 mmap 视图即可）。
    """

    def __init__(self, maxlen, n_users, user_off, seq_item, seq_action, seq_ts,
                 item_feat, user_sparse, user_arrays, has_user, ssl=True,
                 ssl_mask_ratio=0.6, ssl_value_dropout=0.3, neg_alpha=0.0,
                 seed=20252026, indices=None):
        self.maxlen = maxlen
        self.user_off = user_off
        self.seq_item = seq_item
        self.seq_action = seq_action
        self.seq_ts = seq_ts
        self.item_feat = item_feat
        self.user_sparse = user_sparse
        self.user_arrays = list(user_arrays)
        self.has_user = has_user
        self.ssl = ssl
        self.ssl_mask_ratio = ssl_mask_ratio
        self.ssl_value_dropout = ssl_value_dropout
        self.indices = np.arange(n_users) if indices is None else np.asarray(indices)

        # 负采样池：全量出现次数（= logq 用的那份 count）** neg_alpha。
        # alpha=0 时退化为均匀分布，与 O_o 的默认 neg_pop_alpha=0 一致。
        #
        # **池子只装出现过的 item**（counts > 0）：`np.maximum(counts, 1)` 那种写法会
        # 让全量 4.78M 个从未出现过的 item 也拿到权重，与 logQ 的语义直接冲突
        # （它们的 logQ = log(1e-12)），而且 item_feat 之外的行在主 id 上仍是合法的，
        # 会静默污染负样本。过滤之后 counts 恒 > 0，power 也不需要 floor。
        counts = np.bincount(np.asarray(seq_item), minlength=config.VOCAB_SIZE)[:config.VOCAB_SIZE]
        self.neg_pool_ids = np.flatnonzero(counts > 0).astype(np.int32)
        w = np.power(counts[self.neg_pool_ids].astype(np.float64), float(neg_alpha))
        self.neg_pool_cdf = np.cumsum(w) / w.sum()
        self.seed = seed
        # 每个 worker 进程都会 fork 出自己的一份，按种子 + 用户号派生，保证可复现
        self.item_cols = np.array([config.ITEM_STATIC_INDEX[c]
                                   for c in config.ITEM_SPARSE_ITEM], dtype=np.int64)
        self.raw_cols = np.array([config.ITEM_STATIC_INDEX[c]
                                  for c in config.ITEM_RAW], dtype=np.int64)
        self.cross_cols = np.array([config.ITEM_STATIC_INDEX[c]
                                    for c in config.ITEM_CROSS], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        u = int(self.indices[i])
        lo, hi = int(self.user_off[u]), int(self.user_off[u + 1])
        # np.asarray(memmap) 返回的是同一个 mmap 视图，不是拷贝
        items = np.asarray(self.seq_item[lo:hi])
        acts = np.asarray(self.seq_action[lo:hi])
        tss = np.asarray(self.seq_ts[lo:hi])
        has_user = bool(self.has_user[u])

        rng = np.random.RandomState((self.seed + u * 1000003) % (2 ** 31 - 1))
        # user token 的主 id 用**行号 u+1**（user_embbeding 的索引空间），不是原始 user_id
        s = build_user_sample(items, acts, tss, self.maxlen, has_user, u + 1)
        S = self.maxlen + 1
        valid_item_pos = (s["next_token_type"] == 1)

        hist_set = set(int(x) for x in items.tolist() if x > 0)
        for idx in np.flatnonzero(valid_item_pos):
            tid = int(s["pos"][idx])
            if tid != 0:
                neg, _ok = sample_neg(self.neg_pool_ids, self.neg_pool_cdf, hist_set, rng)
                s["neg"][idx] = neg

        # ---- item 侧特征表：三个位置各自 gather ----
        itf = self.item_feat                                 # mmap [V, 16]

        # item 塔 14 列：13 原始 + 901（不含 900 与交叉）
        s["pos_feat"] = itf[s["pos"]][:, self.item_cols]
        s["neg_feat"] = itf[s["neg"]][:, self.item_cols]

        # 序列塔 item 侧 17 列，顺序 = config.ITEM_SEQ_FEAT_COLS
        #   = 13 原始 + 900 + 901 + 116_118 + 118_120
        # 900 是 record 级属性，item_feat 里没有它 —— 由 build_user_sample 逐 token 填好，
        # 这里只负责插到第 13 列的位置。千万不要在 __getitem__ 里反推下标。
        n_raw = self.raw_cols.size
        seq_feat = np.empty((S, len(config.ITEM_SEQ_FEAT_COLS)), dtype=np.int64)
        seq_feat[:, :n_raw] = itf[s["seq"]][:, self.raw_cols]
        seq_feat[:, n_raw] = s["action900"]                  # 900
        seq_feat[:, n_raw + 1] = itf[s["seq"]][:, config.ITEM_STATIC_INDEX[config.CLICK_FEAT]]
        seq_feat[:, n_raw + 2:] = itf[s["seq"]][:, self.cross_cols]
        s["seq_feat"] = seq_feat

        # ---- SSL 两视图：只在有效 neg 位置上做 ----
        if self.ssl:
            id1, id2 = np.zeros(S, dtype=np.int64), np.zeros(S, dtype=np.int64)
            v1 = np.zeros((S, self.item_cols.size), dtype=np.int64)
            v2 = np.zeros((S, self.item_cols.size), dtype=np.int64)
            for idx in np.flatnonzero(valid_item_pos):
                a, b, f1, f2, _m1, _m2 = make_ssl_views(
                    s["neg_feat"][idx].copy(), int(s["neg"][idx]), rng,
                    mask_ratio=self.ssl_mask_ratio)
                id1[idx], id2[idx], v1[idx], v2[idx] = a, b, f1, f2
            s["neg_ssl1"], s["neg_ssl2"] = id1, id2
            s["neg_feat_ssl1"], s["neg_feat_ssl2"] = v1, v2
        else:
            s["neg_ssl1"] = s["neg_ssl2"] = s["neg"].copy()
            s["neg_feat_ssl1"] = s["neg_feat_ssl2"] = s["neg_feat"].copy()

        # ---- user 侧特征：整行广播到 S 个位置 ----
        # 广播而不是只在 is_user 位置填：模型内部按 is_user 掩蔽后再查表（_side_emb），
        # 语义与 O_o 的 `_full_template.copy()` 一致。
        s["user_feat"] = np.repeat(np.asarray(self.user_sparse[u + 1])[None, :], S, axis=0)
        for j, arr in enumerate(self.user_arrays):
            s["user_array%d" % j] = np.repeat(np.asarray(arr[u + 1])[None, :], S, axis=0)

        return tuple(np.ascontiguousarray(s[k]) for k in SAMPLE_KEYS)


def make_collate():
    """把一批 numpy 样本堆成 torch.long 张量。字段顺序 = SAMPLE_KEYS。"""
    def _collate(batch):
        cols = list(zip(*batch))
        return tuple(torch.from_numpy(np.stack(c)).long() for c in cols)
    return _collate


def worker_init_fn(worker_id):
    """按 O_o 的做法从 torch.initial_seed() 派生 numpy / random 种子，保证可复现。"""
    s = int(torch.initial_seed() % (2 ** 31 - 1))
    np.random.seed(s)
    random.seed(s)


def build_loaders(maxlen=101, batch_size=256, num_workers=12, seed=20252026,
                  cache_dir=None, val_ratio=0.1, ssl_mask_ratio=0.6,
                  ssl_value_dropout=0.3, neg_alpha=0.0):
    """组装 train / val DataLoader。

    **划分比例是 9:1**（`val_ratio=0.1`）。抽样方式照 O_o/main.py:334-345：
    `val_count = max(1, round(N * 0.1))` 且
    `np.random.RandomState(seed).choice(N, val_count, replace=False)`，
    训练集取补集，两边用户严格不相交。实测 `N=1,001,845` -> 验证 **100,184** /
    训练 **901,661**（正好 9.0000:1），每 epoch 3,523 步（batch 256）。

    与 O_o 的有意偏离：O_o 用 `round(N * 0.01)`（1%），验证用户只有 1 万，早停
    指标的标准误大、容易在第三位小数上抖；9:1 把它压到 ~1/3.2，代价是训练用户少
    9%、每 epoch 评测耗时涨 10 倍（见 DIFF_vs_O_o.md §6.2 第 11 条）。

    注意验证集**不是**干净的 held-out 测试集：`logq.npy` / 901 点击桶 / 负采样池都
    按整份 `seq`（含验证用户）统计（设计 §1.1 的决策，照抄 O_o），item 特征表也覆盖
    全部 item。所以这是同分布的验证指标，不是零泄漏的泛化估计。要后者得另留一份
    训练期从不接触的测试集（8:1:1）。
    """
    cache = Path(cache_dir or config.CACHE_DIR)
    meta = config.load_meta(cache)
    N = meta["num_users"]
    rng = np.random.RandomState(seed)
    val_count = max(1, int(round(N * val_ratio)))
    val_idx = rng.choice(np.arange(N), size=val_count, replace=False)
    is_val = np.zeros(N, dtype=bool)
    is_val[val_idx] = True
    train_idx = np.flatnonzero(~is_val)

    def _load(name):
        return np.load(cache / name, mmap_mode="r")

    seq_item = _load("seq_item.npy")
    seq_action = _load("seq_action.npy")
    seq_ts = _load("seq_ts.npy")
    user_off = _load("user_off.npy")
    item_feat = _load("item_feat.npy")
    user_sparse = _load("user_sparse.npy")
    has_user = _load("user_has.npy")
    user_arrays = [_load("user_array_%s.npy" % f) for f in config.USER_ARRAY]

    def _ds(indices, ssl):
        return TowerDataset(
            maxlen=maxlen, n_users=N, user_off=user_off, seq_item=seq_item,
            seq_action=seq_action, seq_ts=seq_ts, item_feat=item_feat,
            user_sparse=user_sparse, user_arrays=user_arrays, has_user=has_user,
            ssl=ssl, ssl_mask_ratio=ssl_mask_ratio,
            ssl_value_dropout=ssl_value_dropout, neg_alpha=neg_alpha,
            seed=seed, indices=indices)

    def _loader(ds, shuffle, workers):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                          collate_fn=make_collate(), pin_memory=True, drop_last=False,
                          persistent_workers=(workers > 0),
                          worker_init_fn=worker_init_fn)

    return {
        "train": _loader(_ds(train_idx, True), True, num_workers),
        "val": _loader(_ds(val_idx, False), False, min(4, num_workers)),
        "meta": meta,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }
