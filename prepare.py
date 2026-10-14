"""从 HF arrow 缓存构造 tower 训练所需的稠密缓存。

产出（taac/cache/）：
    seq_item.npy    int32 [总记录数]           全量 seq 拍平后的 item id
    seq_action.npy  int8  [总记录数]           0=曝光 1=点击 2=缺失
    seq_ts.npy      int64 [总记录数]
    user_off.npy    int64 [users+1]            CSR offsets（样本索引 = 行号）
    user_ids.npy    int64 [users]              原始 user_id，用于 join user_feat
    user_has.npy    bool  [users+1]            该用户在 user_feat 里是否有行（行 0 = pad 恒 False）
                                               —— O_o 的 `if u and user_feat` 判据
    item_feat.npy   int32 [VOCAB_SIZE, 16]     静态 item 特征
    user_sparse.npy int32 [users+1, 4]
    user_array_<fid>.npy int32 [users+1, L_fid]
    logq.npy        float32 [VOCAB_SIZE]
    meta.json       词表规模、L_fid、覆盖率自检结果、参数量估算

为什么不 load_dataset：见 baseline/prepare_data.py 的模块 docstring —— 会跑一遍
fingerprint/缓存构建，1.8G 的 seq 要几分钟且要额外磁盘，而根分区只剩 70G。
"""
import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import config


def assign_dense_ids(values):
    """原始值 -> 1 起的稠密 id，按出现频次降序（高频拿小 id）。

    0 与负数一律跳过（0 是 pad/缺失）。返回 (mapping, size)，表大小要 +1 给 pad。

    注意 values 常是 float64：arrow 的 nullable int64 列经 to_numpy 后带 NaN（本数据集里
    特征 115 有 3.25M 个 null）。NaN 与任何值比较都是 False，所以 `values > 0` 天然把它滤掉。
    """
    values = np.asarray(values).ravel()
    values = values[values > 0]
    if values.size == 0:
        return {}, 0
    uniq, counts = np.unique(values, return_counts=True)
    order = np.argsort(-counts, kind="stable")   # 同频次时按值升序，保证可复现
    mapping = {int(uniq[j]): i + 1 for i, j in enumerate(order)}
    return mapping, len(mapping)


def remap(values, mapping):
    """把原始值数组按 mapping 映射为稠密 id；0/缺失 -> 0。

    **必须向量化**：item 的特征 121 有 ~200 万个不同取值，逐 key 做 `out[values == raw]`
    是 O(unique × rows) ≈ 1e13 次比较，永远跑不完。这里改成 searchsorted O(n log m)。
    """
    values = np.asarray(values)
    out = np.zeros(values.size, dtype=np.int32)
    if mapping:
        keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
        vals = np.fromiter(mapping.values(), dtype=np.int32, count=len(mapping))
        order = np.argsort(keys, kind="stable")
        keys, vals = keys[order], vals[order]
        flat = values.ravel().astype(np.float64)
        pos = np.clip(np.searchsorted(keys, flat), 0, keys.size - 1)
        # NaN 会落在末尾，且 keys[pos] != NaN 恒成立 -> 落 0，正是「缺失」的语义
        out = np.where(keys[pos] == flat, vals[pos], 0).astype(np.int32)
    return out.reshape(values.shape)


def click_bucket(c):
    """O_o/dataset.py:433 的分桶，但 c<=0 返回 0。

    O_o 的 `_click_count_to_bucket` 在 c<=0 时返回 1，**但调用点**
    （dataset.py:938/941）写的是 `bucket(cnt) if cnt > 0 else 0`，所以「0 次点击」
    实际落到 0（=pad）。这里按调用点的语义实现。
    """
    if c <= 0:
        return 0
    b = int(math.floor(math.log2(c))) + 1
    return int(max(1, min(config.CLICK_BUCKETS, b)))


# ---------------- arrow 读取 ----------------

def iter_seq_users(glob_pattern=config.SEQ_ARROW_GLOB):
    """逐用户 yield (user_id, item_ids, actions, times)。

    照 baseline/prepare_data.py:52 的做法：arrow 流式读、把 list<struct> 展平再按
    offsets 切回。**与 baseline 的唯一区别**：不把 action_type 的 null 填成 0 ——
    我们要区分「曝光 0」与「缺失 None」，因为 900 的映射不同（None->0，曝光->1）。
    """
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError("没找到 seq arrow: %s" % glob_pattern)
    for f in files:
        reader = pa.ipc.open_stream(pa.memory_map(f))
        for batch in reader:
            col = batch.column("seq")
            user_ids = batch.column("user_id").to_numpy(zero_copy_only=False)
            lengths = pc.fill_null(pc.list_value_length(col), 0).to_numpy(zero_copy_only=False)
            flat = pc.list_flatten(col)
            item_ids = pc.fill_null(flat.field("item_id"), 0).to_numpy(zero_copy_only=False)
            # null -> -1 -> ACT_NONE(2)
            acts = pc.cast(pc.fill_null(flat.field("action_type"), -1), pa.int8())
            acts = acts.to_numpy(zero_copy_only=False)
            # pyarrow 给的是只读视图，不能就地写；-1 -> ACT_NONE(2)
            acts = np.where(acts < 0, np.int8(config.ACT_NONE), acts).astype(np.int8)
            times = pc.fill_null(flat.field("timestamp"), 0).to_numpy(zero_copy_only=False)

            offs = np.zeros(len(lengths) + 1, dtype=np.int64)
            np.cumsum(lengths, out=offs[1:])
            for i in range(len(lengths)):
                yield (int(user_ids[i]),
                       item_ids[offs[i]:offs[i + 1]],
                       acts[offs[i]:offs[i + 1]],
                       times[offs[i]:offs[i + 1]])


def _read_arrow_columns(glob_pattern, columns):
    """把一个 arrow 文件的若干列读成 {列名: np.ndarray}。

    4,783,154 行 × 13 列（nullable int64 经 to_numpy 变 float64）≈ 500MB，
    一次性放进内存没问题（本机 1.5T）。
    """
    files = sorted(glob.glob(glob_pattern))
    if not files:
        raise FileNotFoundError("没找到 arrow: %s" % glob_pattern)
    assert len(files) == 1, "item_feat/user_feat 预期只有 1 个 arrow，实际 %d 个" % len(files)
    reader = pa.ipc.open_stream(pa.memory_map(files[0]))
    parts = {c: [] for c in columns}
    for batch in reader:
        for c in columns:
            parts[c].append(batch.column(c).to_numpy(zero_copy_only=False))
    return {c: np.concatenate(v) for c, v in parts.items()}


def _as_int64(values):
    """原始值 -> int64，缺失(NaN) -> 0。用于交叉特征的打包。"""
    a = np.asarray(values)
    if a.dtype.kind == "f":
        a = np.nan_to_num(a, nan=0.0)
    return a.astype(np.int64)


def build_item_feat(seq_item, seq_action):
    """静态 item 特征矩阵 [VOCAB_SIZE, 16] int32 与词表规模。

    900 是 record 级属性，**不在这里** —— 它在 data.py 里逐 token 从 seq_action 取。
    """
    raw = _read_arrow_columns(config.ITEM_FEAT_ARROW_GLOB,
                              ['item_id'] + config.ITEM_RAW)
    ids = _as_int64(raw['item_id'])
    assert ids.min() >= 1 and ids.max() < config.VOCAB_SIZE, "item_id 越界"
    assert len(np.unique(ids)) == len(ids), "item_id 有重复"

    out = np.zeros((config.VOCAB_SIZE, len(config.ITEM_STATIC_COLS)), dtype=np.int32)
    feat_vocab = {}
    for f in config.ITEM_RAW:
        mapping, size = assign_dense_ids(raw[f])
        feat_vocab[f] = size
        out[ids, config.ITEM_STATIC_INDEX[f]] = remap(raw[f], mapping)
        print("  [item] %-4s vocab=%d" % (f, size), flush=True)

    # 901：item 的「点击次数」分桶（O_o/dataset.py:425 统计的是 a == 1，不是出现次数）
    click_counts = np.bincount(seq_item[seq_action == config.ACT_CLICK],
                               minlength=config.VOCAB_SIZE)[:config.VOCAB_SIZE]
    bucket = np.zeros(config.VOCAB_SIZE, dtype=np.int32)
    nz = np.flatnonzero(click_counts > 0)
    for i in nz:
        bucket[i] = click_bucket(int(click_counts[i]))
    out[:, config.ITEM_STATIC_INDEX[config.CLICK_FEAT]] = bucket
    feat_vocab[config.CLICK_FEAT] = config.CLICK_BUCKETS
    print("  [item] %-4s vocab=%d (被点击过的 item: %d)"
          % (config.CLICK_FEAT, config.CLICK_BUCKETS, nz.size), flush=True)

    # 交叉特征：全表构造（O_o 只覆盖数据里出现过的组合 -> 见 spec §10 第 3 条差异）
    for (a, b), key in zip(config.CROSS_SPECS, config.ITEM_CROSS):
        packed = _as_int64(raw[a]) * (config.VOCAB_SIZE + 1) + _as_int64(raw[b])
        mapping, size = assign_dense_ids(packed)
        feat_vocab[key] = size
        out[ids, config.ITEM_STATIC_INDEX[key]] = remap(packed, mapping)
        print("  [cross] %-9s vocab=%d" % (key, size), flush=True)

    # 900 是 record 级、不由本表承载，但 tower 的 item_item_list 里占一列，
    # 词表规模（= 最大 id 2）必须在这里登记，tower_kwargs 才能统一 +1 建表。
    feat_vocab[config.ACTION_FEAT] = config.ACTION_MAX_ID
    return out, feat_vocab


def build_user_feat(user_ids):
    """user 侧特征表，按 **seq 缓存的行号** 索引（不是 user_id）。

    返回 (sparse [N+1, 4], arrays {fid: [N+1, L_fid]}, feat_vocab, has_user [N+1])

    `has_user` 对应 O_o/dataset.py:709 的 `if u and user_feat:` —— user_feat 与 seq 的
    用户集合互有缺口（user_feat 100,315 行 vs seq 100,014 用户），缺口处的用户
    **一个 user token 都不产**。这个标志必须显式存，不能靠「特征全 0」反推
    （某个用户完全可能合法地全 0）。
    """
    raw = _read_arrow_columns(config.USER_FEAT_ARROW_GLOB,
                              ['user_id'] + config.USER_FEAT_COLS)
    seq_uids = _as_int64(raw['user_id'])
    row_of = {int(u): i + 1 for i, u in enumerate(user_ids)}   # 行 0 留给 pad
    N = len(user_ids)
    # 注意：判据是「该 user_id 在 **user_feat 文件** 里有没有行」，不是「在 row_of 里」——
    # row_of 是从 user_ids 建的，那样写恒为 True。
    has_user = np.zeros(N + 1, dtype=bool)
    has_user[1:] = np.isin(np.asarray(user_ids), seq_uids, assume_unique=False)
    n_missing = int((~has_user[1:]).sum())
    print("[user_feat] 有行 %d / %d，缺口 %d" % (N - n_missing, N, n_missing), flush=True)

    sparse = np.zeros((N + 1, len(config.USER_SPARSE)), dtype=np.int32)
    arrays, feat_vocab = {}, {}
    for f in config.USER_SPARSE:
        mapping, size = assign_dense_ids(raw[f])
        feat_vocab[f] = size
        col = remap(raw[f], mapping)
        for uid, v in zip(seq_uids, col):
            r = row_of.get(int(uid))
            if r is not None:
                sparse[r, config.USER_SPARSE.index(f)] = int(v)
        print("  [user] %-4s vocab=%d" % (f, size), flush=True)
    for f in config.USER_ARRAY:
        # 多值列：每行是 list；先求最长，再定长填充（超长截断，不足补 0）
        vals = raw[f]
        L = int(max((len(v) for v in vals if v is not None), default=1)) or 1
        flat = np.array([x for v in vals if v is not None for x in v], dtype=np.int64)
        mapping, size = assign_dense_ids(flat)
        feat_vocab[f] = size
        arr = np.zeros((N + 1, L), dtype=np.int32)
        for uid, v in zip(seq_uids, vals):
            r = row_of.get(int(uid))
            if r is None or v is None:
                continue
            for j, x in enumerate(v[:L]):
                arr[r, j] = mapping.get(int(x), 0)
        arrays[f] = arr
        print("  [user_array] %s: L=%d vocab=%d" % (f, L, size), flush=True)
    return sparse, arrays, feat_vocab, has_user


def build_logq(seq_item):
    """logQ = log(count / total)，count 用**出现次数**（曝光+点击都算）。

    Q 是「item 被采成负样本的概率」，与它作为正/负样本无关（baseline/data.py:25 同口径）。
    q[0] clamp 到 1e-12 -> log 后约 -27.6，O_o/main.py:447 如此。
    """
    counts = np.bincount(seq_item, minlength=config.VOCAB_SIZE)[:config.VOCAB_SIZE]
    counts = np.maximum(counts, 1).astype(np.float64)
    q = counts / counts.sum()
    q[0] = 1e-12
    return np.log(q).astype(np.float32)


# ---------------- 主流程 ----------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default=str(config.CACHE_DIR))
    p.add_argument("--force", action="store_true", help="已有缓存时也重建")
    p.add_argument("--check", action="store_true", help="只做自检，不写盘")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.check:
        check(out)
        return
    if (out / "meta.json").exists() and not args.force:
        print("[skip] 缓存已存在: %s  (--force 可重建)" % (out / "meta.json"))
        return

    t0 = time.time()
    # ---- seq：一遍扫完，同时拍平成 CSR ----
    items_l, acts_l, ts_l, offs, user_ids = [], [], [], [0], []
    for uid, ids, acts, tss in iter_seq_users():
        user_ids.append(uid)
        items_l.append(ids.astype(np.int32))
        acts_l.append(acts.astype(np.int8))
        ts_l.append(tss.astype(np.int64))
        offs.append(offs[-1] + len(ids))
        if len(user_ids) % 20_000 == 0:
            print("  seq %d users, %d records" % (len(user_ids), offs[-1]), flush=True)
    seq_item = np.concatenate(items_l)
    seq_action = np.concatenate(acts_l)
    seq_ts = np.concatenate(ts_l)
    user_off = np.asarray(offs, dtype=np.int64)
    user_ids = np.asarray(user_ids, dtype=np.int64)
    print("[seq] users=%d records=%d time=%.1fs"
          % (len(user_ids), len(seq_item), time.time() - t0), flush=True)

    # ---- item / user 特征 ----
    item_feat, feat_vocab = build_item_feat(seq_item, seq_action)
    user_sparse, user_arrays, uv, user_has = build_user_feat(user_ids)
    feat_vocab.update(uv)
    logq = build_logq(seq_item)

    # ---- 落盘 ----
    np.save(out / "seq_item.npy", seq_item)
    np.save(out / "seq_action.npy", seq_action)
    np.save(out / "seq_ts.npy", seq_ts)
    np.save(out / "user_off.npy", user_off)
    np.save(out / "user_ids.npy", user_ids)
    np.save(out / "user_has.npy", user_has)
    np.save(out / "item_feat.npy", item_feat)
    np.save(out / "user_sparse.npy", user_sparse)
    for f, arr in user_arrays.items():
        np.save(out / ("user_array_%s.npy" % f), arr)
    np.save(out / "logq.npy", logq)

    meta = {
        "num_users": len(user_ids),
        "num_records": int(len(seq_item)),
        "vocab_size": config.VOCAB_SIZE,
        "feat_vocab": feat_vocab,
        "user_array_lens": {f: int(a.shape[1]) for f, a in user_arrays.items()},
        "item_static_cols": config.ITEM_STATIC_COLS,
        "action_counts": {str(k): int(v) for k, v in
                          zip(*np.unique(seq_action, return_counts=True))},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("[done] %s" % (out / "meta.json"), flush=True)
    check(out)


def check(out):
    """自检：任一条不过就抛异常。训练前必跑。"""
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    item_feat = np.load(out / "item_feat.npy", mmap_mode="r")
    user_off = np.load(out / "user_off.npy")
    seq_item = np.load(out / "seq_item.npy", mmap_mode="r")
    feat_vocab = meta["feat_vocab"]

    # 1. 词表规模必须为正（0 只留给 pad/缺失）—— 由 assign_dense_ids 保证，这里复核
    assert all(v > 0 for v in feat_vocab.values()), "有特征词表规模为 0"

    # 2. 静态特征矩阵每列都必须落在对应表内（越界是最容易静默出错的地方）
    for f in config.ITEM_STATIC_COLS:
        col = np.asarray(item_feat[:, config.ITEM_STATIC_INDEX[f]])
        n = config.CLICK_BUCKETS if f == config.CLICK_FEAT else feat_vocab[f]
        assert col.max() <= n, "%s 越界: max=%d 表大小=%d" % (f, col.max(), n)

    # 3. seq 里 item 的特征覆盖率（实测 100%，仍要断言）
    present = np.zeros(config.VOCAB_SIZE, dtype=bool)
    present[1:] = np.any(np.asarray(item_feat[1:]) != 0, axis=1)
    uniq = np.unique(np.asarray(seq_item))
    uniq = uniq[uniq > 0]
    cov = present[uniq].mean()
    assert cov == 1.0, "seq item 特征覆盖率 %.6f != 1.0" % cov

    # 4. CSR 自洽
    assert np.all(np.diff(user_off) >= 0), "user_off 非单调"
    assert user_off[-1] == meta["num_records"], "user_off 末位 != 记录数"

    # 4b. user_feat 缺口：这些用户一个 user token 都不产（O_o 的 `if u and user_feat`）。
    #     不是错误，但必须显式知道有多少、并让 data.py 拿到正确的 has_user。
    user_has = np.load(out / "user_has.npy")
    assert user_has.shape[0] == user_off.shape[0], "user_has 长度 != users+1"
    assert not user_has[0], "user_has[0] 必须是 False（pad 行）"
    print("[check] user_feat 缺口: %d / %d 个用户无 user token"
          % (int((~user_has[1:]).sum()), len(user_has) - 1))

    # 5. 参数量与显存估算（训练前的门槛）
    emb, hidden = 128, 512
    n = config.VOCAB_SIZE * emb                                  # item_emb
    n += sum(v + 1 for f, v in feat_vocab.items()) * emb         # 稀疏表
    n += (meta["num_users"] + 2) * emb                           # user_emb
    n += (9 + 18 + 3 + 15) * hidden * hidden * 4 // 4            # 两条 DNN（粗估）
    n += 8 * 4 * hidden * hidden * 4 // 4                        # 8 个 HSTU 块（粗估）
    print("[check] OK  参数≈%.2fB  权重 fp32≈%.1fGB  含 AdamW≈%.1fGB"
          % (n / 1e9, n * 4 / 1e9, n * 16 / 1e9))
    print("[check] 各表规模: %s" % json.dumps(feat_vocab, ensure_ascii=False))


if __name__ == "__main__":
    main()
