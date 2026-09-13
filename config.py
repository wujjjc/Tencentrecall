"""tower 数据/训练管线的唯一真源：路径、特征分组、列顺序、派生特征参数。

列顺序只在这里声明一次：data.py 按它拼张量、model.py 按它声明维度。两条 DNN 路径的
输入是手工 concat 的，列序错位不会报错、只会让训练悄悄变差，所以不允许在别处再抄一份。

特征分组照 O_o/dataset.py:834-848，**去掉数据里缺失的 '111'**。
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
LOG_DIR = ROOT / "logs"

# HF 的 arrow 缓存（不是 hub 里的 parquet）。baseline/prepare_data.py 已经走过这条路：
# arrow 是流式 IPC，比 parquet 的列式行组快一个量级（NAS 上实测 92MB row group 会超时）。
HF_CACHE = Path.home() / ".cache/huggingface/datasets/TAAC2025___tencent_gr-1_m"
SEQ_ARROW_GLOB = str(HF_CACHE / "seq/0.0.0/*/*.arrow")
ITEM_FEAT_ARROW_GLOB = str(HF_CACHE / "item_feat/0.0.0/*/*.arrow")
USER_FEAT_ARROW_GLOB = str(HF_CACHE / "user_feat/0.0.0/*/*.arrow")

# seq 里 item_id 实测取值 [1, 4783154] 且稠密，与 item_feat 行数一致，所以 item 主 id
# 不需要映射表，0 天然是 pad（与 baseline/prepare_data.py:30 同口径）。
VOCAB_SIZE = 4_783_155
PAD = 0

# ---------------- 特征分组 ----------------
USER_SPARSE = ['103', '104', '105', '109']
USER_ARRAY = ['106', '107', '108', '110']
ITEM_RAW = ['100', '117', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116']

ACTION_FEAT = '900'    # record 级 action_type
CLICK_FEAT = '901'     # item 级「点击次数」分桶
CROSS_SPECS = [('116', '118'), ('118', '120')]
ITEM_CROSS = ['_'.join(p) for p in CROSS_SPECS]

ITEM_SPARSE_SEQ = ITEM_RAW + [ACTION_FEAT, CLICK_FEAT]   # 序列塔 15 列
ITEM_SPARSE_ITEM = ITEM_RAW + [CLICK_FEAT]               # item 塔 14 列（不含 900）

CLICK_BUCKETS = 16     # O_o/dataset.py:74

# 900 的映射值域：None->0 / 曝光->1 / 点击->2，所以表要 3 行（0=pad/缺失，1，2）。
# prepare.py 把它写进 meta 的 feat_vocab（值 = 最大 id），tower_kwargs 才能统一 +1 建表。
ACTION_MAX_ID = 2

# item_feat 静态矩阵的列顺序（prepare.py 写、data.py 读）
ITEM_STATIC_COLS = ITEM_RAW + [CLICK_FEAT] + ITEM_CROSS
ITEM_STATIC_INDEX = {c: i for i, c in enumerate(ITEM_STATIC_COLS)}

# 两侧的「非主 id」列清单（主 id 单独作为 cols[0] 传）
ITEM_SEQ_FEAT_COLS = ITEM_SPARSE_SEQ + ITEM_CROSS
ITEM_ONLY_FEAT_COLS = ITEM_SPARSE_ITEM
USER_FEAT_COLS = USER_SPARSE + USER_ARRAY

# seq 缓存里 action 的取值：必须把「缺失 None」与「曝光 0」分开 —— 900 的映射不同
ACT_EXPOSURE, ACT_CLICK, ACT_NONE = 0, 1, 2


def action_to_900(a):
    """O_o/dataset.py:439 _map_action_to_id：None->0, 曝光->1, 点击->2。"""
    if a == ACT_NONE:
        return 0
    return 1 if a == ACT_EXPOSURE else 2


def action_to_next_action(a):
    """next_action_type 存**原始** action，供 `act > 0.5` 判定（O_o/main.py:157）。"""
    return 0 if a == ACT_NONE else int(a)


def item_only_cols():
    """item 塔取 item_item_list 里的哪几个下标（0 = 主 id）。"""
    names = ITEM_SPARSE_SEQ + ITEM_CROSS
    return [0] + [1 + names.index(f) for f in ITEM_ONLY_FEAT_COLS]


def user_array_cols():
    """user_array 特征在 user_item_list 里的下标（主 id 占 0）。"""
    return [1 + USER_FEAT_COLS.index(f) for f in USER_ARRAY]


def load_meta(cache_dir=None):
    with open(Path(cache_dir or CACHE_DIR) / "meta.json", encoding="utf-8") as f:
        return json.load(f)


def tower_kwargs(meta, embedding_dim=128, hidden_units=512, dropout=0.2):
    """按 prepare.py 写出的词表规模生成 tower 的构造参数。

    每张表的行数 = 该特征的词表规模 + 1（0 号行是 pad）。
    """
    fv = meta["feat_vocab"]
    return dict(
        user_item_list=[meta["num_users"] + 1] + [fv[f] + 1 for f in USER_FEAT_COLS],
        item_item_list=[VOCAB_SIZE] + [fv[f] + 1 for f in ITEM_SPARSE_SEQ + ITEM_CROSS],
        embbeding_dim=embedding_dim,
        hidden_units=hidden_units,
        dropout=dropout,
        item_only_cols=item_only_cols(),
        user_array_cols=user_array_cols(),
    )
