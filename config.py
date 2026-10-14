"""tower 数据/训练管线的唯一真源：路径、特征分组、列顺序、派生特征参数。

列顺序只在这里声明一次：data.py 按它拼张量、model.py 按它声明维度。两条 DNN 路径的
输入是手工 concat 的，列序错位不会报错、只会让训练悄悄变差，所以不允许在别处再抄一份。

特征分组照 O_o/dataset.py:834-848，**去掉数据里缺失的 '111'**。
"""
import json
import math
import os
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

# user token 在序列里的先后次序。**这里只是默认值**，train.py 的 `--user_seq_order`
# 优先于它（argparse 的 default 就是读的这个常量，所以环境变量不会被静默覆盖）。
#   'o_o'    = 复刻 O_o：user 块在最终序列里是**时间倒序**（默认，行为不变）
#   'chrono' = 按记录实际顺序：user 块时间正序，见 DIFF_vs_O_o.md §2.3
# 两种切法：`--user_seq_order chrono`（推荐，会进 ckpt 的 args）/ `USER_SEQ_ORDER=chrono`。
#
# 模块级 assert 是刻意的：拼错取值必须**当场炸**，不能静默退回默认值 —— 否则一次
# A/B 实验会白跑两小时才发现两边配置其实一样。
USER_SEQ_ORDER = os.environ.get("USER_SEQ_ORDER", "o_o")
assert USER_SEQ_ORDER in ("o_o", "chrono"), \
    "USER_SEQ_ORDER 只能是 'o_o' / 'chrono'，实际 %r" % (USER_SEQ_ORDER)


# HSTU 每个 block 的 FFN 换成 top-k MoE 的专家数与激活数（见 model.py 的 moe）。
#   MOE_EXPERTS = 1 -> 退回原来的 nn.Linear：与 O_o **逐位等价**（test_model_equiv 的前提），
#                     也是 A/B 里「关掉 MoE」那一档，所以它必须是默认之外的合法取值。
#   MOE_TOPK    = 1 -> 每个 token 只激活一个专家（= Switch Transformer 的 top-1）。
# 默认值在这里、真实取值由 train.py 的 --moe_experts / --moe_topk 决定（会进 ckpt 的 args）。
# 只在**序列塔**生效：item 塔没有 HSTU，候选侧编码不受影响。
MOE_EXPERTS = 4
MOE_TOPK = 1

# 共享专家的个数（DeepSeekMoE 式：恒 1 相加、不过门，见 model.py 的 moe docstring）。
#   MOE_SHARED_EXPERTS = 0 -> 一个参数都不建，与改动前**逐位一致**（必须保持默认：
#                             历史 ckpt 的 state_dict 键、之前跑过的 A/B 都靠它）
#   MOE_SHARED_EXPERTS = 1 -> 每个 block 多一路稠密 FFN，每个 token 都过
#
# 它同时改两件事，所以两轮指标不可直接比较（换档务必同时换 OUT_DIR）：
#   1. 结构：每层多 1 个 Linear(input_dim*3, input_dim)。8 层在 1.09B 参数里看不见，
#      ckpt 也能加载（只是多了 16 个键），**没有任何形状变化** —— 是最容易漏记的开关；
#   2. FLOPs：共享专家是稠密的，top-1 下 FFN 的矩阵乘**大约翻倍**（1 共享 + 1 路由）。
#      这比「把 MOE_EXPERTS 调大」贵：后者只涨参数不涨 FLOPs。
#
# 换来的东西在 step 0 就能看到：门控不缩放（见下一条）导致初始 FFN 只有 baseline 的
# 1/E，共享专家把方差补成 (1 + 1/E²)σ²，E=4 时 std ≈ 1.031× baseline —— 正是当初
# 乘 E 想锚住、又因为代价太大而放弃的那个量级，这次代价只有一路稠密 FFN。
MOE_SHARED_EXPERTS = 0

# 路由支路的系数 λ（论文式 (3) 的 λ，见 model.py 的 moe docstring）。
#   MOE_ROUTED_SCALE = 1.0  -> 不缩放，走改动前那条路径，与历史 A/B **逐位一致**
#   MOE_ROUTED_SCALE = λ    -> 只乘路由那一路（共享那一路恒为 1），把两支拉回同量级
#
# **它不进 state_dict、不改任何张量形状、不动任何权重** —— 权重文件与 λ=1 那轮逐位
# 相同，所以「同一份 ckpt 配不同 λ」是合法的，而反过来「哪一轮是哪个 λ」只能靠日志
# 里的 moe_routed_scale 字段去分辨（run_meta 记着）。默认 1.0 必须保持：历史 ckpt 与
# 之前跑过的 A/B 全靠它。
#
# 取值不由这里推导 —— 它由论文那条原则给出：**使两支在初始化阶段模长接近一致**，
# 即 `λ = √s / √(Σρ²)`。本仓库 `router_weight` 零初始化 -> logits 恒 0 -> softmax
# 精确均匀，`Σρ² = k/E²` 是个确定值，于是闭式：
#
#     λ = √s · E / √k          （s = MOE_SHARED_EXPERTS，E = MOE_EXPERTS，k = MOE_TOPK）
#
# 常用档位：15/3/1 -> 8.66；16/4/1 -> 8.00；16/1/1 -> 16.00；不共享（s=0）-> 0（无意义，
# 那种配置下这个原则没有参照物，λ 只能实扫）。注意 σ：论文假设 `logits ~ N(0,1)` 并
# 据此做数值模拟，在这个仓库会得到 ~3.55 —— **那个数直接用会偏小 2.4 倍**。
#
# 一个必须知道的前提：λ 只在**初始化**那一点上被标定。router 训练中会变尖（实测
# gate_mean 六个 epoch 涨 55%，等效 σ 从 0 到 ~0.75），原则要求的 λ 随之从 8.66 掉到
# ~3.8。所以闭式给的是「init 严格正确」的那一档，想覆盖全程要取更小的折中值。
MOE_ROUTED_SCALE = 1.0

# HSTU block 的 FFN 激活函数。'none' = 每个专家是裸 nn.Linear（默认，与改动前逐位
# 一致，也是与 O_o 对齐的那一档 —— O_o/model.py 的 out_linear 同样是线性的）；
# silu / gelu / relu = 每个专家变成两层 MLP: W2 · act(W1 x)。
#
# 为什么需要它：专家是裸 nn.Linear 时，MoE 的形式是 y = Σ g_i(x)·W_i x —— 一个线性
# 映射的加权和，16 个专家之间可以互相蕴含、超出需要的那些字面意义上冗余，router
# 没有理由去用它们。实测层 5 的 moe_expert_frac 有 5 个专家拿不到 0.05% 的 token，
# 而 aux_free 的偏置为掰它涨到 4.81-11.00（其余 7 层都在 0.20-1.80）仍没掰动。
# 两层 MLP 给每个专家一个自己的非线性区域，让它重新变得不可替代。
#
# 代价：参数量与 FLOPs 各 ×1.334（top-4 时 3.15M -> 4.19M MACs/token）。
# step-0 的 FFN 输出模长会明显下降（量级在腰斩上下），会让 MOE_ROUTED_SCALE 那条
# 闭式 λ = √s·E/√k 的前提（初始化阶段两支模长一致）不再严格成立 —— 两个开关在
# step-0 上不正交。**具体数值不要引用**：绝对值依赖 torch 版本与设备，
# 实测口径与量级见设计文档 §6.2。
#
# 特意**不**在这里加 `assert MOE_FFN_ACT in (...)`：白名单是 model.py:97 的 `_FFN_ACT`
# 字典（外加独立的 'none' 档），在 config 里再抄一份就成了第二处真相 —— 将来给
# `_FFN_ACT` 添一档，这里会在 import 期把合法值判成非法。校验留在 `_make_ffn`
# （model.py:126），报错信息会念出全部合法值。
#
# 四档指标不可直接比较，换档务必换 OUT_DIR。
MOE_FFN_ACT = "none"

# MoE 的负载均衡策略（见 model.py 的 moe docstring、DIFF_vs_O_o.md §2.4 第 9 小节）。
#   'aux_loss'（默认）= Switch 式辅助损失 `alpha * aux`；**与改动前逐位一致**
#   'aux_free'        = DeepSeek-V3 式逐专家偏置 b（`b += γ·sign(mean_load − load)`）；
#                       且 `alpha * aux` **不进损失**（aux 仍算、仍记日志当诊断量）
#   'both'            = 两者同时开（偏置照常更新，aux 损失也照常加）
#
# 三档的指标不可直接比较（均衡压力的来源完全不同），换档务必同时换 OUT_DIR。
# 档名本身就声明了 alpha 生不生效，所以不是「静默丢弃」。
# 细节（含下面 §4.2 的引用）见设计文档（下称 spec）：
#   docs/superpowers/specs/2026-09-18-taac-moe-aux-free-balance-design.md §2.1
MOE_BALANCE = "aux_loss"
assert MOE_BALANCE in ("aux_loss", "aux_free", "both"), \
    "MOE_BALANCE 只能是 'aux_loss' / 'aux_free' / 'both'，实际 %r" % (MOE_BALANCE,)

# aux_free / both 档的偏置更新步长 γ（DeepSeek-V3 的 bias update speed）。
#
# **这只是起点、不是推导出来的**，与 MOE_AUX_ALPHA 同一个处境：γ 太小 -> 均衡跟不上
# 路由变尖的速度；γ 太大 -> b 在专家之间来回过冲、maxf 抖。判据同 MOE_AUX_ALPHA 那条：
# 看 train.jsonl 的 moe_expert_frac 里 maxf 落在哪，均匀值是 1/E。
#
# 量纲：γ 进的是 **logits 空间**，与 batch size / 序列长度 / token 数**无关**，
# 所以不用像学习率那样随 batch 缩放。量级参照：实测 router 的 gate_mean 六个 epoch
# 涨 55%、反推等效 σ 从 0 升到 ~0.75（见 MOE_ROUTED_SCALE 的注释），即训练中期的
# logits 尺度在 ~1 量级；γ=1e-3 约 750 步累积到 0.75，一个 epoch 3,523 步。
#
# ⚠️ 偏置 buffer **必须是 fp32**：γ=1e-3 加在 bf16 上，|b| 到 0.75 那个量级就原地
# 不动了（实测 0.75 ± 1e-3 == 0.75，加减两个方向都冻住），而 b 正是要长到 ~0.75 才
# 压得住 logits —— 停摆阈值落在额定工作区间里，症状是 maxf 先降后卡住。
#
# 举例取 0.75 而不是 0.5 是有意的：**0.5 是 binade 边界，那里只有 +γ 冻住** ——
# 减完落进 [0.25, 0.5)，ulp 减半，-γ 照样动（实测 0.5 - 1e-3 = 0.498046875）。
# 拿边界值举例会让人以为「一个数要么全冻要么全不冻」，而真实规律是按 binade 走的。
# 见上面 spec 那个文件 §4.2 的量化表。
MOE_BIAS_GAMMA = 1e-3

# 负载均衡项的权重。0 = 只把 aux 当诊断量记日志、不进损失（路由仍照常硬选）。
#
# **这个数只能靠实扫，下面给的只是起点，不是推导出来的。** 唯一能解析传递的关系是
# 「同 top_k 下 alpha ∝ 1/E」：门控是概率（`softmax[选中]`，不缩放），路由从 task 拿到的
# 梯度 `p*(δ-p)` 比 `E*softmax` 时代小 E 倍，而 aux 走 `probs`、完全不受影响。于是
# r = ‖g_task‖/‖g_aux‖ ∝ 1/E —— 这条在 step 0 实测过（E=4 比值 ~3.4-3.7、E=16 ~14.3-15.0，
# 略低于 E 是因为前向本身也变了）。**旧的 alpha 标定（全部在 ×E 前提下）不能直接搬**：
# 0.1 按这个换算是 0.025。
#
# 但 0.025 只是「参数状态不动」的换算，真跑起来会**系统性偏低**：alpha 变小 -> 路由更
# 集中 -> p_选中 变大 -> r 变大 -> 又需要更大的 alpha（负反馈，线性换算没算进去）。
# 21 步的探针也钉不住它：(4,1) 的 maxf@20 在 alpha = 0.025 / 0.05 / 0.1 上是
# 0.731 / 0.826 / 0.511 —— **非单调**，说明这个步数下轨迹噪声压过了 alpha 的效应。
# top_k>1 没有现成公式，必须单独扫。
#
# 正式跑之前扫 {0.025, 0.05, 0.1}，看 train.jsonl 的 moe_expert_frac 里 maxf 落在哪：
# 均匀值是 1/E，**明显大于它说明 alpha 偏小**；钉在 1/E 不动说明 alpha 偏大、路由被推平。
# 专家塌缩在 loss 上完全看不出来，只有这个量能发现。
MOE_AUX_ALPHA = 0.05

# 易负样本那一族的 -logQ 修正开不开（见 losses.info_nce 的 docstring 与
# DIFF_vs_O_o.md §6.2 第 17 条：问题的形状、三档配置 A/B/C、与实测的偏置量级）。
#
#   True （默认）= 逐字保持 O_o 的写法：logits = sim/T - log(counts/total)[neg_col]
#   False        = 易负只用 sim/T；正样本与难负两族的修正完全不受影响
#
# **默认 True 不是因为它对，而是因为它是那个「已经跑过 5 轮」的基线**：默认值一改，
# 历史 val.jsonl / train.jsonl 与新日志就没法直接比了。
#
# 背景：三族负样本的采样分布并不一样，而 logq.npy 只编码了「流行度」这一种。
# 正样本与难负两族的列都来自数据（∝ counts），修正天然匹配；只有易负那一族是由
# `--neg_pop_alpha` 决定的 —— alpha=0（当前默认）时它**均匀采样**，此时正确修正是
# 个整行常数（softmax 不变 -> 等于不修正），再减流行度的 logQ 等于往 logits 里灌
# item 频率。所以这是一个已知的不自洽，关掉它和把 alpha 调成 1.0 都能修，方向不同。
EASY_NEG_LOGQ = True

# 相对时间偏置表：**全局一张**（默认）还是**每层一张**（8 张）。
#
#   False（默认）= 逐字保持 O_o 的写法：单个 `nn.Embedding(129, 1)`，8 层共用同一份
#                  ts_w，rel_ts 在层循环外只算一次（O_o/model.py:324/520）
#   True         = 每个 HSTU block 一张自己的表，桶号仍然只算一次（它只由 seq_ts
#                  决定、与表无关），查表下放到循环里
#
# 动机：现在是「一张表 + 每桶一个标量」，8 层被迫共用同一个「多久算久」的尺度 ——
# 第 1 层和第 8 层没法各自表达不同的时间衰减。而时间这一路是 HSTU 里**唯一可正可负、
# 且直接乘进 V** 的通道（`hstublock.forward` 的 time_out），尺度被锁死的影响不是零。
#
# 与官方的差距还剩一半：HSTU 论文的 rab 是 **per-head** 的（每个 head 一份，官方实现
# 里 rab 最后一维就是 num_heads）。这里先只做 per-layer —— 8 张表共 8 × 129 = 1032 个
# 参数，相对共享档**净增 7 × 129 = 903**（总参数约 1.09B，实测序列塔 37,818,497 ->
# 37,819,400，可忽略），改的也只是「哪张表」，不动任何张量的形状；per-head
# 需要把 rel_ts 从 [b, s, s] 改成 [b, head, s, s]、连带改 hstublock 里那句 einsum，
# 是另一个实验，别和这个一起开。
TIME_BIAS_PER_LAYER = False

# 第四路注意力：把 `model.Trope`（用时间戳旋转 Q/K）作为**第四路**接进 HSTU block 的三路
# 注意力（content / RoPE / 相对时间表 × V），打开后 `out = cat([attn, rope, time, trope])`
# 是 4·d，而不是现在的 3·d。
#
#   TROPE = False（默认）= 一个参数都不建、`uqkv` 仍是 `nn.Linear(D, 6D)`、FFN 输入仍是
#                          3D —— **与改动前逐位一致**，也是与 O_o 对齐的那一档
#                          （O_o 只有三路，见 DIFF_vs_O_o.md §2.6）
#   TROPE = True          = 每个 block 多一路 trope 注意力 + 一个自己的 Trope 模块
#
# 它是**结构**开关，且是这里唯一会改张量形状的：`uqkv` 的输出从 6D 变 7D（只多一块 U），
# FFN 的输入从 3D 变 4D（1536 -> 2048）。所以两轮的 ckpt **不能互相加载**，
# `load_state_dict` 会直接报 size mismatch —— 比 `--moe_ffn_act` 那种「键名不匹配」更好认，
# 但仍然是不可比的两套权重。换档务必同时换 OUT_DIR。
#
# 代价（实测前的核算，落地后要复核）：
#   参数  uqkv +D²/层（+2.10M）；FFN 输入 1536->2048，每个专家 +512²，E=16 时 +4.19M/层
#         -> 合计约 +35.6M（1.083B -> ~1.119B，+3.3%）
#   FLOPs QK^T + AV 从 2 路变 3 路（注意力矩阵乘 +50%）；FFN MACs/token +33%
#   显存  FFN 激活按 +33% 估；当前 --trope 关闭档的实测峰值是 50.4GB（logs/result.json）
#
# 三条语义要点，都写进了 model.py 的 Trope docstring：
#   1. 依赖的是 `t_i + t_j` 不是 `t_i − t_j`（来自 `k_pos = -k_pos`），只有 α_q 学到
#      ≈ −α_k 时才退化成相对时间 —— 已知且接受，有单测正面钉住
#   2. 喂进去的时间戳是**居中的**（tower.forward 里减去每条序列首个有效 token 的时间），
#      因为绝对 Unix 秒在 fp32 下只有 ~128 秒的分辨率
#   3. 与第三路（rel_ts 偏置 × V）都编码时间，两者**暂时共存**；是否关掉第三路是下一步
TROPE = False

# trope 那一路的 query 要不要用「下一个事件」的时间戳（最后一格退化成用自己）。
# 与本仓库 next-item 预测的目标更贴，但源码默认是 False，这里也保持 False ——
# 把它留成下一档 A/B，不要和 TROPE 本身混成一次实验。
TROPE_SHIFTED_QUERY = False

# Trope 的 β 刻度范围（**不可学**，对数等分成 p 个平面）。
# 第 p 个平面的周期是 `2π·β_p/α_p`，α 初始 N(0, σ=1) 所以起步 ≈ `2πβ_p`；
# 取 β = 周期/2π 就能让周期直接等于想要的秒数。
#
# 取值依据（**实测**，不再照抄 FourierTimeEncoding 的 1~40 天）：
#   - 判据是「这条序列内部扫过几个整周期」。因为相位依赖 `t_i + t_j` 而不是差值，
#     序列内的变化量是 `2W`（W = 序列内时间跨度），所以 `周期数 = 2W / P`。
#   - 实测 W：p10 = 13.8 天、**中位 26.8 天**、p90 = 73.7 天、均值 38.2 天。
#   - 快端留 1 天：给细粒度分辨率（约 0.5 rad/小时），混叠不影响它当高频特征用。
#   - 慢端取 **180 天 ≈ 2 × W(p90)**，让最慢的平面在中位用户上只扫过 0.30 个周期 ——
#     否则没有任何一个平面能单调分辨「这条记录离窗口开头多久」。
#     （慢端 40 天时最慢平面也要绕 1.34 圈，32 个平面里只有 4 个在 2 圈以内。）
#
# 类里原本写死的是 `100` / `1e8`，换算成周期是 ~10 分钟 ~ ~20 年 —— 一半平面在本数据的
# 会话尺度上几乎不动、另一半快到按秒混叠，所以这里重标定。注意这是**纯正值重标定**：
# β 不可学，把它整体除以 2π 与在 forward 里多乘一个 2π 完全等价（α 学得回来）。
#
# ⚠️ β 只决定这个先验，**α 是可学的**（`P = 2πβ/α`），模型能把每个平面拉长或压短
# 1~2 个数量级。所以这套数偏了也不会让实验作废，只是初始化离中心远一点。
TROPE_BETA_MIN_SEC = 86400.0 / (2 * math.pi)              # ≈ 13 751，周期 1 天
TROPE_BETA_MAX_SEC = 180 * 86400.0 / (2 * math.pi)        # ≈ 2 475 176，周期 180 天

# 第四路的时间原点：**不做成开关**，固定为「每条序列减掉自己第一个有效 token 的时间」
# （见 `model.tower.forward` 里 `rot_ts` 的构造）。
#
# 为什么两个「减」都需要：fp32 下 1.7e9 的 ulp 是 **128 秒**，相邻很近的两条行为会不可分；
# 减掉原点后取值落到会话跨度（实测中位 26.8 天、p90 73.7 天），ulp 降到 **2 秒**。
#
# 曾经有一版是「减掉数据集最早一条记录」（1728921670），**实测后废弃**，理由与证据：
#   - 那样每条序列自带一项 `(α_k+α_q)·T_user/β` 的常数相位，`T_user` 是会话在数据集里的
#     绝对位置，**逐用户随机**。快平面（β=13751）在 231.3 天上扫过 ~2900 rad（≈460 整周期），
#     连最慢的平面（180 天周期）也要绕 ~1.7 圈 —— 所有平面都中招，且这一项比信号项还大。
#   - 后果是 `Δ → 注意力权重` 的映射逐用户不一致，梯度互相抵消：α 训练前后几乎没动
#     （`|α|` 均值 0.62~0.97，正对上 `N(0,1)` 初始化的 `E|α| = 0.798`），best.pt(ep4) 与
#     last.pt(ep6) 逐位几乎相同，死平面只有 1~7 个。
#   - 指标上：6 epoch 的 best score 0.095115 vs 不加第四路的基线 0.094985，只差 +0.14%，
#     在噪声内。
# 详见 `DIFF_vs_O_o.md` §2.6 与 `model.Trope` 的 docstring。
#
# ⚠️ 逐序列居中的代价：原点会随 `--user_seq_order` 变。`o_o` 的 user 块是时间倒序，首位
# 拿到的是 user 块里**最新**的那条；`chrono` 下首位是最旧的（= 窗口开头）。本仓库的正式
# 跑都用 `chrono`。


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


def tower_kwargs(meta, embedding_dim=128, hidden_units=512, dropout=0.2,
                 num_experts=MOE_EXPERTS, top_k=MOE_TOPK,
                 time_bias_per_layer=TIME_BIAS_PER_LAYER,
                 num_shared=MOE_SHARED_EXPERTS,
                 routed_scale=MOE_ROUTED_SCALE,
                 balance=MOE_BALANCE,
                 ffn_act=MOE_FFN_ACT,
                 trope=TROPE,
                 trope_shifted_query=TROPE_SHIFTED_QUERY):
    """按 prepare.py 写出的词表规模生成 tower 的构造参数。

    每张表的行数 = 该特征的词表规模 + 1（0 号行是 pad）。
    num_experts / top_k 是 MoE 的开关，num_shared 是共享专家的个数，
    routed_scale 是路由支路的系数 λ，time_bias_per_layer 是相对时间偏置表的开关，
    balance 是负载均衡策略，ffn_act 是 FFN 的激活函数，trope / trope_shifted_query 是
    第四路时间旋转注意力的开关（默认都取本模块的常量，见那里的说明）。第四路的时间
    原点不是参数：固定为逐序列居中，见 `TROPE` 那段注释。

    ⚠️ 默认值在 `def` 求值时绑定模块常量，所以 `monkeypatch.setattr(config, "TROPE", ...)`
    **不会**改变本函数的行为 —— 与其它开关同构，测试要显式传参。
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
        num_experts=num_experts,
        top_k=top_k,
        num_shared=num_shared,
        routed_scale=routed_scale,
        balance=balance,
        ffn_act=ffn_act,
        time_bias_per_layer=time_bias_per_layer,
        trope=trope,
        trope_shifted_query=trope_shifted_query,
    )
