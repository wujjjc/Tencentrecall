# tower 双塔召回

tower 是一个双塔召回模型加一套训练/评测流水线。本文件只讲结构，不复述结论。

## 目录

```
config.py     唯一真源：路径、特征分组、列顺序、派生特征参数、开关默认值
prepare.py    入口 1：从 HF arrow 构建 cache/ 下的稠密缓存（.npy）
data.py       Dataset + collate：用户记录序列 -> 逐位置训练张量
model.py      glu / 三路注意力 / HSTU block（含 MoE）/ hstu / tower
losses.py     InfoNCE(+ -logQ)、SSL 对比损失、权重初始化
eval.py       全量候选口径的验证评测（库，无 __main__）
train.py      入口 2：训练 + 逐 epoch 评测 + 早停
run.sh        预处理 -> 自检 -> 训练 -> 打结果的串场脚本
tests/        132 条用例（pytest）
cache/        数据缓存
logs/         训练输出：train.jsonl / val.jsonl / result.json / *.pt / tb/
docs/         设计文档与实施计划
baseline/     只读参考资料
```

**列顺序只在 `config.py` 声明一次**：`data.py` 按它拼张量、`model.py` 按它声明维度。
两条 DNN 路径的输入是手工 concat 的，列序错位不报错、只会让训练悄悄变差，所以不允许
在别处再抄一份。

## 数据流

```
HF arrow ──prepare.py──> cache/*.npy ──data.py──> DataLoader ──> train.py ──> logs/
                                          │                        │
                                     model.tower              losses + eval
```

`prepare.py --check` 是自检，`run.sh` 每次都先跑它。

## 模型

`model.tower` 是**一个类装两个塔**，按 `token_type` 分派：`token_type is None` 走 item
塔（候选侧），否则走序列塔。

- **item 塔**：主 id + 13 列稀疏特征 → `glu`（门控特征交互块）→ L2 归一化。
  没有 HSTU，候选侧编码因此不受 MoE 影响。
- **序列塔**：user/item 两侧稀疏特征 + hour/dow/weekend + 绝对时间傅里叶编码 → 8 层
  HSTU（head=8, hidden=512）→ L2 归一化。
- **HSTU block**：三路注意力（content / RoPE / 相对时间偏置×V，共用一张 129 桶时间表）
  → U 门控 → RMSNorm 残差 → FFN。
- **FFN 换成了 MoE**（4 专家 top-1）：`--moe_experts 1` 退回 `nn.Linear`，与基线逐位
  等价，也是 A/B 的基线档。门控、dispatch、负载均衡的细节见 `model.py` 的 `moe`
  docstring 与 `tests/test_moe.py`（那里钉的是行为契约）。

模型 1.06B 参数，fp32 权重单份 4.2GB。

## 损失

- **InfoNCE + -logQ 修正**：负样本 = 批内易负样本 + 全量难负样本（不子采样），损失张量
  是 `[M, 1+2M]`，内存随 batch **平方**增长 —— 调大 batch 时这里是第一处 OOM。
- **SSL**（`--ssl_alpha`，默认 0.5）。
- **MoE 负载均衡**（`--moe_aux_alpha`，默认 0.1），`--moe_experts 1` 时自动退化成 0。

## 评测

全量候选口径（4,783,155 个，不是批内候选），位置口径两种都支持：`all_click`（默认，
每个点击位置一个 query）与 `last_click`（留一法，每个用户只取最后一个点击位置，
与 `baseline/` 可比）。两者共用同一遍前向。
早停看 `score = 0.31*HR@10 + 0.69*NDCG@10`，patience 2。

## 跑

```bash
bash run.sh                       # 预处理（缺缓存时）+ 自检 + 3 epoch 训练 + 评测
bash run.sh --prepare             # 强制重建缓存
OUT_DIR=logs_xxx bash run.sh --moe_experts 1     # 换开关 -> 必须同时换 OUT_DIR
```

`run.sh` 的 `--out_dir` / `--val_log` 用环境变量传（`OUT_DIR` / `VAL_LOG`），写在参数里
会让脚本里 tee / cat 的路径对不上。除这几个之外，其余参数原样转给 `train.py`。

**几组值之间指标不可直接比较，换开关必须同时换 `OUT_DIR`**：`--user_seq_order`
（`o_o` / `chrono`）、`--moe_experts`、`--moe_topk`、`--moe_aux_alpha`。这三个 MoE 开关
连同 `user_seq_order` 都会写进日志的 `[env]` 行、`result.json` 与 `val.jsonl`。
它们的定义与理由在 `config.py` 的常量注释和 `model.py` 的 `moe` docstring 里。

`train.jsonl` 每 20 步一行：`loss_*`、`LR`、`moe_expert_frac`（每层每个专家的占比）、
`moe_gate_mean`。**专家塌缩在 loss 上完全看不出来**，只有 `moe_expert_frac` 能发现。

## 测试

```bash
python -m pytest tests/ -q          # 132 条
python -m pytest tests/ -q -m "not slow"    # 跳过需要 cache/ 的端到端
```

`tests/conftest.py` 把仓库根加进 `sys.path`（平铺模块，不是包）。
