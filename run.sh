#!/usr/bin/env bash
# tower 全流程：预处理 -> 自检 -> 训练 -> 全量候选评测
# 用法：bash run.sh [--prepare] [--force] [<train.py 的其它参数...>]
#   --prepare  强制重建缓存（等价于 prepare.py --force）
#   --force    与 --prepare 同义；缓存不存在时无条件重建
# 其余参数原样转给 train.py（如 --batch_size 128 --num_epochs 1）。
# 输出目录用环境变量指定，**不要**在参数里写 --out_dir（那会让下面 tee/cat 的路径对不上）：
#   OUT_DIR=logs_3ep bash run.sh
# 逐 epoch 的验证指标文件名同样用环境变量给，**不要**在参数里写 --val_log：
#   VAL_LOG=val_3ep.jsonl bash run.sh      改名（相对路径按 OUT_DIR 解析）
#   VAL_LOG=none bash run.sh               这次不记
# 默认 val.jsonl。
#
# user token 的先后次序（见 DIFF_vs_O_o.md §2.3）：--user_seq_order 会被原样转给
# train.py，默认复刻 O_o。环境变量 USER_SEQ_ORDER 是同一件事的另一条路（train.py 的
# argparse 默认值读的就是它），两者都给时以 --user_seq_order 为准。
#   OUT_DIR=logs_chrono bash run.sh --user_seq_order chrono
# 跑哪一套会写进日志的 [env] 行、result.json 与 val.jsonl。
# 两种布局的指标**不可直接比较**，所以换布局时务必同时换 OUT_DIR。
#
# HSTU 的 FFN 换成了 top-k MoE（4 专家激活 1 个，见 DIFF_vs_O_o.md §2.4）。
# 下面这些开关同样原样转给 train.py，默认 4 专家 top-1 + aux 权重 0.05（= config.MOE_AUX_ALPHA）：
#   --moe_experts 1              关掉 MoE（退回 nn.Linear，与 O_o 逐位等价）= A/B 的基线档
#   --moe_topk 2                 每个 token 激活 2 个专家（初始门控仍是 1/E，但选中两个
#                                相加 -> 初始量级是 √2/E 倍 baseline，**仍低于 baseline**，
#                                见 DIFF §2.4 的 √k 那条。别再引用 1.414 那种乘 E 时代的数）
#   --moe_aux_alpha 0            只把负载均衡项当诊断量记日志，不进损失
#   --moe_shared_experts 1       DeepSeekMoE 式共享专家：每个 token 都过的稠密 FFN，
#                                输出恒 1 相加、不过门。补回 step-0 的量级（1/E -> ~1.03x），
#                                代价是 top-1 下 FFN 的 FLOPs **翻倍**。要求 moe_experts>1。
#                                默认 0 = 一个参数都不建，与改动前逐位一致。
#                                见 DIFF_vs_O_o.md §2.4 的第 6 小节。
#   --moe_routed_scale 8.66     路由支路的系数 λ（论文式 (3) 的 λ，见 DIFF §2.4 第 8
#                                小节）：只乘路由那一路，共享那一路恒为 1。默认 1.0 =
#                                **逐位走改动前那条路径**。
#                                取值来自「使两支在初始化阶段模长一致」那条原则，本仓库
#                                零初始化下是闭式 λ = √s·E/√k：
#                                  --moe_experts 15 --moe_topk 3 --moe_shared_experts 1  -> 8.66
#                                  --moe_experts 16 --moe_topk 4 --moe_shared_experts 1  -> 8.00
#                                  --moe_experts 16 --moe_topk 1 --moe_shared_experts 1  -> 16.00
#                                注意论文按 logits~N(0,1) 模拟出来的值（同组 ~3.55）**不能
#                                直接用**，差 2.4 倍全在 σ 上，而这个仓库的 σ 恒为 0。
#                                它不改结构、不改 FLOPs、**不改任何权重文件**（不进
#                                state_dict），所以同一份 ckpt 配不同 λ 是合法的 ——
#                                两轮的区别只记在日志的 moe_routed_scale 字段里。
#                                要求 moe_experts>1。
#   --moe_ffn_act silu           HSTU block 内 FFN 的激活函数（见 DIFF §2.4 第 10 小节）：
#                                每个专家由裸 Linear 变成两层 MLP W2·act(W1·x)，参数量与
#                                FLOPs 各 ×1.334。默认 none = **逐位走改动前那条路径**。
#                                非 none 的三档（silu/gelu/relu）权重文件**逐位同形**，
#                                唯一能反查档位的就是日志里的 moe_ffn_act 字段。
#                                也作用于 --moe_shared_experts 与 --moe_experts 1 的稠密兜底路径。
# 换了上面任何一个值的两轮训练指标**不可直接比较**，务必同时换 OUT_DIR：
#   OUT_DIR=logs_moe4 bash run.sh
#   OUT_DIR=logs_shared bash run.sh --moe_shared_experts 1
#   OUT_DIR=logs_lam866 bash run.sh --moe_experts 15 --moe_topk 3 \
#                                    --moe_shared_experts 1 --moe_routed_scale 8.66
#   OUT_DIR=logs_act bash run.sh --moe_experts 16 --moe_topk 4 --moe_ffn_act silu
#
# 负载均衡策略（见 DIFF_vs_O_o.md §2.4 第 9 小节）：
#   --moe_balance aux_free       DeepSeek-V3 式「无辅助损失」均衡：用逐专家偏置 b 决定
#                                选谁，**且 alpha*aux 不进损失**（aux 仍算、仍记日志）。
#                                b 由 train.py 在每个 optimizer.step() 之后更新。
#   --moe_balance both           两者同时开（偏置照常更新，aux 损失也照常加）。
#   默认 aux_loss = 改动前那套（Switch 式辅助损失），逐位一致。
#   --moe_bias_gamma 1e-3        偏置的更新步长 γ（只在 aux_free / both 档生效）。
#                                DSeek-V3 的 bias update speed，取 1e-3；**要扫**
#                                （{3e-4, 1e-3, 3e-3}），判据见 config.MOE_BIAS_GAMMA。
# 三档的指标不可直接比较（均衡压力的来源完全不同），换档务必同时换 OUT_DIR：
#   OUT_DIR=logs_auxfree bash run.sh --moe_balance aux_free
#   OUT_DIR=logs_both bash run.sh --moe_balance both
#
# 易负样本那一族的 -logQ 修正（见 DIFF_vs_O_o.md §6.2 第 17 条）：
#   --no-easy_neg_logq   关掉它。α=0（我们的生效值）时易负是均匀采样的，修正项本应
#                        是常数（softmax 里等于不修正），减流行度的 logQ 属于不自洽。
#                        默认 True = 保持 O_o 写法，所以本轮及之前的口径不变。
# 它与 --neg_pop_alpha 成对：α=0 + 关掉（本轮建议的 A/B 单变量）才自洽，
# α=1.0 + 开着同样自洽但**还额外改了采样分布**，两者不是同一个实验。
#   OUT_DIR=logs_noeasyneg bash run.sh --no-easy_neg_logq
#
# HSTU 的相对时间偏置表（见 DIFF_vs_O_o.md §2.5）：
#   --time_bias_per_layer   把「8 层共用一张 129 行的表」改成每层一张（8 张，共 1032 个
#                           参数）。桶、初始化、三路拼接都不动，改的只是「几张表」。
#                           默认关 = 保持 O_o 写法，所以本轮及之前的口径不变。
#                           注意它是**结构**开关：参数总量与张量形状都不变、ckpt 也照常
#                           加载，只有指标会不一样 —— 所以务必同时换 OUT_DIR。
#   OUT_DIR=logs_tb8 bash run.sh --time_bias_per_layer
#
# 第四路注意力：时间旋转位置编码（见 DIFF_vs_O_o.md §2.6）：
#   --trope               在三路（content / RoPE / 相对时间表×V）之外**追加**一路用时间戳
#                         旋转 Q/K 的注意力（model.Trope）。默认关 = 与 O_o 逐位一致的三路。
#                         ⚠️ 它是这里**唯一会改张量形状**的开关：uqkv 6D->7D、FFN 输入
#                         3D->4D，所以**两轮的 ckpt 不能互相加载**（直接报 size mismatch）。
#                         代价（实测）：参数 +35.7M（1.158B->1.194B，+3.1%）、注意力矩阵乘
#                         +50%、FFN MACs +33%。
#   --trope_shifted_query 第四路的 query 用「下一个事件」的时间戳旋转（最后一个 slot 没有
#                         下一个事件，退化成用自己）。只在 --trope 打开时生效（关着时一个
#                         参数都不建）。默认关。它与 --trope 是两个变量，别合成一次实验。
# 三个前提，用之前要知道：
#   1. 它依赖的是 `t_i + t_j` **不是** `t_i − t_j`（来自 model.py 里那句 `k_pos = -k_pos`）。
#      这是刻意保留的，tests/test_trope.py 有一条用例**正面钉住**它，别顺手改掉。
#      只有 α_q 学到 ≈ −α_k 时才退化成相对时间 —— 能不能学到是实验问题。
#   2. 喂进去的时间戳是**逐序列居中**的：每条序列减去自己第一个有效 token 的时间。
#      为什么必须减：fp32 下 1.7e9 的 ulp 是 128 秒，相邻很近的行为会不可分；减完落到
#      会话跨度（实测中位 26.8 天），ulp 变 2 秒。
#      为什么必须是「逐序列」而不是「数据集最早一条记录」：相位里有一项
#      `(α_k+α_q)·T_user/β`，`T_user` 是会话在数据集里的绝对位置。逐序列居中把它按构造
#      成 0；用全局原点时它逐用户随机（快平面在 231 天上扫 ~300 个整周期），梯度互相抵消。
#      ⚠️ 首轮实测（用全局原点）已经验证过这一点：α 训练前后几乎没动，指标与基线持平；
#      所以改回逐序列居中。代价是原点会随 --user_seq_order 变（o_o 的 user 块是时间倒序，
#      首位拿到最新的那条），正式跑都用 chrono，此时原点 = 窗口开头。
#   3. 它与第三路（相对时间表 × V）都编码时间，**暂时共存**。是否关掉第三路是下一步。
#   OUT_DIR=logs_trope bash run.sh --trope --user_seq_order chrono
#   OUT_DIR=logs_trope_sq bash run.sh --trope --trope_shifted_query
#
# 选卡：**由 train.py 自己挑**（见那里的 auto_pick_gpu / free_gpu）——
# 在「空闲显存 >= NEED_GB」的卡里选利用率最低的一张，一张都不达标时退化为空闲最多的。
# 本脚本只负责把手动指定传下去，不重复实现挑卡逻辑：
#   GPU=5        手动指定卡号（导出 CUDA_VISIBLE_DEVICES，train.py 会沿用、不再自动挑）
#   NEED_GB=60   调整自动挑卡的显存门槛（默认 50；实测训练峰值 ~44G）
#
# 与计划书的一处修正：计划书里把 `${1:+}`（也就是 "--prepare"）直接转给了
# prepare.py，但它的 argparse 只认 --out_dir/--force/--check，`set -e` 下会直接退出。
set -euo pipefail
cd "$(dirname "$0")"

PY=${PY:-python}
CACHE=cache

RUN_PREPARE=0
FORCE=0
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare) RUN_PREPARE=1; shift ;;
    --force)   FORCE=1; shift ;;
    *)         ARGS+=("$1"); shift ;;
  esac
done

if [[ $RUN_PREPARE -eq 1 || ! -f "$CACHE/meta.json" ]]; then
  echo "== 1/2 预处理（arrow -> cache） =="
  if [[ $RUN_PREPARE -eq 1 || $FORCE -eq 1 ]]; then
    $PY prepare.py --force
  else
    $PY prepare.py
  fi
fi

echo "== 自检 =="
$PY prepare.py --check

# 只在真的指定了卡号时导出：CUDA_VISIBLE_DEVICES="" 在 CUDA 里表示「一张卡都看不见」
if [[ -n "${GPU:-}" ]]; then
  echo "== 选卡：GPU=$GPU（手动指定，train.py 会沿用） =="
  export CUDA_VISIBLE_DEVICES="$GPU"
else
  echo "== 选卡：交给 train.py 自动挑最空闲的一张（NEED_GB=${NEED_GB:-50}） =="
fi

echo "== 2/2 训练 =="
OUT_DIR=${OUT_DIR:-logs}
VAL_LOG=${VAL_LOG:-val.jsonl}
mkdir -p "$OUT_DIR"
# tee 到文件而不是只打屏：崩了也留下 traceback（之前在终端里裸跑，进程一死什么都没剩下）
$PY train.py --num_epochs 3 --batch_size 256 --amp bf16 --out_dir "$OUT_DIR" --val_log "$VAL_LOG" \
  ${ARGS[@]+"${ARGS[@]}"} 2>&1 | tee "$OUT_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "== 结果 =="
cat "$OUT_DIR/result.json"
# 逐 epoch 的验证指标（一行一个 epoch）；result.json 只有最后一个 epoch。
# 用 if 而不是 `[ -f ] && cat`：后者在文件不存在时返回 1，set -e 下会让脚本在这一行退出。
if [[ "$VAL_LOG" = /* ]]; then VAL_LOG_PATH="$VAL_LOG"; else VAL_LOG_PATH="$OUT_DIR/$VAL_LOG"; fi
if [[ "$VAL_LOG" != "none" && -f "$VAL_LOG_PATH" ]]; then
  echo "== 逐 epoch（$VAL_LOG_PATH） =="
  cat "$VAL_LOG_PATH"
fi
