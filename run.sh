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
