"""tower 的训练与全量候选评测。

对齐 O_o/main.py 的配方：InfoNCE(+ -logQ) + SSL(rfm_no_compl, α=0.5) + AdamW
+ warmup/cosine + clip 1.0 + 按 score=0.31*HR@10+0.69*NDCG@10 早停（patience 2）。

与 O_o 的有意偏离（设计文档 §10）：
  1. 评测用全量 4,783,154 候选（不是批内候选），与 baseline/ 可比、与 O_o 日志不可比。
  2. batch_size=256（O_o 是 128）。
  3. 每个 epoch 只保留 last.pt + best.pt（1.08B 参数的 fp32 权重单份就 4.0GB）。
另外：`--neg_pop_alpha 0` 配全量 `item_logQ` —— 采样是均匀的、修正项却是流行度的
（O_o 的 dataset 默认 α=0.15 而 main.py 默认 0，它自己也不自洽）。这里照抄 O_o 的
损失式不改，写进 DIFF。
"""
import argparse
import json
import math
import os
import random
import subprocess
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import config
import data as data_mod
import eval as eval_mod
import losses
from model import tower


def resolve_val_log(spec, out_dir):
    """`--val_log` 的取值 -> 最终路径，或 None 表示不记。

    名字不写死：相对路径按 `--out_dir` 解析（跟随本次运行的输出目录），绝对路径原样用。
    `none` / `off` / 空串是「这次不要记」的开关。
    """
    if spec is None or str(spec).strip().lower() in ("", "none", "off"):
        return None
    p = Path(str(spec))
    return p if p.is_absolute() else out_dir / p


def append_jsonl(path, record):
    """一行一个 JSON 追加写（与 `train.jsonl` 同一约定）。

    追加而不是覆盖是关键：`result.json` 每个 epoch 都被重写，跑完只剩最后一个 epoch
    的指标 —— 早停一旦被冻住（见 `score_is_improvement`），事后连「哪几个 epoch 在变好」
    都看不出来，只能去 `best.pt` 的二进制里刨。崩了也只丢最后一行。
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def run_meta(epoch, train_sec, eval_sec, peak_gb, args=None):
    """一次 epoch 的元信息（**不含**指标键）。`result.json` 与 `val.jsonl` 共用这一份，
    避免只在一处记了某个字段、另一处漏掉。

    `user_seq_order` 必须落在这里：`result.json` 每个 epoch 被重写、`val.jsonl` 是逐
    epoch 追加的，两者都得能**脱离 ckpt 单独读懂**。`last.pt` 里虽然存了 `vars(args)`
    （含 `--user_seq_order`），但要刨二进制才拿得到 —— 而且两种布局的指标不可直接比较，
    事后才发现记错的代价是整轮训练白跑。

    同一个理由适用于 MoE 的五个开关：换了 `--moe_experts` / `--moe_topk` /
    `--moe_aux_alpha` / `--moe_shared_experts` / `--moe_routed_scale` 的两轮训练，
    指标同样不可直接比较，而「专家数写没写进 ckpt」在日志里完全看不出来。`args=None`
    时（旧调用点 / 测试）退回 config 的默认值。共享专家是其中最容易漏记的一个：它不改
    任何张量的形状，ckpt 照常加载，唯一的表面差异只是 step-0 的 loss 高一点。

    `moe_routed_scale`（λ）在这几个开关里是**最**该记的：它不进 state_dict、不改任何
    权重、不改任何张量形状 —— 同一份 ckpt 配不同 λ 跑出来的两轮，权重文件**逐位相同**，
    唯一能区分它们的就是这一个字段。而它的默认值 1.0 又是个「看着很正常」的数，所以
    「日志里没有这个键」和「日志里是 1.0」必须分得开（测试两个方向都钉着）。

    `moe_balance` 与 `moe_bias_gamma` 是同一类：前者决定均衡压力从哪来（辅助损失 /
    逐专家偏置 / 两者），三档指标不可直接比较；后者只改偏置的更新步长，看起来无关紧要，
    但它改的那个 buffer **会进 state_dict** —— 两轮的权重文件已经不同，而能区分它们
    的只有这一个字段。`aux_loss` / `1e-3` 都是「看着很正常」的默认值，所以
    「日志里没有这个键」与「日志里是默认值」必须分得开。

    `moe_ffn_act` 也是同一类：它**会改权重文件**（非 none 档每个专家多一层 Linear，键名
    从 `experts.0.weight` 变成 `experts.0.0.weight`、单个专家的参数量 ×1.334）。更强的
    理由是**非 none 的三档彼此完全同形** —— 激活函数本身没有参数，`silu` / `gelu` /
    `relu` 建出来的键名与张量形状逐位相同（实测三档都是 `['0.weight','0.bias',
    '2.weight','2.bias']`），事后拿到两份 ckpt 根本分不清是哪一档，唯一能反查的就是
    这个字段。

    同理还有 `--easy_neg_logq`（易负那一族的 -logQ，见 losses.info_nce）：它尤其
    容易在日志里丢 —— 因为最省事的 A/B 手法是**直接改 config 常量**而不给命令行参数。

    `time_bias_per_layer` 也在这里（见 config.TIME_BIAS_PER_LAYER）：它是**结构**开关，
    改的是模型里有没有那 8 张表。它比别的开关更容易漏记 —— 参数总量、张量形状、ckpt
    的可加载性全都不变，只有指标会不一样。

    与指标合并由调用方 `{**run_meta(...), **val}` 完成（val 后展开，同名键以 val 为准）。
    """
    if args is None:
        args = argparse.Namespace(moe_experts=config.MOE_EXPERTS,
                                  moe_topk=config.MOE_TOPK,
                                  moe_aux_alpha=config.MOE_AUX_ALPHA,
                                  moe_shared_experts=config.MOE_SHARED_EXPERTS,
                                  moe_routed_scale=config.MOE_ROUTED_SCALE,
                                  moe_balance=config.MOE_BALANCE,
                                  moe_bias_gamma=config.MOE_BIAS_GAMMA,
                                  moe_ffn_act=config.MOE_FFN_ACT,
                                  easy_neg_logq=config.EASY_NEG_LOGQ,
                                  time_bias_per_layer=config.TIME_BIAS_PER_LAYER)
    return {'epoch': epoch,
            'train_sec': round(train_sec, 1),
            'eval_sec': round(eval_sec, 1),
            'peak_mem_gb': round(peak_gb, 2),
            'user_seq_order': config.USER_SEQ_ORDER,
            'moe_experts': args.moe_experts,
            'moe_topk': args.moe_topk,
            'moe_aux_alpha': args.moe_aux_alpha,
            'moe_shared_experts': args.moe_shared_experts,
            'moe_routed_scale': args.moe_routed_scale,
            'moe_balance': args.moe_balance,
            'moe_bias_gamma': args.moe_bias_gamma,
            'moe_ffn_act': args.moe_ffn_act,
            'easy_neg_logq': args.easy_neg_logq,
            'time_bias_per_layer': args.time_bias_per_layer}


def score_is_improvement(new_score, best_score):
    """早停的「更好」判据：非有限分一律不算更好。

    让 inf 当上 best 的代价极大 —— 之后每个 epoch 与它比较都失败，patience 很快
    耗尽、训练**假性早停**。实测：epoch 1 有 1 个 query 的 rank = -1（掩蔽扣除用了
    与计数不同的 kernel），NDCG 的 1/log2(rank+2) 变成 1/log2(1) = inf，于是
    `--num_epochs 100` 在第 3 个 epoch 就停了，而模型当时还在变好
    （HR@10 0.0875 -> 0.1184）。根因已在 eval.ranks_full_candidates 修掉，
    这一层只是不让任何一个非有限值再把早停冻住。
    """
    if not math.isfinite(new_score):
        return False
    return new_score > best_score + 1e-12


def moe_loss_weight(args):
    """aux 项进损失的系数。aux_free 档恒为 0 —— 负载均衡由逐专家偏置承担，
    `aux` 只当诊断量记日志（`loss_aux` 照记原始值，只有 `loss_moe` 归零）。

    抽成函数而不是内联在训练循环里，是为了让「aux_free 到底有没有抑制损失」有一条
    能单测的断言 —— 内联的话只有跑一次 `--max_steps 1` 才能验证。

    注意 `both` 档**返回 alpha 而不是 0**：两者同时开的时候把损失项也砍掉，等于把
    `both` 静默降级成 `aux_free`，两轮指标会一模一样而日志上写着 `balance=both`。
    """
    return 0.0 if args.moe_balance == "aux_free" else float(args.moe_aux_alpha)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default=str(config.CACHE_DIR))
    p.add_argument("--out_dir", default=str(config.LOG_DIR))
    p.add_argument("--val_log", default="val.jsonl",
                   help="每个 epoch 的验证指标追加到这个文件。相对路径按 --out_dir 解析，"
                        "绝对路径原样使用；给 none/off/空串则不记。默认 val.jsonl")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--maxlen", type=int, default=101)
    # default 取 config.USER_SEQ_ORDER（= 环境变量 USER_SEQ_ORDER 或 'o_o'），**不能**写死
    # "o_o"：那样 `USER_SEQ_ORDER=chrono python train.py` 会被这里的默认值静默覆盖回 'o_o'，
    # 环境变量看起来生效了、实际没生效。
    p.add_argument("--user_seq_order", choices=["o_o", "chrono"],
                   default=config.USER_SEQ_ORDER,
                   help="user token 在序列里的先后次序（默认跟随环境变量 USER_SEQ_ORDER）。"
                        "'o_o' = 复刻 O_o，user 块时间倒序；'chrono' = 按记录实际顺序，"
                        "user 块时间正序。两种布局的指标不可直接比较，"
                        "见 DIFF_vs_O_o.md §2.3")
    p.add_argument("--moe_experts", type=int, default=config.MOE_EXPERTS,
                   help="序列塔 HSTU 每个 block 的 FFN 换成 E 个专家的 top-k MoE。"
                        "1 = 关掉（退回 nn.Linear，与 O_o 逐位等价，见 test_model_equiv）。"
                        "只影响序列塔；item 塔没有 HSTU")
    p.add_argument("--moe_topk", type=int, default=config.MOE_TOPK,
                   help="每个 token 激活几个专家（默认 1 = Switch 式 top-1）。"
                        "初始门控恒为 1/E，与 top_k 无关；k>1 时选中专家的输出相加，"
                        "初始量级因此是 √k/E 倍 baseline（仍低于 baseline），"
                        "见 model.py moe 的 docstring")
    p.add_argument("--moe_aux_alpha", type=float, default=config.MOE_AUX_ALPHA,
                   help="Switch 式负载均衡项的权重（loss += alpha * aux，aux 值域 [1, E]）。"
                        "0 = 不进损失，只在日志里记专家使用率；"
                        "aux_free 档此值不生效，见 --moe_balance")
    p.add_argument("--moe_shared_experts", type=int, default=config.MOE_SHARED_EXPERTS,
                   help="共享专家个数（DeepSeekMoE 式：输出恒 1 相加、不过门）。"
                        "每个 token 都要过，所以 top-1 下 FFN 的 FLOPs 大约翻倍；"
                        "它只与 num_experts>1 搭配（否则被断言拦住）。"
                        "默认 0 = 一个参数都不建，与改动前逐位一致。"
                        "两轮指标不可直接比较，见 config.MOE_SHARED_EXPERTS")
    p.add_argument("--moe_routed_scale", type=float, default=config.MOE_ROUTED_SCALE,
                   help="路由支路的系数 λ（论文式 (3) 的 λ）：只乘路由那一路，共享专家"
                        "那一路恒为 1。1.0 = 不缩放，与改动前逐位一致（默认）。"
                        "取值由论文那条「使两支在初始化阶段模长一致」给出，本仓库零初始化"
                        "下是闭式 λ = √s · E / √k（15/3/1 -> 8.66），见 config.MOE_ROUTED_SCALE。"
                        "它不进 state_dict、不改任何权重，所以同一份 ckpt 配不同 λ 是合法的；"
                        "两轮的区别只记在日志的 moe_routed_scale 字段里，换档务必换 OUT_DIR")
    p.add_argument("--moe_balance", choices=["aux_loss", "aux_free", "both"],
                   default=config.MOE_BALANCE,
                   help="MoE 的负载均衡策略。aux_loss（默认）= Switch 式辅助损失，"
                        "与改动前逐位一致；aux_free = DeepSeek-V3 式逐专家偏置，"
                        "**且 alpha*aux 不进损失**（aux 仍算、仍记日志）；"
                        "both = 两者同时开。三档指标不可直接比较，换档务必换 OUT_DIR")
    p.add_argument("--moe_bias_gamma", type=float, default=config.MOE_BIAS_GAMMA,
                   help="aux_free / both 档的偏置更新步长 γ（DSeek-V3 的 bias update "
                        "speed），只在更新时用、前向不碰。改 γ 会改偏置的历史，所以"
                        "**权重文件会不一样**，两轮只能靠日志的这个字段区分")
    p.add_argument("--moe_ffn_act", choices=["none", "silu", "gelu", "relu"],
                   default=config.MOE_FFN_ACT,
                   help="HSTU block 的 FFN 激活函数。none（默认）= 每个专家是裸 "
                        "nn.Linear，与改动前逐位一致；silu / gelu / relu = 每个专家变成"
                        "两层 MLP W2·act(W1·x)，参数量与 FLOPs 各 ×1.334。"
                        "本开关也作用于共享专家与 --moe_experts 1 那条兜底路径（此时 "
                        "FFN 是单个 nn.Linear），所以 `--moe_experts 1 --moe_ffn_act "
                        "silu` 是一个稠密的两层 MLP FFN，可用来分离「非线性容量」与"
                        "「专家数量」。"
                        "四档指标不可直接比较，换档务必换 OUT_DIR")
    p.add_argument("--seed", type=int, default=20252026)
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--hidden_units", type=int, default=512)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.025)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--neg_pop_alpha", type=float, default=0.0)
    p.add_argument("--easy_neg_logq", action=argparse.BooleanOptionalAction,
                   default=config.EASY_NEG_LOGQ,
                   help="易负样本那一族是否用 -logQ 修正（默认跟随 config.EASY_NEG_LOGQ）。"
                        "它与 --neg_pop_alpha 是一对：alpha=0 时易负是均匀采样的，"
                        "--no-easy_neg_logq 才自洽；alpha=1.0 时反过来，两个都是"
                        "「要么修 proposal、要么修修正项」。正样本/难负两族不受影响。"
                        "两轮指标不可直接比较，见 losses.info_nce 的 docstring")
    p.add_argument("--time_bias_per_layer", action=argparse.BooleanOptionalAction,
                   default=config.TIME_BIAS_PER_LAYER,
                   help="HSTU 的相对时间偏置表：默认全局一张（8 层共用，对齐 O_o）；"
                        "打开后每层一张（8 张 × 129 行 = 1032 个参数）。桶的定义、初始化、"
                        "三路拼接都不变，只改「几张表」—— 但两档的指标不可直接比较，"
                        "务必同时换 OUT_DIR/OUT_LOG，见 config.TIME_BIAS_PER_LAYER")
    p.add_argument("--ssl_alpha", type=float, default=0.5)
    p.add_argument("--ssl_mask_ratio", type=float, default=0.6)
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="验证集用户比例，默认 0.1（训练:验证 = 9:1，实测 901,661:100,184）。"
                        "O_o 原版是 0.01，见 DIFF_vs_O_o.md §6.2 第 11 条")
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--eval_all_clicks", action=argparse.BooleanOptionalAction, default=True,
                   help="每个点击位置都算一次 rank（默认，对齐 O_o/main.py:258）；"
                        "--no-eval_all_clicks 则每用户只取最后一个点击位置 t*（留一法）")
    p.add_argument("--eval_last_click_too", action=argparse.BooleanOptionalAction, default=True,
                   help="除了主口径，再顺带算一遍 t* 口径存进 result.json（*_last_click），"
                        "早停仍只看主口径的 score。同一遍前向，代价很小")
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--eval_cand_chunk", type=int, default=262144)
    p.add_argument("--max_steps", type=int, default=0,
                   help=">0 时只跑这么多步就结束 epoch（冒烟用；正式跑保持 0）")
    return p.parse_args()


Card = namedtuple("Card", "index free_gb util uuid")


def free_gpu(need_gb=None):
    """挑一张最空闲的卡，返回 `Card(index, free_gb, util, uuid)`；挑不出来返回 None。

    规则：**在「空闲显存 >= need_gb」的卡里选利用率最低的**，同利用率取空闲多的；
    一张都不达标时退化为「空闲显存最多」的一张（宁可变慢，也别因为显存不够崩在半路）。
    `need_gb` 默认取环境变量 `NEED_GB`，再默认 50 —— 实测训练峰值 ~44G（评测另需 ~10G）。

    为什么用 nvidia-smi 而不是 torch：这个决定必须在 CUDA 初始化**之前**做，否则设
    `CUDA_VISIBLE_DEVICES` 已经晚了（torch 一旦枚举过设备就定死了）。所以这里连
    `torch.cuda.device_count()` 都不能调。

    **必须带上 uuid**：本机 A100/H100 混插，nvidia-smi 按 PCI 顺序列卡，而 CUDA 默认按
    「最快优先」枚举（4 张 H100 排在前）——两套卡号完全错位。实测 `CUDA_VISIBLE_DEVICES=1`
    （想选 nvidia-smi 的 1 号 A100）落到了 3 号 H100 上。UUID 与枚举顺序无关，选卡只认它。
    """
    if need_gb is None:
        need_gb = float(os.environ.get("NEED_GB", "50"))
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu,uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        cards = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 5:
                continue
            idx, used, total, util = (int(float(p)) for p in parts[:4])
            cards.append(Card(idx, (total - used) / 1024.0, util, parts[4]))
    except Exception:
        return None
    if not cards:
        return None
    ok = [c for c in cards if c.free_gb >= need_gb]
    if ok:
        return min(ok, key=lambda c: (c.util, -c.free_gb))
    return max(cards, key=lambda c: c.free_gb)


def auto_pick_gpu():
    """没显式指定 `CUDA_VISIBLE_DEVICES` 时自动选卡；返回选中的卡号或 None。

    必须在任何 CUDA 调用（含 `set_seed` 里的 `torch.cuda.manual_seed_all`）之前调用。
    """
    # 统一枚举顺序，好让手写的 `CUDA_VISIBLE_DEVICES=3` / `GPU=3` 与 nvidia-smi 的卡号
    # 指同一张卡（否则 cuda:0 是「最快的卡」而不是「0 号卡」，混插机器上会悄悄错位）
    if "CUDA_DEVICE_ORDER" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("[gpu] 沿用已有的 CUDA_VISIBLE_DEVICES=%s（不再自动挑）"
              % os.environ["CUDA_VISIBLE_DEVICES"], flush=True)
        return None

    card = free_gpu()
    if card is None:
        print("[gpu] nvidia-smi 不可用，交给 torch 默认（第一张可见卡）", flush=True)
        return None
    if card.free_gb < float(os.environ.get("NEED_GB", "50")):
        print("[gpu] 没有卡满足 NEED_GB=%s，退化为空闲最多的一张"
              % os.environ.get("NEED_GB", "50"), flush=True)
    # 用 UUID 而不是卡号：与 CUDA/nvidia-smi 的枚举顺序都无关
    os.environ["CUDA_VISIBLE_DEVICES"] = card.uuid
    print("[gpu] 自动选卡 GPU=%d（利用率 %d%%，空闲 %.0fG，%s）"
          % (card.index, card.util, card.free_gb, card.uuid), flush=True)
    return card.index


def set_seed(seed):
    """与 baseline/train.py:65 同口径。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(meta, args, device):
    kwargs = config.tower_kwargs(meta, args.embedding_dim, args.hidden_units, args.dropout_rate,
                                 num_experts=args.moe_experts, top_k=args.moe_topk,
                                 num_shared=args.moe_shared_experts,
                                 routed_scale=args.moe_routed_scale,
                                 balance=args.moe_balance,
                                 ffn_act=args.moe_ffn_act,
                                 time_bias_per_layer=args.time_bias_per_layer)
    model = tower(**kwargs).to(device)
    # init_weights 会给每个 nn.Linear 装 xavier —— 路由权重是**裸 Parameter**，
    # 因此不受影响、保持零初始化（初始门控精确等于 1/E 的来源，见 model.py 的 moe）。
    # 共享专家**是** nn.Linear，所以这里照常吃到 xavier，正是它该有的初始化。
    # λ（routed_scale）是 python float、不是 buffer/Parameter，init_weights 碰不到它，
    # state_dict 里也没有它的键 —— 这是「同一份权重配不同 λ」能成立的前提。
    model.apply(losses.init_weights)
    return model


def main():
    args = parse_args()
    # 开关落在 config 上，而不是逐层传参：data.py 的 build_user_sample 是在 DataLoader
    # 的 worker 进程里被调用的，透传要改 build_loaders / TowerDataset / build_user_sample
    # 三层签名。**必须在 build_loaders 之前赋值** —— worker 是 fork 出来的，晚于它们启动
    # 再改就已经是旧值了，而且不报错、只是静默跑错布局。
    config.USER_SEQ_ORDER = args.user_seq_order
    auto_pick_gpu()          # 必须在 set_seed 之前：它内部会初始化 CUDA
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cache = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_log = resolve_val_log(args.val_log, out_dir)
    if val_log is not None:
        val_log.parent.mkdir(parents=True, exist_ok=True)
    meta = config.load_meta(cache)
    device = torch.device("cuda")
    # user_seq_order 打在这里：日志是唯一「扫一眼就知道这轮跑的哪套布局」的地方，
    # 而两种布局的指标不可直接比较。结构/损失开关同理，一并打在这一行。
    print("[env] %s | amp=%s | batch=%d | workers=%d | user_seq_order=%s"
          " | moe=%dx top%d shared%d scale=%s balance=%s gamma=%s alpha=%s ffn_act=%s"
          " | neg_pop_alpha=%s | easy_neg_logq=%s"
          " | time_bias_per_layer=%s"
          % (torch.cuda.get_device_name(0), args.amp, args.batch_size, args.num_workers,
             config.USER_SEQ_ORDER, args.moe_experts, args.moe_topk,
             args.moe_shared_experts, args.moe_routed_scale, args.moe_balance,
             args.moe_bias_gamma, args.moe_aux_alpha, args.moe_ffn_act,
             args.neg_pop_alpha, args.easy_neg_logq, args.time_bias_per_layer),
          flush=True)
    # 打出来，省得事后找「这轮的逐 epoch 指标写哪去了」
    print("[log] 每 epoch 验证指标 -> %s"
          % (val_log if val_log is not None else "(本次不记，--val_log none)"), flush=True)

    loaders = data_mod.build_loaders(
        maxlen=args.maxlen, batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, cache_dir=cache, val_ratio=args.val_ratio,
        ssl_mask_ratio=args.ssl_mask_ratio, neg_alpha=args.neg_pop_alpha)
    train_loader, val_loader = loaders["train"], loaders["val"]
    print("[data] 训练用户=%d 验证用户=%d" % (loaders["n_train"], loaders["n_val"]), flush=True)

    item_feat = np.load(cache / "item_feat.npy", mmap_mode="r")
    item_cols = [config.ITEM_STATIC_INDEX[c] for c in config.ITEM_SPARSE_ITEM]
    item_logQ = torch.from_numpy(np.load(cache / "logq.npy")).to(device)

    model = build_model(meta, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print("[model] 参数量 %.3fB | 权重 fp32 %.1fGB"
          % (n_params / 1e9, n_params * 4 / 2 ** 30), flush=True)

    # 与 O_o/main.py:497-502 一致：betas=(0.9,0.98)，wd=1e-4，不排除任何参数（含 embedding）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.98), weight_decay=args.weight_decay)
    steps_per_epoch = args.max_steps if args.max_steps > 0 else len(train_loader)
    total_steps = steps_per_epoch * args.num_epochs
    warmup = max(1, int(total_steps * 0.1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup,
                                                       eta_min=1e-6),
        ],
        milestones=[warmup],
    )

    amp_enabled = args.amp != "none"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp, None)
    # GradScaler 只对 fp16 有意义；bf16 不需要（O_o/main.py:505-508 同）
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    log_file = open(out_dir / "train.jsonl", "a", encoding="utf-8")
    writer = SummaryWriter(str(out_dir / "tb"))
    global_step = 0
    best_score, epochs_no_improve = -1e30, 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(train_loader):
            if args.max_steps > 0 and step >= args.max_steps:
                break
            (seq, pos, neg, token_type, next_token_type, next_action_type,
             seq_feat, pos_feat, neg_feat, seq_ts,
             neg_feat_ssl1, neg_feat_ssl2, neg_ssl1, neg_ssl2, user_feat) = batch[:15]
            user_arrays = [t.to(device, non_blocking=True) for t in batch[15:]]
            seq = seq.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            neg = neg.to(device, non_blocking=True)
            token_type = token_type.to(device, non_blocking=True)
            next_token_type = next_token_type.to(device, non_blocking=True)
            next_action_type = next_action_type.to(device, non_blocking=True)
            seq_feat = seq_feat.to(device, non_blocking=True)
            pos_feat = pos_feat.to(device, non_blocking=True)
            neg_feat = neg_feat.to(device, non_blocking=True)
            seq_ts = seq_ts.to(device, non_blocking=True)
            user_feat = user_feat.to(device, non_blocking=True)
            neg_ssl1 = neg_ssl1.to(device, non_blocking=True)
            neg_ssl2 = neg_ssl2.to(device, non_blocking=True)
            neg_feat_ssl1 = neg_feat_ssl1.to(device, non_blocking=True)
            neg_feat_ssl2 = neg_feat_ssl2.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                # 三次前向取代 O_o 的单个 fused forward（O_o 的 forward 内部就是这三件事）
                # 序列塔那次顺带取 aux（MoE 的负载均衡项，标量），只多一个返回值、不多一次前向
                log_feats, moe_aux = model(seq, token_type, user_feat, seq_feat, seq_ts,
                                          user_arrays, return_aux=True)
                pos_embs = model(pos, item_feat=pos_feat)
                neg_embs = model(neg, item_feat=neg_feat)
                loss_main, stats = losses.info_nce(
                    pos_embs, neg_embs, log_feats, temperature=args.temperature,
                    next_token_type=next_token_type, next_action_type=next_action_type,
                    pos_ids=pos, neg_ids=neg, item_logQ=item_logQ,
                    easy_neg_logq=args.easy_neg_logq)
                if args.ssl_alpha > 0.0:
                    z1 = model(neg_ssl1, item_feat=neg_feat_ssl1)
                    z2 = model(neg_ssl2, item_feat=neg_feat_ssl2)
                    m_ssl = (token_type == 1)
                    if m_ssl.dtype is not torch.bool:
                        m_ssl = m_ssl.bool()
                    loss_ssl = losses.ssl_loss(z1[m_ssl], z2[m_ssl], temperature=args.temperature)
                else:
                    loss_ssl = torch.zeros((), device=device)
                # MoE 的负载均衡项：值域 [1, E]，均匀=1（下界）；要取到上界 E 需要 f 与 P
                # **同时**塌到一个专家（只有 f 塌、P 还均匀时是下界 1，见 moe._load_balance）。
                # α=0、experts=1、或 balance=aux_free 时这一项恒为 0，自动退化成
                # 「什么都不加」—— 注意第三种情况下 aux 本身仍被算出来、仍记在
                # `loss_aux` 那一列里，只有进损失的那份权重归零（见 moe_loss_weight）。
                loss_moe = moe_loss_weight(args) * moe_aux
                loss = loss_main + float(args.ssl_alpha) * loss_ssl + loss_moe

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            # aux_free / both 档：按**刚过去那次序列塔前向**的专家负载更新偏置。
            # 必须在 optimizer.step() 之后（DSeek 原版就是「每个训练 step 之后」），
            # 且读的就是本 step 的 f —— pos/neg/ssl 走 item 塔，不会覆盖 expert_frac。
            # aux_loss 档这是 no-op（expert_bias is None），所以不加分支。
            model.update_expert_bias(args.moe_bias_gamma)
            scheduler.step()

            if step % 20 == 0:
                # MoE 诊断量取自**序列塔**最近一次前向（item 塔没有 HSTU，不覆盖它）。
                # 塌缩在 loss 上完全看不出来，只有 expert_frac 能发现 —— 所以每 20 步
                # 采样一次、整份写进 jsonl（每层 4 个比例），控制台只打最能说明问题的一行。
                moe = model.moe_stats()
                frac = [max(f) for f in moe['expert_frac']]          # 每层最挤的专家占比
                rec = {'global_step': global_step, 'epoch': epoch, 'step': step,
                       'loss_main': float(loss_main.item()), 'loss_ssl': float(loss_ssl.item()),
                       'loss_aux': float(moe_aux.item()), 'loss_moe': float(loss_moe.item()),
                       'loss_total': float(loss.item()),
                       'LR': optimizer.param_groups[0]['lr'],
                       'mean_pos_sim': stats['mean_pos_sim'],
                       'mean_neg_sim': stats['mean_neg_sim'],
                       'mean_hard_neg_sim': stats['mean_hard_neg_sim'],
                       'mask_rate_hard': stats['mask_rate_hard'],
                       'moe_expert_frac': moe['expert_frac'],
                       'moe_gate_mean': moe['gate_mean'],
                       'moe_bias': moe['expert_bias'],
                       'peak_gb': torch.cuda.max_memory_allocated() / 2 ** 30,
                       'time': time.time()}
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                writer.add_scalar('Loss/main', rec['loss_main'], global_step)
                writer.add_scalar('Loss/ssl', rec['loss_ssl'], global_step)
                writer.add_scalar('Loss/aux', rec['loss_aux'], global_step)
                # aux 的理论下界是 1.0（均匀）、上界 E；maxf 是「最挤那一层最挤的那个专家」
                # 的占比：0.25 = 完美均匀，1.0 = 完全塌缩。两者一起看才知道路由在不在干活。
                if frac:
                    gate_mean = sum(moe['gate_mean']) / len(frac)
                    writer.add_scalar('MoE/gate_mean', gate_mean, global_step)
                    moe_txt = "aux=%.4f gate=%.3f maxf=%.3f" % (rec['loss_aux'], gate_mean,
                                                               max(frac))
                else:
                    moe_txt = "moe=off"
                print("[e%d s%d/%d] loss=%.4f main=%.4f ssl=%.4f lr=%.2e hard=%d peak=%.1fG %s"
                      % (epoch, step, steps_per_epoch, rec['loss_total'], rec['loss_main'],
                         rec['loss_ssl'], rec['LR'], int((next_token_type == 1).sum()),
                         rec['peak_gb'], moe_txt),
                      flush=True)
            global_step += 1

        # ---- 验证：全量候选口径 ----
        t_eval = time.time()
        cand = eval_mod.build_candidate_matrix(model, item_feat, item_cols,
                                               device=device, amp_dtype=amp_dtype)
        val = eval_mod.evaluate_full(model, val_loader, cand, device=device,
                                     amp_dtype=amp_dtype, cand_chunk=args.eval_cand_chunk,
                                     all_clicks=args.eval_all_clicks,
                                     last_click_too=args.eval_last_click_too)
        eval_sec = time.time() - t_eval
        peak_gb = torch.cuda.max_memory_allocated() / 2 ** 30
        # 早停只看主口径的 score（默认 = 每个点击位置；`*_last_click` 仅作对照）
        print("[epoch %d] %.1fs train, %.1fs eval | %s HR@10=%.6f NDCG@10=%.6f score=%.6f "
              "(n=%d, skipped=%d, mean_rank=%.0f, peak=%.1fG)"
              % (epoch, t_eval - t0, eval_sec,
                 "all-click" if args.eval_all_clicks else "last-click",
                 val['hr10'], val['ndcg10'],
                 val['score'], val['n_eval'], val['n_skipped'], val['mean_rank'], peak_gb),
              flush=True)
        if "score_last_click" in val:
            print("[epoch %d]   t* 口径: HR@10=%.6f NDCG@10=%.6f score=%.6f (n=%d)"
                  % (epoch, val['hr10_last_click'], val['ndcg10_last_click'],
                     val['score_last_click'], val['n_eval_last_click']),
                  flush=True)
        del cand
        torch.cuda.empty_cache()

        # 只留 last.pt + best.pt：单份 fp32 权重就 4.0GB，全存会吃掉磁盘
        with open(out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump({**run_meta(epoch, t_eval - t0, eval_sec, peak_gb, args), **val},
                      f, ensure_ascii=False, indent=2)

        ckpt = out_dir / "last.pt"
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, ckpt)

        improved = score_is_improvement(val['score'], best_score)
        # 每个 epoch 的全部验证指标追加进 val.jsonl（一行一个 epoch）。
        # `val` 里的两种口径（主口径 + `*_last_click` 那一组）原样带进来；
        # 元信息用 update 补，避免将来 eval 那边多出一个同名的键就把这里打崩
        # —— 记录日志的代码不该有把训练搞挂的能力。
        rec_val = dict(val)
        rec_val.update(run_meta(epoch, t_eval - t0, eval_sec, peak_gb, args),
                       lr=optimizer.param_groups[0]['lr'], is_best=improved,
                       time=time.time())
        if val_log is not None:          # --val_log none/off 就是这一次不记
            append_jsonl(val_log, rec_val)

        if not math.isfinite(val['score']):
            print("[epoch %d] 警告：验证分非有限（score=%r），不计入早停判据"
                  % (epoch, val['score']), flush=True)
        if improved:
            best_score, epochs_no_improve = val['score'], 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'metrics': val},
                       out_dir / "best.pt")
            print("[epoch %d] new best %.6f" % (epoch, best_score), flush=True)
        else:
            epochs_no_improve += 1
            print("[epoch %d] no improvement (%d/%d)" % (epoch, epochs_no_improve, args.patience),
                  flush=True)
            if epochs_no_improve >= args.patience:
                print("[early stop] best=%.6f" % best_score, flush=True)
                break

    log_file.close()
    writer.close()


if __name__ == "__main__":
    main()
