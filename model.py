import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class glu(nn.Module):
    """门控特征交互块，逐字等于 O_o 的 `FeatureInteractionEncoder`（O_o/model.py:257-285）。

    `X + X * sum_r H` 与 O_o 的 `einsum("bsd,bsrd->bsd", X, H) + X` 是同一条式子：
    后者里 `d` 在输出中故为自由维、`r` 不在输出中故被求和。
    两处 Linear 都是 bias=False（对齐 O_o 的 down_proj / gate）。
    """

    def __init__(self, input_dim, output_dim, expand_dim, dropout=0.1):
        super(glu, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)  # O_o: down_proj
        self.expand_dim = expand_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim * expand_dim, bias=False),  # O_o: gate
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [b, s, input_dim]
        X = self.linear(x)
        g = self.mlp(X).view(x.shape[0], x.shape[1], self.expand_dim, self.output_dim)
        return X + X * g.sum(dim=2)
        
def rope(q, k, base=10000):
    # q: [b, head, s, d] k: [b, head, s, d]
    pos = torch.arange(0, q.shape[-2], device=q.device).float() #[s]
    theta = torch.arange(0, q.shape[-1], 2, device=q.device).float() / q.shape[-1] #[d/2]
    theta = base ** theta #[d/2]
    theta = pos.unsqueeze(-1) / theta.unsqueeze(0) #[s, d/2]
    cos_, sin_ = torch.cos(theta), torch.sin(theta) #[s, d/2]
    # theta 由 arange(...).float() 构造，恒为 fp32。不转回 q.dtype 的后果：
    #  - 手动 .to(bfloat16) 前向时直接抛 RuntimeError（q1 * cos_ 类型不匹配）
    #  - autocast 下不报错，但 q1*cos_ 会把结果升到 fp32，导致 rope 那一路的
    #    注意力与另外两路精度不同（O_o/model.py:68 用 _build_cos_sin(..., q.dtype) 避免）
    cos_ = cos_.to(q.dtype)
    sin_ = sin_.to(q.dtype)
    cos_ = cos_.view(1, 1, theta.shape[0], -1) #[1, 1, s, d/2]
    sin_ = sin_.view(1, 1, theta.shape[0], -1)
    q1, q2 = q[..., ::2], q[..., 1::2] #[b, head, s, d/2]
    k1, k2 = k[..., ::2], k[..., 1::2] #[b, head, s, d/2]
    q_real = q1 * cos_ - q2 * sin_
    q_imag = q1 * sin_ + q2 * cos_
    k_real = k1 * cos_ - k2 * sin_
    k_imag = k1 * sin_ + k2 * cos_
    q = torch.stack([q_real, q_imag], dim=-1).flatten(-2) #[b, head, s, d]
    k = torch.stack([k_real, k_imag], dim=-1).flatten(-2) #[b, head, s, d]
    return q, k

    



class FourierTimeEncoding(nn.Module):
    """绝对时间戳（秒）-> 多频正余弦 -> 线性投到 hidden_units。

    作为 hour / dow / weekend 三张离散表的补充：那三张表只能表达「几点的」
    「周几的」这种循环位置，表达不了「两次行为隔了多少天」的绝对尺度。
    周期对数等分，是 buffer 不参与训练。

    ⚠️ min_period_seconds 默认 1 天，**对齐的是 O_o 的实际行为而非它的注释**：
    `O_o/model.py:139` 的注释写「使用 8*86400.0 以防止与 hour、weekday 重复」，
    但同一行的默认值是 `86400.0`，且 `O_o/model.py:333` 的调用
    `FourierTimeEncoding(hidden_units=...)` 没有覆盖它 —— 所以 O_o 实际跑的是
    1~40 天，其中 6 个频率落在 8 天以下（含一个日周期分量）。
    要回到「8 天起步」的版本，把默认值改回 `8 * 86400.0` 即可。
    """
    def __init__(self, hidden_units, num_frequencies=12,
                 min_period_seconds=86400.0, max_period_seconds=40 * 86400.0):
        super(FourierTimeEncoding, self).__init__()
        assert num_frequencies >= 1
        assert max_period_seconds > min_period_seconds > 0
        periods = torch.logspace(
            math.log10(min_period_seconds),
            math.log10(max_period_seconds),
            steps=num_frequencies,
        ) #[L]
        self.register_buffer('periods', periods)
        self.register_buffer('two_pi', torch.tensor(2.0 * math.pi))
        self.proj = nn.Linear(2 * num_frequencies, hidden_units, bias=False)

    def forward(self, ts):
        # ts: [b, s] 单位秒（int/float 均可）
        t = ts.to(device=self.periods.device, dtype=torch.float32) #[b, s]
        angle = (t.unsqueeze(-1) / self.periods) * self.two_pi #[b, s, L]
        feats = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1) #[b, s, 2L]
        return self.proj(feats) #[b, s, hidden_units]


# HSTU block 的 FFN 激活函数白名单。**键必须与 train.py 的 --moe_ffn_act choices
# 逐字一致** —— 两边对不上时 CLI 能过、这里断言失败，报错点离原因很远。
_FFN_ACT = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU}


def _make_ffn(input_dim, output_dim, act):
    """按 act 造一个 FFN 单元：'none' -> 裸 nn.Linear，其余 -> W2 · act(W1 x)。

    供三处使用：moe 的路由专家、moe 的共享专家、hstublock 在 num_experts<=1 时的稠密
    兜底 —— 最后那个不是「专家」，所以名字取 FFN 而不是 expert。

    act='none' 时**恰好**返回 nn.Linear，不是「建了 Sequential 再跳过激活」：
    state_dict 的键名（`experts.0.weight` 而不是 `experts.0.0.weight`）与随机数消耗
    都必须与改动前逐位一致 —— 之前跑过的 A/B 靠它成立。工厂自身的这条性质由
    tests/test_ffn_act.py::test_none_returns_a_plain_linear 钉住（直接测工厂本身）；
    moe.experts 的**集成**键名由同文件的 ::test_none_keeps_the_experts_as_plain_linears
    钉住。
    **注意 tests/test_model_equiv.py 不覆盖这条** —— 它只走 num_experts=1 的稠密兜底
    （读的是 blk.ffn.weight），既不构造 moe、也不碰 experts.* 的键。

    act 非 none 时返回 Sequential(W1, act, W2)：两层都吃 losses.init_weights 的
    xavier_uniform_（model.apply 会递归进 Sequential）、bias 都清零，与既有专家的
    初始化口径一致。

    step-0 时多一层 + 激活会明显压低 FFN 输出（三个档压得不一样），这会连带削弱
    config.MOE_ROUTED_SCALE 那条闭式 λ = √s·E/√k 的前提（它假设初始化阶段路由支路
    与共享专家支路模长一致），两个开关在 step-0 上不正交。设计文档 §6.2 的**代理**测量
    （iid 高斯输入）可迁移的是「明显变小」的方向与「腰斩上下」的量级，绝对数值不要引用。
    """
    if act == "none":
        return nn.Linear(input_dim, output_dim)
    assert act in _FFN_ACT, "act 只能是 'none' / %s，实际 %r" % (" / ".join(_FFN_ACT), act)
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),    # W1
        _FFN_ACT[act](),
        nn.Linear(output_dim, output_dim),   # W2
    )


class moe(nn.Module):
    """HSTU block 的 FFN 换成 top-k MoE（Switch Transformer 式硬路由）。

    **门控 = `softmax(logits)[选中槽位]`，不做任何缩放** —— 门控就是路由概率本身
    （∈ (0, 1]），与 Switch/Mixtral 一致。零初始化的路由 -> logits 恒 0 -> softmax
    均匀 -> 初始门控**精确**等于 1/E，选中几个槽位就加起来几个（k=1 时是 1/E）。

    曾经乘回专家数 `E`，为的是让 step 0 的门控恰为 1.0、MoE 块与 baseline 的
    `nn.Linear` 同量级（block 末尾是 `norm(ffn(x) + x)`，RMSNorm 归一化不掉残差配比，
    同 DIFF §1.10 的教训）。那条路被放弃了，三个理由：
      1. 门控不再是概率（上界变成 E），训练中 FFN 分支被**系统性放大**，A/B 里除路由
         外还混着「分支强度偏大」这个变量 —— 换掉的混淆换进来一个新混淆；
      2. Switch 不缩放、Mixtral 在选中集内 renormalize，**两家的门控都 ≤ 1**，
         `E * softmax` 是本仓库的本地发明，没有对齐基准；
      3. 它只锚住 step 0 这一个点，而上面第 1 条的代价贯穿整个训练。
    代价要认：step 0 的 FFN 分支比 baseline 弱 1/E，A/B 的初始可比性改由「前若干步的
    loss 轨迹」来保证，不再靠初始化量级对齐。（`num_shared > 0` 时这个锚点由共享专家
    那一路补回来 —— 见下面「共享专家」一段，那是本模块唯一一条不乘门控的路径。）

    **连带：`moe_aux_alpha` 的旧标定作废，默认从 0.1 改成 0.025。** 去掉 `E` 之后路由从
    task 拿到的梯度（`∂gate/∂logits`，从 `E*p*(δ-p)` 变成 `p*(δ-p)`）小了 E 倍，而 aux
    走的是 `probs`、完全不受影响，所以 `r = ‖g_task‖/‖g_aux‖` 变成原来的 1/E（step 0
    实测：E=4 比值 3.4–3.7、E=16 14.3–15.0）。**但 `0.025` 只是这个梯度尺度换算，不是
    标定结果，而且系统性偏低** —— α 变小 -> 路由更集中 -> `p_选中` 变大 -> r 变大 ->
    又需要更大的 α，线性换算没算这个负反馈。21 步的探针也钉不住它：(4,1) 的 `maxf@20`
    在 α = 0.025 / 0.05 / 0.1 上是 0.731 / 0.826 / 0.511，**非单调**，轨迹噪声压过了
    α 的效应。所以 α 只能在正式跑里扫，详见 config.py 该常量处的说明。

    k>1 时门控之和 = 选中槽位的概率之和（top-k mass），k=1 时就是 1/E —— 不再需要
    任何「补 k」的修正。注意那 k 个专家的输出是**相加**的（见 `_dispatch_topk`），
    k 个独立初始化的 Linear 之和方差相加，所以初始量级是 **√k/E 倍** baseline
    （实测 E=4：k=1/2/3/4 -> 0.249/0.352/0.431/0.497 倍 std）—— 比 top-1 的 1/E 好，
    但**仍然低于 baseline**：多选几个专家补不回「门控不缩放」丢掉的那 E 倍。
    （曾记成 `√k` = 1.414/1.732/2.000，那是**乘 E 时代**的数 —— 门控为 1.0 时方差才是
    `k·σ²`。那条路连同那两个数一起作废，别再引用。）跨 top_k 的 A/B 仍要认这条。

    **别改成「先 topk 再在选中集上 softmax」来压那个 √k/E**（Mixtral 的 renormalized
    top-k）：门控之和恒为 1 的代价是 k=1 时 softmax 作用在单个元素上恒等于
    `exp(x)/exp(x) = 1`，**门控对 logits 的梯度精确为 0**（实测 ∂gate/∂logits：该写法
    0.000000、本写法 0.1552）。路由于是只剩 aux 一个梯度，而 aux 只会把它推向均匀，
    专家再也不会因为「更适合这个 token」被选中 —— 而 top_k=1 正是本实验唯一的档位。
    更一般地：**「门控之和恒为 1」与「门控对全部 E 个 logits 可导」在 k=1 时互斥** ——
    恒为 1 要求在选中集内重新归一化，那必然掐断非选中 logits 的梯度，k=1 时选中集只剩
    一个元素，连它自己也一起掐断。

    专家只在「分到自己的 token 子集」上做矩阵乘，top-1 的 FLOPs 因此与单个 FFN 相同，
    不随专家数增长。子集怎么切见 `_dispatch_top1`（默认 top_k=1 的快路径）与
    `_dispatch_topk`（k>1 的通用慢路径）。

    **共享专家（`num_shared > 0`，DeepSeekMoE 式）：**
        y = Σ_s `shared[s](x)` + Σ_{k 个选中槽位} `softmax(logits)[槽位] · expert(x)`
    共享那一路的系数**恒为 1、不过门** —— 这不是省略，是 DeepSeekMoE 的原版语义：
    「所有 token 都要走的、不参与路由的那部分知识」交给一个稠密 FFN 去学，路由专家
    因此可以专心做特化，而不用每个专家都重复学一遍公共模式；负载均衡的压力也小了
    （公共部分不再需要靠路由去分摊）。因为**没有门控，就没有概率可乘** —— 写成
    `gate * shared` 需要先给共享专家造一个门（原版没有），写成 `(1/E) * shared` 则是
    把路由的初始劣势又抄了一遍。恒 1 的直接收益是补回了上面那条「step 0 比 baseline
    弱 1/E」：初始方差变成 `(1 + 1/E²)·σ²`，E=4 时 std ≈ **1.031×** baseline
    （k>1 时是 `√(k/E² + 1)`，k 个路由专家那部分还是相加），量级锚点回来了，而
    代价只有「多一路稠密 FFN」，不再需要「训练全程门控被放大」。

    代价要说清楚，它是**算力**不是参数量：共享专家是稠密的，每个 token 都要过。top-1
    时 FFN 的矩阵乘 FLOPs 因此**翻倍**（1 个共享 + 1 个路由 vs 原来 1 个路由），序列塔
    8 层每层如此；参数量却只涨 6.3M（占全模型 0.578%）。**方向与「把 E 调大」正好相反**
    —— 那条路只涨参数、FLOPs 一动不动（见 `_dispatch_top1`），共享专家则是拿 2x FLOPs
    换「每个 token 都有一份不受路由噪声影响的 FFN 通路」这个结构性质。其实测的收益在
    step 0 那一个锚点上（1/E -> 1.031x），更大的那部分值不值只能靠 A/B 说话。

    **共享专家与 λ 都不参与 `f` / `P`**：`_load_balance` 收到的仍是纯路由的 `probs` 与
    `idx`，所以 `aux` 的值与 `num_shared`、`routed_scale` 都无关（逐位相同，测试钉着）。
    把共享专家的 logits 也喂进路由会让「负载均衡」这个概念失去基准 —— 它本来就该只
    描述**被路由的那部分**；λ 则是个输出端的常数，与「谁分到了多少 token」无关。
    副作用是好的：改 λ 不会顺带把 `alpha * aux` 那一项也动掉，A/B 的变量是干净的。

    `num_shared > 0` 且 `num_experts == 1` 是**非法组合、当场断言**：那一档走 `nn.Linear`，
    本模块根本不会被构造，共享专家会被无声丢弃，而 `loss + alpha * aux` 里的 aux 会退化
    成恒定的 1.0（不是 0，见 `hstublock` 的 `has_moe`）—— 损失被垫高一个常数，日志上
    完全看不出来。一个「打开了共享专家但什么都没发生」的 A/B 是白跑几小时，所以由
    断言拦住。同一个断言在 `hstublock` 上也有一份（那里的 `has_moe` 才是真正的分支）。
    λ 的边界**同一个**（也是 `num_experts <= 1`，也是长在 `hstublock` 上）：那一档 FFN
    是 nn.Linear，λ 会被静默丢掉，而「λ=1」与「λ=8.66」的日志只差一个字段，最容易
    被当成「跑过了」。注意这里**不**要求 `num_shared > 0` —— 不开共享专家时 λ 照样
    乘在路由输出上（实测 `routed/x` 从 0.00128 抬到 0.00128·λ），那是真实生效的配置，
    断言掉它反而挡了一条合法路径。

    **路由支路的系数 λ（`routed_scale`，默认 1.0）：**
        y = Σ_s `shared[s](x)` + **λ** · Σ_{k 个选中槽位} `softmax(logits)[槽位] · expert(x)`
    λ 只乘**路由**那一路（在上面那段公式里它在求和号外面，与论文式 (3) 一致），共享那
    一路恒为 1。默认 1.0 时走原路径、与改动前**逐位一致**。它是个 Python float、不是
    Parameter，所以不进 state_dict、`losses.init_weights` 也碰不到 —— 换 λ 只改指标，
    **不改任何权重文件**（同一份 ckpt 配不同 λ 是两轮不同的实验，这就是 run_meta 要记
    它的原因）。

    加它的理由：上面那条「门控不做缩放」（见本 docstring 开头）使初始的路由支路只有
    baseline 的 `√(k)/E`，而共享专家的模长是 `√s` —— E=15/k=3/s=1 时前者只有后者的
    **1/8.7**。两支量级差这么多时，路由专家能分到的梯度微乎其微，等于白开路由。λ 的
    作用就是把两支拉回同一量级，取值由论文那条原则给出：**适当的 λ 应使得两者在初始
    化阶段模长接近一致**（`y = Σe + λ Σρe`，令 `λ√(Σρ²) = √s`）。

    本仓库**不用**论文里的数值模拟：`router_weight` 是零初始化的裸 Parameter，logits
    恒为 0、softmax 精确均匀，于是 `Σρ²` 是个确定值而不是一个分布，模拟退化成一行闭式
    （论文假设 `logits ~ N(0,1)` 才需要模拟）：

        λ = √s / √(Σρ²) = √s · E / √k        （E=15/k=3/s=1 -> 8.66）

    按论文那个 N(0,1) 假设模拟会得到 ~3.55，差这 2.4 倍全在 σ 上 —— **本仓库的 σ 恒为
    0**，别把那个数搬过来。tests/test_moe.py 的 test_paper_lambda... 钉着这条闭式。

    代价与边界，三条都要说清楚：
      - λ>1 抬高的只是 FFN 与残差流的**配比**（实测 ffn_out/std(x)：dense 0.0105、
        无共享 0.0013、共享+λ=1 0.0106、共享+λ=8 0.0149）。`hstublock` 末尾是
        `norm(out + x)`，post-norm 的 RMSNorm 把块输出模长钉死在 1.0，**不跨层累积**。
      - λ 只在**初始化**那一点上被标定。router 在训练中会变尖（实测 gate_mean 六个
        epoch 涨 55%，等效 σ 从 0 到 ~0.75），原则要求的 λ 随之从 8.66 掉到 ~3.8 ——
        到 score 见顶的 ep3 只剩 ~4.2。所以这个闭式是「init 严格正确、训练中期偏大」
        的一档，想覆盖全程得取更小的折中值。这不是缺陷，是这条原则的定义域。
      - λ=0 合法（只留共享专家的消融档，输出对门控的导数是 `λ·expert`），但它会把
        路由的 task 梯度整体掐断、只剩 aux 推它 —— 那一档的 router 不是「学会了不
        路由」，是**没有任务信号**，别拿它的路由统计当结论。

    路由权重是**裸 Parameter**、不是 nn.Linear：`losses.init_weights` 会给每个
    nn.Linear 装 xavier，而路由必须从严格均匀起步（与 `hstu.time` 同一个处境）。
    改这里之前先看 tests/test_moe.py 的对照断言。

    零初始化时所有 logits 相等，于是第一步**全部 token 落同一个专家** —— 落哪一个由
    `torch.topk` 的并列打破规则决定，而文档并未规定它（本机实测 CPU 落专家 2、
    CUDA 落专家 0；换个设备/版本可能又不同）。这不影响结果（各专家同分布，且下一步
    路由权重就吃到梯度、logits 不再并列），但**测试不能断言具体下标**，只能断言
    「恰好一个专家拿到全部 token」。这一步之后 aux 与门控两路梯度都会把路由推开，
    塌缩是起点状态、不是稳态。

    pad 位置在进入本模块时 hidden 恒为 0（`_side_emb` 双侧掩蔽 + `glu` 首层
    bias=False），0 输入给不出任何 logits，所以 pad 与其它 token 一样只是「并列中的
    一员」，不需要单独处理。实测 pad 只占全部槽位的 0.93%（2.6% 的用户有 pad），
    对 f 的污染可忽略，因此没有为它单独传 valid 掩码；换 maxlen 或换数据后要重新
    算这笔账。（若将来要把 pad 排除在 f 之外，入口在这里加一个 valid 参数。）

    **无辅助损失负载均衡（`balance != 'aux_loss'`，DeepSeek-V3 §4.2）：**
        y = Σ_s shared[s](x) + λ · Σ_{选中槽位} softmax(logits)[槽位] · expert(x)
                        ↑ 选谁由 `logits + b` 决定，门控仍是无偏 softmax 的值

    偏置 b 是个**不进梯度**的逐专家向量，由 `update_expert_bias(gamma)` 按最近一次
    前向的负载更新（`b_i += γ · sign(mean_load − load_i)`，过载则降）。它只参与
    **选择**，不参与组合权重的计算 —— 这是 DSeek 的原版语义，也让「门控 = 概率、
    初始精确 1/E」这条契约原样保留（见上面「门控不做任何缩放」一段）。

    由此得到一条别处要专门处理、这里不需要的性质：Σ_i sign(e_i) 不必为 0，所以 b 的
    均值会漂 —— 但所有专家同加一个常数既不改变 topk 结果、也不进门控（取自无偏
    probs），于是连一个可观测量都影响不到。
    """
    def __init__(self, input_dim, output_dim, num_experts=4, top_k=1, num_shared=0,
                 routed_scale=1.0, balance="aux_loss", ffn_act="none"):
        super(moe, self).__init__()
        assert num_experts >= 1
        assert 1 <= top_k <= num_experts, "top_k 必须落在 [1, num_experts]"
        # num_experts=1 时本模块不会被构造（见 hstublock.has_moe），共享专家会随之
        # 被静默丢弃 —— 与其让一次 A/B 白跑，不如当场炸。理由见类 docstring。
        assert num_shared == 0 or num_experts > 1, \
            "num_shared>0 需要 num_experts>1：num_experts=1 走 nn.Linear，moe 根本不会被构造"
        # λ 没有上界（它就是个配比，论文那条闭式在 E/k 大时会很大），但负数没有解释：
        # 那等于把路由专家的输出反过来加，不属于任何一档有意义的消融。
        assert routed_scale >= 0, \
            "routed_scale 不能为负，实际 %r" % (routed_scale,)
        # 负载均衡策略：'aux_loss'（默认，辅助损失）/ 'aux_free'（逐专家偏置）/
        # 'both'（两者都开）。见类 docstring 的「无辅助损失负载均衡」一段。
        assert balance in ("aux_loss", "aux_free", "both"), \
            "balance 只能是 'aux_loss' / 'aux_free' / 'both'，实际 %r" % (balance,)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared = num_shared
        # float 而不是 nn.Parameter/0-dim tensor：它必须**不进 state_dict**，
        # 否则「同一份权重配不同 λ」的两轮就没法共享 ckpt 了（见类 docstring）。
        self.routed_scale = float(routed_scale)
        self.balance = balance
        # 偏置是**条件创建**的 buffer（同 self.shared 那条约定）：aux_loss 档一个键都
        # 不建，旧 ckpt 的 state_dict 与 test_model_equiv 才都不受影响。
        #
        # 它**进** state_dict（与 λ 相反）：b 的当前值影响输出，必须随权重一起保存。
        # dtype 恒 fp32 且不受 autocast 影响 —— γ=1e-3 加在 bf16 上，|b| 到 0.75 那个
        # 量级就原地不动了（实测 0.75 ± 1e-3 == 0.75），而 b 要长到 ~0.75 才压得住
        # logits：静默失效，logging 上看不出来。
        if balance == "aux_loss":
            self.expert_bias = None
        else:
            self.register_buffer("expert_bias",
                                 torch.zeros(num_experts, dtype=torch.float32))
        # 「有没有偏置」与「哪一档」必须是同一件事：forward 走的是前者，日志与
        # train.py 只看后者。断言把这两处知识钉在一起，免得将来加个运行时改档的
        # 口子时，把 buffer 留在原地、静默地继续偏置选择。
        assert (balance == "aux_loss") == (self.expert_bias is None)
        self.experts = nn.ModuleList(
            [_make_ffn(input_dim, output_dim, ffn_act) for _ in range(num_experts)])
        # 关掉时**不建任何参数**（不是「建了但不加」）：这样旧 ckpt 的 state_dict 键
        # 一个不多、一个不少，且随机数消耗与改动前逐位一致 —— test_model_equiv 与
        # 之前跑过的 A/B 都还成立。
        self.shared = (None if num_shared == 0 else
                       nn.ModuleList([_make_ffn(input_dim, output_dim, ffn_act)
                                      for _ in range(num_shared)]))
        self.router_weight = nn.Parameter(torch.zeros(num_experts, input_dim))
        # 诊断量（都已 detached，不进图）：训练日志靠它们发现专家塌缩
        self.aux = torch.zeros(())
        self.expert_frac = torch.zeros(num_experts)
        self.gate_mean = torch.zeros(())
        self.last_expert = torch.zeros(0, dtype=torch.long)

    def _load_balance(self, probs, idx):
        """Switch Transformer 的负载均衡项 `E * Σ_e f_e · P_e`（值域 [1, E]）。

        f_e = 分到专家 e 的 token 比例 —— argmax 的统计量，**不可微**（常量）；
        P_e = 全体 token 对专家 e 的平均 softmax 概率 —— 可微，负载均衡的梯度
        **全部**从这一项回流到路由权重（argmax 那一路是断的）。

        「f 塌缩但 P 还均匀」时它取下界 1，「f 与 P 同时塌在同一个专家」时才到上界 E，
        所以它盯的是两者的**联合**分布。
        """
        f = torch.zeros(self.num_experts, device=probs.device)
        f.scatter_add_(0, idx.reshape(-1),
                       torch.ones(idx.numel(), device=probs.device))
        f = f / idx.numel()          # 除以「token 数 x top_k」，于是 Σf 恒为 1
        self.expert_frac = f         # 由 ones 累加而来，天然 detached
        return self.num_experts * (f * probs.mean(dim=0)).sum()

    @torch.no_grad()
    def update_expert_bias(self, gamma):
        """按**最近一次前向**的专家负载，把偏置推离过载的专家（DSeek-V3 §4.2）。

            b_i += γ · sign(mean_load − load_i)        mean_load = 1/E

        **由 train.py 在 `optimizer.step()` 之后显式调用**，不在 forward 里自动更新：
        一个训练 step 里有 5 次前向（默认 `ssl_alpha>0` 时：序列塔 + pos + neg + ssl×2，
        `--ssl_alpha 0` 则只有 3 次），「在哪一次之后更新」会变成一条没人写得下来的隐式
        契约；而评测期每个 epoch 要跑几百次前向，自动更新还得再挂一个 `self.training`
        判断。显式调用让「什么算一个 step」由 train.py 定义。
        每步只调用一次（调用两次就是 2γ）；ckpt 里存的是 `b` 本身，恢复训练不会重放更新。

        调用点读到的 `expert_frac` 是**序列塔**那一份：pos / neg / ssl 走 item 塔，
        没有 HSTU、不覆盖它（见 `hstublock` / tower.forward）。

        `aux_loss` 档 `expert_bias is None` -> 无条件 no-op，调用点不用加分支。

        前向之前调用不会破坏 `b`：`expert_frac` 的初值全零 -> 每个专家同加 γ，而整体
        平移是可证明的 no-op（见 §3.1(b) 那条用例）。所以调用点不必自己保证「先前向」。

        已知的退化态：零初始化 + 完全并列时（全部 token 落同一个专家），sign(e) 是
        确定性的 —— 若 router 权重一直不动，b 会推着全部 token 在专家之间周期性轮转
        而不是收敛。真实训练里 router 每步都在吃梯度，这个态只在最初若干步存在。
        （顺带把塌缩从当前专家挪走 —— E 路并列变成 E−1 路并列，真正收敛仍要靠后续
        router 梯度。）
        """
        if self.expert_bias is None:
            return
        assert gamma >= 0, \
            "γ 不能为负（实际 %r）：那等于把符号反过来，是正反馈" % (gamma,)
        f = self.expert_frac                    # [E]，Σf = 1（top_k>1 时分母是 token 数×k）
        e = (1.0 / self.num_experts) - f        # 正 = 欠载
        self.expert_bias += gamma * torch.sign(e)

    def _expert_out(self, e, chunk, w):
        """第 e 个专家在 `chunk` 上做前向，再乘上门控 `w`（[m, 1]，就是路由概率本身）。

        末尾的 **`.to(chunk.dtype)` 不是可选的**：bf16 autocast 下 torch 会把 softmax
        提成 fp32（数值稳定性），于是 w 是 fp32、专家输出是 bf16，而下面两处结果都要
        靠下标写入拼回原张量，`index_put` 要求源与目标 dtype **完全一致** —— 不转就是
        RuntimeError（不是静默降精度）。`--amp bf16` 是 train.py 的默认档，这条路径
        每轮训练都会走到，tests/test_moe.py 里有一条 CUDA-only 的用例钉着它。
        """
        return (self.experts[e](chunk) * w).to(chunk.dtype)

    def _shared_out(self, flat):
        """共享专家：每个 token 都过，输出**原样相加**（系数恒 1，没有门控）。

        `num_shared` 个共享专家是求和的 —— 与「多个路由专家按各自门控加权求和」不同，
        这里没有「谁的输出更重要」这个问题，它们只是同一路稠密 FFN 的并联。

        末尾的 `.to(flat.dtype)` 与 `_expert_out` 同理：autocast 下 nn.Linear 的输出
        类型由 autocast 决定，而 `out + shared` 的张量提升规则会把整个 block 抬成
        fp32（不是报错，是静默吃显存、并让下游 dtype 与 baseline 不一致）。显式对齐
        到输入 dtype 是防御性的 —— 当前档位下两者本来就相等。
        """
        y = self.shared[0](flat)
        for s in self.shared[1:]:
            y = y + s(flat)
        return y.to(flat.dtype)

    def _dispatch_top1(self, flat, gate, e_id):
        """top-1 的快路径：一次 argsort 把同一专家的 token 排到一起，循环里只做切片。

        不用 `flat[idx == e]` 那套布尔掩码：掩码索引内部要走 `nonzero`，实测这条比它
        快 ~4 倍（3232x1536 的 token，fwd+bwd，单层单步 6.7ms vs 25.9ms，H100），而
        FLOPs 完全一样 —— 省掉的全是索引/同步开销，这正是「top-1 的算力应当等于单个
        FFN」这句话在 wall-clock 上兑现的地方，所以放在默认路径上。

        `counts.tolist()` 会同步一次设备（切片要 Python int）：每层一次，远小于掩码
        路径里那几十次 nonzero 的代价。
        """
        order = torch.argsort(e_id)                       # 同一专家的 token 连续排布
        counts = torch.bincount(e_id, minlength=self.num_experts).tolist()
        perm, w = flat[order], gate[order]
        parts, pos = [], 0
        for e, c in enumerate(counts):
            parts.append(self._expert_out(e, perm[pos:pos + c],
                                          w[pos:pos + c].unsqueeze(-1)))
            pos += c
        y = torch.cat(parts)                              # cat 的顺序就是 counts 的顺序
        out = torch.empty_like(y)
        out[order] = y                                    # 还原成 token 的原始顺序
        return out

    def _dispatch_topk(self, flat, gate, idx):
        """top-k（k>1）的通用路径：布尔掩码逐专家累加。慢，但只在小流量档位用。

        两条都是被测试钉住的坑：
        1) 归属用 **槽位**（`idx == e`）判定，不用 `routing > 0` —— bf16 下概率可能
           下溢成 0（logits 差 ~88 以上），那会让该 token 被静默丢掉、输出变 0 而不是
           「贡献极小」。idx 是精确的。
        2) 累加必须是读-改-写，不能写成 `out[sel] = contrib`：同一个 token 会落进多个
           专家的 sel，直接赋值会让循环里**最后一个**专家把前面的整行覆盖掉。
           tests/test_moe.py::test_top_k_two_sums_the_selected_experts 钉住这条。
        """
        # topk 的槽位互异，scatter_（不是 scatter_add_）足够
        routing = torch.zeros(flat.shape[0], self.num_experts,
                              dtype=gate.dtype, device=flat.device)
        routing.scatter_(1, idx, gate)
        out = torch.zeros(flat.shape[0], self.output_dim,
                          dtype=flat.dtype, device=flat.device)
        for e in range(self.num_experts):
            sel = (idx == e).any(dim=1)                   # [n]
            out[sel] = out[sel] + self._expert_out(
                e, flat[sel], routing[sel, e].unsqueeze(-1))
        return out

    def forward(self, x):
        # x: [b, s, input_dim]
        shape = x.shape
        flat = x.reshape(-1, shape[-1])                              # [n, input_dim]
        logits = F.linear(flat, self.router_weight)                  # [n, E]
        probs = torch.softmax(logits, dim=-1)                        # [n, E]，**无偏**
        if self.expert_bias is None:
            # aux_loss 档：改动前那条路径，一个算子都不多（test_model_equiv 靠它）
            gate, idx = probs.topk(self.top_k, dim=-1)               # [n, k]，门控即概率
        else:
            # 偏置只加在**选择分数**上；门控仍从无偏 probs 里 gather —— DSeek 的原版
            # 语义（bias 不参与组合权重）。三种写错的方式都被测试钉着：
            #   - 门控取自含偏 softmax（把 b 灌进了概率。注意 b==0 时含偏与无偏
            #     **逐位相同**、初始 1/E 的契约照旧成立 —— 所以任何「初始态」用例
            #     都抓不住它，能钉死这条的只有 b != 0 的那些）
            #   - 忘了加 b（偏置形同不存在）
            #   - 把 b 加进了 probs 再 topk（前向看似对，但门控已不是路由概率）
            # 和**故意**留在 fp32（bf16 的 logits + fp32 的 b 自动提升），不能反过来
            # 再 `.to(logits.dtype)` 压回去 —— 选谁的分数被量化回 bf16 后，|sel|≈1 处
            # 的量化步长(~2-4e-3)比 γ=1e-3 还粗，正是 fp32 buffer 要避开的那件事。
            sel = logits + self.expert_bias                          # [n, E]，仅用于选谁
            idx = sel.topk(self.top_k, dim=-1).indices               # [n, k]
            gate = probs.gather(-1, idx)                             # [n, k]，无偏概率
        if self.top_k == 1:
            out = self._dispatch_top1(flat, gate[:, 0], idx[:, 0])
        else:
            out = self._dispatch_topk(flat, gate, idx)
        # λ 只乘**路由**那一路，在共享专家之前 —— 论文式 (3) 里它在求和号外面。
        # `!= 1.0` 这个守卫不是优化：它保证默认档（λ=1）走的是改动前那条一模一样的
        # 路径，连一次多余的乘法都没有，于是 test_model_equiv 的逐位比对仍然成立；
        # λ=0 也走这里（乘 0 得 0，路由输出干净地归零）。
        if self.routed_scale != 1.0:
            out = out * self.routed_scale
        # 共享专家在**路由结果之上**相加，不参与 topk、也不进门控（见类 docstring）。
        if self.shared is not None:
            out = out + self._shared_out(flat)

        aux = self._load_balance(probs, idx)
        with torch.no_grad():                                        # 诊断量不进图
            # self.aux 只是**副本**，给日志/测试看的；带图的那份走返回值 ——
            # 存在属性上会让这张图活到下一次前向才被覆盖。
            self.aux = aux.detach()
            self.gate_mean = gate.mean()
            self.last_expert = idx[:, 0]
        return out.view(*shape[:-1], self.output_dim), aux


class hstublock(nn.Module):
    def __init__(self, input_dim, head, num_experts=1, top_k=1, num_shared=0,
                 routed_scale=1.0, balance="aux_loss", ffn_act="none"):
        super(hstublock, self).__init__()
        self.uqkv = nn.Sequential(
            # u: 3 * input_dim, q: input_dim, k: input_dim, v: input_dim
            nn.Linear(input_dim, input_dim * 6, bias=False),  # O_o/model.py:186
            nn.SiLU()
        )
        self.head = head
        self.norm = nn.RMSNorm(input_dim, eps=1e-8)  # O_o/model.py:182
        # num_experts=1 走原来的 nn.Linear：与 O_o 逐位等价（test_model_equiv 靠它），
        # 也是 A/B 的「关掉 MoE」那一档。>1 时才是带路由的 MoE。
        #
        # 下面三条断言必须长在这里、而不是只长在 moe 里：num_experts=1 时 moe 压根
        # 不会被构造，参数被静默丢掉、FFN 还是那个 nn.Linear，而 MoE 关着时 aux 恒为 0
        # （tower.forward 那一支）—— 于是损失曲线、日志字段全都一样，一个「开了开关却
        # 什么都没变」的 A/B 是白跑的。三条共用这一个理由，所以只写一遍。
        assert not (num_shared > 0 and num_experts <= 1), \
            "num_shared>0 需要 num_experts>1（否则 FFN 是 nn.Linear，共享专家会被静默忽略）"
        # λ 不要求 num_shared>0：不开共享专家时 λ 照样真实作用在路由输出上。
        assert not (routed_scale != 1.0 and num_experts <= 1), \
            "routed_scale != 1 需要 num_experts>1（否则 FFN 是 nn.Linear，λ 会被静默忽略）"
        assert not (balance != "aux_loss" and num_experts <= 1), \
            "balance != 'aux_loss' 需要 num_experts>1（否则 FFN 是 nn.Linear，偏置会被静默忽略）"
        self.has_moe = num_experts > 1
        self.ffn = (moe(input_dim * 3, input_dim, num_experts, top_k, num_shared,
                        routed_scale, balance, ffn_act)
                    if self.has_moe else _make_ffn(input_dim * 3, input_dim, ffn_act))

    def forward(self, x, mask, rel_ts):
        # x: [b, s, input_dim]  mask: [b, s, s] True=屏蔽  rel_ts: [b, s, s] 相对时间偏置
        # 返回 (输出, 负载均衡项)——aux 走返回值而不是 self.ffn.aux 这类属性引用：
        # 属性会把带 grad_fn 的标量（连同它引用的图）留到下一次前向才被覆盖，
        # 在 no_grad 之外的前向里等于多留一份反向图。
        U, Q, K, V = torch.split(
            self.uqkv(x), 
            [x.shape[-1] * 3, 
            x.shape[-1], 
            x.shape[-1],
            x.shape[-1]
            ],
            dim=-1
        )
        Q = Q.view(x.shape[0], x.shape[1], self.head, -1).transpose(1, 2) # [b, head, s, d]
        K = K.view(x.shape[0], x.shape[1], self.head, -1).transpose(1, 2) # [b, head, s, d]
        V = V.view(x.shape[0], x.shape[1], self.head, -1).transpose(1, 2) # [b, head, s, d]
        qk_attn = torch.matmul(Q, K.transpose(-2, -1)) / Q.shape[-2]# [b, head, s, s]
        
        time_out = torch.einsum('bmn, bhnd -> bmhd', rel_ts, V) # [b, s, head, d]
        time_out = time_out.reshape(x.shape[0], x.shape[1], -1) # [b, s, d]
        
        qk_attn = torch.masked_fill(qk_attn, mask.unsqueeze(1), float('0')) # [b, head, s, s]
        qk_attn = torch.relu(qk_attn) # [b, head, s, s]
        attn_out = torch.matmul(qk_attn, V) # [b, head, s, d]
        attn_out = attn_out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1) # [b, s, d]

        q_rope, k_rope = rope(Q, K)
        rope_attn = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / Q.shape[-2]# [b, head, s, s]
        rope_attn = torch.masked_fill(rope_attn, mask.unsqueeze(1), float('0')) # [b, head, s, s]
        rope_attn = torch.relu(rope_attn) # [b, head, s, s]
        rope_out = torch.matmul(rope_attn, V) # [b, head, s, d]
        rope_out = rope_out.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1) # [b, s, d]
        out = torch.cat([attn_out, rope_out, time_out], dim=-1) # [b, s, 3 * d]
        out = out * U
        if self.has_moe:
            out, aux = self.ffn(out) # [b, s, d] + 负载均衡项（标量）
        else:
            aux = None
            out = self.ffn(out) # [b, s, d]
        return self.norm(out + x), aux # [b, s, d]

    def update_expert_bias(self, gamma):
        """把偏置更新转给 FFN —— MoE 关着时没有偏置可更新（那一档是 nn.Linear）。"""
        if self.has_moe:
            self.ffn.update_expert_bias(gamma)

class hstu(nn.Module):
    def __init__(self, input_dim, head, num_layers, num_experts=1, top_k=1,
                 time_bias_per_layer=False, num_shared=0, routed_scale=1.0,
                 balance="aux_loss", ffn_act="none"):
        super(hstu, self).__init__()
        self.layers = nn.ModuleList([hstublock(input_dim, head, num_experts, top_k,
                                               num_shared, routed_scale, balance,
                                               ffn_act)
                                     for _ in range(num_layers)])
        self.time_bias_per_layer = bool(time_bias_per_layer)
        # 相对时间偏置表：桶 0 = 对角线 / padding / 上三角，1..128 = Δ 的分桶，共 129 行。
        #
        #   False（默认）= **全局一张**，8 层共用 —— 对齐 O_o/model.py:324 的
        #                  self.rel_time_bias（单个 RelativeTimeBias 实例，共用 ts_w）
        #   True         = 每层一张（见 config.TIME_BIAS_PER_LAYER 的动机与代价）
        #
        # 两条支路只建真正会用到的那一套（与 hstublock.has_moe 同路子），不留一张
        # 永远不会被查的表。两支的**初始化与桶的定义完全一致**，变的只是「几张表」——
        # 所以「每层表都取共享表那份权重」时必须逐位复现共享档（tests/test_time_bias.py
        # 的 test_equal_tables_reproduce_the_shared_path_bit_for_bit 钉的就是这条）。
        #
        # 初始化为什么不用 nn.Embedding 的默认 N(0, 1)：第 3 路注意力没有 ReLU、没有
        # 归一化，是唯一可正可负、且直接乘进 V 的通道。实测初始时时间路的 std 比两条
        # 注意力路大 ~24x，8 层堆叠后会把内容相关的注意力压掉。对齐 O_o 的 ts_w
        # （Parameter(...).normal_(std=0.02)，O_o/model.py:91），让三路初始量级相当。
        # 注意：不要给这些 Embedding 设 padding_idx —— 桶 0 是对角线的可学习自连接
        # 偏置，是真参数，设了会被 init_weights 的清零分支抹掉。另外 losses.init_weights
        # 给所有 Embedding 的初始化恰好也是 N(0, 0.02)，与下面这行一致，apply 之后不必
        # 再单独排除它们（见那里的 docstring）。
        tables = [nn.Embedding(129, 1) for _ in range(num_layers if self.time_bias_per_layer else 1)]
        if self.time_bias_per_layer:
            self.time_layers = nn.ModuleList(tables)
        else:
            self.time = tables[0]
        for t in tables:
            nn.init.normal_(t.weight, mean=0.0, std=0.02)

    def time_bucket(self, seq_ts):
        """seq_ts: [b, s] -> [b, s, s] 桶号（long）。

        与用哪张表无关，所以每层一张时可以只算这一次。
        桶号闭区间 1..128，与 O_o/model.py:327-330 的 clamp(min=1, max=128) 一致
        （40 天窗口实际只用到桶 1~22，128 只是上界）。
        """
        s = seq_ts.shape[1]
        time_i = seq_ts.unsqueeze(-1) #[b, s, 1]
        time_j = seq_ts.unsqueeze(-2) #[b, 1, s]
        # 单向差值并整流：负值只出现在上三角；不 clamp 会让 log2(负) -> NaN 污染反传
        time_diff = torch.clamp(time_i - time_j, min=0) #[b, s, s]
        # 必须 .long()：nn.Embedding 只接受 Long/Int 索引，float 会直接抛 RuntimeError
        bucket = (torch.floor(torch.log2(time_diff.clamp(min=1).float())) + 1).clamp(min=1, max=128).long() #[b, s, s]

        # 对角线（Δ=0）单独走桶 0：给每个位置一个可学习的自连接偏置
        diag = torch.eye(s, dtype=torch.bool, device=seq_ts.device)
        return torch.where(diag, torch.zeros_like(bucket), bucket) #[b, s, s]

    def time_bias(self, bucket, mask, table):
        """桶号 + 一张表 -> [b, s, s] 相对时间偏置。无效点对（上三角 / padding）真·归零。"""
        time_emb = table(bucket).squeeze(-1) #[b, s, s]
        return time_emb.masked_fill(mask, 0.0) #[b, s, s]

    def relative_time(self, seq_ts, mask):
        """seq_ts: [b, s]  mask: [b, s, s] True=屏蔽  ->  [b, s, s] 相对时间偏置。

        在位次上等价于 O_o 的 RelativeTimeBias.forward + HSTUBlock 里那句
        `rel_ts_bias.masked_fill(attn_mask.logical_not(), 0.0)`：
        - 对角线 O_o 显式改写成桶 0（`pair_valid & ~diag`），这里同样置 0；而
          attn_mask 允许对角线，所以两边都保留 `ts_w[0]` 这个可学习的自连接偏置。
        - 上三角 / padding 两边最终都归零。

        **共享表专用**：time_bias_per_layer=True 时没有 self.time，逐层取偏置要用
        `time_bias(time_bucket(seq_ts), mask, 第 i 张表)`（见 _rel_ts_per_layer）。
        """
        return self.time_bias(self.time_bucket(seq_ts), mask, self.time)

    def _rel_ts_per_layer(self, seq_ts, mask):
        """逐层产出 rel_ts —— 两种表配置的唯一差别就收在这里。

        共享表（默认）：**层循环外只算一次**，同一个张量交给全部 8 层（对齐
        O_o/model.py:520）。每层一张：桶号仍然只算一次，查表进循环、每层一份；
        用生成器是为了让上一层的 [b, s, s] 随循环变量一起被释放，不在 8 层之间
        一直挂着 8 份活跃张量（batch=256、s=102 时每份约 10MB）。
        """
        if not self.time_bias_per_layer:
            rel_ts = self.relative_time(seq_ts, mask)
            for _ in self.layers:
                yield rel_ts
            return
        bucket = self.time_bucket(seq_ts)
        for table in self.time_layers:
            yield self.time_bias(bucket, mask, table)

    def forward(self, x, mask, seq_ts, return_aux=False):
        aux = None
        for layer, rel_ts in zip(self.layers, self._rel_ts_per_layer(seq_ts, mask)):
            x, a = layer(x, mask, rel_ts)
            if a is not None:
                aux = a if aux is None else aux + a
        if not return_aux:
            return x
        # 取各层**均值**而不是求和：改 num_layers 时 α 的等效强度不该跟着变。
        return x, (aux / len(self.layers) if aux is not None else x.new_zeros(()))

    def update_expert_bias(self, gamma):
        """逐层更新偏置（见 moe.update_expert_bias）。

        由 train.py 在 `optimizer.step()` 之后调一次。MoE 关着时每层都是 no-op，
        所以调用点不用加分支。
        """
        for layer in self.layers:
            layer.update_expert_bias(gamma)

class tower(nn.Module):
    def __init__(self, user_item_list, item_item_list, embbeding_dim, hidden_units=512,
                 dropout=0.2, item_only_cols=None, user_array_cols=(),
                 num_experts=1, top_k=1, time_bias_per_layer=False, num_shared=0,
                 routed_scale=1.0, balance="aux_loss", ffn_act="none"):
        super(tower, self).__init__()
        assert hidden_units % 8 == 0, "hidden_units 必须能被 head 数 8 整除（O_o/model.py:181）"
        self.user_item_list = user_item_list
        self.item_item_list = item_item_list
        self.embbeding_dim = embbeding_dim
        self.hidden_units = hidden_units
        # item 塔取 item_item_list 里的哪几列（None = 全部，保持旧行为）。
        # item 塔与序列塔**共享** self.item_embbeding 这批表，所以这里存的是下标：
        # 候选侧没有 record 级的 action_type(900)，也不参与交叉特征，因此只用
        # 13 原始 + 901 = 14 列 + 主 id。glu 首层 bias=False 下，O_o 那列恒零的 900
        # 对输出的贡献恒为 0，所以「少一列」与 O_o 逐位等价（见设计文档 §3.1）。
        self.item_only_cols = (list(range(len(item_item_list))) if item_only_cols is None
                               else list(item_only_cols))
        # user_array 特征在 user_item_list 里的下标（值维在 sum 里塌掉，维度公式不变）
        self.user_array_cols = tuple(user_array_cols)
        self.user_embbeding = nn.ModuleList([nn.Embedding(num_embeddings=x, embedding_dim=embbeding_dim, padding_idx=0) for x in user_item_list])
        self.item_embbeding = nn.ModuleList([nn.Embedding(num_embeddings=item_item_list[i], embedding_dim=embbeding_dim, padding_idx=0)
                                             for i in range(len(item_item_list))])
        # 两条路径各一个 glu（= O_o 的 FeatureInteractionEncoder），不再在前面套 MLP。
        # O_o 的输入维度见 O_o/model.py:347-356：每个特征路径都是 emb * (特征列数 + 1 个主 id)，
        # 主 id 的 +1 已经包含在 len(user_item_list) / len(item_item_list) 里。
        # item 塔的列数用 len(self.item_only_cols) 而不是 len(item_item_list)：候选侧
        # 不含 900 与交叉特征，输入维度是 1920（O_o 是 2048，差额正是 900 那一列 + 交叉两列）。
        self.item_dnn = glu(embbeding_dim * len(self.item_only_cols),
                            output_dim=hidden_units, expand_dim=4, dropout=dropout)
        # user 路径的输入比 item 路径多 3 * embbeding_dim：hour / dow / weekend
        # 三张表按 O_o 的做法当 item 侧稀疏特征拼进 DNN 输入。item 塔的输入维度
        # 不变——候选侧拿不到 seq_ts，也不该有用户侧时间特征。
        self.user_dnn = glu(embbeding_dim * (len(user_item_list) + len(item_item_list) + 3),
                            output_dim=hidden_units, expand_dim=4, dropout=dropout)
        # MoE 只作用在序列塔的 HSTU 上（换掉每个 block 的 FFN）。item 塔没有 HSTU、
        # 只过一次 glu，候选侧编码因此不受影响 —— 召回是双塔，两侧编码器不同构是常态。
        self.num_experts = num_experts
        self.num_shared = num_shared
        self.routed_scale = routed_scale
        self.balance = balance
        self.ffn_act = ffn_act
        self.HSTU = hstu(input_dim=hidden_units, head=8, num_layers=8,  # head=8: O_o/main.py:38
                         num_experts=num_experts, top_k=top_k,
                         time_bias_per_layer=time_bias_per_layer,
                         num_shared=num_shared, routed_scale=routed_scale,
                         balance=balance, ffn_act=ffn_act)

        # 时间离散特征表：hour 1..24 / dow 1..7 / weekend 1..2，0 一律留给 pad
        self.hour_emb = nn.Embedding(24 + 1, embbeding_dim, padding_idx=0)
        self.dow_emb = nn.Embedding(7 + 1, embbeding_dim, padding_idx=0)
        self.weekend_emb = nn.Embedding(2 + 1, embbeding_dim, padding_idx=0)
        # 绝对时间编码，加性叠到序列表示上（不是拼到特征维）
        self.time_abs_enc = FourierTimeEncoding(hidden_units=hidden_units)
        self.emb_dropout = nn.Dropout(p=dropout)
    
    def _side_emb(self, tables, seq, feat, side, cols, arrays=None, array_cols=()):
        """按 side 掩码拼出单侧特征向量 -> [b, s, emb * len(cols)]。

        seq 是主 id，走 cols[0] 号表；cols[1:] 依次消费 feat 的列（多值特征改吃
        arrays 的第 ai 个张量），两者不重叠 —— 与 O_o/model.py:426-440 的
        `user_emb(seq)` + `sparse_emb[k](tens)` 同构。
        非本侧位置强制置 0 走各自的 padding_idx=0，于是该位置在另一侧的特征块恒为
        零向量 —— 每个位置只携带一种实体。O_o 靠上游把非本侧填默认 0 来保证
        （`_full_template.copy()`），这里收进模型内部，上游填错也不会静默污染。

        cols：该侧用到的表下标（不是列数）。item 塔与序列塔共享同一批表，但用不同的
              列子集，所以必须给下标。
        arrays / array_cols：多值特征。arrays[i] 是 [b, s, L_i]（0 = pad），
              对应表下标 array_cols[i]；查表后在 L 维求和，即 O_o/model.py:442 的 `.sum(2)`。
        """
        parts = [tables[cols[0]](torch.where(side, seq, 0))]
        fi, ai = 0, 0
        for j in cols[1:]:
            if ai < len(array_cols) and array_cols[ai] == j:
                col = arrays[ai]                                     # [b, s, L]
                col = torch.where(side.unsqueeze(-1), col, torch.zeros_like(col))
                parts.append(tables[j](col).sum(dim=2))              # [b, s, emb]
                ai += 1
            else:
                parts.append(tables[j](torch.where(side, feat[..., fi], 0)))
                fi += 1
        return torch.cat(parts, dim=-1)

    def forward(self, seq, token_type=None, user_feat=None, item_feat=None, seq_ts=None,
                user_array=None, return_aux=False):
        """token_type is None -> 物品塔，否则 -> 混合序列塔。

        物品塔（候选侧，对应 O_o 的 feat2emb(include_user=False)）:
            seq       [b, s]        候选 item 主 id
            item_feat [b, s, li-1]  其余 item 特征 id，不含主 id（li==1 时可传 None）
            -> [b, s, H]，已归一化
        序列塔（对应 O_o 的 feat2emb(include_user=True)）:
            seq       [b, s]        主 id：item 位置放 item_id，user 位置放 user_id，pad = 0
            token_type[b, s]        0=pad, 1=item, 2=user（与 O_o 逐值一致）
            user_feat [b, s, lu-1]  user 侧其余特征，不含主 id 与 array 特征
            item_feat [b, s, li-1]  item 侧其余特征，不含主 id
            seq_ts    [b, s]        绝对时间戳（秒），pad 位置为 0：user token 用它自己
                                    那条 record 的时间
            user_array list[[b, s, L_i]] x A  user_array 特征，顺序 = user_array_cols()
                                    的顺序，item 塔忽略此参数
            -> [b, s, H]，已归一化

        lu = len(user_item_list), li = len(item_item_list)。主 id 占 1 列、其余走 [1:]，
        合计仍是 lu / li 列，所以 user_dnn 的输入维度 emb*(lu+li+3) 与改动前一致。

        return_aux=True 时返回 (output, aux)：aux 是序列塔各层 MoE 负载均衡项的均值
        （标量，见 hstu.forward）。item 塔没有 HSTU，或 MoE 关着时 aux 恒为 0 ——
        于是 `loss + alpha * aux` 在两种情况下都不改损失。
        """
        # 列数契约：列序错位不会报错、只会让训练悄悄变差，所以在这里显式钉住。
        # （顺序本身定义在 config.py，两侧 import 同一份。）
        if token_type is None:
            if item_feat is not None:
                assert item_feat.shape[-1] == len(self.item_only_cols) - 1, (
                    "item 塔的特征列数 %d != %d" % (item_feat.shape[-1],
                                                  len(self.item_only_cols) - 1))
        else:
            if user_feat is not None:
                assert user_feat.shape[-1] == len(self.user_item_list) - 1 - len(self.user_array_cols), (
                    "user 塔的特征列数 %d != %d" % (user_feat.shape[-1],
                                                  len(self.user_item_list) - 1 - len(self.user_array_cols)))
            if item_feat is not None:
                assert item_feat.shape[-1] == len(self.item_item_list) - 1, (
                    "序列塔 item 侧的特征列数 %d != %d" % (item_feat.shape[-1],
                                                        len(self.item_item_list) - 1))
            if self.user_array_cols:
                assert user_array is not None and len(user_array) == len(self.user_array_cols), (
                    "user_array 需要 %d 个张量，实际 %s"
                    % (len(self.user_array_cols),
                       "None" if user_array is None else len(user_array)))

        #item_input: [b, i]
        aux = None
        if token_type is None: #物品塔
            # 候选侧没有 user 槽位，整条序列都属于 item 侧
            side = torch.ones_like(seq, dtype=torch.bool)
            item_emb = self._side_emb(self.item_embbeding, seq, item_feat, side,
                                      self.item_only_cols) #[b, s, emb * len(item_only_cols)]
            output = self.item_dnn(item_emb) #[b, s, H]
            output = torch.nn.functional.normalize(output, p=2, dim=-1)
        else:
            # 每个位置由 token_type 决定身份，只有一侧查表、另一侧强制置 0
            valid = token_type != 0 #[b, s]
            is_item = token_type == 1 #[b, s]
            is_user = token_type == 2 #[b, s]
            user_emb = self._side_emb(self.user_embbeding, seq, user_feat, is_user,
                                      range(len(self.user_item_list)),
                                      arrays=user_array,
                                      array_cols=self.user_array_cols) #[b, s, emb * lu]
            item_emb = self._side_emb(self.item_embbeding, seq, item_feat, is_item,
                                      range(len(self.item_item_list))) #[b, s, emb * li]
            # 时间离散特征：从 seq_ts 算出 hour / dow / weekend，当 item 侧稀疏
            # 特征拼进 user 路径的 DNN 输入。按 is_item 门控而不是按 valid——
            # 这是 O_o 的语义（is_item = (mask == 1)），user token 不参与
            # 「几点/周几」；pad 位置同样置 0，避免 seq_ts=0 算出一个假的「周四 01 点」。
            ts = seq_ts.long() #[b, s]
            hour_idx = ((ts % 86400) // 3600) + 1          # 1..24
            dow_idx = (((ts // 86400) + 4) % 7) + 1        # 1=周日 .. 7=周六
            weekend_idx = (dow_idx >= 6).long() + 1        # 1=工作日, 2=周末
            zero = torch.zeros_like(hour_idx)
            hour_idx = torch.where(is_item, hour_idx, zero)
            dow_idx = torch.where(is_item, dow_idx, zero)
            weekend_idx = torch.where(is_item, weekend_idx, zero)

            emb = torch.cat([user_emb, item_emb,
                             self.hour_emb(hour_idx),
                             self.dow_emb(dow_idx),
                             self.weekend_emb(weekend_idx)], dim=-1) #[b, s, emb * (lu + li + 3)]
            output = self.user_dnn(emb) #[b, s, H]
            # 先把序列表示放大，再把绝对时间傅里叶特征加性叠上去，pad 位置为 0。
            # 乘法必须排在加法前：顺序反了 sqrt(H) 会把时间编码一起放大。
            # 这里按 valid 而不是 is_item：user token 的 ts 是它自己那条 record 的
            # 时间，绝对时间对它同样有意义（也是同一 user 的多个 token 之间唯一的
            # 区分量），与 O_o/model.py:497-499 的 (mask != 0) 一致。
            output = output * (self.hidden_units ** 0.5)
            output = output + self.time_abs_enc(ts) * valid.unsqueeze(-1)
            output = self.emb_dropout(output)
            # block 内部要的是 [b, s, s] 的布尔屏蔽矩阵（True = 屏蔽），且必须补上
            # 因果下三角：只屏蔽 key 侧 padding 的话，位置 i 仍能注意到 i+1 及以后，
            # 而下一个 item 就在 i+1，等于直接把标签喂给模型。
            s = token_type.shape[1]
            tril = torch.tril(torch.ones(s, s, dtype=torch.bool, device=token_type.device))
            attn_mask = ~(tril & valid.unsqueeze(2) & valid.unsqueeze(1)) #[b, s, s]，True = 屏蔽
            if return_aux:
                output, aux = self.HSTU(output, attn_mask, seq_ts, return_aux=True)
            else:
                output = self.HSTU(output, attn_mask, seq_ts)
            output = torch.nn.functional.normalize(output, p=2, dim=-1)
        if not return_aux:
            return output
        # item 塔（或 MoE 关着）没有 aux —— 给一个 0 标量，让调用方不必分支
        return output, (aux if aux is not None else output.new_zeros(()))

    def update_expert_bias(self, gamma):
        """按最近一次**序列塔**前向的专家负载更新偏置（见 `moe.update_expert_bias`）。

        必须在 `optimizer.step()` 之后、且在**同一个 step 的序列塔前向之后**调用 ——
        它读的是 `layer.ffn.expert_frac`。item 塔没有 HSTU，pos/neg/ssl 那几次前向
        不会覆盖它（`test_item_tower_forward_does_not_touch_the_sequence_tower_load`）。
        """
        self.HSTU.update_expert_bias(gamma)

    def moe_stats(self):
        """诊断量（训练日志用，不参与前向）：每层 MoE 的专家使用率与门控均值。

        `expert_frac[l]` 是第 l 层**最近一次前向**的 f（Σ=1），`gate_mean[l]` 是对应
        的平均门控值。塌缩时 expert_frac 会集中到一个专家上（aux 随之升到 ~E）——
        光看 loss 是看不出来的，这是唯一能发现它的窗口。MoE 关着时三个列表都是空的。

        `expert_bias` 是每层每个专家的当前偏置（`aux_loss` 档为空列表）。它是
        aux_free 档唯一的「机制是否在动」的直接观测 —— `maxf` 能告诉你结果，
        但只有 `b` 能告诉你它有没有卡住（bf16 停摆那个坑就是靠它才看得见）。
        """
        return {
            "expert_frac": [layer.ffn.expert_frac.tolist() for layer in self.HSTU.layers
                            if layer.has_moe],
            "gate_mean": [layer.ffn.gate_mean.item() for layer in self.HSTU.layers
                          if layer.has_moe],
            # 只有 aux_free / both 档才有偏置；**没有时给空列表**而不是一列 None，
            # 与 MoE 关掉时 expert_frac == [] 一致（判据是 `expert_bias is not None`
            # 而不是 `has_moe` —— 否则 aux_loss 档会在这行 AttributeError）。
            #
            # 这个额外的判据会不会让三个列表**错位**（train.jsonl 是按下标对齐的）？
            # 不会：balance 是 tower 级的单个开关、逐层一致，所以有偏置的层恰好就是
            # has_moe 的那些层。手工逐层配不同 balance 才可能错位，那条路本仓库没走。
            "expert_bias": [layer.ffn.expert_bias.tolist() for layer in self.HSTU.layers
                            if layer.has_moe and layer.ffn.expert_bias is not None],
        }

    @torch.no_grad()
    def predict(self, seq, token_type, user_feat, item_feat, seq_ts, user_array=None):
        """检索 query -> [b, H]，已归一化。对应 O_o/model.py:538-544。

        因果 mask 下只有最后一个位置看过整条序列，所以取 [:, -1]。
        ⚠️ 只在**右对齐**（真实 token 靠右、pad 在左）时正确。若上游改成左对齐或
        中间 padding，-1 会静默取到 pad，需要改成按 valid 的长度 gather。
        """
        return self.forward(seq, token_type, user_feat, item_feat, seq_ts, user_array)[:, -1, :]
            
        