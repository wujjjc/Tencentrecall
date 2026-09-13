import math
import torch
import torch.nn as nn
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


class hstublock(nn.Module):
    def __init__(self, input_dim, head):
        super(hstublock, self).__init__()
        self.uqkv = nn.Sequential(
            # u: 3 * input_dim, q: input_dim, k: input_dim, v: input_dim
            nn.Linear(input_dim, input_dim * 6, bias=False),  # O_o/model.py:186
            nn.SiLU()
        )
        self.head = head
        self.norm = nn.RMSNorm(input_dim, eps=1e-8)  # O_o/model.py:182
        self.ffn = nn.Linear(input_dim * 3, input_dim)

    def forward(self, x, mask, rel_ts):
        # x: [b, s, input_dim]  mask: [b, s, s] True=屏蔽  rel_ts: [b, s, s] 相对时间偏置
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
        out = self.ffn(out * U) # [b, s, d]
        return self.norm(out + x) # [b, s, d]
        
class hstu(nn.Module):
    def __init__(self, input_dim, head, num_layers):
        super(hstu, self).__init__()
        self.layers = nn.ModuleList([hstublock(input_dim, head) for _ in range(num_layers)])
        # 相对时间偏置表。**全局一张**，不是每层一张 —— 对齐 O_o/model.py:324 的
        # self.rel_time_bias（单个 RelativeTimeBias 实例，8 层共用同一份 ts_w）。
        # 桶 0 = 对角线 / padding / 上三角，1..128 = Δ 的分桶，共 129 行。
        self.time = nn.Embedding(129, 1)
        # 第 3 路注意力没有 ReLU、没有归一化，是唯一可正可负、且直接乘进 V 的通道。
        # nn.Embedding 默认初始化是 N(0, 1)，实测初始时时间路的 std 比两条注意力路
        # 大 ~24x，8 层堆叠后会把内容相关的注意力压掉。对齐 O_o 的 ts_w
        # （Parameter(...).normal_(std=0.02)，O_o/model.py:91），让三路初始量级相当。
        # 注意：不要给这个 Embedding 设 padding_idx —— 桶 0 是对角线的可学习
        # 自连接偏置，是真参数。另外若日后接入全局 init_weights 之类的遍历式
        # 初始化，这里会被重新覆盖回 N(0,1)，需要在那里排除 self.time。
        nn.init.normal_(self.time.weight, mean=0.0, std=0.02)

    def relative_time(self, seq_ts, mask):
        """seq_ts: [b, s]  mask: [b, s, s] True=屏蔽  ->  [b, s, s] 相对时间偏置。

        在位次上等价于 O_o 的 RelativeTimeBias.forward + HSTUBlock 里那句
        `rel_ts_bias.masked_fill(attn_mask.logical_not(), 0.0)`：
        - 对角线 O_o 显式改写成桶 0（`pair_valid & ~diag`），这里同样置 0；而
          attn_mask 允许对角线，所以两边都保留 `ts_w[0]` 这个可学习的自连接偏置。
        - 上三角 / padding 两边最终都归零。
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
        bucket = torch.where(diag, torch.zeros_like(bucket), bucket) #[b, s, s]

        time_emb = self.time(bucket).squeeze(-1) #[b, s, s]
        # 无效点对（上三角 / padding）真·归零，时间那路自包含
        time_emb = time_emb.masked_fill(mask, 0.0) #[b, s, s]
        return time_emb

    def forward(self, x, mask, seq_ts):
        # 对齐 O_o/model.py:520：rel_ts_bias 在层循环**外**只算一次，8 层共用同一份。
        rel_ts = self.relative_time(seq_ts, mask)
        for layer in self.layers:
            x = layer(x, mask, rel_ts)
        return x

class tower(nn.Module):
    def __init__(self, user_item_list, item_item_list, embbeding_dim, hidden_units=512,
                 dropout=0.2, item_only_cols=None, user_array_cols=()):
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
        self.HSTU = hstu(input_dim=hidden_units, head=8, num_layers=8)  # head=8: O_o/main.py:38

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
                user_array=None):
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
            output = self.HSTU(output, attn_mask, seq_ts)
            output = torch.nn.functional.normalize(output, p=2, dim=-1)
        return output

    @torch.no_grad()
    def predict(self, seq, token_type, user_feat, item_feat, seq_ts, user_array=None):
        """检索 query -> [b, H]，已归一化。对应 O_o/model.py:538-544。

        因果 mask 下只有最后一个位置看过整条序列，所以取 [:, -1]。
        ⚠️ 只在**右对齐**（真实 token 靠右、pad 在左）时正确。若上游改成左对齐或
        中间 padding，-1 会静默取到 pad，需要改成按 valid 的长度 gather。
        """
        return self.forward(seq, token_type, user_feat, item_feat, seq_ts, user_array)[:, -1, :]
            
        