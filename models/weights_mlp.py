"""权重 MLP —— 退化码字 -> 三个深度专家的融合权重。

    w = softmax(MLP([one_hot(code) ; soft/(K-1)]) / tau)     -> (B, 3)

这是 SQuaDE 唯一「自适应」的自由度,也是全部 novelty 的所在。README 的消融链里
A2(三专家 + 均匀权重) -> A3(+ 本模块) 那一格增益,量化的就是「退化路由」值多少。
所以这个模块的每一条约束都是为了让那格增益**可归因**:

1. **输入只有码字,绝不含图像特征**  forward 的签名里永远不会出现 img_feat。
   一旦权重 MLP 能看见图像内容,它就会学到「这种内容的图用深层专家更好」——
   那是内容驱动路由(HIT-VIRLAB 做的事),不是退化驱动路由。novelty 和可解释性
   会同时失效,而且从最终指标上**看不出来**。这里用运行时断言把它焊死。

2. **输出层 weight 与 bias 严格为 0**  于是 logits 恒为 0,softmax 输出严格
   (1/3, 1/3, 1/3)。含义是:Stage 2 的起点在数值上**精确等于** Stage 1 的终点,
   本模块起手时严格等价于「不存在」。A2 -> A3 的差值因此干净地归因于路由本身,
   而不是初始化扰动。

3. **温度 tau: 2.0 -> 1.0 退火**  高温 = 权重平滑(接近均匀),低温 = 允许锐化。
   初期高温防止权重过早锁死到某个专家(MoE 坍缩的温和防线),后期放开让它果断选择。

4. **一层隐层,宽 32**  输入 24 维、输出 3 维,本质是定义在 3^6 = 729 个离散点上的
   小函数,32 已经过参数化。加深只会增加对「训练集里出现过的退化组合」的过拟合风险,
   损害对未见组合(如 held-out WebP 档)的泛化。参数量 24*32+32 + 32*3+3 = 899。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_features import DEG_DIMS, N_LEVELS  # noqa: E402

N_EXPERTS = 3           # 浅 / 中 / 深
TAU_START, TAU_END = 2.0, 1.0


class WeightMLP(nn.Module):
    """退化码字 -> softmax 权重 (B, N_EXPERTS)。

    输入维度 = n_dims * n_levels (one-hot) + n_dims (soft) = 6*3 + 6 = 24。

    one-hot 与 soft 并用是刻意的分工:one-hot 表达「档位身份」(哪一档),不假设档位之间
    的距离;soft 表达「有序强度」(多重),保留 one-hot 四舍五入掉的档间信息。两者都只
    来自 6 维码字,瓶颈宽度不变 —— 24 维是 6 维码字的无损重编码,不是新增信息通道。
    """

    def __init__(self, n_dims: int = len(DEG_DIMS), n_levels: int = N_LEVELS,
                 n_experts: int = N_EXPERTS, hidden: int = 32):
        super().__init__()
        self.n_dims, self.n_levels, self.n_experts = n_dims, n_levels, n_experts
        self.in_dim = n_dims * n_levels + n_dims

        self.fc1 = nn.Linear(self.in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_experts)

        # 约束 2:输出层严格置零 -> logits 恒 0 -> softmax 严格 (1/3, 1/3, 1/3)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # tau 存成 buffer:随 state_dict 保存、随 .to(device) 搬运,恢复训练时不会丢
        self.register_buffer("tau", torch.tensor(float(TAU_START)))

    # --- 温度 -------------------------------------------------------------
    def set_temperature(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError(f"温度必须为正,收到 {tau}")
        self.tau.fill_(float(tau))

    def anneal(self, progress: float, schedule: str = "cosine") -> float:
        """按训练进度 progress ∈ [0,1] 把 tau 从 TAU_START 退火到 TAU_END。

        cosine 前期降得慢,把高温期(防坍缩)拉长,后期再快速放开锐化;linear 供对照。
        在 Stage 2 的每个 step 调用:mlp.anneal(step / total_steps)
        """
        p = min(max(progress, 0.0), 1.0)
        f = p if schedule == "linear" else 0.5 * (1 - math.cos(math.pi * p))
        tau = TAU_START + (TAU_END - TAU_START) * f
        self.set_temperature(tau)
        return tau

    # --- 编码 -------------------------------------------------------------
    def encode(self, code: torch.Tensor, soft: torch.Tensor | None = None) -> torch.Tensor:
        """(B, n_dims) 码字 -> (B, in_dim) 网络输入。

        soft 为 None 时退化成 code.float() —— A3-o oracle 只有真值硬码字,没有估计器
        的连续输出,这条分支让同一个模块能直接跑 oracle 对照。
        """
        oh = F.one_hot(code, self.n_levels).flatten(1).float()          # (B, n_dims*n_levels)
        s = (code.float() if soft is None else soft.float())
        s = s / max(self.n_levels - 1, 1)                               # 归到 [0,1],与 one-hot 同尺度
        return torch.cat([oh, s], dim=-1)

    # --- 前向 -------------------------------------------------------------
    # 签名里永远不会有 img_feat / feats / x 这类参数。见约束 1。
    def forward(self, code: torch.Tensor, soft: torch.Tensor | None = None,
                return_logits: bool = False):
        """code: (B, n_dims) 整数张量 ∈ {0..K-1};soft: (B, n_dims) 浮点 ∈ [0, K-1]"""
        # 约束 1 的运行时防线:只接受形状恰为 (B, n_dims) 的整数码字。
        # 图像特征(几百上千维、浮点)会在这里直接崩掉,而不是被静默接受。
        if code.dtype not in (torch.long, torch.int32, torch.int16, torch.uint8):
            raise TypeError(
                f"code 必须是整数码字,收到 {code.dtype}。权重 MLP 只能看退化状态 —— "
                "喂图像特征会把退化驱动路由变成内容驱动路由,novelty 与可解释性同时失效。"
            )
        if code.ndim != 2 or code.shape[-1] != self.n_dims:
            raise ValueError(f"code 形状应为 (B, {self.n_dims}),收到 {tuple(code.shape)}")
        if code.min() < 0 or code.max() >= self.n_levels:
            raise ValueError(f"码字越界:取值应在 0..{self.n_levels - 1}")
        if soft is not None and soft.shape != code.shape:
            raise ValueError(f"soft 形状应与 code 一致,收到 {tuple(soft.shape)}")

        logits = self.fc2(F.relu(self.fc1(self.encode(code, soft))))    # (B, n_experts)
        w = F.softmax(logits / self.tau, dim=-1)
        return (w, logits) if return_logits else w

    # --- 诊断 -------------------------------------------------------------
    @torch.no_grad()
    def route_table(self, soft: torch.Tensor | None = None) -> dict:
        """把学到的路由摊开成一张表,用于论文里的可解释性分析。

        对每个退化维度,固定其余维为 0、只让该维走遍 0..K-1,看权重怎么变。
        README 的核心主张(压缩重 -> 听深层,裁剪碎 -> 听浅层)在这张表上应当直接可读。
        """
        dev = self.fc1.weight.device
        out = {}
        for d, name in enumerate(DEG_DIMS[: self.n_dims]):
            codes = torch.zeros(self.n_levels, self.n_dims, dtype=torch.long, device=dev)
            codes[:, d] = torch.arange(self.n_levels, device=dev)
            out[name] = self(codes).cpu().tolist()
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = WeightMLP()
    print(f"输入维度  : {m.in_dim}  = {m.n_dims}x{m.n_levels} one-hot + {m.n_dims} soft")
    print(f"参数量    : {count_params(m)}")
    print(f"初始温度  : {m.tau.item()}")

    print("\n--- 约束 2:初始化后输出必须严格均匀 ---")
    codes = torch.randint(0, N_LEVELS, (5, len(DEG_DIMS)))
    softs = torch.rand(5, len(DEG_DIMS)) * (N_LEVELS - 1)
    w = m(codes, softs)
    print("任意输入的输出:", [[round(v, 6) for v in r] for r in w.tolist()[:2]])
    print("与 1/3 的最大偏差:", (w - 1 / 3).abs().max().item())
    print("逐比特等于 1/3 :", bool(torch.equal(w, torch.full_like(w, 1 / 3))))
    print("soft=None(oracle 模式)也均匀:", bool(torch.equal(m(codes), torch.full_like(w, 1 / 3))))

    print("\n--- 约束 3:温度退火 ---")
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        print(f"  progress={p:.2f}  cosine tau={m.anneal(p):.3f}   "
              f"linear tau={m.anneal(p, 'linear'):.3f}")
    m.set_temperature(TAU_START)

    print("\n--- 约束 1:图像特征必须被拒绝 ---")
    for bad, why in [(torch.randn(5, 3840), "3840 维图像特征"),
                     (torch.rand(5, 6), "浮点而非整数码字"),
                     (torch.randint(0, 3, (5, 4)), "维度不是 6")]:
        try:
            m(bad)
            print(f"  ❌ {why} 竟被接受")
        except (TypeError, ValueError) as e:
            print(f"  ✅ {why} 被拒绝: {type(e).__name__}")

    print("\n--- 梯度可达性(输出层零初始化的一步延迟) ---")
    w, logits = m(codes, softs, return_logits=True)
    logits.sum().backward()
    print(f"  fc2.weight 梯度范数 = {m.fc2.weight.grad.norm():.4f}  (非零 -> 下一步就活了)")
    print(f"  fc1.weight 梯度范数 = {m.fc1.weight.grad.norm():.4f}  (第 0 步为 0,符合预期)")
