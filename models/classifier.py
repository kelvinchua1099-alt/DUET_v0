"""退化估计器 —— 直接吃原图,输出 6 维退化码字。可独立训练。

模型定义与加载入口 —— 训练 / 评估 / 导出码字在 training/train_classifier.py。

这是 README 架构图里那条独立分支:小 CNN,免费监督(退化标签随数据给定,零标注成本),
训完冻结,永不接触真假分类的梯度。梯度隔离是可归因性的前提 —— 路由信号必须是退化状态
的函数,而不能被端到端训练吸收成"看图内容"。

七条设计纪律:

1. **stem 不下采样**  JPEG 块效应周期 8 像素、噪声是像素级的。stem 一旦 stride=2 或
   patch 化,证据在第一步就没了 —— ViT 的 16x16 patch 化损失高频指纹正是这个原因,
   估计器不能重蹈覆辙。stem 的 4 个卷积全部 stride=1,在原分辨率上把高频榨出来再降采样。
   深度取 4 是为了感受野:3x3 每层给 RF +2,四层得 1+4*2 = 9 > 8,恰好盖住一整个 JPEG
   块周期。RF 只有 5(两层)时看不见完整周期,判不了块效应。

2. **GroupNorm 而非 BatchNorm**  BN 依赖 batch 统计量,而一个 batch 里混着各种退化强度,
   BN 会把它们的统计量搅在一起 —— 那恰恰就是要估计的东西。GN 逐样本归一化,安全。

3. **六个独立的头**  六维是可以并存的属性(一张图可以同时被压缩和模糊),不是互斥类别。
   用单个头预测 3^6=729 种组合会组合稀疏、无法泛化到未见组合。六个头各自输出。

4. **CORAL 序数回归**  档位有序(无 < 轻 < 重),普通 CE 把它们当无序类别,"错成相邻档"
   和"错成最远档"惩罚相同。这里把 K 档拆成 K-1 个二分类"是否 >= k",且用共享权重 +
   单调偏置(CORAL),保证 P(y>=1) >= P(y>=2) 恒成立,不会给出自相矛盾的分布。
   副产品:sum_k sigmoid(logit_k) 是一个连续的强度估计,可作为消融里的软码字。

5. **mean + std 双路池化**  噪声强度本质是方差量,只做 GAP 会把它平均掉。

6. **尺度旁路**  GN 虽然避开了 BN 的 batch 间搅拌,但它逐样本归一化,同样会抹掉每一层的
   全局尺度。实测代价很具体:blur(改变频率分布 *形状*,比例量)能学到 0.73,而 noise
   (改变高频能量 *幅度*,尺度量)死在随机水平 0.35 —— 证据被逐层归一化系统性抹除。
   修法不是放弃 GN,而是让 stem 的 **归一化之前** 的全分辨率激活直接旁路到分类头:
   那里是像素级噪声唯一完好的地方。log1p 压一下动态范围以免重噪声样本主导优化。
   抽头取 stem 的首尾两层:第 0 层 RF=3 离原始像素最近、尺度最纯(噪声能量),末层 RF=9
   盖住整个块周期(块边界对比度)。只挂末层是错的 —— 那时前面已经过了 3 个 GN,要抢救的
   绝对尺度早就被归掉了。

7. **块边界统计量**  纪律 6 救回了尺度,但救不了 *周期*。JPEG 的证据是相位锁定的 8
   像素周期结构,而 stages 的 4 次 stride=2 会把它混叠掉、末端池化又对所有位置求平均,
   恰好抹平「边界位置 vs 其余位置」的差。实测 RF 5 -> 9 对 jpeg 维零增益(0.360 ->
   0.383,均在随机水平),证明瓶颈在池化而非感受野。解法是直接把该比值算成 2 个标量拼
   进特征。代价要诚实说明:这是硬编码了「块为 8x8」的手工先验,对 held-out WebP 这类
   块结构不同的未见编码可能失效甚至误导 —— 故做成 --no-blockiness 开关,消融里给出
   有/无两行数字让它自己说话。

数据增广只允许保退化统计量的操作:随机裁剪(对齐到 8 像素以免打乱 JPEG 网格)、翻转、
90 度旋转。**禁止** resize / 重压缩 / 调色 / 模糊 / 加噪 —— 那些会直接改变标签。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_features import DEG_CODE_VALUES, DEG_DIMS, N_LEVELS  # noqa: E402

JPEG_GRID = 8   # 裁剪偏移对齐到 8 的倍数,避免打乱 JPEG 的 8x8 块网格


# --------------------------------------------------------------------------- 模型

def gn(c: int) -> nn.GroupNorm:
    """GroupNorm,组数取不超过 8 且能整除通道数的最大值(纪律 2)。"""
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


def conv_block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
        gn(cout),
        nn.SiLU(inplace=True),
    )


def blockiness_stats(x: torch.Tensor, grid: int = JPEG_GRID) -> torch.Tensor:
    """块效应统计量 -> (B, 2)。纪律 7,见模块 docstring。

    JPEG 按 grid x grid 独立量化,相邻块的误差互不协调,于是交界处留下细微台阶,而块
    内部因高频被量化掉反而更平滑。所以「差分落在 mod grid == grid-1 的位置」与「落在
    其余位置」的比值,就是压缩强度的直接读数。实测(合成集,150 张/档):

        无压缩  6.95 6.85 7.08 6.91 6.98 6.91 6.84 6.98  -> 比值 1.007
        q=75    6.88 6.79 6.89 6.74 6.88 6.87 6.81 8.06  -> 比值 1.179
        q=32    4.89 5.08 4.99 4.87 4.88 5.04 4.82 7.30  -> 比值 1.478
                                              ↑ 块边界

    网络本身拿不到这个信号:stages 的 4 次 stride=2 会把周期 grid 混叠掉,而末端的
    mean/std 池化对所有空间位置求平均,恰好把「边界位置 vs 其余位置」的差抹平。看得见
    不等于测得出 —— 把 RF 从 5 加到 9 对 jpeg 维毫无增益,正是因为瓶颈在池化不在感受野。

    返回值按降序排列(max, min)而非 (横, 纵):训练增广里的 90 度旋转会交换两个轴,
    排序后特征对该增广不变。JPEG 对两轴的影响本就对称,不损失信息。
    """
    g = x.mean(1)                                                   # (B, H, W) 转灰度
    out = []
    for gr in (g.diff(dim=-1).abs(), g.diff(dim=-2).transpose(-1, -2).abs()):   # 横向、纵向
        idx = torch.arange(gr.shape[-1], device=x.device) % grid == grid - 1
        edge = gr[..., idx].mean((-2, -1))
        inner = gr[..., ~idx].mean((-2, -1))
        out.append(edge / (inner + 1e-6))
    return torch.stack(out, -1).sort(-1, descending=True).values     # (B, 2)


class CoralHead(nn.Module):
    """一个退化维度的 CORAL 序数头(纪律 4)。

    K 档 -> K-1 个 "y >= k" 的二分类 logit,共享同一个权重向量 w,只有偏置不同:
        logit_k = w . h + b_k
    因为 w 共享,logit 之间的差恒等于偏置之差,与输入无关 —— 这就保证了秩一致性。
    再把偏置参数化成严格递减(b_k = b_1 - cumsum(softplus(delta))),于是
        P(y>=1) >= P(y>=2) >= ...
    对任意输入恒成立,不可能给出自相矛盾的分布。
    """

    def __init__(self, in_dim: int, n_levels: int = N_LEVELS):
        super().__init__()
        self.k = n_levels - 1                       # 阈值个数
        self.w = nn.Linear(in_dim, 1, bias=False)
        self.b0 = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.full((self.k - 1,), -1.0)) if self.k > 1 else None

    def biases(self) -> torch.Tensor:
        if self.delta is None:
            return self.b0
        gaps = F.softplus(self.delta)               # > 0
        return torch.cat([self.b0, self.b0 - torch.cumsum(gaps, 0)])   # 严格递减

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.w(h) + self.biases()            # (B, K-1)


class DegradationCNN(nn.Module):
    """小 CNN:512x512 原图 -> 6 个维度各 K-1 个序数 logit。"""

    def __init__(self, n_dims: int = len(DEG_DIMS), n_levels: int = N_LEVELS,
                 widths: tuple[int, ...] = (32, 64, 96, 128, 160), dropout: float = 0.1,
                 scale_bypass: bool = True, stem_depth: int = 4, blockiness: bool = True):
        super().__init__()
        self.n_dims, self.n_levels = n_dims, n_levels
        self.widths, self.scale_bypass, self.blockiness = tuple(widths), scale_bypass, blockiness
        w0 = widths[0]

        # 纪律 1:stem 全 stride=1,在原分辨率上保住 8 像素周期与像素级噪声。
        # 深度 4 -> 感受野 1+4*2 = 9 像素,刚好盖住一整个 JPEG 块周期(8);
        # 深度 2 时 RF 只有 5,看不见完整周期,判不了块效应。
        self.stem_convs = nn.ModuleList(
            [nn.Conv2d(3 if i == 0 else w0, w0, 3, padding=1, bias=False) for i in range(stem_depth)]
        )
        self.stem_norms = nn.ModuleList(
            [nn.Sequential(gn(w0), nn.SiLU(inplace=True)) for _ in range(stem_depth)]
        )
        # 双抽头(见纪律 6):第 0 层 RF=3,离原始像素最近,尺度信息最纯 -> 噪声能量;
        # 最后一层 RF=9,盖住整个块周期,但前面已过 stem_depth-1 个 GN -> 块边界对比度。
        # 只挂在末层是错的:旁路要抢救的绝对尺度在到达抽头点前已被归掉好几次。
        self.taps = (0, stem_depth - 1) if stem_depth > 1 else (0,)

        stages = []
        for cin, cout in zip(widths[:-1], widths[1:]):
            stages += [conv_block(cin, cout, 2), conv_block(cout, cout, 1)]
        self.stages = nn.Sequential(*stages)

        self.drop = nn.Dropout(dropout)
        feat = (widths[-1] * 2                                        # 纪律 5
                + (w0 * 2 * len(self.taps) if scale_bypass else 0)    # 纪律 6
                + (2 if blockiness else 0))                           # 纪律 7
        self.heads = nn.ModuleList([CoralHead(feat, n_levels) for _ in range(n_dims)])

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h, taps = x, []
        for i, (conv, norm) in enumerate(zip(self.stem_convs, self.stem_norms)):
            raw = conv(h)                           # 归一化 *之前* 的全分辨率激活
            if self.scale_bypass and i in self.taps:
                taps.append(raw)
            h = norm(raw)

        h = self.stages(h)                          # (B, C, h, w)
        flat = h.flatten(2)
        out = [flat.mean(-1), flat.std(-1)]                          # 纪律 5:mean + std
        for raw in taps:                                             # 纪律 6:尺度旁路
            r = raw.flatten(2)
            # log1p 压动态范围,否则重噪声样本的量级会主导优化
            out += [torch.log1p(r.abs().mean(-1)), torch.log1p(r.std(-1))]
        if self.blockiness:                                          # 纪律 7:块边界统计量
            out.append(blockiness_stats(x))
        return torch.cat(out, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """-> (B, n_dims, K-1) 的序数 logit"""
        h = self.drop(self.features(x))
        return torch.stack([head(h) for head in self.heads], dim=1)

    # --- 解码 -------------------------------------------------------------
    @staticmethod
    def decode(logits: torch.Tensor, snap: bool = True) -> torch.Tensor:
        """序数 logit -> 离散码字 (B, n_dims),取值 {0..K-1}

        snap:把预测吸附到该维**合法**的取值上。码字取值有空洞(blur 只有 0/1/2/4),
        CORAL 数阈值时可能数出 3 —— 那是个真值里不存在的档位,必错。吸到最近的合法值
        是白捡的准确率,且保证输出的码字下游一定认得。关掉它可用于诊断原始输出分布。
        """
        code = (logits > 0).sum(-1)                 # sigmoid(l)>0.5 等价于 l>0
        if not snap:
            return code
        out = code.clone()
        for i, d in enumerate(DEG_DIMS[: code.shape[-1]]):
            legal = torch.tensor(DEG_CODE_VALUES[d], device=code.device)
            out[:, i] = legal[(code[:, i, None] - legal[None, :]).abs().argmin(-1)]
        return out

    @staticmethod
    def soft_code(logits: torch.Tensor) -> torch.Tensor:
        """连续强度估计 (B, n_dims) ∈ [0, K-1]。CORAL 的副产品,留给软码字消融。"""
        return torch.sigmoid(logits).sum(-1)

    @staticmethod
    def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """CORAL 损失:每个阈值一个 BCE。target (B, n_dims) ∈ {0..K-1}"""
        k = logits.shape[-1]
        levels = torch.arange(1, k + 1, device=target.device).view(1, 1, k)
        y = (target.unsqueeze(-1) >= levels).float()          # (B, D, K-1)
        return F.binary_cross_entropy_with_logits(logits, y)

    def freeze(self):
        """训完冻结。README 纪律:估计器永不接触分类梯度。"""
        self.eval()
        self.requires_grad_(False)
        return self


# --------------------------------------------------------------------------- 加载

def load_estimator(ckpt: str | Path, device=None) -> DegradationCNN:
    """给下游用的入口:载入并**冻结**估计器。"""
    d = torch.load(ckpt, map_location=device or "cpu")
    c = d["cfg"]
    m = DegradationCNN(n_dims=c["n_dims"], n_levels=c["n_levels"],
                       widths=tuple(c.get("widths", (32, 64, 96, 128, 160))),
                       stem_depth=c.get("stem_depth", 4),
                       scale_bypass=c.get("scale_bypass", True),
                       blockiness=c.get("blockiness", True))
    m.load_state_dict(d["model"])
    if device:
        m.to(device)
    return m.freeze()
