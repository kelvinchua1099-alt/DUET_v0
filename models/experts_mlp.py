"""三个深度专家 —— 各读一个抽头层的缓存特征,各出一个裸 logit,在 logit 空间融合。

    z = sum_i w_i * z_i        w 来自 weights_mlp.WeightMLP(退化码字)

六条设计决策:

1. **两层,隐层 256**  一层线性太弱:输入是 cls+mean+std 的拼接,三种统计量之间的交互
   (如「std 高但 mean 低」)是有判别力的组合信号,线性切不出来;而且三个专家若都线性,
   logit 融合后整个系统会退化成一个大线性模型。三层以上无先例,DCPT 明确警告过「架构
   组件在有限数据上过拟合」。

2. **固定统计量标准化,不是 LayerNorm**  这是最容易写错的一处。用 register_buffer 存
   离线算好的均值方差:只在**训练集**上算(防泄漏),每个抽头层**各算一组**(不同深度的
   特征尺度差异极大 —— 实测 prenorm 范数 L0=42 / L32=12373)。
   LN 是逐样本逐通道动态算的,会把「这张图的浅层激活整体被 JPEG 压弱了」这个能量证据
   抹平 —— 那恰恰是浅层专家唯一的翻身机会。

3. **输出裸 logit,squeeze(-1) 不能忘**  忘了 squeeze 的话 z 是 (B,1),加权求和时会广播
   成 (B,B) 而**不报错**,静默出错。这里 squeeze 之后再加断言兜底。
   融合必须在 logit 空间:sigmoid 在两端极度压缩,概率平均会把「很确定」和「极其确定」
   钝化成同一个东西。

4. **Dropout 放两层之间,冻结时必须配 eval()**  Stage 2 冻结专家只训权重 MLP 时,只关
   梯度不关 dropout,专家输出就带随机性,权重 MLP 学的是噪声上的平均。这两句必须成对,
   所以这里把它们封进 freeze(),不给调用方漏写的机会。

5. **三个专家同一个类、同一组超参**  不给深层更宽的隐层。唯一允许的差异是 in_dim,
   因为三者的**池化方式**应当不同 —— 证据形态决定聚合算子:

       shallow: mean+std+prenorm   浅层证据是分布性质,要高阶矩,不要 attention
       mid:     cls+mean+std
       deep:    cls+mean           深层语义异常局部化,CLS/attention 有活干

   这个差异是设计意图,要在报告里写明,不是隐藏的容量不公。

6. **保留 return_hidden 接口**  五行代码的诊断工具:训完把三个专家的隐层表征做 t-SNE,
   若高度重合,就是「三个专家其实在做同一件事」的直接证据 —— 正好用来判断整个系统有没有
   退化成只用一层。
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

BANDS = ("shallow", "mid", "deep")

# 决策 5:唯一允许的差异。cache_features.py 以 "cls+mean+std" 缓存,三块各 hidden_size 维,
# 顺序 [cls | mean | std];这里按需切片,不需要为每个专家重跑一次缓存。
POOL = {
    "shallow": ("mean", "std"),      # 浅层证据是分布性质 -> 高阶矩,不要 attention
    "mid": ("cls", "mean", "std"),
    "deep": ("cls", "mean"),         # 深层语义异常局部化 -> CLS 有活干
}
USE_PRENORM = {"shallow": True, "mid": False, "deep": False}   # 浅层额外吃 2 个能量标量
BLOCK_ORDER = ("cls", "mean", "std")                           # 与 cache_features.pool 的拼接顺序一致


def band_slices(band: str, hidden_size: int) -> list[slice]:
    return [slice(BLOCK_ORDER.index(b) * hidden_size, (BLOCK_ORDER.index(b) + 1) * hidden_size)
            for b in POOL[band]]


def band_in_dim(band: str, hidden_size: int) -> int:
    return len(POOL[band]) * hidden_size + (2 if USE_PRENORM[band] else 0)


class Expert(nn.Module):
    """一个深度专家:标准化 -> Linear -> GELU -> Dropout -> Linear -> 裸 logit。"""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1,
                 zero_init_head: bool = True):
        super().__init__()
        self.in_dim, self.hidden = in_dim, hidden

        # 决策 2:固定统计量,不是 LayerNorm。默认恒等,必须由 fit_normalization() 填。
        self.register_buffer("feat_mean", torch.zeros(in_dim))
        self.register_buffer("feat_std", torch.ones(in_dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)          # 决策 4:放在两层之间
        self.fc2 = nn.Linear(hidden, 1)

        if zero_init_head:
            # 起手输出恒为 0 -> p=0.5「没有意见」,符合 README「零初始化」纪律。
            # 代价是第 0 步 fc1 拿不到梯度(dL/dh = W2^T dL/dz = 0),但 W2 本身有梯度,
            # 一步之后就活了。不是死局,别当 bug 修。
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    # --- 决策 2 -----------------------------------------------------------
    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-6) -> None:
        if mean.shape != (self.in_dim,) or std.shape != (self.in_dim,):
            raise ValueError(f"统计量形状应为 ({self.in_dim},),收到 {tuple(mean.shape)}/{tuple(std.shape)}")
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std.clamp_min(eps))
        self.fitted.fill_(True)

    # --- 决策 3 + 6 -------------------------------------------------------
    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """x: (B, in_dim) -> logit (B,);return_hidden 时附带 (B, hidden) 隐层表征。"""
        if not bool(self.fitted) and self.training:
            warnings.warn("专家未标定标准化统计量,正在用恒等变换训练。"
                          "先调 fit_normalization()(只用训练集!)。", stacklevel=2)
        # 缓存以 fp16 存(见 cache_features.py),numpy 运算又常升到 fp64,
        # 两种都会让 Linear 报 dtype 不匹配。统一对齐到 buffer 的精度。
        x = x.to(self.feat_mean.dtype)
        x = (x - self.feat_mean) / self.feat_std
        h = self.drop(self.act(self.fc1(x)))
        z = self.fc2(h).squeeze(-1)                       # 决策 3:(B,1) -> (B,)
        assert z.ndim == 1, f"logit 应为 1 维,得到 {tuple(z.shape)};squeeze 漏了会广播成 (B,B)"
        return (z, h) if return_hidden else z

    # --- 决策 4 -----------------------------------------------------------
    def freeze(self):
        """关梯度 **且** 关 dropout。两句必须成对,故封在一起。"""
        self.eval()                      # 关 dropout —— 极易漏掉
        self.requires_grad_(False)       # 关梯度
        return self


class ExpertBank(nn.Module):
    """三个专家 + logit 空间融合。

    forward 的 feats 必须按 [shallow, mid, deep] 的顺序给,即 cache_features 里
    probe_layers 选出的浅/中/深三个抽头层。
    """

    def __init__(self, hidden_size: int = 1280, hidden: int = 256, dropout: float = 0.1,
                 zero_init_head: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        # 决策 5:同一个类、同一组超参,只有 in_dim 因池化方式而异
        self.experts = nn.ModuleDict({
            b: Expert(band_in_dim(b, hidden_size), hidden, dropout, zero_init_head) for b in BANDS
        })

    # --- 决策 5:按 band 切片 ------------------------------------------------
    def slice_band(self, band: str, pooled: torch.Tensor, prenorm: torch.Tensor | None) -> torch.Tensor:
        """pooled: (B, 3*hidden_size) 该层的 cls|mean|std;prenorm: (B, 2)"""
        parts = [pooled[:, s] for s in band_slices(band, self.hidden_size)]
        if USE_PRENORM[band]:
            if prenorm is None:
                raise ValueError(f"{band} 专家配置了 prenorm,但调用时没传 prenorm")
            parts.append(prenorm)
        return torch.cat(parts, dim=-1)

    def forward(self, feats: torch.Tensor, prenorm: torch.Tensor | None = None,
                weights: torch.Tensor | None = None, return_parts: bool = False):
        """feats: (B, 3, 3*hidden_size);prenorm: (B, 3, 2);weights: (B, 3) 或 None(均匀)

        返回融合后的裸 logit (B,);return_parts 时附带各专家 logit 与隐层表征。
        """
        if feats.ndim != 3 or feats.shape[1] != len(BANDS):
            raise ValueError(f"feats 形状应为 (B, {len(BANDS)}, D),收到 {tuple(feats.shape)}")

        zs, hs = [], []
        for i, b in enumerate(BANDS):
            x = self.slice_band(b, feats[:, i], None if prenorm is None else prenorm[:, i])
            z, h = self.experts[b](x, return_hidden=True)
            zs.append(z)
            hs.append(h)
        z_stack = torch.stack(zs, dim=-1)                 # (B, 3)

        if weights is None:                               # Stage 1:均匀权重,防饿死
            weights = z_stack.new_full(z_stack.shape, 1.0 / len(BANDS))
        elif weights.shape != z_stack.shape:
            raise ValueError(f"weights 形状应为 {tuple(z_stack.shape)},收到 {tuple(weights.shape)}")

        z = (weights * z_stack).sum(-1)                   # 决策 3:logit 空间融合
        assert z.ndim == 1, f"融合后 logit 应为 1 维,得到 {tuple(z.shape)}"
        if return_parts:
            return z, {"expert_logits": z_stack, "hidden": torch.stack(hs, dim=1), "weights": weights}
        return z

    # --- 决策 4 -----------------------------------------------------------
    def freeze_experts(self):
        """Stage 2 用:专家全部冻结(梯度 + dropout),权重 MLP 单独训。"""
        for e in self.experts.values():
            e.freeze()
        return self

    # --- 决策 6 -----------------------------------------------------------
    @torch.no_grad()
    def collect_hidden(self, feats: torch.Tensor, prenorm: torch.Tensor | None = None) -> torch.Tensor:
        """-> (B, 3, hidden)。喂给 t-SNE:三簇若高度重合,说明三个专家在做同一件事。"""
        was_training = self.training
        self.eval()
        _, parts = self(feats, prenorm, return_parts=True)
        self.train(was_training)
        return parts["hidden"]


# --------------------------------------------------------------------------- 标准化标定

@torch.no_grad()
def fit_normalization(bank: ExpertBank, feats, prenorm, train_idx, eps: float = 1e-6) -> None:
    """在**训练集**上标定三个专家的标准化统计量(决策 2)。

    Args:
        feats:     (M, 3, 3*hidden_size) 全量缓存,可以是 np.memmap
        prenorm:   (M, 3, 2)
        train_idx: 训练集下标。**必须显式给出** —— 用全量算会把验证/测试集的分布信息
                   泄漏进标准化参数,让所有消融数字一起虚高,且极难事后察觉。

    每个 band 各算一组:不同深度的特征尺度差异极大(实测 prenorm 范数 L0=42 / L32=12373),
    共用一组统计量等于让深层压过浅层。
    """
    idx = torch.as_tensor(train_idx, dtype=torch.long)
    if idx.numel() == 0:
        raise ValueError("train_idx 为空")
    f = np_to_torch(feats)[idx].to(torch.float32)                          # (N, 3, D)
    p = None if prenorm is None else np_to_torch(prenorm)[idx].to(torch.float32)

    for i, b in enumerate(BANDS):
        x = bank.slice_band(b, f[:, i], None if p is None else p[:, i])    # (N, in_dim)
        bank.experts[b].set_normalization(x.mean(0), x.std(0), eps)


def np_to_torch(a):
    """memmap / ndarray / tensor 统一成可索引对象(不整体载入内存)。"""
    if isinstance(a, torch.Tensor):
        return a
    import numpy as np
    return torch.from_numpy(np.asarray(a))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(0)
    H = 1280                                    # DINOv3 ViT-H+/16
    bank = ExpertBank(hidden_size=H)

    print("--- 决策 5:同类同超参,只有 in_dim 因池化而异 ---")
    for b in BANDS:
        e = bank.experts[b]
        pool = "+".join(POOL[b]) + ("+prenorm" if USE_PRENORM[b] else "")
        print(f"  {b:<8} pool={pool:<20} in_dim={e.in_dim:<5} hidden={e.hidden}  "
              f"参数={count_params(e):,}")
    print(f"  {'合计':<8} {'':<25} {'':<12} 参数={count_params(bank):,}")

    B, M = 8, 200
    feats = np.random.randn(M, 3, 3 * H).astype(np.float32) * np.array([1.0, 5.0, 50.0], np.float32).reshape(1, 3, 1)
    prenorm = np.abs(np.random.randn(M, 3, 2)).astype(np.float32) * np.array([42, 380, 12373], np.float32).reshape(1, 3, 1)
    train_idx = np.arange(0, 160)               # 刻意只用前 80%,验证集不参与标定

    print("\n--- 决策 2:标定前后 ---")
    print(f"  标定前 fitted = {[bool(bank.experts[b].fitted) for b in BANDS]}")
    fit_normalization(bank, feats, prenorm, train_idx)
    print(f"  标定后 fitted = {[bool(bank.experts[b].fitted) for b in BANDS]}")
    for b in BANDS:
        e = bank.experts[b]
        print(f"  {b:<8} mean 幅度={e.feat_mean.abs().mean():8.3f}  std 幅度={e.feat_std.mean():8.3f}"
              f"   <- 各层各一组,尺度差异被吸收")

    x = torch.from_numpy(feats[:B]); pn = torch.from_numpy(prenorm[:B])

    print("\n--- 决策 3:输出形状与 logit 空间融合 ---")
    z, parts = bank(x, pn, return_parts=True)
    print(f"  融合 logit    {tuple(z.shape)}   (必须是 (B,),不是 (B,1))")
    print(f"  各专家 logit  {tuple(parts['expert_logits'].shape)}")
    print(f"  均匀权重      {parts['weights'][0].tolist()}")
    print(f"  零初始化 -> 输出恒 0: {bool((z == 0).all())}")

    w = torch.softmax(torch.randn(B, 3), -1)
    print(f"  带权重时形状仍为 {tuple(bank(x, pn, weights=w).shape)}  (若漏 squeeze 这里会是 (B,B))")

    print("\n--- 决策 4:freeze 必须同时关梯度和 dropout ---")
    bank.train()
    print(f"  freeze 前: training={bank.experts['mid'].training}  "
          f"可训练参数={count_params(bank):,}")
    bank.freeze_experts()
    print(f"  freeze 后: training={bank.experts['mid'].training}  "
          f"可训练参数={count_params(bank):,}")
    bank.train()                                  # 再次 train() 也不该解冻 dropout 之外的东西
    a = bank(x, pn); b_ = bank(x, pn)
    print(f"  冻结后两次前向是否确定(dropout 已关): {bool(torch.equal(a, b_))}")

    print("\n--- 决策 6:return_hidden / collect_hidden ---")
    h = bank.collect_hidden(x, pn)
    print(f"  隐层表征 {tuple(h.shape)}  -> 直接喂 t-SNE,三簇重合 = 三个专家在做同一件事")
