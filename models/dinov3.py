"""Frozen Hugging Face DINO backbone for intermediate-layer feature extraction.

The released DUET checkpoints use facebook/dinov2-giant with a 504×504 crop.
The historical module and class names are retained for source compatibility.
The backbone is frozen, every tapped layer is normalised consistently, image
resizing is avoided, and deterministic crops are used for reproducible caches.
"""

from __future__ import annotations

import hashlib
import warnings

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

DEFAULT_MODEL = "facebook/dinov2-giant"
CROP_SIZE = 504      # 16 的整数倍 -> 32x32 = 1024 个 patch token,天然 patch 对齐


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DINOv3Preprocessor:
    """不做 resize 的预处理:rescale -> 512x512 crop -> ImageNet 标准化。

    mean/std 从 checkpoint 自带的 AutoImageProcessor 读,保持与官方权重一致。

    小于 512 的图:默认保持原生尺寸(裁到 patch 整数倍),不做 padding。理由是 crop
    本身就是本项目要研究的退化档之一,退化流水线裁出来的小图若再 reflect-pad 回 512,
    接缝处的人工高频对取证模型等同于伪造痕迹。代价是这些图 token 数不同,需要按尺寸
    分桶 batch。若确实要求全批严格 512,把 on_small 设为 "pad_reflect"。
    """

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        patch_size: int = 16,
        crop_size: int = CROP_SIZE,
        crop: str = "center",
        on_small: str = "native",
    ):
        if crop not in ("center", "random"):
            raise ValueError(f"crop 只能是 center / random,收到 {crop!r}")
        if on_small not in ("native", "pad_reflect"):
            raise ValueError(f"on_small 只能是 native / pad_reflect,收到 {on_small!r}")
        if crop_size % patch_size:
            raise ValueError(f"crop_size={crop_size} 不是 patch_size={patch_size} 的整数倍")

        proc = AutoImageProcessor.from_pretrained(name)
        proc.do_resize = False          # 纪律 3
        proc.do_center_crop = False
        self.hf_processor = proc
        self.image_mean = torch.tensor(proc.image_mean).view(-1, 1, 1)
        self.image_std = torch.tensor(proc.image_std).view(-1, 1, 1)
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.crop = crop
        self.on_small = on_small

    def __call__(self, image, image_id: str | None = None) -> torch.Tensor:
        """image: PIL.Image / HWC uint8 ndarray / CHW tensor -> (1, 3, H, W) float32

        image_id: crop="random" 时用来派生确定性种子。同一个 id 永远裁到同一个位置,
                  所以缓存仍然可复现。crop="center" 时忽略。
        """
        x = self._to_chw_float(image)
        h, w = x.shape[-2:]
        s = self.crop_size

        if h >= s and w >= s:
            top, left = self._crop_origin(h - s, w - s, image_id)
            x = x[..., top : top + s, left : left + s]
        elif self.on_small == "pad_reflect":
            x = self._pad_reflect_to(x, s, s)
        else:
            # 原生尺寸,只裁到 patch 整数倍(至多丢右/下各 patch_size-1 个像素)
            p = self.patch_size
            hh, ww = min(h, s) // p * p, min(w, s) // p * p
            if hh == 0 or ww == 0:
                raise ValueError(f"图像 {(h, w)} 小于一个 patch ({p}x{p})")
            top, left = self._crop_origin(min(h, s) - hh, min(w, s) - ww, image_id)
            x = x[..., top : top + hh, left : left + ww]

        return ((x - self.image_mean) / self.image_std).unsqueeze(0)

    def _crop_origin(self, max_top: int, max_left: int, image_id: str | None) -> tuple[int, int]:
        if self.crop == "center" or (max_top == 0 and max_left == 0):
            return max_top // 2, max_left // 2
        # 由 image_id 派生种子:随机但可复现,重跑缓存结果一致(纪律 4)
        seed = int(hashlib.sha256((image_id or "").encode()).hexdigest()[:16], 16)
        g = torch.Generator().manual_seed(seed)
        top = int(torch.randint(max_top + 1, (1,), generator=g))
        left = int(torch.randint(max_left + 1, (1,), generator=g))
        return top, left

    @staticmethod
    def _pad_reflect_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        import torch.nn.functional as F

        ph, pw = max(0, h - x.shape[-2]), max(0, w - x.shape[-1])
        if ph or pw:
            x = F.pad(x.unsqueeze(0), (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), mode="reflect")[0]
        return x[..., :h, :w]

    @staticmethod
    def _to_chw_float(image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            x = image.float()
            if x.ndim == 4 and x.shape[0] == 1:
                x = x[0]
            if x.max() > 1.5:           # 还是 0..255
                x = x / 255.0
            return x
        import numpy as np

        # copy():PIL 给出的 buffer 是只读的,直接 from_numpy 会让 torch 告警
        arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


class DINOv3Backbone(nn.Module):
    """冻结的 DINOv3,输出各层已归一化、已按 CLS / register / patch 拆分的 token。"""

    def __init__(
        self,
        name: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """dtype 默认 bf16 而非 fp16 —— 见模块 docstring 纪律 5。"""
        super().__init__()
        self.name = name
        self.device_ = pick_device(device)
        if self.device_.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32     # CPU 上半精度多数算子无加速甚至不支持
        self.dtype = dtype

        model = AutoModel.from_pretrained(name, dtype=dtype)
        model.eval().requires_grad_(False)        # 纪律 1
        self.model = model.to(self.device_)

        cfg = model.config
        self.n_layers: int = cfg.num_hidden_layers
        self.hidden_size: int = cfg.hidden_size
        self.patch_size: int = cfg.patch_size
        # register token 数:DINOv3 一定有这个字段;DINOv2 只有 -with-registers 变体才有,
        # 普通 dinov2 没有 register,前缀就只有 CLS 一个。取错会把 patch token 当成
        # register 切掉(或反之),池化出来的 mean/std 全是错的,而且不会报错。
        self.n_registers: int = int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.n_prefix: int = 1 + self.n_registers  # CLS + registers,见 modeling 里的 cat 顺序

        # 最后那层 LayerNorm 的属性名各家不同(纪律 2 要求全层统一过它):
        #   DINOv3ViTModel -> .norm      Dinov2Model -> .layernorm
        for attr in ("norm", "layernorm"):
            if hasattr(model, attr):
                self.final_norm = getattr(model, attr)
                self.final_norm_name = attr
                break
        else:
            raise SystemExit(f"{name}: 找不到最后的 LayerNorm(试过 .norm / .layernorm)")

    def train(self, mode: bool = True):
        """屏蔽解冻。backbone 一旦进入 train mode,RoPE 位置增广会让前向变得不确定。"""
        if mode:
            warnings.warn("DINOv3Backbone 恒为 eval;忽略 train(True)。", stacklevel=2)
        return super().train(False)

    def make_preprocessor(self, **kw) -> DINOv3Preprocessor:
        return DINOv3Preprocessor(self.name, patch_size=self.patch_size, **kw)

    @torch.inference_mode()
    def forward_layers(
        self,
        pixel_values: torch.Tensor,
        layers: list[int] | None = None,
        return_norm_stats: bool = True,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """单次前向,取出指定层的 token。

        Args:
            pixel_values: (B, 3, H, W),H/W 须为 patch_size 整数倍(走 DINOv3Preprocessor
                          自动满足;512 crop 时是 512x512 -> 32x32 grid)。
            layers:       层索引,0 是 embedding 输出,i>=1 是第 i 个 block 的输出。
                          None 表示全取。训练阶段只传探针选出的 3 层可大幅省内存。
            return_norm_stats: 额外返回归一化 *之前* 的 patch token 激活范数统计量。
                          LayerNorm 是 per-token 的,会抹掉每个 token 的绝对激活强度,
                          而 JPEG / 噪声等退化的能量证据恰好住在那里。每层 2 个标量,
                          缓存体积几乎不变,是否喂给专家留作消融开关。

        Returns:
            {layer_idx: {"cls": (B,C), "regs": (B,R,C), "patches": (B,N,C),
                         "grid": (h,w), "prenorm_stats": (B,2)}}
        """
        if pixel_values.shape[-1] % self.patch_size or pixel_values.shape[-2] % self.patch_size:
            raise ValueError(
                f"输入 {tuple(pixel_values.shape[-2:])} 不是 patch_size={self.patch_size} 的整数倍;"
                "请走 DINOv3Preprocessor,不要让 HF processor 偷偷 resize。"
            )

        h_grid = pixel_values.shape[-2] // self.patch_size
        w_grid = pixel_values.shape[-1] // self.patch_size

        pixel_values = pixel_values.to(self.device_, self.dtype)
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)

        idx = list(range(len(out.hidden_states))) if layers is None else list(layers)

        # 一次性堆起来算,而不是逐层循环。LayerNorm 沿最后一维逐元素,对
        # (L,B,T,C) 调一次与逐层调 L 次数值等价,但把 ~5L 次小 kernel 启动压成 ~5 次。
        # 实测 MPS 上 33 层的逐层写法要 ~4 s/图,其中约 3 s 是启动开销而非计算。
        stack = torch.stack([out.hidden_states[i] for i in idx])          # (L,B,T,C)

        stats = None
        if return_norm_stats:
            # 存 fp32:实测 L32 的范数已达 1.2e4,半精度在该量级的间距太大
            # (fp16 为 8, bf16 为 64),而这两个标量本身就是要拿来比较退化强度的。
            #
            # **逐层算,不要对整个 stack 调 .float()**:stack 是 (L,B,T,C),
            # ViT-g 上 (41,24,1297,1536) bf16 就是 3.9 GB,.float() 一次性翻倍成 7.8 GB
            # 直接 OOM(DINOv3 的 (33,16,1029,1280) 只有 1.4 GB,所以一直没暴露)。
            # 逐层只需要 (B,T,C) fp32 ≈ 191 MB,而且结果逐位相同。
            per = []
            for k in range(stack.shape[0]):
                n = stack[k, :, self.n_prefix :].float().norm(dim=-1)      # (B,N)
                per.append(torch.stack([n.mean(-1), n.std(-1)], dim=-1))   # (B,2)
            stats = torch.stack(per)                                       # (L,B,2)

        h = self.final_norm(stack)                                        # 纪律 2:全层统一 norm

        feats: dict[int, dict[str, torch.Tensor]] = {}
        for k, i in enumerate(idx):
            entry: dict[str, torch.Tensor] = {
                "cls": h[k][:, 0],
                "regs": h[k][:, 1 : self.n_prefix],
                "patches": h[k][:, self.n_prefix :],
                "grid": (h_grid, w_grid),
            }
            if stats is not None:
                entry["prenorm_stats"] = stats[k]
            feats[i] = entry
        return feats

    def forward_blocks(
        self,
        pixel_values: torch.Tensor,
        layers: list[int],
        max_block: int | None = None,
        resume: torch.Tensor | None = None,
        start_block: int = 0,
    ) -> tuple[dict[int, dict[str, torch.Tensor]], torch.Tensor]:
        """逐 block 前向,**可以在 max_block 处停下** —— 提前退出真正省算力的前提。

        `forward_layers` 走的是 `model(output_hidden_states=True)`,它**总是跑完全部 40 个
        block**,哪怕只要第 14 层。推理时门判"干净"就该在 L27 停住,那条路径必须自己跑 block。

        Args:
            layers:      要取的层号(0 = embedding 输出, i>=1 = 第 i 个 block 之后)
            max_block:   跑到第几个 block 为止(含)。None = 跑满
            resume:      上一次调用返回的 hidden,从 start_block+1 接着跑
            start_block: resume 对应的 block 号

        Returns:
            (与 forward_layers 同构的 dict, 最后一层的 pre-norm hidden)
            返回 hidden 是为了让调用方能从这里继续往下跑,不必重跑前面的 block。
        """
        if pixel_values is not None and (pixel_values.shape[-1] % self.patch_size
                                         or pixel_values.shape[-2] % self.patch_size):
            raise ValueError(f"输入 {tuple(pixel_values.shape[-2:])} 不是 patch_size="
                             f"{self.patch_size} 的整数倍;请走 DINOv3Preprocessor")
        blocks = self.model.encoder.layer
        top = len(blocks) if max_block is None else int(max_block)
        if top > len(blocks):
            raise ValueError(f"max_block={top} 超过 block 数 {len(blocks)}")

        if resume is None:
            h_grid = pixel_values.shape[-2] // self.patch_size
            w_grid = pixel_values.shape[-1] // self.patch_size
            x = self.model.embeddings(pixel_values.to(self.device_, self.dtype))
            self._grid = (h_grid, w_grid)
        else:
            x = resume

        want = set(layers)
        got: dict[int, torch.Tensor] = {}
        if resume is None and 0 in want:
            got[0] = x
        for i in range(start_block + 1, top + 1):
            # 这一版 transformers 的 Dinov2Layer.forward 返回**裸 Tensor**,旧版返回 tuple。
            # 无脑写 [0] 会把 batch 维索引掉:(B,T,C) -> (T,C),下一个 block 当成
            # batch=T、seq=C,注意力矩阵瞬间涨到几十 GB 直接 OOM。两种都要认。
            o = blocks[i - 1](x)
            x = o[0] if isinstance(o, (tuple, list)) else o
            if i in want:
                got[i] = x

        feats: dict[int, dict[str, torch.Tensor]] = {}
        for i, raw in got.items():
            # 纪律 2:全层统一 final_norm。prenorm 统计量要在 norm **之前**取
            n = raw[:, self.n_prefix :].float().norm(dim=-1)               # (B,N)
            hh = self.final_norm(raw)
            feats[i] = {"cls": hh[:, 0], "regs": hh[:, 1 : self.n_prefix],
                        "patches": hh[:, self.n_prefix :], "grid": self._grid,
                        "prenorm_stats": torch.stack([n.mean(-1), n.std(-1)], dim=-1)}
        return feats, x

    def forward_deltas(
        self,
        pixel_values: torch.Tensor,
        layers: list[int],
        return_norm_stats: bool = True,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """层间残差增量 —— 差分在 **归一化之前** 的 token 上做,之后再统一 norm。

        给 layers=[a,b,c],返回三个"band":

            band a: norm(h_a)          累积到第 a 层
            band b: norm(h_b - h_a)    第 a->b 段新增的残差
            band c: norm(h_c - h_b)    第 b->c 段新增的残差

        与「先 norm 再相减」的关键区别:LayerNorm 是逐 token 的仿射 + 缩放,
        norm(h_b) - norm(h_a) != norm(h_b - h_a),前者不是残差增量。ViT 的残差流
        h_l = h_{l-1} + attn + mlp 的可加性只在 **归一化之前** 成立,所以差分必须
        在这里做。差完再 norm 是为了守住纪律 2:跨层范数差 293 倍,不归一化的话
        下游测到的是范数而不是信息量。

        返回结构与 forward_layers 一致(cls/regs/patches/grid/prenorm_stats),
        key 用原始层号,方便复用 pool() 和缓存机制。
        """
        if len(layers) < 2:
            raise ValueError(f"至少要两层才能做差分,给了 {layers}")
        if pixel_values.shape[-1] % self.patch_size or pixel_values.shape[-2] % self.patch_size:
            raise ValueError(
                f"输入 {tuple(pixel_values.shape[-2:])} 不是 patch_size={self.patch_size} 的整数倍;"
                "请走 DINOv3Preprocessor,不要让 HF processor 偷偷 resize。"
            )
        h_grid = pixel_values.shape[-2] // self.patch_size
        w_grid = pixel_values.shape[-1] // self.patch_size

        pixel_values = pixel_values.to(self.device_, self.dtype)
        out = self.model(pixel_values=pixel_values, output_hidden_states=True)

        raw = torch.stack([out.hidden_states[i] for i in layers])          # (L,B,T,C) pre-norm
        # band 0 保持累积,其余为相邻两层之差 —— 三者拼起来仍能重构 h_c,不丢信息
        delta = torch.cat([raw[:1], raw[1:] - raw[:-1]], dim=0)            # (L,B,T,C)

        stats = None
        if return_norm_stats:
            nrm = delta[:, :, self.n_prefix :].float().norm(dim=-1)        # (L,B,N)
            stats = torch.stack([nrm.mean(-1), nrm.std(-1)], dim=-1)       # (L,B,2)

        h = self.final_norm(delta)                                         # 纪律 2

        feats: dict[int, dict[str, torch.Tensor]] = {}
        for k, i in enumerate(layers):
            entry: dict[str, torch.Tensor] = {
                "cls": h[k][:, 0],
                "regs": h[k][:, 1 : self.n_prefix],
                "patches": h[k][:, self.n_prefix :],
                "grid": (h_grid, w_grid),
            }
            if stats is not None:
                entry["prenorm_stats"] = stats[k]
            feats[i] = entry
        return feats

    @staticmethod
    def pool(entry: dict[str, torch.Tensor], mode: str = "cls+mean+std") -> torch.Tensor:
        """把一层的 token 压成定长向量,供缓存落盘。-> (B, K*C)

        默认 cls+mean+std:CLS 给全局语义,patch 均值给平均响应,patch 标准差给空间
        异质性。std 是 *跨 patch* 算的,不会被 per-token 的 LayerNorm 抹平,是浅层
        专家能拿到的高频统计量代理。
        """
        cls, patches = entry["cls"], entry["patches"]
        parts = {
            "cls": [cls],
            "cls+mean": [cls, patches.mean(1)],
            "cls+mean+std": [cls, patches.mean(1), patches.std(1)],
            "mean+std": [patches.mean(1), patches.std(1)],
        }
        if mode not in parts:
            raise ValueError(f"未知 pool mode: {mode};可选 {list(parts)}")
        return torch.cat(parts[mode], dim=-1)


if __name__ == "__main__":
    import numpy as np

    bb = DINOv3Backbone()
    print(f"model         : {bb.name}")
    print(f"device/dtype  : {bb.device_} / {bb.dtype}")
    print(f"n_layers      : {bb.n_layers}   (hidden_states 长度 = {bb.n_layers + 1})")
    print(f"hidden_size   : {bb.hidden_size}")
    print(f"patch_size    : {bb.patch_size}")
    print(f"n_registers   : {bb.n_registers}   -> prefix token 数 = {bb.n_prefix}")

    pre = bb.make_preprocessor()
    big = (np.random.rand(768, 1024, 3) * 255).astype(np.uint8)
    x = pre(big)
    print(f"\n768x1024 --504 center crop--> {tuple(x.shape)}")

    feats = bb.forward_layers(x)
    for i in (0, bb.n_layers // 2, bb.n_layers):
        e = feats[i]
        print(
            f"  layer {i:>2}: cls{tuple(e['cls'].shape)} "
            f"regs{tuple(e['regs'].shape)} patches{tuple(e['patches'].shape)} "
            f"grid{e['grid']} prenorm_mean={e['prenorm_stats'][0, 0].item():.1f}"
        )

    small = (np.random.rand(200, 173, 3) * 255).astype(np.uint8)   # 被 crop 退化砍小的图
    xs = pre(small)
    print(f"\n200x173 --on_small=native--> {tuple(xs.shape)} (裁到 patch 整数倍,不 pad)")

    pooled = bb.pool(feats[bb.n_layers])
    per_layer_kb = pooled.shape[-1] * 2 / 1024
    print(f"\npool(cls+mean+std) -> {tuple(pooled.shape)}")
    print(
        f"缓存估算(fp16): 池化后 {per_layer_kb:.1f} KB/层/图, "
        f"全 {bb.n_layers + 1} 层 {per_layer_kb * (bb.n_layers + 1) / 1024:.2f} MB/图, "
        f"选定 3 层 {per_layer_kb * 3 / 1024:.2f} MB/图"
    )
