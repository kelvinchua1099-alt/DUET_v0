"""退化生成 —— 给数据集里每张图旁边造一张干扰图,并产出 manifest。

    python utils/preprocess.py --data data/raw --out-csv data/manifest.csv
    python utils/preprocess.py --data data/raw --out-csv data/manifest.csv --dry-run
    python utils/preprocess.py --data data/raw --out-csv data/manifest.csv --no-clean-rows

每张图随机抽 **1..--max-deg 种退化**(不重复维度),每种再随机抽一个强度,施加后存在
原图旁边。`--max-deg 1`(默认)= 历史行为,码字里至多一维非零。

调大 --max-deg 的代价,动手前先算清楚:

  * probe_layers.py 的「单退化隔离」分桶不再干净 —— 一张 jpeg+blur 图会同时落进
    jpeg 桶和 blur 桶,桶间不再互斥,逐桶最优层的解读要跟着改口径;
  * 码本规模从 1 + 14 = 15 个码字涨到 1 + 14 + 78 = 93(--max-deg 2)。A3-o 的
    逐码字最优权重是按码字分格估的,格子多了每格样本就薄 —— 10,000 张图摊到 93 格
    平均 100 出头,搜出来的「oracle 上界」有多少是噪声,必须自己心里有数。
    这正是当初定「每图一种」的理由,不是随手选的默认值。

反过来的好处:码字不再是「一维非零、五维全零」的极稀疏标签,退化估计器不容易塌到
全零退化解上(全零解在 --max-deg 1 时就能拿到 per-dim 0.917 / 全对 0.500)。

退化谱与真实世界的对应(见 DEGRADATIONS):
    jpeg   q=90/70/50/30      社交媒体转存、聊天软件重编码
    blur   σ=0.5/1.0/2.0      失焦
    resize 0.5x/0.25x 再放回   缩略图生成
    noise  σ=0.02/0.05/0.10   弱光传感器噪声
    jitter 亮度/对比度/饱和度 ±20%   滤镜、自动增强
    crop   保留 80% 面积        头像裁剪、构图

三条纪律:

1. **一律存 PNG**  即使源图是 JPEG。若把加噪后的图存成 JPEG,就等于额外施加了一次
   **没有记录在码字里的**压缩 —— 标签与像素从此对不上,而且不会报错。jpeg 退化本身
   也是在内存里编解码一遍再存 PNG,保留块效应像素但不引入二次损失。

2. **确定性**  随机数种子由 (--seed, 图片相对路径) 派生。重跑得到完全相同的分配,
   删掉重来也一样。数据集的退化分配是实验的一部分,不该每次跑都变。

3. **跳过自己的产物**  输出写在原图旁边,重跑时必须认出上次生成的干扰图并跳过,
   否则会在干扰图上再叠一层退化,码字却只记录新的那层。

4. **归一化的顺序**  输出统一为 512x512,但归一化必须在退化**之前**做,不能之后:
   先退化再缩放会把 JPEG 的 8x8 块网格重采样掉(blockiness_stats 与整个 jpeg 维
   随之失效)、把噪声平均掉、把模糊的有效 σ 改掉。正确顺序是

       ① normalize -> 512x512    ② 施加退化    ③ 尺寸变了再 normalize 回来

   第 ③ 步只有 crop 会触发(其余算子都保尺寸)。

   **叠加时这一步会咬人。** 若按任意顺序施加再统一缩放回 512,`crop + jpeg` 会变成
   裁剪 -> 压缩(留下 8x8 块网格)-> 缩放(把块网格重采样掉):码字记着 jpeg=3,
   像素上的证据却没了,blockiness_stats 与整个 jpeg 维静默失效。

   所以施加顺序不是随机的,而是固定的 PIPELINE_ORDER,且 crop 一做完就立刻归一化
   回 512,之后的算子全部作用在 512x512 上,末尾不再有缩放。jpeg 永远排最后 ——
   它是「传输编码」,块网格必须活到存盘那一刻。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cache_features import DEG_CODE_VALUES, DEG_DIMS, N_LEVELS, TAXONOMY_NAME  # noqa: E402

# 强度表 + 「越大越坏」方向的 severity 函数。
# 码字**不是**按列表下标递增,而是把 severity 线性归一化后映射到 1..4:
#     code = round(1 + (N_LEVELS-2) * (sev - sev.min()) / (sev.max() - sev.min()))
# 这样「档位 4」在任何一维都表示「该维最重」,跨维语义一致 —— weights_mlp 的序数
# 输入 level/(N_LEVELS-1) 才有可比的含义。副作用是档数少的维度留有空洞
# (blur 只有 0/1/2/4),这没问题,DEG_CODE_VALUES 记录了每维的合法取值。
_OURS: dict[str, list[float]] = {
    "jpeg": [90, 70, 50, 30],        # JPEG 质量因子
    "blur": [0.5, 1.0, 2.0],         # 高斯核 σ(像素)
    "resize": [0.5, 0.25],           # 下采样倍率,之后放回原尺寸
    "noise": [0.02, 0.05, 0.10],     # 高斯噪声 σ(0-1 尺度)
    "jitter": [0.20],                # 亮度/对比度/饱和度扰动幅度
    "crop": [0.80],                  # 中心裁剪保留的**面积**比例(边长比 = sqrt)
}

# 我们的档位全保留,再掺入官方 distortion_range 里**隔一档取一档**的三档。
# 算子仍是我们的;三处语义差异与处理方式写在 deg_taxonomy.SYNTH6X 的注释里。
# 每维按「越靠后越坏」排好序,码字 = 下标 + 1。
_MIXED: dict[str, list[float]] = {
    "jpeg":   [90, 70, 50, 43, 30, 24, 4],        # 官方掺入 43 / 24 / 4
    "blur":   [0.1, 0.5, 1.0, 2.0, 5.0],          # 官方掺入 0.1 / 1 / 5(1.0 与我们重合)
    "resize": [0.8, 0.55, 0.5, 0.3, 0.25],        # 官方掺入 0.8 / 0.55 / 0.3
    "noise":  [0.02, 0.032, 0.05, 0.10],          # 官方 var 0.001/0.003/0.01 -> σ 0.032/0.055/0.100
    "jitter": [0.05, 0.20, 1.0],                  # 只借官方数值,算子仍是我们的调色扰动
    "crop":   [0.80, 0.60, 0.40],                 # 官方掺入 0.6 / 0.4(0.8 与我们重合)
}

DEGRADATIONS: dict[str, list[float]] = _MIXED if TAXONOMY_NAME == "synthetic6x" else _OURS

# severity:把参数转成「越大越坏」。质量因子和缩放倍率是反的,要翻过来。
SEVERITY = {
    "jpeg": lambda v: 100.0 - v,     # 质量越低越坏
    "blur": lambda v: v,
    "resize": lambda v: 1.0 / v,     # 倍率越小越坏
    "noise": lambda v: v,
    "jitter": lambda v: v,
    "crop": lambda v: 1.0 - v,       # 保留的面积越少越坏
}

# 唯一一处主观约定:jitter 和 crop 各只有一个强度,没有「彼此之间的大小关系」可循,
# 无法从 severity 推出它该落在 1..4 的哪一格。这里取 1(该维最轻),理由是 ±20% 调色
# 与 20% 裁剪在绝对意义上确实属于轻度退化。若日后为它们补上更多强度,这行自动失效。
SINGLE_LEVEL_CODE = 1


def derive_code_values() -> dict[str, list[int]]:
    """由 DEGRADATIONS + SEVERITY 推出每维的合法码字,并与 cache_features 的表核对。"""
    out = {}
    if TAXONOMY_NAME == "synthetic6x":
        # 表已按严重度排好序,码字 = 下标+1。不用 severity 归一化 —— 表变长后
        # 会有两个参数落到同一个码字(实测 jpeg 的 50 与 43 都映射到 4),
        # code_to_param 取 index 会静默返回错的那个参数。
        return {d: [0] + list(range(1, len(v) + 1)) for d, v in DEGRADATIONS.items()}
    for d, vals in DEGRADATIONS.items():
        sev = np.array([SEVERITY[d](float(v)) for v in vals], dtype=np.float64)
        if len(sev) == 1:
            codes = [SINGLE_LEVEL_CODE]
        else:
            norm = (sev - sev.min()) / (sev.max() - sev.min())
            codes = [int(round(1 + (N_LEVELS - 2) * x)) for x in norm]
        out[d] = [0] + codes
    return out


def code_to_param(dim: str, code: int) -> float:
    """码字取值 -> 该维的实际参数。"""
    codes = derive_code_values()[dim][1:]            # 去掉 0
    return DEGRADATIONS[dim][codes.index(code)]

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_HINTS = {  # 目录名 -> label(0=真, 1=假)
    "real": 0, "0": 0, "authentic": 0, "natural": 0, "nature": 0, "pristine": 0,
    "fake": 1, "1": 1, "ai": 1, "generated": 1, "synthetic": 1, "gen": 1,
}


def rng_for(seed: int, key: str) -> np.random.Generator:
    """由 (全局种子, 相对路径) 派生,保证重跑一致(纪律 2)。"""
    h = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


# --------------------------------------------------------------------------- 退化算子

def apply_jpeg(img: Image.Image, q: float, _r) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(q))
    buf.seek(0)
    return Image.open(buf).convert("RGB")          # 解码回像素,之后统一存 PNG(纪律 1)


def apply_blur(img: Image.Image, sigma: float, _r) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def apply_resize(img: Image.Image, scale: float, _r) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)     # 放回原尺寸,留下重采样痕迹


def apply_noise(img: Image.Image, sigma: float, r) -> Image.Image:
    a = np.asarray(img, dtype=np.float32)
    a = a + r.normal(0.0, float(sigma) * 255.0, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def apply_jitter(img: Image.Image, amt: float, r) -> Image.Image:
    for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enh(img).enhance(1.0 + float(r.uniform(-amt, amt)))
    return img


def apply_crop(img: Image.Image, keep_area: float, _r) -> Image.Image:
    """keep_area 是保留的**面积**比例,不是边长比例。

    退化谱里写的「crop 80%」指丢掉 20% 的画面,所以边长比是 sqrt(0.8)=0.894,
    512 -> 458。若按边长 0.8 裁,面积只剩 64%,退化强度会比规格重得多。
    参数表里保持 0.80 这个数,与规格文档一致;换算在这里做。
    """
    scale = keep_area ** 0.5
    w, h = img.size
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    return img.crop(((w - nw) // 2, (h - nh) // 2, (w - nw) // 2 + nw, (h - nh) // 2 + nh))


CODE_VALUES = derive_code_values()

# 重采样内核池。随机挑一个而不是固定用某一个 —— 每种内核都有自己的频域指纹
# (LANCZOS 的振铃、BOX 的块状、BICUBIC 的过冲),固定一种就等于给数据盖了统一的水印,
# 模型会去学它,换个来源的数据就失效。随机化把这条捷径堵死,代价是每张图的插值痕迹
# 不再一致 —— 但那正是真实世界的样子。
# 与 OpenCV 的对应:INTER_LINEAR/CUBIC/AREA/LANCZOS4 -> BILINEAR/BICUBIC/BOX/LANCZOS
INTERP_POOL = [Image.BILINEAR, Image.BICUBIC, Image.BOX, Image.LANCZOS]
INTERP_NAMES = {Image.BILINEAR: "bilinear", Image.BICUBIC: "bicubic",
                Image.BOX: "box(area)", Image.LANCZOS: "lanczos"}


def pick_interp(seed: int, rel: str):
    """按 (seed, 相对路径) 确定性地挑一个内核。

    **用独立的 RNG 流**(key 后缀 |interp),不动主 RNG —— 否则多抽一个随机数会让
    退化分配整体错位,历史 manifest 再也复现不出来(纪律 2)。
    """
    r = rng_for(seed, rel + "|interp")
    return INTERP_POOL[int(r.integers(len(INTERP_POOL)))]


def area_crop(img: Image.Image, seed: int, rel: str, lo: float, hi: float,
              prob: float) -> tuple[Image.Image, float | None]:
    """以 prob 的概率随机裁一块占原图面积 lo~hi 的区域(位置也随机)。

    **动机**:SID / WildFake 里大量图是原生 512 或 1024 的整数尺寸,而"原生分辨率"
    本身与来源(进而与标签)相关。裁一块再缩到 512,等于强行打散"这张图被缩放了多少倍"
    这条线索,同时给模型看不同的视野范围。

    **必须四个来源同比例施加**。若只对 SID/WildFake 做,没被处理的图就 100% 是
    Complementary 的真图,"有没有重采样痕迹"直接等于标签 —— 比原来的捷径更糟。

    裁完之后由 normalize 缩到 size:原生 512 的图会被**上采样**(裁出 280~486 再放大),
    原生 1024 的图是下采样(裁出 561~971 再缩小)。方向随原生尺寸变,这是有意的 ——
    目的就是让缩放倍率不再是一个可读出的常量。

    **用独立 RNG 流**(key 后缀 |areacrop),不动主 RNG,也不动 |interp 那条流 ——
    否则退化分配会整体错位,历史 manifest 复现不出来(纪律 2)。
    """
    if prob <= 0:
        return img, None
    r = rng_for(seed, rel + "|areacrop")
    if float(r.random()) >= prob:
        return img, None
    f = float(r.uniform(lo, hi))
    w, h = img.size
    sc = f ** 0.5                                   # 面积比 -> 边长比
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    l = int(r.integers(0, w - nw + 1))
    t = int(r.integers(0, h - nh + 1))
    return img.crop((l, t, l + nw, t + nh)), f


def normalize(img: Image.Image, size: int, resample=None, fit: str = "resize") -> Image.Image:
    """中心裁成正方 + 缩放到 size x size。已经是目标尺寸时原样返回,不做无谓重采样。

    先裁成正方再缩放,而不是直接 resize 到 size x size —— 后者会拉伸长宽比,
    把「非正方」这个与退化无关的属性变成一个可被模型利用的伪线索。

    **重采样内核按方向分开**(顺序:先 crop,再按下面的规则缩放):

        下采样(源 >= size)  LANCZOS   抗混叠最好
        上采样(源 <  size)  BILINEAR  见下

    上采样为什么不用 LANCZOS:它的负瓣会在边缘制造振铃,而振铃强度随放大倍率变化。
    若两类的原生分辨率分布不同(CIFAKE 那种 32x32 的极端情形放大 16 倍),
    "振铃有多强"就成了一条与内容无关、却和标签相关的伪线索 —— 模型会去学它。
    BILINEAR 无负瓣,放大时只是平滑,不引入这种依赖倍率的高频结构。
    代价是更糊,但糊是对两类一视同仁的。

    resize **退化**本身用的是 BICUBIC,那是在模仿缩略图生成的真实管线,
    与这里的归一化目的不同,三者不要合并。

    resample 显式给定时覆盖上面的规则(--interp random 会从 INTERP_POOL 里按图确定性
    地挑一个)。同一张图的两次 normalize 调用必须传同一个内核,否则一张图身上会留下
    两种插值痕迹。

    fit 决定大图怎么处理:

        resize  裁正方后缩到 size(历史行为)。**每张图的缩放倍率不同**,
                而倍率会留下痕迹(抗混叠强度、整数倍 vs 分数倍的相位混叠)。
                若两类的原生分辨率分布不同,"被缩了多少倍"就编码了标签 ——
                实测某些数据上假图 100% 是 1024x1024 或 512x512、真图尺寸散落,
                缩放路径下三类的倍率恰好是三段离散取值,几乎等于标签本身。
        crop    短边 >= size 时**直接中心裁 size x size,一次重采样都不做**;
                短边 < size 时才裁正方 + 上采样。重采样痕迹这条捷径随之消失,
                而且保留了原生高频(生成器指纹住在那里)。
                代价:裁剪覆盖的画面比例随原生分辨率变化,"视野范围"仍与类别弱相关,
                但那是语义线索,远弱于重采样痕迹。
    """
    if img.size == (size, size):
        return img
    w, h = img.size
    if fit == "crop" and min(w, h) >= size:
        l, t = (w - size) // 2, (h - size) // 2
        return img.crop((l, t, l + size, t + size))          # 不重采样
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m))
    kernel = resample if resample is not None else (Image.LANCZOS if m >= size else Image.BILINEAR)
    return img.resize((size, size), kernel)


OPS = {"jpeg": apply_jpeg, "blur": apply_blur, "resize": apply_resize,
       "noise": apply_noise, "jitter": apply_jitter, "crop": apply_crop}


# 固定的施加顺序,模仿真实传输管线:几何 -> 调色 -> 光学/传感器 -> 压缩。
# 两条硬约束(不要凭「看起来更自然」改动):
#   * crop 必须最先 —— 它是唯一改尺寸的算子,做完立刻归一化回 512,
#     后续算子才不会被末尾那次缩放抹掉证据(见 docstring 纪律 4)。
#   * jpeg 必须最后 —— 块网格要活到存盘,任何排在它之后的算子都会破坏它。
PIPELINE_ORDER = ["crop", "resize", "jitter", "blur", "noise", "jpeg"]
assert set(PIPELINE_ORDER) == set(DEGRADATIONS), "PIPELINE_ORDER 与 DEGRADATIONS 不同步"

SIZE_CHANGING = {"crop"}          # 施加后尺寸会变,需要就地归一化回来


def degrade(img: Image.Image, dim: str, code: int, r) -> Image.Image:
    """code 是**码字取值**(不是列表下标);0 表示不施加。"""
    if code == 0:
        return img
    return OPS[dim](img, code_to_param(dim, code), r)


def degrade_many(img: Image.Image, code: list[int], r, size: int,
                 resample=None, fit: str = "resize") -> Image.Image:
    """按 PIPELINE_ORDER 依次施加 code 里所有非零维。

    code 的下标对齐 DEG_DIMS。改尺寸的算子做完就地归一化,保证返回值恒为 size x size,
    调用方末尾不需要(也不应该)再缩放一次。
    """
    idx = {d: i for i, d in enumerate(DEG_DIMS)}
    for dim in PIPELINE_ORDER:
        level = code[idx[dim]]
        if not level:
            continue
        img = degrade(img, dim, level, r)
        if dim in SIZE_CHANGING:
            # ③ 必须与 ① 用同一个内核。注意这里 crop 后的图已经是 size 的子集,
            #    走 fit=crop 分支会直接再裁一次;传 fit 保证两次行为一致
            img = normalize(img, size, resample, fit)
    return img


# --------------------------------------------------------------------------- 标签

def infer_label(path: Path, root: Path, default: int | None) -> int:
    for part in reversed(path.relative_to(root).parts[:-1]):
        hit = LABEL_HINTS.get(part.strip().lower())
        if hit is not None:
            return hit
    if default is None:
        raise SystemExit(
            f"无法从路径推断真假标签: {path}\n"
            f"目录名里需要出现 {sorted(LABEL_HINTS)} 之一,或用 --label 0/1 统一指定。")
    return default


# --------------------------------------------------------------------------- 主流程

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="数据集根目录,递归查找图片")
    ap.add_argument("--out-csv", required=True, help="产出的 manifest")
    ap.add_argument("--suffix", default="__deg", help="干扰图的文件名后缀")
    ap.add_argument("--clean-suffix", default="__clean",
                    help="源图尺寸不合规时,归一化副本的后缀")
    ap.add_argument("--size", type=int, default=512, help="统一输出边长")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-deg", type=int, default=1, metavar="K",
                    help="每张退化图最多同时施加 K 种退化(维度不重复),实际种数在 1..K "
                         "均匀抽取。默认 1 = 历史行为。调大前先读 docstring 里的代价说明")
    ap.add_argument("--also-skip", default="", metavar="S1,S2",
                    help="额外的文件名片段,含之一的文件不当作源图。往同一目录里写第二套"
                         "退化图时必须用它排掉第一套,否则会在退化图上再叠一层(纪律 3)")
    ap.add_argument("--label", type=int, default=None, choices=[0, 1],
                    help="统一指定真假标签;不给则从目录名推断")
    ap.add_argument("--no-clean-rows", action="store_true",
                    help="manifest 里不包含原图(码字全 0)。默认包含 —— 干净档是必要的对照")
    ap.add_argument("--split-val", type=float, default=0.2,
                    help="按**源图**分组划出验证集,写进 split 列。默认 0.2 而非 0:"
                         "不写 split 列的话,下游会退回按 image_id 哈希,"
                         "同一源图的干净版与退化版会被拆到两边,造成泄漏")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的干扰图")
    ap.add_argument("--area-prob", type=float, default=0.0,
                    help="以此概率对一张图先随机裁一块再缩到 size(0=关闭)。"
                         "务必对所有来源用同一个值,否则'有没有重采样痕迹'会编码标签")
    ap.add_argument("--area-lo", type=float, default=0.30, help="裁出区域占原面积的下界")
    ap.add_argument("--area-hi", type=float, default=0.90, help="上界")
    ap.add_argument("--fit", default="resize", choices=["resize", "crop"],
                    help="大图(短边>=size)怎么处理。resize=裁正方后缩放(历史行为);"
                         "crop=直接中心裁,不重采样 —— 消除'缩放倍率'这条捷径,"
                         "并保留原生高频。小图两种模式都是上采样")
    ap.add_argument("--interp", default="fixed", choices=["fixed", "random"],
                    help="归一化的重采样内核。fixed = 下采样 LANCZOS / 上采样 BILINEAR"
                         "(历史行为);random = 每图从 INTERP_POOL 里确定性地挑一个,"
                         "防止模型学到单一内核的指纹")
    ap.add_argument("--workers", type=int, default=8,
                    help="并行写图的线程数。0 = 串行(历史行为)。码字分配始终串行,"
                         "每图的 RNG 由 (seed, 相对路径) 独立派生,所以并行不改变结果")
    args = ap.parse_args(argv)

    if CODE_VALUES != DEG_CODE_VALUES:
        raise SystemExit(
            "退化谱与 cache_features.DEG_CODE_VALUES 不一致 —— 两处必须同步。\n"
            f"  由 DEGRADATIONS+SEVERITY 推出: {CODE_VALUES}\n"
            f"  cache_features 里写的      : {DEG_CODE_VALUES}")

    root = Path(args.data).resolve()
    if not root.is_dir():
        raise SystemExit(f"{root} 不是目录")

    # 纪律 3:认出**所有**已生成的干扰图并跳过,否则会叠加退化而码字只记新的那层。
    # 注意子串匹配的陷阱:suffix="__deg2" 时,"X__deg" 里并不含 "__deg2",
    # 上一套的产物会溜进源图集 —— 所以第二套必须显式 --also-skip __deg
    skip_frags = [f for f in ([args.suffix, args.clean_suffix]
                              + [x.strip() for x in args.also_skip.split(",")]) if f]
    srcs = sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in IMG_EXT
                  and not any(f in p.stem for f in skip_frags))
    if args.limit:
        srcs = srcs[: args.limit]
    if not srcs:
        raise SystemExit(f"{root} 下没有找到图片(已排除后缀含 {args.suffix!r} 的文件)")

    # 只为算下面那一个 n_ok 统计数。串行跑的话,10 万张图在网络盘上要十几分钟
    # (每次 Image.open 是一个往返),而这期间一张图都不会被写出来 —— 看着像卡死。
    # Image.open 只读文件头,是纯 IO,线程池能直接吃满。
    def _probe(q):
        try:
            with Image.open(q) as im:
                return str(q), im.size
        except Exception:
            return str(q), None

    fmts = Counter(q.suffix.lower() for q in srcs)
    if len(srcs) > 2000:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(8, args.workers)) as _ex:
            src_sizes = dict(_ex.map(_probe, srcs))
    else:
        src_sizes = dict(_probe(q) for q in srcs)

    print(f"源目录    : {root}")
    print(f"源图      : {len(srcs)} 张   格式 " + dict(fmts).__repr__())
    n_ok = sum(1 for v in src_sizes.values() if v == (args.size, args.size))
    print(f"归一化    : 中心裁正方 + LANCZOS -> {args.size}x{args.size};"
          f" 已合规 {n_ok} 张,需处理 {len(srcs) - n_ok} 张")
    lossy = sum(v for k, v in fmts.items() if k in (".jpg", ".jpeg", ".webp"))
    if lossy:
        print(f"[警告] 有 {lossy} 张源图本身是有损格式。它们自带一层**未记录在码字里**的"
              f"压缩,所谓「干净档」并不真的干净。这是数据来源的性质,脚本无法修复,"
              f"但论文里报 clean 基线时必须说明。")
    print("退化谱(参数 -> 码字,按 severity 线性映射到 1..%d):" % (N_LEVELS - 1))
    for d in DEGRADATIONS:
        pairs = "  ".join(f"{v}→{c}" for v, c in zip(DEGRADATIONS[d], CODE_VALUES[d][1:]))
        print(f"  {d:<8} {pairs}")
    if args.max_deg < 1:
        raise SystemExit("--max-deg 至少为 1")
    if args.max_deg > len(DEGRADATIONS):
        raise SystemExit(f"--max-deg 最多 {len(DEGRADATIONS)}(维度不重复)")
    print(f"策略      : 每图随机 1..{args.max_deg} 种退化(维度不重复) x 各自 1 个强度,"
          f"{'不含' if args.no_clean_rows else '含'}干净行")
    if args.max_deg > 1:
        print(f"施加顺序  : {' -> '.join(PIPELINE_ORDER)}  "
              f"(crop 做完就地归一化回 {args.size};jpeg 恒最后,块网格活到存盘)")
    print(f"排除片段  : {skip_frags}")
    print(f"模式      : {'DRY RUN(不写任何文件)' if args.dry_run else '写入'}\n")

    dims = list(DEGRADATIONS)
    rows, stats, skipped, failed = [], Counter(), 0, 0
    n_applied = Counter()
    jobs: list = []

    for p in srcs:
        rel = p.relative_to(root).as_posix()
        try:
            label = infer_label(p, root, args.label)
        except SystemExit:
            raise
        r = rng_for(args.seed, rel)
        # 先定种数再定维度:维度用不放回抽样,同一维不会被抽两次(叠两次 jpeg
        # 只会留下更重的那次的痕迹,码字却记两个值 —— 无法表示)
        k = 1 if args.max_deg == 1 else int(r.integers(1, args.max_deg + 1))
        picked = [dims[i] for i in r.choice(len(dims), size=k, replace=False)]
        code = [0] * len(DEG_DIMS)
        for dim in picked:
            legal = CODE_VALUES[dim][1:]                   # 该维的非零合法码字
            level = int(legal[int(r.integers(len(legal)))])
            code[DEG_DIMS.index(dim)] = level
        out_path = p.with_name(f"{p.stem}{args.suffix}.png")
        # clean 行**永远**写一份归一化 PNG,即使源图已经是 size x size。
        #
        # 曾经这里有个"已经合规就直接指向原图"的优化,它引入了一个静默的标签泄漏:
        # 生成器出图恰好 512x512 的比例远高于真实照片,于是"clean 行是 .jpg"这件事
        # 与 label 高度相关(实测 2060 假 : 44 真)。模型可以直接学"码字说干净、
        # 像素里却有 JPEG 块效应 = 假图",而这条捷径在任何指标上都看不出来。
        # 多存几千个文件,换掉一个查不出来的伪线索,划算。
        clean_path = p.with_name(f"{p.stem}{args.clean_suffix}.png")

        # 分组键取**源图**身份,而不是每行的 image_id。同一张源图的干净版与退化版
        # 必须落在同一侧 —— 否则模型能从训练集的干净版记住图像内容,再在验证集里
        # 认出它的退化版,验证指标虚高且事后极难察觉。这是本脚本最容易踩的坑。
        group = rel.rsplit(".", 1)[0]
        gh = int(hashlib.sha256(f"split|{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        split = "val" if gh < args.split_val else "train"

        def row(path: Path, c: list[int], iid: str) -> dict:
            d = {"path": str(path), "image_id": iid, "group": group, "label": label}
            d.update({f"d{i}": c[i] for i in range(len(DEG_DIMS))})
            if args.split_val > 0:
                d["split"] = split
            return d

        if not args.no_clean_rows:
            rows.append(row(clean_path, [0] * len(DEG_DIMS), clean_path.stem))
        rows.append(row(out_path, code, out_path.stem))
        for dim in picked:
            stats[f"{dim}={code[DEG_DIMS.index(dim)]}"] += 1
        n_applied[len(picked)] += 1

        if args.dry_run:
            continue
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        # 攒起来最后并行执行。码字、行、统计都已在上面串行定好,
        # 这里只剩纯 I/O + 像素运算,是唯一值得并行的部分。
        rs = pick_interp(args.seed, rel) if args.interp == "random" else None
        jobs.append((p, clean_path, out_path, list(code), r, rs, args.fit, rel))

    # ---- 并行写图 ----
    # 串行版里读盘、算、写盘首尾相接,单线程在网络盘上跑不满。
    # 每图的 r 由 rng_for(seed, rel) 独立派生,互不干扰,所以并行**不改变**任何一张图的
    # 像素(纪律 2)。改完请用 --workers 0 跑一遍小样本逐位比对。
    def write_one(job):
        p, clean_path, out_path, code, r, rs, fit, rel = job
        try:
            with Image.open(p) as im:
                im2, frac = area_crop(im.convert("RGB"), args.seed, rel,
                                      args.area_lo, args.area_hi, args.area_prob)
                # 被面积裁剪过的图必须走 resize 分支缩到 size。留在 fit=crop 下的话,
                # 裁出来的 793x793 会被直接中心裁 512、一次不缩放 —— 那就等于没做。
                fit_i = "resize" if frac is not None else fit
                base = normalize(im2, args.size, rs, fit_i)               # ① 先归一化
                # 幂等:归一化副本是确定性的,已存在就别重写 —— 往同一目录写第二套
                # 退化图时,重写会去动别的进程正在读的文件
                if args.overwrite or not clean_path.exists():
                    base.save(clean_path, "PNG")
                # ② 再退化。degrade_many 内部按 PIPELINE_ORDER 施加,
                #    并在 crop 之后就地归一化,返回值恒为 size x size(③ 已内含)
                out = degrade_many(base, code, r, args.size, rs, fit_i)
                out.save(out_path, "PNG")                        # 纪律 1
        except Exception as e:
            return str(out_path), f"[跳过] {p}: {type(e).__name__}: {e}"
        return None

    if jobs:
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **kw):
                return x
        dead = set()
        if args.workers > 0:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                it = ex.map(write_one, jobs)
                for res in tqdm(it, total=len(jobs), desc="写退化图", unit="img"):
                    if res:
                        dead.add(res[0]); print("\n" + res[1]); failed += 1
        else:
            for job in tqdm(jobs, desc="写退化图", unit="img"):
                res = write_one(job)
                if res:
                    dead.add(res[0]); print("\n" + res[1]); failed += 1
        if dead:
            rows = [x for x in rows if x["path"] not in dead]

    # ---- 报告 ----
    print(f"退化分配(每图 1..{args.max_deg} 种,下表按维度计数,叠加图会被计入多个维度):")
    for d in dims:
        line = f"  {d:<8}"
        for v, c in zip(DEGRADATIONS[d], CODE_VALUES[d][1:]):
            line += f"  码{c}({v})={stats[f'{d}={c}']:<5}"
        print(line)
    codes_seen = Counter(tuple(x[f"d{i}"] for i in range(len(DEG_DIMS))) for x in rows)
    print(f"\n每图施加种数分布: {dict(sorted(n_applied.items()))}")
    print(f"实际出现的码字种类: {len(codes_seen)}")
    thin = sum(1 for c in codes_seen.values() if c < 30)
    print(f"  其中样本数 < 30 的码字: {thin}  "
          f"(码字格越薄,A3-o 的逐码字 oracle 上界越接近噪声)")
    multi = sum(1 for x in rows if sum(x[f"d{i}"] > 0 for i in range(len(DEG_DIMS))) > 1)
    print(f"多重退化样本: {multi}"
          + ("  (应为 0 —— 每图只施加一种)" if args.max_deg == 1 else ""))
    lbl = Counter(x["label"] for x in rows)
    print(f"标签分布: 真={lbl[0]} 假={lbl[1]}")
    if args.split_val > 0:
        sp = Counter(x["split"] for x in rows)
        # 一次扫过building一个 group -> split 集合的映射。
        # 原来写成 "对每个 group 再遍历一遍 rows",是 O(组数 x 行数):
        # 1 万组时约 1e8 次还能跑完,10 万组就是 2e10,实测卡死 40 分钟没出结果。
        by_group = defaultdict(set)
        for x in rows:
            by_group[x["group"]].add(x["split"])
        bad = {g for g, v in by_group.items() if len(v) > 1}
        print(f"划分: train={sp['train']} val={sp['val']}  "
              f"跨 split 的源图组: {len(bad)} (必须为 0,否则干净版/退化版泄漏)")
    if skipped:
        print(f"已存在而跳过: {skipped} 张(加 --overwrite 强制重做)")
    if failed:
        print(f"读取失败: {failed} 张")

    if args.dry_run:
        print("\nDRY RUN,未写任何文件。")
        return 0

    outp = Path(args.out_csv); outp.parent.mkdir(parents=True, exist_ok=True)
    cols = ["path", "image_id", "group", "label"] + [f"d{i}" for i in range(len(DEG_DIMS))] \
        + (["split"] if args.split_val > 0 else [])
    with open(outp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} 行 -> {outp}")
    print(f"列序 d0..d5 = {DEG_DIMS}")
    print(f"下一步: python cache_features.py --manifest {outp} --out cache/probe --layers all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
