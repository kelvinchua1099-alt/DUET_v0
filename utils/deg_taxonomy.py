"""退化码本的**唯一定义处** —— 码字有几维、每维几档、哪些维属于哪个"退化族"。

原本这三样常量硬写在 `cache_features.py` 里。抽出来是因为本项目现在要跑两套数据:

* `synthetic` —— `utils/preprocess.py` 自造的 6 维退化谱(jpeg/blur/resize/noise/jitter/crop)。
  历史默认值,改动它会让已有的缓存与 manifest 全部对不上,所以保持逐字不变。
* `ntire`     —— NTIRE 2026 Robust AIGI Detection 官方退化管线的分组。
  由 `utils/ntire_aug/`(官方 aug_utils_train 原样 vendored)推导,不是我们自己编的。

用环境变量切换,默认仍是 `synthetic`:

    export SQUADE_TAXONOMY=ntire

**为什么是环境变量而不是命令行参数**:`DEG_DIMS` / `N_LEVELS` 是 import 期常量,
被 `models/weights_mlp.py` 与 `models/classifier.py` 当作默认参数值(`n_dims=len(DEG_DIMS)`)
在类定义时求值。走参数就得把六个文件的构造函数全改成延迟求值,得不偿失。
代价是同一个 shell 里不能同时跑两套码本 —— 缓存目录的 `config.json` 会记录
`deg_dims`,下游读到不一致会直接拒绝,不会静默串味。

---------------------------------------------------------------------------
NTIRE 码本的推导(来自官方 `aug_utils_train/utils_data.py`)

官方把 12 个畸变函数编成 **7 个组**,`get_distortions_composition` 先从组里无放回抽
1~3 个组,每组再抽一个变体、抽一个强度档。所以「组」才是官方自己的一级分类,
把码字的维度定在组上(而不是 12 个函数上)有三个好处:

  1. 与官方语义一致 —— 同组两个变体(gausblur / lensblur)本就是可互换的实现;
  2. 维度 7 而非 12,码本小一半,A3-o 的逐码字最优权重每格样本才够;
  3. 正好对上 README 「退化**族**离散编码」那句主张 —— 族是官方给的,不是我们凑的。

档位 0 保留给「该组未施加」,1..5 直接是官方 `distortion_range` 的下标 +1,
因此 `N_LEVELS = 6`。官方的 range 列表本身按严重度递增排列(见下),序数语义成立。

    blur        gausblur   σ      0.1  0.5  1    2    5
                lensblur   radius 1    2    4    6    8
    color       colorshift amount 1    3    6    8    12
                colorsat   factor 0.4  0.2  0.1  0    -0.4
    jpeg        jpeg       q      43   36   24   7    4
    noise       whitenoise var    .001 .002 .003 .005 .01
                impulsenoise d    .001 .005 .01  .02  .03
    brightness  brighten   amount 0.1  0.2  0.4  0.7  1.1
                darken     amount 0.05 0.1  0.2  0.4  0.8
    spatial     jitter     amount 0.05 0.1  0.2  0.5  1
                quantization levels 20 16   13   10   7
    contrast    lincontrchange    0.   0.15 -0.4 0.3  -0.6

两处**官方表自带的**不规则,原样保留,但报告里必须提:

  * `contrast` 档 1 的 amount 是 `0.`,即近似恒等变换 —— 该档样本实质是干净图。
  * `contrast` 的 |amount| 是 0 / .15 / .4 / .3 / .6,档 3 比档 4 更重,不严格单调。

---------------------------------------------------------------------------
第 8 维 `geometric` 的来历与**诚实边界**

NTIRE **训练集**的 12 个变换全是光度/信号域的,**一个几何变换都没有**
(Random Crop / Random Aspect Crop / Downscale / Pixelation / Perspective 只出现在
val / hard-val / test 的变换表里)。

这对 SQuaDE 是个直接问题:README 的核心二分是「smudged 偏深 vs shattered 偏浅」,
而只用官方训练变换的话 **shattered 一侧是空的**,证伪检查根本测不到那半边。

所以这里补了第 8 组 `geometric`,取 NTIRE val/test 变换表里点名的 Random Crop /
Random Aspect Crop / Downscale。**它的算子名来自 NTIRE,强度参数是我们自己定的**
—— 官方只公开了训练管线的脚本,没公开 val/test 的参数。报告里必须这样写,
不能说成「NTIRE 的几何退化档」。

不想要这一维就用 `SQUADE_TAXONOMY=ntire7`,严格只留官方训练管线的 7 组。
"""

from __future__ import annotations

import os

# --------------------------------------------------------------------------- synthetic(历史默认)

SYNTHETIC = {
    "name": "synthetic",
    "dims": ["jpeg", "blur", "resize", "noise", "jitter", "crop"],
    "n_levels": 5,
    "code_values": {
        "jpeg": [0, 1, 2, 3, 4],
        "blur": [0, 1, 2, 4],
        "resize": [0, 1, 4],
        "noise": [0, 1, 2, 4],
        "jitter": [0, 1],
        "crop": [0, 1],
    },
    # probe_layers.py 的汇总桶:README 的核心二分
    "families": {
        "smudged": ("jpeg", "blur", "noise"),
        "shattered": ("crop", "resize"),
    },
}

# --------------------------------------------------------------------------- NTIRE

# 官方 7 组(distortion_groups 的键,重命名成短名以便打表)-> 官方组名
NTIRE_GROUP_ALIAS = {
    "blur": "blur",
    "color": "color_distortion",
    "jpeg": "jpeg",
    "noise": "noise",
    "brightness": "brightness_change",
    "spatial": "spatial_distortion",
    "contrast": "sharpness_contrast",
}
NTIRE_OFFICIAL_DIMS = list(NTIRE_GROUP_ALIAS)          # 7 组,顺序即码字列序
NTIRE_GEOMETRIC_DIM = "geometric"                      # 第 8 维,见模块 docstring
NTIRE_N_LEVELS = 6                                     # 0=未施加, 1..5=官方五档

_NTIRE_FAMILIES = {
    # 抹掉浅层信号指纹 -> 应听深层。blur/jpeg/noise 无争议;
    # spatial 里 jitter 是像素级重采样、quantization 砍掉细微灰阶,同样杀高频。
    "smudged": ("blur", "jpeg", "noise", "spatial"),
    # 毁掉深层全局构图 -> 应听浅层。官方训练管线里没有,只有补上的第 8 维。
    "shattered": (NTIRE_GEOMETRIC_DIM,),
    # 单调点变换为主,高频结构基本保留 —— 既不 smudged 也不 shattered,
    # 单列一族,免得硬塞进二分里污染那两个汇总桶。
    "photometric": ("color", "brightness", "contrast"),
}


def _ntire(dims: list[str], name: str) -> dict:
    return {
        "name": name,
        "dims": dims,
        "n_levels": NTIRE_N_LEVELS,
        # 官方每组都是齐整的五档,没有 synthetic 那种空洞
        "code_values": {d: list(range(NTIRE_N_LEVELS)) for d in dims},
        "families": {k: tuple(x for x in v if x in dims) for k, v in _NTIRE_FAMILIES.items()},
    }


NTIRE = _ntire(NTIRE_OFFICIAL_DIMS + [NTIRE_GEOMETRIC_DIM], "ntire")
NTIRE7 = _ntire(list(NTIRE_OFFICIAL_DIMS), "ntire7")

# --------------------------------------------------------------------------- NTIRE val(官方退化真值)

# NTIRE 的 val / val_hard 标签文件里带 `distortions` 与 `distortion_scales` 两列 ——
# **官方逐图给出了施加了哪些畸变、各自多强**。于是码字不用我们自己造,直接抄真值。
#
# 这比 `ntire` 那套(自己按官方脚本施加退化)强在三点:
#   1. 生成器更难 —— val 用 FLUX.1 Dev/Kontext、SD3 Medium、Qwen Image、Playground v2.5、
#      Ideogram v3、ImageGen-4,而 train 是 SD1.4/1.5/2.1、PixArt、Kandinsky 2/3 那一代。
#      在 train 上冻结 DINOv3 + 线性探针干净图 AUC 已经 99.99,退化最重档只掉 0.47 ——
#      没有线索被杀死,就没有可路由的对象,探针门禁必然过不了。
#   2. **几何退化齐全** —— train 的 12 个变换全是光度/信号域的,shattered 一侧是空的;
#      val 里 74% 的退化图带 downscale,randomcrop / randomaspectcrop / pixelate 也都有。
#   3. 退化是官方真实施加的,不是我们的复刻,外部效度直接成立。
#
# 代价(必须写进论文):
#   * 退化是**复合**的,每图 1~4 种,边缘分桶有混杂。共现矩阵里 downscale 与其余各维的
#     相关性会很高(它出现在 74% 的退化图上),那些桶的结论要谨慎读。
#   * 干净图与退化图是**不同的图**,不存在「同一源图的干净版/退化版」配对 ——
#     所以按 image_name 划分即可,不需要 preprocess.py 的源图分组逻辑。
#   * 样本量 10,000(val)/ 2,500(val_hard),比自己造退化少一个量级。

# 维度 = NTIRE 自己的畸变名,不做二次分组。顺序固定(按 val 里的出现频次降序),
# 一旦定下就不能改 —— manifest 的 d0..d18 列序依赖它。
NTIREVAL_DIMS = [
    "downscale", "lincontrchange", "jpeg", "randomcrop", "randomaspectcrop",
    "darken", "brighten", "impulsenoise", "rgbshift", "colorshift",
    "gausblur", "pixelate", "whitenoise", "colorsat", "quantization",
    "lensblur", "multnoise", "motionblur", "jitter",
]
NTIREVAL_N_LEVELS = 6      # 0 = 未施加, 1..5 = 官方五档

# scale -> 档位的方向。官方 distortion_range 的列表顺序即严重度递增,这里据此定方向;
# 官方训练表里没有的 6 个畸变(downscale/randomcrop/randomaspectcrop/rgbshift/
# pixelate/multnoise/motionblur)按物理含义定。
#   "asc"  数值越大越坏
#   "desc" 数值越小越坏(质量因子、保留比例、量化级数、饱和度系数)
NTIREVAL_DIRECTION = {
    "downscale": "desc",          # 缩放倍率,越小丢的像素越多
    "jpeg": "desc",               # 质量因子
    "randomcrop": "desc",         # 保留边长比例
    "randomaspectcrop": "desc",   # 保留比例
    "quantization": "desc",       # 量化级数
    "colorsat": "desc",           # 饱和度系数 0.4 -> -0.4
    "darken": "asc", "brighten": "asc", "impulsenoise": "asc", "rgbshift": "asc",
    "colorshift": "asc", "gausblur": "asc", "pixelate": "asc", "whitenoise": "asc",
    "lensblur": "asc", "multnoise": "asc", "motionblur": "asc", "jitter": "asc",
    # lincontrchange 不单调:官方表是 [0., 0.15, -0.4, 0.3, -0.6],|amount| 是
    # 0 / .15 / .4 / .3 / .6,且正负号在「提对比」与「降对比」之间来回跳。
    # 只能照抄官方顺序,不能按数值排 —— 见下面的 NTIREVAL_EXPLICIT_ORDER。
    "lincontrchange": "explicit",
}
NTIREVAL_EXPLICIT_ORDER = {"lincontrchange": [0.0, 0.15, -0.4, 0.3, -0.6]}

# downscale 的 scale 是**连续**的(val 里 3708 个互不相同的值),其余 18 个都是 5 个离散值。
# 按等宽分箱到 5 档,与其余维保持同样的「1..5」语义。
NTIREVAL_CONTINUOUS = {"downscale": (0.3, 0.8)}

_NTIREVAL_FAMILIES = {
    # 抹掉浅层信号指纹 -> 应听深层
    "smudged": ("jpeg", "gausblur", "lensblur", "motionblur", "whitenoise",
                "impulsenoise", "multnoise", "quantization", "jitter"),
    # 毁掉深层全局构图 / 砍掉分辨率 -> 应听浅层。
    # 把 downscale / pixelate 归到这里是沿用 synthetic 码本的约定(那里 resize 就在
    # SHATTERED),保持两套码本的族定义可比。
    "shattered": ("downscale", "randomcrop", "randomaspectcrop", "pixelate"),
    # 单调点变换为主,高频结构基本保留 —— 单列一族,免得污染上面两个汇总桶
    "photometric": ("lincontrchange", "darken", "brighten", "rgbshift",
                    "colorshift", "colorsat"),
}

NTIREVAL = {
    "name": "ntireval",
    "dims": list(NTIREVAL_DIMS),
    "n_levels": NTIREVAL_N_LEVELS,
    "code_values": {d: list(range(NTIREVAL_N_LEVELS)) for d in NTIREVAL_DIMS},
    "families": dict(_NTIREVAL_FAMILIES),
}

# --------------------------------------------------------------------------- NTIRE val_hard

# val_hard 用的是 val 的 19 种之外**再加 10 种**,而且新增的那批正是这份数据"hard"的
# 由来:两种对抗嵌入、四种 AI/多轮重压缩、以及 clahe / randomtonecurve / perspective /
# isonoise。val 里一张都没有 —— 所以在 val 上拟合、到 val_hard 上打分,测的就是
# "深度偏好能不能迁移到没见过的退化类型",这是自建数据根本测不了的东西。
#
# 前 19 维**沿用 NTIREVAL_DIMS 的原顺序**,新增 10 维追加在后面。这样 val 的码字
# 直接零填充到 29 维就与 val_hard 对齐,两份 manifest 能拼进同一个缓存。
NTIREHARD_NEW_DIMS = [
    "jpeg_ai", "adv_embed_resnet", "adv_embed_clip", "randomtonecurve",
    "jpeg_recompression_1", "jpeg_recompression_2", "jpeg_recompression_comb_jpegai",
    "clahe", "perspective", "isonoise",
]
NTIREHARD_DIMS = list(NTIREVAL_DIMS) + NTIREHARD_NEW_DIMS      # 29 维

# 新增维的强度方向。实测取值全是等间隔的序号或单调递增的幅度
# (adv_embed_* = 4/6/8/10/12 的扰动预算, clahe/perspective/isonoise = 0..4 的档号,
#  randomtonecurve = 0.05..0.4 的幅度, jpeg_recompression_* = 重压轮数),一律 asc。
#
# **jpeg_ai 的 0..3 是个预设编号,官方没公开它与压缩率的对应关系** —— 这里按 asc 处理,
# 但这个方向未经证实。它只影响 --bucket-mode level;dim 分桶只看"是否 >0",不受影响。
NTIREHARD_DIRECTION = dict(NTIREVAL_DIRECTION)
NTIREHARD_DIRECTION.update({d: "asc" for d in NTIREHARD_NEW_DIMS})

_NTIREHARD_FAMILIES = {
    "smudged": _NTIREVAL_FAMILIES["smudged"] + (
        "jpeg_ai", "jpeg_recompression_1", "jpeg_recompression_2",
        "jpeg_recompression_comb_jpegai", "isonoise"),
    "shattered": _NTIREVAL_FAMILIES["shattered"] + ("perspective",),
    "photometric": _NTIREVAL_FAMILIES["photometric"] + ("clahe", "randomtonecurve"),
    # 对抗嵌入既不抹高频也不毁构图 —— 它是**针对检测器的定向扰动**,
    # 硬塞进上面三族会污染它们。单列一族,也正好当迁移实验的主角。
    "adversarial": ("adv_embed_resnet", "adv_embed_clip"),
}

NTIREHARD = {
    "name": "ntirehard",
    "dims": list(NTIREHARD_DIMS),
    "n_levels": NTIREVAL_N_LEVELS,
    "code_values": {d: list(range(NTIREVAL_N_LEVELS)) for d in NTIREHARD_DIMS},
    "families": dict(_NTIREHARD_FAMILIES),
}

# --------------------------------------------------------------------------- synthetic6:6 维 + 官方强度
#
# 与最早的 SYNTHETIC 是**同样六类退化**,但每一维换成 NTIRE 官方管线里对应的算子与强度表:
#
#     我们原来的维   -> 官方算子      强度来源
#     jpeg            jpeg            官方 distortion_range [43,36,24,7,4]
#     blur            gausblur        官方 [0.1,0.5,1,2,5]
#     noise           whitenoise      官方 [0.001,0.002,0.003,0.005,0.01]
#     jitter          jitter          官方 [0.05,0.1,0.2,0.5,1]
#     resize          downscale       官方 val 标注观测值,范围 [0.30,0.80]
#     crop            randomcrop      官方 val 标注观测值 [0.4,0.5,0.6,0.7,0.8]
#
# 前四维的**算子与参数都是官方的**(vendored 副本);后两维官方没公开 val/test 的实现,
# 算子是我们写的,参数取自 val 标注 —— 报告里必须这样区分。
#
# 维度直接用官方算子名而不是我们原来的短名(blur/resize/noise),是为了让码字与
# ntireval / ntirehard 的同名维**语义完全一致**,三套码本之间可以直接对照。
SYNTH6_DIMS = ["jpeg", "gausblur", "downscale", "whitenoise", "jitter", "randomcrop"]

SYNTH6 = {
    "name": "synthetic6",
    "dims": list(SYNTH6_DIMS),
    "n_levels": 6,                                  # 0 = 未施加, 1..5 = 官方五档
    "code_values": {d: list(range(6)) for d in SYNTH6_DIMS},
    "families": {
        # 抹掉浅层信号指纹
        "smudged": ("jpeg", "gausblur", "whitenoise"),
        # 毁掉全局构图 / 砍分辨率
        "shattered": ("downscale", "randomcrop"),
        # 单调点变换
        "photometric": ("jitter",),
    },
}

# --------------------------------------------------------------------------- synthetic6x:我们的六维 + 掺入一半官方档位
#
# **算子仍是我们自己的**(utils/preprocess.py 里那六个),只把强度表加长:
# 我们原来的档位全部保留,再从官方 distortion_range 里**隔一档取一档**并进去。
#
# 三处算子语义不同,不能无脑并参数,逐条说明:
#
#   jpeg    我们 PIL 质量因子 / 官方 也是质量因子        -> 直接并
#   blur    我们 GaussianBlur(radius=σ) / 官方 σ        -> 直接并
#   resize  我们 bicubic 缩放倍率 / 官方 downscale 倍率  -> 直接并
#   noise   我们参数是 **σ** / 官方 white_noise 参数是 **方差**
#           -> 官方值先开平方转成 σ 再并(0.001->0.032, 0.003->0.055, 0.01->0.100)
#   jitter  我们是亮度/对比度/饱和度扰动 / 官方 jitter 是**像素位移场**,完全不同的算子
#           -> **只借数值不借算子**:把官方的 amount 档位用到我们的调色扰动上。
#              这是一个选择,不是等价替换,报告里必须写清楚。
#   crop    我们**中心**裁剪 / 官方 randomcrop 是**随机位置**
#           -> 保留边长比例可比,直接并;位置仍用我们的中心裁剪。
#
# 档位不再按 severity 归一化映射,而是**按严重度排序后直接编号 1..n**。
# 原来的归一化映射在表变长后会出现两个参数撞到同一个码字(实测 jpeg 的 50 和 43 都落在 4),
# code_to_param 取 index 会返回错的那个 —— 静默错误。
SYNTH6X_DIMS = ["jpeg", "blur", "resize", "noise", "jitter", "crop"]
SYNTH6X_N_LEVELS = 8                       # 0 = 未施加, 1..7 = 各维自己的档位数

SYNTH6X = {
    "name": "synthetic6x",
    "dims": list(SYNTH6X_DIMS),
    "n_levels": SYNTH6X_N_LEVELS,
    "code_values": {
        "jpeg":   [0, 1, 2, 3, 4, 5, 6, 7],   # 90 70 50 43 30 24 4
        "blur":   [0, 1, 2, 3, 4, 5],         # 0.1 0.5 1.0 2.0 5.0
        "resize": [0, 1, 2, 3, 4, 5],         # 0.8 0.55 0.5 0.3 0.25
        "noise":  [0, 1, 2, 3, 4],            # 0.02 0.032 0.05 0.10
        "jitter": [0, 1, 2, 3],               # 0.05 0.20 1.0
        "crop":   [0, 1, 2, 3],               # 0.80 0.60 0.40
    },
    "families": {
        "smudged": ("jpeg", "blur", "noise"),
        "shattered": ("crop", "resize"),
        "photometric": ("jitter",),
    },
}

TAXONOMIES = {"synthetic": SYNTHETIC, "ntire": NTIRE, "ntire7": NTIRE7,
              "ntireval": NTIREVAL, "ntirehard": NTIREHARD, "synthetic6": SYNTH6,
              "synthetic6x": SYNTH6X}

# --------------------------------------------------------------------------- 选中的那一套

ENV_VAR = "SQUADE_TAXONOMY"
_choice = os.environ.get(ENV_VAR, "synthetic").strip().lower()
if _choice not in TAXONOMIES:
    raise SystemExit(
        f"{ENV_VAR}={_choice!r} 不认识。可选: {sorted(TAXONOMIES)}"
    )

ACTIVE = TAXONOMIES[_choice]
DEG_DIMS: list[str] = list(ACTIVE["dims"])
N_LEVELS: int = ACTIVE["n_levels"]
DEG_CODE_VALUES: dict[str, list[int]] = {k: list(v) for k, v in ACTIVE["code_values"].items()}
FAMILIES: dict[str, tuple[str, ...]] = dict(ACTIVE["families"])
TAXONOMY_NAME: str = ACTIVE["name"]


if __name__ == "__main__":
    for n, t in TAXONOMIES.items():
        mark = "  <- 当前生效" if n == TAXONOMY_NAME else ""
        print(f"\n=== {n} ==={mark}")
        print(f"  维度 ({len(t['dims'])}): {t['dims']}")
        print(f"  档数    : {t['n_levels']}  (0=未施加, 1..{t['n_levels'] - 1})")
        print(f"  合法取值: {t['code_values']}")
        print(f"  退化族  : {t['families']}")
        n_codes = 1 + sum(len(v) - 1 for v in t["code_values"].values())
        print(f"  单退化码字数: {n_codes}  (每图只施加一种时,码本的大小)")
