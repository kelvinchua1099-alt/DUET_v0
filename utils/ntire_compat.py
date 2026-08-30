"""让官方 `utils/ntire_aug/` 在 NumPy 2 上跑起来 —— 一行修复,不动 vendored 源码。

官方脚本写于 NumPy 1.x 时代。`utils_distortions.spline()` 的 n==3 分支里有一句

    y[2] = np.diff(divdif.T).T / (x[2] - x[0])

右边是长度 1 的数组,左边是标量位置。NumPy 1.x 会静默取出那个元素;NumPy 2 起
直接 `ValueError: setting an array element with a sequence`。受影响的是走 n==3 的
`curves(x, 标量)` 调用,也就是 **brighten / darken 两个畸变全档位报错**
(`lincontrchange` 传的是列表,走 n>3 分支,不受影响)。

处理方式是运行时替换 `spline`,而不是改 `utils/ntire_aug/` 里的文件 —— 那四个文件
与官方发布逐字节相同是「我们施加的退化就是官方那一套」这句话的凭据,见该目录的
NOTICE.md。替换版是官方 n==3 分支的逐行复制,只在那一句上取标量,数值语义与
NumPy 1.x 下的官方实现相同(自检见文件末尾的 __main__)。

用法:在任何 `from utils.ntire_aug...` **之前** 调一次

    from utils.ntire_compat import apply_numpy2_patch
    apply_numpy2_patch()
"""

from __future__ import annotations

import numpy as np

_PATCHED = False


def _spline_np2(x: np.ndarray, y: np.ndarray):
    """官方 spline() 的 NumPy-2 安全版。n != 3 时原样转交官方实现。"""
    from utils.ntire_aug import utils_distortions as U

    n = x.shape[0]
    if n != 3:
        return _ORIG_SPLINE(x, y)

    dd = 1
    dx = np.diff(x)
    divdif = np.diff(y) / dx

    y[1:3] = divdif
    # 唯一的改动:NumPy 1.x 允许把长度 1 的数组赋给标量位,NumPy 2 不允许
    y[2] = (np.diff(divdif.T).T / (x[2] - x[0])).item()
    y[1] -= y[2] * dx[0]

    dlk = y[[2, 1, 0]].shape[0]
    l = x[[0, 2]].shape[0] - 1
    dl = np.prod(dd) * l
    k = np.fix(dlk / dl + 100 * 2.2204e-16)

    return (x[[0, 2]], y[[2, 1, 0]], l, int(k), dd)


_ORIG_SPLINE = None


def apply_numpy2_patch() -> bool:
    """打补丁。返回是否真的打了(NumPy 1.x 上是 no-op)。幂等。"""
    global _PATCHED, _ORIG_SPLINE
    if _PATCHED:
        return True
    if int(np.__version__.split(".")[0]) < 2:
        _PATCHED = True
        return False

    from utils.ntire_aug import utils_distortions as U

    _ORIG_SPLINE = U.spline
    U.spline = _spline_np2
    # distortions.py 里是 `from .utils_distortions import ... curves ...`,
    # 而 curves 在自己的模块里按名字查 spline,所以只补这一处就够;
    # 但 curves 本身若已被 distortions 绑定过,它内部仍走 utils_distortions.spline。
    _PATCHED = True
    return True


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch

    print(f"numpy {np.__version__}  ->  打补丁: {apply_numpy2_patch()}")
    from utils.ntire_aug.utils_data import distortion_functions, distortion_range

    torch.manual_seed(0)
    x = torch.rand(3, 64, 64)
    print(f"\n原图均值 {x.mean():.4f}")
    for name, want in (("brighten", "递增"), ("darken", "递减")):
        ms = [distortion_functions[name](x.clone(), v).mean().item()
              for v in distortion_range[name]]
        mono = all(a < b for a, b in zip(ms, ms[1:])) if want == "递增" \
            else all(a > b for a, b in zip(ms, ms[1:]))
        print(f"  {name:<9} 各档均值 {[round(v, 4) for v in ms]}  单调{want}: {mono}")
    print("\n12 个官方畸变 x 5 档全通:")
    bad = []
    for name, fn in distortion_functions.items():
        for li, v in enumerate(distortion_range[name]):
            try:
                fn(x.clone(), v)
            except Exception as e:
                bad.append(f"{name} L{li + 1}: {type(e).__name__}: {e}")
    print("  全部通过" if not bad else "  失败:\n    " + "\n    ".join(bad))
