# SQuaDE 运行环境 —— `source env.sh`
#
# 为什么整套装在 /workspace 而不是系统 site-packages:
#   容器的 / 是 20 GB 的 overlay(装完 torch 就用掉一半),/workspace 有 900 T 且持久化。
#   HF 的模型缓存同理 —— DINOv3 ViT-H+ 光权重就 3.2 GB。
#
# 这台机器的两个坑,都已经在下面处理掉了:
#   1. GPU 是 Blackwell(sm_120),PyPI 默认的 cu124 轮子里没有对应 kernel,
#      跑任何 CUDA 算子都会 `RuntimeError: no kernel image is available`。
#      必须装 cu128 轮子(torch 2.11.0+cu128,arch list 里能看到 sm_120)。
#   2. DINOv3 是 gated repo,token 要放在 HF_HOME 下,否则 from_pretrained 报 401
#      而不是「网络不好」—— 很容易误判。

export SQUADE_ROOT=/workspace/SQuaDE
export VIRTUAL_ENV=/workspace/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

export HF_HOME=/workspace/.hf_cache          # 模型权重 + token 都在这
export TOKENIZERS_PARALLELISM=false

# 码本:synthetic = utils/preprocess.py 的 6 维谱(默认,当前实验用的就是它)
#       ntire / ntire7 = NTIRE 官方退化分组,见 utils/deg_taxonomy.py
export SQUADE_TAXONOMY=${SQUADE_TAXONOMY:-synthetic}

cd "$SQUADE_ROOT" 2>/dev/null || true

echo "SQuaDE env:"
echo "  python    $(python -V 2>&1)  ($(which python))"
echo "  torch     $(python -c 'import torch;print(torch.__version__, "cuda" if torch.cuda.is_available() else "NO-CUDA")' 2>/dev/null)"
echo "  HF_HOME   $HF_HOME"
echo "  码本      $SQUADE_TAXONOMY"
