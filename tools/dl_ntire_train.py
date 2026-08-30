import os, sys, time
os.environ.setdefault("HF_HOME", "/workspace/.hf_cache")
from huggingface_hub import hf_hub_download
REPO = "deepfakesMSU/NTIRE-RobustAIGenDetection-train"
for fn in sys.argv[1:]:
    t0 = time.time()
    p = hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset",
                        local_dir="/workspace/data/ntire/_zips")
    print(f"{fn} -> {p}  ({time.time()-t0:.0f}s, {os.path.getsize(p)/2**30:.2f} GiB)", flush=True)
