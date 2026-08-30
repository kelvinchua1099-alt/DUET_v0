import os, sys, time, zipfile
os.environ.setdefault("HF_HOME", "/workspace/.hf_cache")
from huggingface_hub import hf_hub_download
REPO = "deepfakesMSU/NTIRE-RobustAIGenDetection-val"
out = "/workspace/data/ntire_val"
os.makedirs(out, exist_ok=True)
for fn in ["val_labels.csv", "val_hard_labels.csv", "val_images.zip", "val_images_hard.zip"]:
    t0 = time.time()
    p = hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset",
                        local_dir=f"{out}/_dl")
    print(f"下载 {fn}  {os.path.getsize(p)/2**30:.2f} GiB  {time.time()-t0:.0f}s", flush=True)
    if fn.endswith(".zip"):
        t0 = time.time()
        with zipfile.ZipFile(p) as z:
            n = len(z.namelist()); z.extractall(out)
        print(f"  解压 {n} 项 -> {out}  {time.time()-t0:.0f}s", flush=True)
print("目录:", sorted(os.listdir(out)))
