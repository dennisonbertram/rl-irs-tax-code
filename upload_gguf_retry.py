import os
import sys
import traceback
sys.path.insert(0, '/Users/dennisonbertram/Develop/rl-irs-tax-code/.venv/lib/python3.14/site-packages')

from huggingface_hub import HfApi, login

print("Logging in...", flush=True)
login(token=os.environ["HF_TOKEN"])  # set HF_TOKEN env var or run `huggingface-cli login`
api = HfApi()

USERNAME = "dennisonb"
REPO = f"{USERNAME}/qwen25-tax-3b-GGUF"

print(f"Uploading GGUF to {REPO} ...", flush=True)
try:
    result = api.upload_file(
        path_or_fileobj="/tmp/qwen25-tax-3b-q8_0.gguf",
        path_in_repo="qwen25-tax-3b-q8_0.gguf",
        repo_id=REPO,
    )
    print(f"SUCCESS: {result}", flush=True)
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 2. Upload adapters repo
ADAPTER_REPO = f"{USERNAME}/qwen25-tax-3b-adapters"
print(f"\nCreating repo {ADAPTER_REPO}...", flush=True)
api.create_repo(repo_id=ADAPTER_REPO, repo_type="model", exist_ok=True)

for stage in ["sft", "dpo", "grpo"]:
    print(f"Uploading {stage} adapters...", flush=True)
    try:
        api.upload_folder(
            folder_path=f"/Users/dennisonbertram/Develop/rl-irs-tax-code/outputs/{stage}/adapters",
            path_in_repo=f"{stage}/",
            repo_id=ADAPTER_REPO,
        )
        print(f"  {stage} done.", flush=True)
    except Exception as e:
        print(f"  ERROR uploading {stage}: {e}", flush=True)
        traceback.print_exc()

print(f"\nGGUF repo: https://huggingface.co/{REPO}", flush=True)
print(f"Adapters repo: https://huggingface.co/{ADAPTER_REPO}", flush=True)
print("Done!", flush=True)
