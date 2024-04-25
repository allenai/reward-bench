from huggingface_hub import hf_hub_download, snapshot_download
import os 
repo_id = "ai2-adapt-dev/HERM_BoN_candidates"  
local_dir = f"bon_data/" 
os.makedirs(local_dir, exist_ok=True) 
# snapshot_download(repo_id=repo_id, allow_patterns="alpaca_eval_n=16/*", local_dir=local_dir, repo_type="dataset")
# hf_hub_download(repo_id=repo_id, filename="gpt-3.5-turbo-0613.ae.json", local_dir=local_dir, repo_type="dataset")
snapshot_download(repo_id="allenai/reward-bench-results", allow_patterns="best-of-n/alpaca_eval/tulu-13b/*", local_dir=f"{local_dir}/rm_bon_eval_results", repo_type="dataset")