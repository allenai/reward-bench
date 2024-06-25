import os 
import json
file_paths = {}
# list all filepaths under `reward-bench-results/best-of-n/alpaca_eval/tulu-13b/`
for root, dirs, files in os.walk("bon_data/rm_bon_eval_results/best-of-n/alpaca_eval/tulu-13b/"):
    for file in files:
        filepath = str(os.path.join(root, file))
        if "/eval_results/" in filepath:
            continue
        rm_name = filepath.replace("bon_data/rm_bon_eval_results/best-of-n/alpaca_eval/tulu-13b/", "").replace(".json", "" )
        url = f"https://huggingface.co/datasets/allenai/reward-bench-results/raw/main/best-of-n/alpaca_eval/tulu-13b/{rm_name}.json"
        if rm_name == "bon_eval_results":
            continue
        file_paths[rm_name] = {"url": url, "localpath": filepath}
with open("bon_data/rm_mapping.json", "w") as f:
    json.dump(file_paths, f, indent=4)

