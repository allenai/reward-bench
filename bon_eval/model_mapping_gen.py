import os 
import json
file_paths = {}
# list all filepaths under `reward-bench-results/best-of-n/alpaca_eval/tulu-13b/`
for root, dirs, files in os.walk("reward-bench-results/best-of-n/alpaca_eval/tulu-13b/"):
    for file in files:
        rm_name = str(os.path.join(root, file)).replace("reward-bench-results/best-of-n/alpaca_eval/tulu-13b/", "").replace(".json", "" )
        url = f"https://huggingface.co/datasets/allenai/reward-bench-results/raw/main/best-of-n/alpaca_eval/tulu-13b/{rm_name}.json"
        file_paths[rm_name] = url
with open("rm_mapping.json", "w") as f:
    json.dump(file_paths, f, indent=4)

