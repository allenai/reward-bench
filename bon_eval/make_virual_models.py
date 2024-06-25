import json

N = 16
input_file = "bon_data/alpaca_eval_n=16/tulu-2-dpo-13b.json"
output_file = "bon_data/alpaca_eval_n=16/virtual/tulu-2-dpo-13b.{model_id}.json"

with open(input_file, "r") as f:
    data = json.load(f)

virtual_models = {}
for item in data:
    for i in range(N):
        item_copy = item.copy()
        item_copy["generator"] = f'{item["generator"]}.{i}'
        item_copy["output"] = item["output"][i]
        if i not in virtual_models:
            virtual_models[i] = []
        virtual_models[i].append(item_copy)

for i in range(N):
    with open(output_file.format(model_id=i), "w") as f:
        json.dump(virtual_models[i], f, indent=2)
