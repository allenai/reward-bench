import json 
import requests
from bon_utils import evaluate
import sys 
import os 
import pandas as pd


# Load all tulu results by the ids 
model_oputputs_all = {}
for i in range(16):
    file_outputs = f"alpaca_eval_n=16/annotations/tulu-2-dpo-13b.{i}.json"
    with open(file_outputs) as f:
        model_oputputs = json.load(f)
        model_oputputs_all[i] = model_oputputs

# Load all annotations by the ids
annotations_all = {}
for i in range(16):
    file_annotations = f"alpaca_eval_n=16/annotations/annotations_ref=GPT35t/tulu-2-dpo-13b.{i}/weighted_alpaca_eval_gpt4_turbo/annotations.json"
    with open(file_annotations) as f:
        annotations = json.load(f)
        annotations_all[i] = annotations

def extract_score(score_item):
    if type(score_item) == list:
        return score_item[0]
    elif type(score_item) == float:
        return score_item
    else:
        raise ValueError("Invalid score item")
    
def compute_rm_bon_eval(pretty_rm_name, rm_result_url, model_oputputs_all, annotations_all, mode="rm"):
    if mode == "rm":
        # load json file from http url 
        # to make it a valid json 
        text_content = requests.get(rm_result_url).text
        text_content = text_content.replace("}\n{", "},\n{") 
        text_content = "[" + text_content + "]"   
        rm_result = json.loads(text_content)

        assert len(rm_result) == 805 * 16

        # split the results by grouping them by each of the 805 example 
        rm_result_grouped = [rm_result[i:i+16] for i in range(0, len(rm_result), 16)]
        
        # rank the items with scores and take the top one in each group 
        rm_bon_selection = []
        
        for group in rm_result_grouped:
            group = sorted(group, key=lambda x: extract_score(x["scores"]), reverse=True)
            # select the top one as the submitted one 
            rm_bon_selection.append(group[0])
    elif mode == "oracle":
        pass
    elif mode == "longest":
        pass 
    elif mode == "shortest":
        pass 
    # Example item in rm_bon_selection: 
    # {'config': 'top_p=0.9;temp=1.0', 'dataset_details': 'helpful_base', 'id': [0, 9], 'model': 'allenai/tulu-2-dpo-13b', 'scores': [7.94921875]}


    # generate the selection of virutal model output by selection 
    # generate the annotations of the virtual model output by selection

    rm_bon_model_outputs = []
    rm_bon_annotations = []

    for item in rm_bon_selection:
        example_id = item["id"][0]
        virutal_model_id = item["id"][1]
        output_item = model_oputputs_all[virutal_model_id][example_id].copy()
        original_generator = output_item["generator"]
        output_item["generator"] = pretty_rm_name+"-BoN"
        anno_item = annotations_all[virutal_model_id][example_id].copy()
        if anno_item["generator_1"] == original_generator:
            anno_item["generator_1"] = pretty_rm_name+"-BoN"
        elif anno_item["generator_2"] == original_generator:
            anno_item["generator_2"] = pretty_rm_name+"-BoN"
        rm_bon_model_outputs.append(output_item)
        rm_bon_annotations.append(anno_item)

    file_model_outputs = f"rm_bon_eval_results/{pretty_rm_name}.model_outputs.json"
    file_annotations = f"rm_bon_eval_results/{pretty_rm_name}.annotations.json"
    # create folder if not exists

    os.makedirs(os.path.dirname(file_model_outputs), exist_ok=True)
    os.makedirs(os.path.dirname(file_annotations), exist_ok=True)
    with open(file_model_outputs, "w") as f: 
        json.dump(rm_bon_model_outputs, f, indent=2) 
    with open(file_annotations, "w") as f:
        json.dump(rm_bon_annotations, f, indent=2)

    df_leaderboard, _ = evaluate(model_outputs=file_model_outputs, annotaitons_file=file_annotations, is_return_instead_of_print=True)
    # print(df_leaderboard)
    df_leaderboard = df_leaderboard.reset_index()

    # convert the dataframe to json
    rm_row_json = df_leaderboard.to_dict(orient="records", index=True,)
    # print(json.dumps(rm_row_json, indent=2))
    # find the one with pretty_rm_name 
    for row in rm_row_json:
        if row["index"] == pretty_rm_name + "-BoN":
            target_rm_row_json = row
            break
    target_rm_row_json["reward_model"] = pretty_rm_name
    del target_rm_row_json["index"]
    # print(json.dumps(target_rm_row_json, indent=2))
    return target_rm_row_json

def extract_random(eval_results):
    
    table_file = "alpaca_eval_n=16/annotations/annotations_ref=GPT35t/merged_leaderboard.csv"
    # load as dataframe
    df = pd.read_csv(table_file) 
    # convert to list of dict 
    df_json = df.to_dict(orient="records")
    # find the one with maximum length_controlled_winrate value
    eval_results["random_max"] = max(df_json, key=lambda x: x["length_controlled_winrate"])
    # find the one with minimum length_controlled_winrate value
    eval_results["random_min"] = min(df_json, key=lambda x: x["length_controlled_winrate"])
    # find the one with median length_controlled_winrate value
    length_controlled_winrate_values = [x["length_controlled_winrate"] for x in df_json]
    median_value = sorted(length_controlled_winrate_values)[len(length_controlled_winrate_values)//2]
    eval_results["random_median"] = [x for x in df_json if x["length_controlled_winrate"] == median_value][0]
    # give the average values of all columns 
    # eval_results["random_avg"] = {}
    # for column in df.columns:
    #     if column == "index":
    #         continue
    #     values = [x[column] for x in df_json]
    #     avg_value = sum(values) / len
    #     eval_results["random_avg"][column] = avg_value
    # change model_name with reward_model name 
    eval_results["random_max"]["reward_model"] = eval_results["random_max"]["model_name"]
    eval_results["random_min"]["reward_model"] = eval_results["random_min"]["model_name"]
    eval_results["random_median"]["reward_model"] = eval_results["random_median"]["model_name"] 
    del eval_results["random_max"]["model_name"]
    del eval_results["random_min"]["model_name"]
    del eval_results["random_median"]["model_name"]


if __name__ == "__main__": 
    eval_results = {}
    extract_random(eval_results)
    # print(json.dumps(eval_results, indent=2))
    # exit()

    with open("rm_mapping.json") as f:
        rm_mapping = json.load(f) 
    for pretty_rm_name in rm_mapping:
        rm_result_url = rm_mapping[pretty_rm_name]
        rm_result_url = rm_result_url.replace("huggingface.co", "hf-mirror.com")
        print(f"Running evaluation for {pretty_rm_name} with url {rm_result_url}")
        rm_result = compute_rm_bon_eval(pretty_rm_name, rm_result_url, model_oputputs_all, annotations_all)
        eval_results[pretty_rm_name] = rm_result
        print(rm_result)
    with open("bon_eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)