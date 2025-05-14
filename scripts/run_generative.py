# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# run a generative RM. For now, this requires openai and anthropic to be installed
# Examples:
# python scripts/run_generative.py --model gpt-3.5-turbo
# python scripts/run_generative.py --model=claude-3-haiku-20240307

# note: for none API models, this script uses vllm
# pip install vllm

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer

from rewardbench import load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.generative import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_answers,
    process_judgement,
    run_judge_pair,
    run_judge_ratings,
)
from rewardbench.utils import calculate_scores_per_section

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",  # allow list of models (ensemble)
        required=True,
        help="name of model to use",
    )
    parser.add_argument("--chat_template", type=str, default=None, help="fastchat chat template (optional)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument(
        "--score_w_ratings", action="store_true", default=False, help="score with ratings instead of pairwise ranking"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.9, help="gpu utilization for vllm")
    # parser.add_argument("--vllm_max_seq_length", type=int, default=None, help="max sequence length for vllm")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    model_type = "Generative RM"

    # if model is list, make type + PoLL and check multiple is odd
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        model_type += " PoLL"
        # assert that is odd and > 1
        assert len(args.model) % 2 == 1

    # define variable if is API or local
    if args.force_local:
        is_api_models = False
    else:
        is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST

    # if model isn't API, load via vllm
    if not is_api_models:
        from vllm import LLM, SamplingParams

        # if multi gpu, set multiproc method to spawn
        if args.num_gpus > 1:
            # Set the environment variable
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # load model
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.vllm_gpu_util,
            # max_seq_length=args.vllm_max_seq_length,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "Llama-3" in args.model or "llama3-8b" in args.model and "3.1" not in args.model:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1,
            max_tokens=2048,
            stop_token_ids=stop_token_ids,
        )

    # handle off-case models
    # use different prompt for prometheus/gemini models
    if "prometheus" in args.model:
        model_modifier = "prometheus"
    elif "Con-J" in args.model:
        model_modifier = "Con-J"
    elif "OffsetBias" in args.model:
        model_modifier = "offsetbias"
    elif "Atla" in args.model:
        logger.info("Using ATLA model")
        model_modifier = "Atla"
    elif "gemini" in args.model:
        model_modifier = "gemini"
    elif "RISE-Judge" in args.model:
        model_modifier = "RISE-Judge"
    else:
        model_modifier = None

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    dataset, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        conv=get_conv_template("raw"),  # not used in this script (handled later)
        custom_dialogue_formatting=True,  # handle formatting later
        tokenizer=None,
        logger=logger,
        keep_columns=["text_chosen", "text_rejected", "id"],
        max_turns=4,
    )

    # copy id for saving, then remove
    ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]

    if is_api_models:
        ############################
        # Run inference via API
        ############################
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_judgement(batch, debug=args.debug):
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if len(batch["text_chosen"]) <= 4:  # set up only for 1 or 2 turns
                if not args.score_w_ratings:
                    winner, request, judgement = run_judge_pair(
                        prompt, answer_a, answer_b, args.model, multi_turn=mult_turn, model_modifier=model_modifier
                    )
                    if debug:
                        print(f"Prompt: {request}")
                        print(f"Judgement: {judgement}")
                else:
                    winner, request, judgement = run_judge_ratings(
                        prompt, answer_a, answer_b, args.model, multi_turn=mult_turn, model_modifier=model_modifier
                    )
                    if debug:
                        print(f"Prompt: {request}")
                        print(f"Judgement: {judgement}")

                # handle voting
                if isinstance(winner, list):
                    # print votes if debug
                    if debug:
                        print(winner)
                    winner = max(set(winner), key=winner.count)

                if winner == winner_text:
                    return 1
                elif winner == loser_text:
                    return 0
                else:  # if "error"
                    return 0.5  # effectively a tie
            else:
                return 0.5

        # if debug, do not multi-thread
        if args.debug:
            num_threads = 1
        else:
            num_threads = args.num_threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads
            # results = list(executor.map(get_judgement, dataset))

            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(get_judgement, x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Print newline after progress bar
            print()
    else:
        ############################
        # Run model weights with vllm
        ############################

        def format_judgements(batch, optional_chat_template=None):
            prompt_ids = []  # Prevent crash if it's unused
            # TODO expand this to include fastchat chat templates if needed
            mult_turn = True if len(batch["text_chosen"]) > 2 else False
            prompt = batch["text_chosen"][0]["content"]
            answer_a = batch["text_chosen"]
            answer_b = batch["text_rejected"]

            # shuffle a and b randomly for position bias
            is_shuffled = np.random.rand() > 0.5
            if is_shuffled:
                answer_a, answer_b = answer_b, answer_a

            system_prompt, user_prompt = format_judge_answers(
                prompt, answer_a, answer_b, multi_turn=mult_turn, model_modifier=model_modifier
            )

            if optional_chat_template is not None:
                optional_chat_template.set_system_message(system_prompt)
                optional_chat_template.messages = []
                optional_chat_template.append_message(optional_chat_template.roles[0], user_prompt)
                optional_chat_template.append_message(optional_chat_template.roles[1], None)
                prompt = optional_chat_template.get_prompt()
            else:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # chat template already include special tokens
                # when vllm runs model.generate on prompts, the tokenizer is applied to the prompts
                # defaulting to add_special_tokens=True - this will end up duplicating the special tokens
                # so we need to tokenize without adding special tokens
                tokenized_prompt = tokenizer(prompt, add_special_tokens=False, return_length=True)
                prompt_ids = tokenized_prompt["input_ids"]
            batch["text"] = prompt
            batch["is_shuffled"] = is_shuffled
            batch["prompt_ids"] = prompt_ids
            return batch

        # format the dataset for the model, with optional fastchat templating
        if args.chat_template is not None:
            chat_template = get_conv_template(args.chat_template)
        else:
            chat_template = None
        dataset_prompts = dataset.map(format_judgements, fn_kwargs={"optional_chat_template": chat_template})
        # collect texts of dataset in list
        prompts = dataset_prompts["text"]
        prompt_ids = dataset_prompts["prompt_ids"]
        is_shuffled = dataset_prompts["is_shuffled"]

        # generate
        logger.info("*** Run inference ***")
        if model_modifier == "Atla":
            logger.info("Using Atla model for inference")
            outputs = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
        else:
            outputs = model.generate(prompts, sampling_params=sampling_params)
        logger.info("*** Inference done ***")

        answers = [o.outputs[0].text for o in outputs]
        winners = [process_judgement(a, model_modifier) for a in answers]

        def process_shuffled(win, shuffle):
            if shuffle:
                winner_text = "B"
                loser_text = "A"
            else:
                winner_text = "A"
                loser_text = "B"

            if win == winner_text:
                return 1
            elif win == loser_text:
                return 0
            else:  # if "error"
                return 0.5  # effectively a tie

        results = [process_shuffled(w, s) for w, s in zip(winners, is_shuffled)]

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    out_dataset = out_dataset.add_column("id", ids)

    # model name concat if list
    if isinstance(args.model, list):
        model_name = "_".join(args.model)
        model_name = "PoLL/" + model_name
    else:
        model_name = args.model
    # if model in openai or Anthropic list, append org to model name
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template

    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)

    ############################
    # Upload results to hub
    #############################
    # args.score_w_ratings because results not comprable to those already existing, can change later
    do_not_save = args.do_not_save or args.score_w_ratings

    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"
    results_url = save_to_hub(
        results_grouped,
        model_name,
        sub_path,
        args.debug,
        local_only=do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
    )
    if not do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    logger.info("Not uploading chosen-rejected text with scores due to model compatibility")

    ############################
    # Save per-prompt results to hub
    ############################
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = model_name
    scores_dict["model_type"] = model_type

    sub_path_scores = "eval-set-scores/" if not args.pref_sets else "pref-sets-scores/"

    scores_url = save_to_hub(scores_dict, model_name, sub_path_scores, args.debug, local_only=args.do_not_save)
    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()
