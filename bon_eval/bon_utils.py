import logging
import sys
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence, Union

import fire
import pandas as pd

from alpaca_eval import analyze, annotators, constants, decoders, metrics, utils
from alpaca_eval.types import AnyData, AnyLoadableDF, AnyPath
 

def evaluate(
    model_outputs: Optional[AnyLoadableDF] = None,
    reference_outputs: AnyLoadableDF = constants.ALPACAEVAL_REFERENCE_OUTPUTS,
    annotators_config: AnyPath = constants.DEFAULT_ANNOTATOR_CONFIG,
    name: Optional[str] = None,
    output_path: Optional[Union[AnyPath, str]] = "auto",
    precomputed_leaderboard: Optional[Union[str, AnyPath, AnyData]] = "auto",
    is_overwrite_leaderboard: bool = False,
    leaderboard_mode_to_print: Optional[Union[str, Sequence[str]]] = "minimal",
    current_leaderboard_mode: str = "community",
    is_return_instead_of_print: bool = False,
    fn_metric: Union[str, callable] = "get_length_controlled_winrate" if constants.IS_ALPACA_EVAL_2 else "get_winrate",
    metric_kwargs: Optional[dict[str, Any]] = None,
    is_recompute_metrics_only: bool = False,
    sort_by: str = "length_controlled_winrate" if constants.IS_ALPACA_EVAL_2 else "win_rate",
    is_cache_leaderboard: Optional[bool] = None,
    max_instances: Optional[int] = None,
    annotation_kwargs: Optional[dict[str, Any]] = None,
    Annotator=annotators.PairwiseAnnotator,
    annotaitons_file: Optional[AnyPath] = None,
    **annotator_kwargs,
):
    """Evaluate a model based on its outputs. This is the default entrypoint if no command is specified.

    Parameters
    ----------
    model_outputs : path or data or dict
        The outputs of the model to add to the leaderboard. Accepts data (list of dictionary, pd.dataframe,
        datasets.Dataset) or a path to read those (json, csv, tsv) or a function to generate those. Each dictionary
        (or row of dataframe) should contain the keys that are formatted in the prompts. E.g. by default `instruction`
        and `output` with optional `input`. If None, we just print the leaderboard.

    reference_outputs : path or data, optional
        The outputs of the reference model. Same format as `model_outputs`. If None, the reference outputs are a
        specific set of Davinci 003 outputs on the AlpacaEval set:
        https://huggingface.co/datasets/tatsu-lab/alpaca_eval.

    annotators_config : path or list of dict, optional
        The path the (or list of dict of) the annotator's config file. For details see the docstring of
        `PairwiseAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard. If None we check if `generator is in model_outputs` if not
        we use "Current model".

    output_path : path, optional
        Path to the directory where the new leaderboard and the annotations should be stored. If None we don't save.
        If `auto` we use `model_outputs` if it is a path, and otherwise use the directory from which we call the script.

    precomputed_leaderboard : path or data, optional
        The precomputed leaderboard or a path to it (json, csv, or tsv). The leaderboard should contain at least the
        column `win_rate`. If `auto` we will try to use the corresponding leaderboard for the reference outputs (only if
        in CORRESPONDING_OUTPUTS_LEADERBOARDS). If `None` we won't add other models from the leaderboard.

    is_overwrite_leaderboard : bool, optional
        Whether to overwrite the leaderboard if the model is already in it.

    leaderboard_mode_to_print : {"minimal", "verified", "community", None} or list, optional
        The mode of the leaderboard to use. Only used if the precomputed leaderboard has a column `mode`, in which case
        it will filter the leaderboard by this mode. If None keeps all. If a list, will print all the models in the
        list.

    current_leaderboard_mode : {"minimal", "verified", "community"}, optional
        The mode of the leaderboard for the current method.

    is_return_instead_of_print : bool, optional
        Whether to return the metrics instead of printing the results.

    fn_metric : str or callable, optional
        The function or function name in `metrics` that will be used to convert preference to metrics. The function
        should take a sequence of dict annotations. Each dict has a preference key (1.5 for draw, 1 for base win,
        2 when the model to compare wins) and return a dictionary of metrics and the key by which to sort the
        leaderboard. Common choices: `get_winrate`, `get_length_controlled_winrate`, `get_length_controlled_elo`.

    metric_kwargs : dict, optional
        Additional arguments to pass to `fn_metric`.

    is_recompute_metrics_only : bool, optional
        Whether to recompute the metrics. Useful if all you want to recompute the metrics without reannotating.

    sort_by : str, optional
        The key by which to sort the leaderboard.

    is_cache_leaderboard : bool, optional
        Whether to save the result leaderboard to `precomputed_leaderboard`. If None we save only if max_instances
        not None. A preferred way of adding models to the leaderboard is to set `precomputed_leaderboard` to the
        previously saved leaderboard at `<output_path>/leaderboard.csv`.

    max_instances : int, optional
        The maximum number of instances to annotate. Useful for testing.

    annotation_kwargs : dict, optional
        Additional arguments to pass to `PairwiseAnnotator.annotate_head2head`.

    Annotator : class, optional
        The annotator class to use.

    annotator_kwargs :
        Additional arguments to pass to `PairwiseAnnotator`.
    """
    if (
        isinstance(current_leaderboard_mode, str)
        and current_leaderboard_mode not in constants.ORDERED_LEADERBOARD_MODES
    ):
        raise ValueError(f"current_leaderboard_mode should be one of {constants.ORDERED_LEADERBOARD_MODES}")

    annotation_kwargs = annotation_kwargs or dict()

    leaderboard, precomputed_leaderboard = utils.get_precomputed_leaderboard(
        precomputed_leaderboard, reference_outputs, annotators_config
    )
    annotations = None

    arg_model_outputs = model_outputs
    if False and model_outputs is not None:
        model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
        reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)
        name = utils.get_generator_name(name, model_outputs)

        if (name not in leaderboard) or is_overwrite_leaderboard or is_recompute_metrics_only:
            
            logging.info(f"Evaluating the {name} outputs.")

            if not is_recompute_metrics_only:
                leaderboard[name] = {}
                if max_instances is not None:
                    # first we shuffle both outputs with a fix seed => more representative
                    if len(model_outputs) != len(reference_outputs):
                        logging.warning(
                            "model_outputs and reference_outputs have different lengths, so we cannot shuffle before taking the first max_instances."
                        )
                    else:
                        seed = 123
                        model_outputs = model_outputs.sample(frac=1, random_state=seed)
                        reference_outputs = reference_outputs.sample(frac=1, random_state=seed)

                    model_outputs = model_outputs[:max_instances]
                    reference_outputs = reference_outputs[:max_instances]

                annotator = Annotator(annotators_config=annotators_config, **annotator_kwargs)
                annotations = annotator.annotate_head2head(
                    outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs
                )

                leaderboard[name]["mode"] = current_leaderboard_mode
                leaderboard[name]["avg_length"] = int(model_outputs["output"].str.len().mean())

            else:
                # load previously computed annotations so that we can recompute metrics
                assert output_path is not None and name in leaderboard
                output_path = utils.get_output_path(
                    output_path, arg_model_outputs, name, annotators_config=annotators_config
                )
                annotations = pd.read_json(output_path / "annotations.json")

            # Note: I'm using _ to make clear that we may change the annotations in-place. This is bad practice
            # but gives much more control for saving annotations with desired metrics. E.g. that's how we save
            # "glm_preference" in the annotations
            # TODO: change this and use classes
            if isinstance(fn_metric, str):
                fn_metric_ = getattr(metrics, fn_metric)
            else:
                fn_metric_ = fn_metric

            leaderboard[name].update(fn_metric_(annotations, **(metric_kwargs or {})))

        else:
            logging.info(f"Skipping evaluation of {name} as it is already in the precomputed leaderboard.")
    else:
        model_outputs = utils.load_or_convert_to_dataframe(model_outputs)
        reference_outputs = utils.load_or_convert_to_dataframe(reference_outputs)
        name = utils.get_generator_name(name, model_outputs)
        leaderboard[name] = {}
        if max_instances is not None:
            # first we shuffle both outputs with a fix seed => more representative
            if len(model_outputs) != len(reference_outputs):
                logging.warning(
                    "model_outputs and reference_outputs have different lengths, so we cannot shuffle before taking the first max_instances."
                )
            else:
                seed = 123
                model_outputs = model_outputs.sample(frac=1, random_state=seed)
                reference_outputs = reference_outputs.sample(frac=1, random_state=seed)

            model_outputs = model_outputs[:max_instances]
            reference_outputs = reference_outputs[:max_instances]

        # annotator = Annotator(annotators_config=annotators_config, **annotator_kwargs)
        # annotations = annotator.annotate_head2head(
        #     outputs_1=reference_outputs, outputs_2=model_outputs, **annotation_kwargs
        # )

        leaderboard[name]["mode"] = current_leaderboard_mode 
        leaderboard[name]["avg_length"] = int(model_outputs["output"].str.len().mean())
        annotations = pd.read_json(annotaitons_file)
        # print(f"len(annotations): {len(annotations)}")
        if isinstance(fn_metric, str):
            fn_metric_ = getattr(metrics, fn_metric)
        else:
            fn_metric_ = fn_metric

        leaderboard[name].update(fn_metric_(annotations, **(metric_kwargs or {})))
 

    # output_path = utils.get_output_path(output_path, arg_model_outputs, name, annotators_config=annotators_config)

    df_leaderboard = pd.DataFrame.from_dict(leaderboard, orient="index").sort_values(by=sort_by, ascending=False)
    df_leaderboard = df_leaderboard[
        utils.prioritize_elements(list(df_leaderboard.columns), ["win_rate", "standard_error"])
    ]

    # if output_path is not None:
    #     logging.info(f"Saving all results to {output_path}")
    #     df_leaderboard.to_csv(output_path / "leaderboard.csv")
    #     if annotations is not None:
    #         utils.convert_to_dataframe(annotations).to_json(
    #             output_path / "annotations.json", orient="records", indent=2
    #         )

    if is_cache_leaderboard is None:
        is_cache_leaderboard = max_instances is None

    if is_cache_leaderboard:
        if isinstance(precomputed_leaderboard, AnyPath):
            logging.info(f"Saving result to the precomputed leaderboard at {precomputed_leaderboard}")
            df_leaderboard.to_csv(precomputed_leaderboard)
        else:
            logging.info(
                f"Not saving the result to the cached leaderboard because precomputed_leaderboard is not a "
                f"path but {type(precomputed_leaderboard)}."
            )

    if is_return_instead_of_print:
        return df_leaderboard, annotations
    else:
        utils.print_leaderboard(
            df_leaderboard,
            leaderboard_mode_to_print,
            current_name=name,
            cols_to_print=[sort_by, "win_rate", "standard_error", "n_total", "avg_length"],
        )

 
if __name__ == "__main__":
    # fire.Fire(ALL_FUNCTIONS)
    fire.Fire(evaluate)
    file_model_outputs =  "alpaca_eval_n=16/annotations/tulu-2-dpo-13b.0.json"
    file_annotations = "alpaca_eval_n=16/annotations/annotations_ref=GPT35t/tulu-2-dpo-13b.0/weighted_alpaca_eval_gpt4_turbo/annotations.json"
    # reference_outputs = "gpt-3.5-turbo-0613.ae.json"
    # evaluate(model_outputs=file_model_outputs, reference_outputs=reference_outputs, annotators_config=constants.DEFAULT_ANNOTATOR_CONFIG, output_path="auto", precomputed_leaderboard="auto", is_overwrite_leaderboard=False, leaderboard_mode_to_print="minimal", current_leaderboard_mode="community", is_return_instead_of_print=False, fn_metric="get_length_controlled_winrate", metric_kwargs=None, is_recompute_metrics_only=False, sort_by="length_controlled_winrate", is_cache_leaderboard=None, max_instances=None, annotation_kwargs=None, Annotator=annotators.PairwiseAnnotator)
    # evaluate(model_outputs=file_model_outputs, annotaitons_file=file_annotations)

"""
python src/bon_eval.py --model_outputs alpaca_eval_n=16/annotations/tulu-2-dpo-13b.0.json --annotaitons_file alpaca_eval_n=16/annotations/annotations_ref=GPT35t/tulu-2-dpo-13b.0/weighted_alpaca_eval_gpt4_turbo/annotations.json
"""