import re
import argparse
import json
import os
import time
import random
import string
from datetime import datetime, timezone

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray
from tqdm import tqdm
from transformers import AutoTokenizer

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    LLMPerfResults,
    randomly_sample_sonnet_lines_prompt,
    sample_random_positive_int,
)


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
    tokenizer_id="hf-internal-testing/llama-tokenizer",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

    clients = construct_clients(llm_api=llm_api, num_clients=num_concurrent_requests)
    req_launcher = RequestsLauncher(clients)
    completed_requests = []
    num_completed_requests = 0
    # make up prompts outside of send loop for faster benchmarking loop
    num_output_tokens_list = []
    prompts = []
    for i in range(max_num_completed_requests):
        num_output_tokens = (sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        ))
        num_output_tokens_list.append(num_output_tokens)

        prompts.append(randomly_sample_sonnet_lines_prompt(
            prompt_tokens_mean=mean_input_tokens,
            prompt_tokens_stddev=stddev_input_tokens,
            expect_output_tokens=num_output_tokens,
            tokenizer=tokenizer
        ))
    start_time = time.monotonic()
    iter = 0
    pbar = tqdm(total=max_num_completed_requests)
    while (
        time.monotonic() - start_time < test_timeout_s
        and len(completed_requests) < max_num_completed_requests
    ):
        iter += 1

        default_sampling_params = {"max_tokens": num_output_tokens_list.pop()}
        default_sampling_params.update(additional_sampling_params)
        request_config = RequestConfig(
            model=model,
            prompt=prompts.pop(),
            sampling_params=default_sampling_params,
            llm_api=llm_api,
        )
        req_launcher.launch_requests(request_config)
        # Retrieving results less frequently allows for more concurrent requests
        # to be launched. This will overall reduce the amount of time it takes
        # for the test to run.
        if not (iter % num_concurrent_requests):
            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                num_output_tokens = get_token_length(gen_text)
                if num_output_tokens:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / request_metrics[common_metrics.E2E_LAT]
                all_metrics.append(request_metrics)
            completed_requests.extend(all_metrics)
        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        num_output_tokens = get_token_length(gen_text)
        if num_output_tokens:
            request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
        else:
            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
        request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = num_output_tokens / request_metrics[common_metrics.E2E_LAT]
        all_metrics.append(request_metrics)
    completed_requests.extend(all_metrics)

    print(f"\Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)
    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
        "tokenizer_id": tokenizer_id,
    }

    metadata["results"] = ret
    return metadata, completed_requests


def sanitize_name(value):
    return re.sub(
        rf"[{re.escape(string.punctuation)}]+",
        "-",
        value.encode("ascii", "ignore").decode("utf-8"),
    )


def log_metrics(
    summary: Dict[str, Any],
    model: str,
    ml_repo: str,
    run_name: Optional[str] = None,
) -> None:
    import mlfoundry

    client = mlfoundry.get_client()
    client.create_ml_repo(ml_repo=ml_repo)
    if not run_name:
        job_run_name = os.getenv("TFY_INTERNAL_JOB_RUN_NAME")
        if job_run_name:
            fallback_run_name = f"bench-{sanitize_name(model)}-{job_run_name}"
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
            fallback_run_name = f"bench-{sanitize_name(model)}-{timestamp}"
        run_name = fallback_run_name
    run = client.create_run(ml_repo=ml_repo, run_name=run_name)
    results = summary.pop("results", {})
    additional_sampling_params = summary.pop("additional_sampling_params", {})
    params = {**summary, **additional_sampling_params}
    run.log_params(param_dict=params)

    key_mapping = {
        "end_to_end_latency_s": "end_to_end_latency_seconds",
        "inter_token_latency_s": "inter_token_latency_seconds",
        "number_input_tokens": "input_tokens",
        "number_output_tokens": "output_tokens",
        "request_output_throughput_token_per_s": "output_tokens_per_second",
        "ttft_s": "time_to_first_token_seconds",
    }

    # steps_to_metrics = collections.defaultdict(dict)
    # for metric_name, metric_values in results.items():
    #     if metric_name in key_mapping:
    #         metric_name = key_mapping[metric_name]
    #     if isinstance(metric_values, dict) and "quantiles" in metric_values:
    #         # TODO: We can turn quantiles into charts
    #         for quantile_key in metric_values["quantiles"]:
    #             step = int(quantile_key[1:])
    #             quantile_value = metric_values["quantiles"][quantile_key]
    #             steps_to_metrics[step][metric_name] = quantile_value
    #         steps_to_metrics[0][metric_name] = metric_values["min"]
    #         steps_to_metrics[100][metric_name] = metric_values["max"]
    #     elif isinstance(metric_values, (int, float)):
    #         # If the metric is a single variable, add it directly
    #         steps_to_metrics[0][metric_name] = metric_values
    # for step_number, metrics in steps_to_metrics.items():
    #     run.log_metrics(metrics, step=step_number)

    metrics = {}
    for metric_name, metric_values in results.items():
        if metric_name in key_mapping:
            metric_name = key_mapping[metric_name]
        if isinstance(metric_values, dict) and "quantiles" in metric_values:
            metrics[f"{metric_name}_min"] = metric_values["min"]
            metrics[f"{metric_name}_max"] = metric_values["max"]
            metrics[f"{metric_name}_mean"] = metric_values["mean"]
            metrics[f"{metric_name}_p25"] = metric_values["quantiles"]["p25"]
            metrics[f"{metric_name}_p50"] = metric_values["quantiles"]["p50"]
            metrics[f"{metric_name}_p75"] = metric_values["quantiles"]["p75"]
            metrics[f"{metric_name}_p90"] = metric_values["quantiles"]["p90"]
            metrics[f"{metric_name}_p95"] = metric_values["quantiles"]["p95"]
            metrics[f"{metric_name}_p99"] = metric_values["quantiles"]["p99"]
        elif isinstance(metric_values, (int, float)):
            # If the metric is a single variable, add it directly
            metrics[metric_name] = metric_values
    run.log_metrics(metrics)
    run.end()


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS,
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    additional_sampling_params: str,
    results_dir: str,
    user_metadata: Dict[str, Any],
    tokenizer_id: str = "hf-internal-testing/llama-tokenizer",
    ml_repo: Optional[str] = None,
    run_name: Optional[str] = None,
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
        tokenizer_id=tokenizer_id,
    )

    if ml_repo is not None:
        log_metrics(summary=summary, model=model, ml_repo=ml_repo, run_name=run_name)

    if results_dir:
        filename = f"{model}_{mean_input_tokens}_{mean_output_tokens}"
        filename = re.sub(r"[^\w\d-]+", "-", filename)
        filename = re.sub(r"-{2,}", "-", filename)
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        # Update to metadata.
        summary.update(user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            with open(results_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e


args = argparse.ArgumentParser(
    description="Run a token throughput and latency benchmark."
)

args.add_argument(
    "--model", type=str, required=True, help="The model to use for this load test."
)
args.add_argument(
    "--mean-input-tokens",
    type=int,
    default=550,
    help=(
        "The mean number of tokens to send in the prompt for the request. "
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-input-tokens",
    type=int,
    default=150,
    help=(
        "The standard deviation of number of tokens to send in the prompt for the request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--mean-output-tokens",
    type=int,
    default=150,
    help=(
        "The mean number of tokens to generate from each llm request. This is the max_tokens param "
        "for the completions API. Note that this is not always the number of tokens returned. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--stddev-output-tokens",
    type=int,
    default=80,
    help=(
        "The stdandard deviation on the number of tokens to generate per llm request. "
        "(default: %(default)s)"
    ),
)
args.add_argument(
    "--num-concurrent-requests",
    type=int,
    default=10,
    help=("The number of concurrent requests to send (default: %(default)s)"),
)
args.add_argument(
    "--timeout",
    type=int,
    default=90,
    help="The amount of time to run the load test for. (default: %(default)s)",
)
args.add_argument(
    "--max-num-completed-requests",
    type=int,
    default=10,
    help=(
        "The number of requests to complete before finishing the test. Note "
        "that its possible for the test to timeout first. (default: %(default)s)"
    ),
)
args.add_argument(
    "--additional-sampling-params",
    type=str,
    default="{}",
    help=(
        "Additional sampling params to send with the each request to the LLM API. "
        "(default: %(default)s) No additional sampling params are sent."
    ),
)
args.add_argument(
    "--results-dir",
    type=str,
    default="",
    help=(
        "The directory to save the results to. "
        "(`default: %(default)s`) No results are saved)"
    ),
)
args.add_argument(
    "--llm-api",
    type=str,
    default="openai",
    help=(
        f"The name of the llm api to use. Can select from {SUPPORTED_APIS}"
        " (default: %(default)s)"
    ),
)
args.add_argument(
    "--metadata",
    type=str,
    default="",
    help=(
        "A comma separated list of metadata to include in the results, e.g. "
        "name=foo,bar=1. These will be added to the metadata field of the results. "
    ),
)
args.add_argument(
    "--tokenizer-id",
    type=str,
    default="hf-internal-testing/llama-tokenizer",
    help=("Tokenizer to load from HuggingFace Hub. (default: %(default)s)"),
)
args.add_argument(
    "--ml-repo",
    type=str,
    default=None,
    help=("Name of ML repo in Truefoundry where all the results will be logged."),
)
args.add_argument(
    "--run-name",
    type=str,
    default=None,
    help=(
        "Name of Run Name in ML repo in Truefoundry where all the results will be logged."
    ),
)


if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()
    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value

    run_token_benchmark(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        max_num_completed_requests=args.max_num_completed_requests,
        mean_input_tokens=args.mean_input_tokens,
        stddev_input_tokens=args.stddev_input_tokens,
        mean_output_tokens=args.mean_output_tokens,
        stddev_output_tokens=args.stddev_output_tokens,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
        tokenizer_id=args.tokenizer_id,
        ml_repo=args.ml_repo,
        run_name=args.run_name,
    )
