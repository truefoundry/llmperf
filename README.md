# LLMPerf

A Tool for evaulation the performance of LLM APIs. This repo is forked from https://github.com/ray-project/llmperf and builds upon this awesome project to log benchmarking metrics to Truefoundry.

## Installation

```bash
git clone https://github.com/truefoundry/llmperf.git
cd llmperf
pip install -e .
```

## Quickstart

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>

# For privately used model export your organization's URL as base
# Example: https://*.truefoundry.com/api/llm/openai
export OPENAI_API_BASE="https://llm-gateway.truefoundry.com/openai"

python token_benchmark_ray.py \
--model "truefoundry-public/CodeLlama-Instruct(13B)" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api openai \
--additional-sampling-params '{}'
```

### Parameters

The Quickstart script accepts several parameters:

| Parameter                  | Comments                                                                                                                                                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| model                      | Chat based model from llm-gateway. <br> Example: `truefoundry-public / Falcon-Instruct(7B)`, `truefoundry-public / CodeLlama-Instruct(7B)`, etc                                                                                                 |
| mean-input-tokens          | This is the average number of input tokens in a dataset. For example, if you have 3 sentences with 5, 7, and 8 tokens, the mean-input-tokens would be 6.67.                                                                                     |
| stddev-input-tokens        | This is the standard deviation of the number of input tokens, which measures the amount of variation or dispersion in the token counts. For instance, if your input tokens across different requests vary greatly, this number will be high.    |
| mean-output-tokens         | This is the average number of output tokens generated. For example, if your model generates 3 responses with 10, 12, and 15 tokens respectively, the mean-output-tokens would be 12.33.                                                         |
| stddev-output-tokens       | This is the standard deviation of the number of output tokens. It measures the variability in the number of tokens in the output. A high value indicates a wide range of token counts in the output.                                            |
| max-num-completed-requests | This is the maximum number of requests that must be completed.                                                                                                                                                                                  |
| timeout                    | This is the maximum time allowed for a request to be processed. For instance, if the timeout is set to 30 seconds, any request that takes longer than this will be terminated.                                                                  |
| num-concurrent-requests    | his is the number of requests that can be processed at the same time. For example, if this value is 10, it means the system will handle 10 requests simultaneously.                                                                             |
| results-dir                | The directory where the results will be saved.                                                                                                                                                                                                  |
| llm-api                    | Type of LLM Client to be used. Supported Clients: `openai`, `anthropic`, `litellm`.                                                                                                                                                             |
| additional-sampling-params | These are extra parameters used for sampling.                                                                                                                                                                                                   |
| tokenizer_id               | **[Optional]** This specifies the name of the tokenizer used from [huggingface](https://huggingface.co/models?other=tokenizers), default is [`hf-internal-testing/llama-tokenizer`](https://huggingface.co/hf-internal-testing/llama-tokenizer) |
| ml_repo                    | **[Optional]** This specifies the name of the Machine Learning repository. You need to have access to this repository.                                                                                                                          |
| run_name                   | **[Optional]** The run name which is logged in the Machine Learning repository. It helps in identifying and tracking different runs or experiments.                                                                                             |

#### Generate an API Key

- Navigate to Settings > API Keys tab
- Click on Create New API Key
- Give any name to the API Key
- On Generate, API Key will be gererated.
- Please save the value or download it

For more details visit [here](https://docs.truefoundry.com/docs/generate-api-key).

### Logging params and metrics to ML Repo

ML Repositories are like specialized Git repositories for machine learning, managing runs, models, and artifacts within MLFoundry.

We'll use mlfoundry to log parameters and metrics for the cumulative results of all results within an ML Repo.

#### Script

```bash
export OPENAI_API_KEY=<YOUR_API_KEY>
# For privately used model export your organization's URL as base
# Example: https://*.truefoundry.com/api/llm/openai
export OPENAI_API_BASE="https://llm-gateway.truefoundry.com/openai"

python token_benchmark_ray.py \
--model "<MODEL_NAME>" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 2 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs" \
--llm-api openai \
--tokenizer_id "<TOKENIZER_ID_FROM_HUGGINGFACE>"
--ml_repo "<ML_REPO_NAME>"
--run_name "<RUN_NAME>"
--additional-sampling-params '{}'
```

After logging the results seen are:

![Result](https://github.com/truefoundry/llmperf/assets/60005585/379a9545-9edc-4a44-a2b7-8c39b5c2bd13)

## Basic Usage

We implement a load test for evaluating LLMs to check for performance.

## Load test

The load test spawns a number of concurrent requests to the LLM API and measures the inter-token latency and generation throughput per request and across concurrent requests. The prompt that is sent with each request is of the format:

```
Randomly stream lines from the following text. Don't generate eos tokens:
LINE 1,
LINE 2,
LINE 3,
...
```

Where the lines are randomly sampled from a collection of lines from Shakespeare sonnets.

You can count tokens using any tokenizer from huggingface, by providing tokenizer id in the params of the run.
Default tokenizer used is [`hf-internal-testing/llama-tokenizer`](https://huggingface.co/hf-internal-testing/llama-tokenizer).

To run the most basic load test you can the [`token_benchmark_ray`](./token_benchmark_ray.py) script.

### Caveats and Disclaimers

- The endpoints provider backend might vary widely, so this is not a reflection on how the software runs on a particular hardware.
- The results may vary with time of day.
- The results may vary with the load.
- The results may not correlate with users’ workloads.

## Saving Results

The results of the load test and correctness test are saved in the results directory specified by the `--results-dir` argument. The results are saved in 2 files, one with the summary metrics of the test, and one with metrics from each individual request that is returned.
