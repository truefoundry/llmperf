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
export OPENAI_API_BASE="https://<COMPANY>.truefoundry.tech/api/llm/openai"

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

### Parameters

The Quickstart script accepts several parameters:

| Parameter                  | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| model                      | The model to be used for the benchmark.                 |
| mean-input-tokens          | The mean number of input tokens.                        |
| stddev-input-tokens        | The standard deviation for the number of input tokens.  |
| mean-output-tokens         | The mean number of output tokens.                       |
| stddev-output-tokens       | The standard deviation for the number of output tokens. |
| max-num-completed-requests | The maximum number of completed requests.               |
| timeout                    | The timeout for the benchmark.                          |
| num-concurrent-requests    | The number of concurrent requests to be made.           |
| results-dir                | The directory where the results will be saved.          |
| llm-api                    | The LLM API to be used.                                 |
| tokenizer_id               | The ID of the tokenizer from HuggingFace Hub.           |
| ml_repo                    | The name of the ML repository.                          |
| run_name                   | Name of run in ML_repo                                  |
| additional-sampling-params | Additional parameters for sampling                      |

#### Generate an API Key

- Navigate to Settings > API Keys tab
- Click on Create New API Key
- Give any name to the API Key
- On Generate, API Key will be gererated.
- Please save the value or download it

For more details visit [here](https://docs.truefoundry.com/docs/generate-api-key).

### MLFoundry

ML Repositories are like specialized Git repositories for machine learning, managing runs, models, and artifacts within MLFoundry.

- **Runs** are individual experiments with specific models and hyperparameters.
- **Models** include model files and metadata, with multiple versions possible.
- **Parameters** are the settings defining each experiment.
- **Metrics** are the values used to evaluate and compare runs.
- **Artifacts** are file collections, also versioned.

We'll use mlfoundry to log parameters and metrics for the cumulative results of all results within an ML Repo.

After logging the results seen are:

![Result](https://github.com/truefoundry/llmperf/assets/60005585/379a9545-9edc-4a44-a2b7-8c39b5c2bd13)

## Basic Usage

We implement 2 tests for evaluating LLMs: a load test to check for performance and a correctness test to check for correctness.

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

# Advanced Usage

The correctness tests were implemented with the following workflow in mind:

```python
import ray
from transformers import LlamaTokenizerFast

from llmperf.models import RequestConfig
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.requests_launcher import RequestsLauncher

# Copying the environment variables and passing them to ray.init() is necessary
# For making any clients work.
ray.init(
    runtime_env={
        "env_vars": {
            "OPENAI_API_BASE": "https://<COMPANY>.truefoundry.tech/api/llm/openai",
            "OPENAI_API_KEY": "YOUR_API_KEY",
        }
    }
)

base_prompt = "hello_world"
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
base_prompt_len = len(tokenizer.encode(base_prompt))
prompt = (base_prompt, base_prompt_len)

# Create a client for spawning requests
clients = [OpenAIChatCompletionsClient.remote()]

req_launcher = RequestsLauncher(clients)

req_config = RequestConfig(model="meta-llama/Llama-2-7b-chat-hf", prompt=prompt)

req_launcher.launch_requests(req_config)
result = req_launcher.get_next_ready(block=True)
print(result)


```

