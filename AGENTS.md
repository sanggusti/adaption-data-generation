# Modal Rules and Guidelines for LLMs

This file provides rules and guidelines for LLMs when implementing Modal code.

## General

- Modal is a serverless cloud platform for running Python code with minimal configuration
- Designed for AI/ML workloads but supports general-purpose cloud compute
- Serverless billing model - you only pay for resources used

## Modal documentation

- Extensive documentation is available at: modal.com/docs (and in markdown format at modal.com/llms-full.txt)
- A large collection of examples is available at: modal.com/docs/examples (and github.com/modal-labs/modal-examples)
- Reference documentation is available at: modal.com/docs/reference

Always refer to documentation and examples for up-to-date functionality and exact syntax.

## Core Modal concepts

### App

- A group of functions, classes and sandboxes that are deployed together.

### Function

- The basic unit of serverless execution on Modal.
- Each Function executes in its own container, and you can configure different Images for different Functions within the same App:

  ```python
  image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "numpy", "transformers")
    .apt_install("ffmpeg")
    .run_commands("mkdir -p /models")
  )

  @app.function(image=image)
  def square(x: int) -> int:
    return x * x
  ```

- You can configure individual hardware requirements (CPU, memory, GPUs, etc.) for each Function.

  ```python
  @app.function(
    gpu="H100",
    memory=4096,
    cpu=2,
  )
  def inference():
    ...
  ```

  Some examples specificly for GPUs:

  ```python
  @app.function(gpu="A10G")  # Single GPU, e.g. T4, A10G, A100, H100, or "any"
  @app.function(gpu="A100:2")  # Multiple GPUs, e.g. 2x A100 GPUs
  @app.function(gpu=["H100", "A100", "any"]) # GPU with fallbacks
  ```

- Functions can be invoked in a number of ways. Some of the most common are:
  - `foo.remote()` - Run the Function in a separate container in the cloud. This is by far the most common.
  - `foo.local()` - Run the Function in the same context as the caller. Note: This does not necessarily mean locally on your machine.
  - `foo.map()` - Parallel map over a set of inputs.
  - `foo.spawn()` - Calls the function with the given arguments, without waiting for the results. Terminating the App will also terminate spawned functions.
- Web endpoint: You can turn any Function into an HTTP web endpoint served by adding a decorator:

  ```python
  @app.function()
  @modal.fastapi_endpoint()
  def fastapi_endpoint():
    return {"status": "ok"}

  @app.function()
  @modal.asgi_app()
  def asgi_app():
    app = FastAPI()
    ...
    return app
  ```

- You can run Functions on a schedule using e.g. `@app.function(schedule=modal.Period(minutes=5))` or `@app.function(schedule=modal.Cron("0 9 * * *"))`.

### Classes (a.k.a. `Cls`)

- For stateful operations with startup/shutdown lifecycle hooks. Example:

  ```python
  @app.cls(gpu="A100")
  class ModelServer:
      @modal.enter()
      def load_model(self):
          # Runs once when container starts
          self.model = load_model()

      @modal.method()
      def predict(self, text: str) -> str:
          return self.model.generate(text)

      @modal.exit()
      def cleanup(self):
          # Runs when container stops
          cleanup()
  ```

### Other important concepts

- Image: Represents a container image that Functions can run in.
- Sandbox: Allows defining containers at runtime and securely running arbitrary code inside them.
- Volume: Provide a high-performance distributed file system for your Modal applications.
- Secret: Enables securely providing credentials and other sensitive information to your Modal Functions.
- Dict: Distributed key/value store, managed by Modal.
- Queue: Distributed, FIFO queue, managed by Modal.

## Differences from standard Python development

- Modal always executes code in the cloud, even while you are developing. You can use Environments for separating development and production deployments.
- Dependencies: It's common and encouraged to have different dependency requirements for different Functions within the same App. Consider defining dependencies in Image definitions (see Image docs) that are attached to Functions, rather than in global `requirements.txt`/`pyproject.toml` files, and putting `import` statements inside the Function `def`. Any code in the global scope needs to be executable in all environments where that App source will be used (locally, and any of the Images the App uses).

## Modal coding style

- Modal Apps, Volumes, and Secrets should be named using kebab-case.
- Always use `import modal`, and qualified names like `modal.App()`, `modal.Image.debian_slim()`.
- Modal evolves quickly, and prints helpful deprecation warnings when you `modal run` an App that uses deprecated features. When writing new code, never use deprecated features.

## Common commands

Running `modal --help` gives you a list of all available commands. All commands also support `--help` for more details.

### Running your Modal app during development

- `modal run path/to/your/app.py` - Run your app on Modal.
- `modal run -m module.path.to.app` - Run your app on Modal, using the Python module path.
- `modal serve modal_server.py` - Run web endpoint(s) associated with a Modal app, and hot-reload code on changes. Will print a URL to the web endpoint(s). Note: you need to use `Ctrl+C` to interrupt `modal serve`.

### Deploying your Modal app

- `modal deploy path/to/your/app.py` - Deploy your app (Functions, web endpoints, etc.) to Modal.
- `modal deploy -m module.path.to.app` - Deploy your app to Modal, using the Python module path.

Logs:

- `modal app logs <app_name>` - Stream logs for a deployed app. Note: you need to use `Ctrl+C` to interrupt the stream.

### Resource management

- There are CLI commands for interacting with resources like `modal app list`, `modal volume list`, and similarly for `secret`, `dict`, `queue`, etc.
- These also support other command than `list` - use e.g. `modal app --help` for more.

## Testing and debugging

- When using `app.deploy()`, you can wrap it in a `with modal.enable_output():` block to get more output.

# Adaption rules

Adaption is designed so the full path from raw file to augmented output is repeatable in code: a short sequence of API calls, whether you run once in a notebook or at scale in a pipeline. This page uses the SDK’s convenience methods (upload, import, run, wait, download).

## Authentication

```python
from adaption import Adaption

client = Adaption(api_key="pt_live_...")
# Or set the ADAPTION_API_KEY environment variable and omit the argument
```

Ingest data from where it already lives—local files, Hugging Face, or Kaggle—without hand-built conversion scripts for supported formats. Each path returns a dataset_id you pass into datasets.run for the adapt step.

upload_file creates the dataset, uploads via a presigned URL, and confirms the upload.

Supported extensions: .csv, .json, .jsonl, .parquet.

```python
result = client.datasets.upload_file("training_data.csv")
print(result.dataset_id)
```

Optional custom name (defaults to the filename without extension):

```python
result = client.datasets.upload_file("data.csv", name="my-dataset")
```

import from huggingface

```python
resp = client.datasets.create_from_huggingface(
    url="https://huggingface.co/datasets/org/repo",
    files=["train.csv"],
)
print(resp.dataset_id)
```

import from kaggle

```python
resp = client.datasets.create_from_kaggle(
    url="https://www.kaggle.com/datasets/org/dataset-name",
    files=["data.csv"],
)
print(resp.dataset_id)
```

Adapt applies the platform’s recipes and optimization to your dataset. Map columns to the roles the API expects (prompt is required; others are optional). Call with estimate=True first to see estimated cost and duration before you commit—no surprises on large jobs.

```python
run = client.datasets.run(
    dataset_id,
    column_mapping={
        "prompt": "instruction",  # required — your prompt column
        "completion": "response",  # optional — completion column
        # "chat": "conversation",  # optional — chat column
        # "context": ["source", "ref"],  # optional — context columns
    },
)
print(run.run_id)
print(run.estimated_credits_consumed)
```

Estimate cost without starting a run:

```python
estimate = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    estimate=True,
)
print(f"Would cost {estimate.estimated_credits_consumed} credits")
```

Wait for completion

```python
from adaption import DatasetTimeout

try:
    status = client.datasets.wait_for_completion(dataset_id, timeout=600)
    print(f"Done: {status.status}")  # "succeeded" or "failed"
    if status.error:
        print(f"Error: {status.error.message}")
except DatasetTimeout as e:
    print(f"Still running after {e.timeout}s (last status: {e.last_status})")
```

List datasets

```python
for dataset in client.datasets.list():
    print(f"{dataset.dataset_id} — {dataset.status}")

# with filters
for dataset in client.datasets.list(status="succeeded", limit=10):
    print(dataset.name)
```

End to end example

```python
import time

from adaption import Adaption, DatasetTimeout

client = Adaption(api_key="pt_live_...")

# 1. Upload
result = client.datasets.upload_file("training_data.csv")

# 2. Wait for file processing
while True:
    status = client.datasets.get_status(result.dataset_id)
    if status.row_count is not None:
        break
    time.sleep(2)

# 3. Adapt (start run)
run = client.datasets.run(
    result.dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
)
print(
    f"Run started: {run.run_id}, ~{run.estimated_minutes} min, "
    f"{run.estimated_credits_consumed} credits"
)

# 4. Wait for completion
try:
    final = client.datasets.wait_for_completion(result.dataset_id, timeout=1800)
    print(f"Finished: {final.status}")
except DatasetTimeout:
    print("Timed out — check status manually")

# 5. Download
url = client.datasets.download(result.dataset_id)
print(f"Download: {url}")
```

## Async variants

```python
from adaption import AsyncAdaption

client = AsyncAdaption(api_key="pt_live_...")

result = await client.datasets.upload_file("data.csv")
status = await client.datasets.wait_for_completion(result.dataset_id)
async for dataset in client.datasets.list():
    print(dataset.dataset_id)
```

## Enable mitigations

Enable mitigation when outputs must stay fact-aligned: customer support, RAG-style workflows with external truth, compliance-sensitive text, or any pipeline where invented details are expensive to catch later. For creative or purely subjective tasks, you may omit it to save latency or credits; use estimate=True to compare cost before you scale.

```python
import os
import time

from adaption import Adaption, DatasetTimeout

client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
dataset_id = os.environ.get("ADAPTION_DATASET_ID")

if not dataset_id:
    result = client.datasets.upload_file("training_data.csv")
    dataset_id = result.dataset_id
    while True:
        st = client.datasets.get_status(dataset_id)
        if st.row_count is not None:
            break
        time.sleep(2)

estimate = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    brand_controls={"hallucination_mitigation": True},
    estimate=True,
)
print(f"Estimated credits: {estimate.estimated_credits_consumed}")

run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    brand_controls={"hallucination_mitigation": True},
)
print(f"Run started: {run.run_id}")

try:
    final = client.datasets.wait_for_completion(dataset_id, timeout=3600)
    print(f"Finished: {final.status}")
    if final.error:
        raise RuntimeError(final.error.message)
except DatasetTimeout:
    print("Timed out — poll datasets.get or get_status in your environment")

url = client.datasets.download(dataset_id)
print(f"Download: {url}")
```

## Processing large datasets

ingest (or reuse a dataset_id), wait until the dataset is ready, run on a bounded number of rows, wait for completion, then export. Use max_rows when you need a representative trial on production-scale data: confirm mappings and behavior, compare estimate=True quotes, or share a quick pilot without duplicating files. When you are ready for the full corpus, run again without max_rows (or with a higher cap) so the job processes the entire dataset.

```python
import os
import time

from adaption import Adaption, DatasetTimeout

client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
dataset_id = os.environ.get("ADAPTION_DATASET_ID")

if not dataset_id:
    result = client.datasets.upload_file("large_training_data.csv")
    dataset_id = result.dataset_id
    while True:
        st = client.datasets.get_status(dataset_id)
        if st.row_count is not None:
            break
        time.sleep(2)

run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    job_specification={"max_rows": 1_000},
)
print(f"Run started (up to 1,000 rows): {run.run_id}")

try:
    final = client.datasets.wait_for_completion(dataset_id, timeout=3600)
    print(f"Finished: {final.status}")
    if final.error:
        raise RuntimeError(final.error.message)
except DatasetTimeout:
    print("Timed out — poll datasets.get or get_status in your environment")

url = client.datasets.download(dataset_id)
print(f"Download: {url}")
```

## Reasoning traces

Reasoning traces are part of Adaptive Data’s recipe library—alongside options such as deduplication and prompt rephrasing—so you can shape how models reason over your data, not only the final string.

Traces expose an intermediate reasoning path with each adapted completion. That supports auditability (you can inspect how an answer was derived), distillation, and supervised fine-tuning that teaches models to show structured reasoning before you strip traces for production.

In the SDK this is a recipe toggle under recipe_specification, not a brand_controls field. RecipeSpecificationRecipes includes additional toggles (for example deduplication and prompt_rephrase). Set multiple keys under recipes in one run; omitted keys keep backend defaults.

```python
import os
import time

from adaption import Adaption, DatasetTimeout

client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
dataset_id = os.environ.get("ADAPTION_DATASET_ID")

if not dataset_id:
    result = client.datasets.upload_file("training_data.csv")
    dataset_id = result.dataset_id
    while True:
        st = client.datasets.get_status(dataset_id)
        if st.row_count is not None:
            break
        time.sleep(2)

run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    recipe_specification={"recipes": {"reasoning_traces": True}},
)
print(f"Run started: {run.run_id}, ~{run.estimated_credits_consumed} credits")

try:
    final = client.datasets.wait_for_completion(dataset_id, timeout=3600)
    print(f"Finished: {final.status}")
    if final.error:
        raise RuntimeError(final.error.message)
except DatasetTimeout:
    print("Timed out — check status manually")

url = client.datasets.download(dataset_id)
print(f"Download: {url}")
```

## Brand Controls

Adaptive Data treats specification as foundational: goals such as length, safety, and brand voice become objectives that steer what the platform produces. Data configuration should be structural, not a layer of preferences applied after the fact.

In the Python SDK, brand_controls on datasets.run is where you express that specification. It covers:

- `length` — target verbosity of generated completions.
- `safety_categories` — content safety categories to enforce.
- `hallucination_mitigation` — web-search grounding (covered in Mitigating hallucinations).
- `blueprint` — a freeform system prompt for tone, persona, language, or any guideline that does not fit the structured fields.

Together these help ensure adapted data matches how much your product says, what it is allowed to say, and how it says it.

```python
import os
import time

from adaption import Adaption, DatasetTimeout

client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
dataset_id = os.environ.get("ADAPTION_DATASET_ID")

if not dataset_id:
    result = client.datasets.upload_file("training_data.csv")
    dataset_id = result.dataset_id
    while True:
        st = client.datasets.get_status(dataset_id)
        if st.row_count is not None:
            break
        time.sleep(2)

run = client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "instruction", "completion": "response"},
    brand_controls={
        "length": "concise",
        "safety_categories": ["harassment", "hate"],
        "blueprint": "Answer in British English with a warm, concise tone.",
    },
)
print(f"Run started: {run.run_id}")

try:
    final = client.datasets.wait_for_completion(dataset_id, timeout=3600)
    print(f"Finished: {final.status}")
    if final.error:
        raise RuntimeError(final.error.message)
except DatasetTimeout:
    print("Timed out")

url = client.datasets.download(dataset_id)
print(f"Download: {url}")
```

## Evaluating Dataset Quality

After an adaptation run finishes successfully, the platform can produce evaluation signals—scores and related metrics that summarize how the augmented data compares to the original on quality dimensions the pipeline measures.

In the Python SDK you retrieve that information in two complementary ways:

- `datasets.get_evaluation(dataset_id)` — dedicated response with evaluation pipeline status and structured quality metrics.
- `datasets.get(dataset_id)` — full dataset record including evaluation_summary, a compact mirror of the headline metrics when evaluation has finished.

Use get_evaluation when you need explicit evaluation status and full quality details. Use get when you already fetch the dataset and want a single summary on the same object. Pair either approach with estimate=True on future runs (see Processing large datasets and the FAQ) when you are iterating on quality before scaling row counts.

```python

import os

from adaption import Adaption

client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
dataset_id = os.environ["ADAPTION_DATASET_ID"]

ev = client.datasets.get_evaluation(dataset_id)
print(f"Evaluation status: {ev.status}")

ds = client.datasets.get(dataset_id)
print(f"Dataset status: {ds.status}")
if ds.evaluation_summary:
    print(f"Summary: {ds.evaluation_summary.model_dump(exclude_none=True)}")

# Fetch evaluation results
ev = client.datasets.get_evaluation(dataset_id)

print(ev.status)  # pending | running | succeeded | failed | skipped
if ev.quality:
    print(f"Score before: {ev.quality.score_before}")
    print(f"Score after:  {ev.quality.score_after}")
    print(f"Improvement:  {ev.quality.improvement_percent}%")

# Quick view on the dataset record
ds = client.datasets.get(dataset_id)
if ds.evaluation_summary:
    print(ds.evaluation_summary.score_after, ds.evaluation_summary.improvement_percent)

# Poll until evaluation finishes
import time

while True:
    ev = client.datasets.get_evaluation(dataset_id)
    if ev.status in ("succeeded", "failed", "skipped"):
        break
    time.sleep(5)

if ev.status == "succeeded" and ev.quality:
    print(ev.quality.model_dump(exclude_none=True))
```
