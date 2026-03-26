# Evaluating Cerebras Endpoints with lm-evaluation-harness

This guide covers how to evaluate models served via the [Cerebras API](https://inference-docs.cerebras.ai) using the two supported endpoints:

- **`cerebras-chat-completions`** — Chat completions (`/v1/chat/completions`), generation tasks only
- **`cerebras-completions`** — Text completions (`/v1/completions`), generation + loglikelihood tasks

---

## Setup

### 1. Install dependencies

```bash
pip install -e ".[api]"
pip install transformers python-dotenv
```

### 2. Set your API key

Either export it directly:

```bash
export CEREBRAS_API_KEY=your_key_here
```

Or add it to a `.env` file in your working directory:

```
CEREBRAS_API_KEY=your_key_here
```

---

## Available Models

Models currently served by the Cerebras API:

| Model Arg | Description |
|---|---|
| `llama3.1-8b` | Meta Llama 3.1 8B |
| `gpt-oss-120b` | OpenAI GPT OSS |
| `qwen-3-235b-a22b-instruct-2507` | Qwen 3 235B A22B Instruct |
| `zai-glm-4-7` | ZAI GLM-4 7B |

For the always-up-to-date list, query the API directly:
```bash
curl https://api.cerebras.ai/v1/models \
  -H "Authorization: Bearer $CEREBRAS_API_KEY" | python -m json.tool
```

---

## Quick Start

### Chat completions (generation tasks)

```bash
lm_eval --model cerebras-chat-completions \
  --model_args model=llama3.1-8b \
  --tasks gsm8k \
  --apply_chat_template \
  --num_fewshot 0 \
  --limit 100
```

### Text completions (generation tasks)

```bash
lm_eval --model cerebras-completions \
  --model_args model=llama3.1-8b \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 100
```

### Text completions (loglikelihood / multiple-choice tasks)

A tokenizer is required to compute context lengths for loglikelihood tasks.
Use `NousResearch/Meta-Llama-3.1-8B` as an ungated HF mirror of the Llama 3.1 tokenizer:

```bash
lm_eval --model cerebras-completions \
  --model_args model=llama3.1-8b,tokenizer=NousResearch/Meta-Llama-3.1-8B \
  --tasks mmlu,hellaswag,arc_easy \
  --num_fewshot 0
```

---

## Endpoint Comparison

| Feature | `cerebras-chat-completions` | `cerebras-completions` |
|---|---|---|
| API endpoint | `/v1/chat/completions` | `/v1/completions` |
| Input format | `messages: [{"role": ...}]` | Raw text prompt |
| Generation tasks | ✅ | ✅ |
| Loglikelihood tasks | ❌ | ✅ (requires `tokenizer=`) |
| Multiple-choice tasks | ❌ | ✅ (requires `tokenizer=`) |
| Perplexity tasks | ❌ | ✅ (requires `tokenizer=`) |
| Requires `--apply_chat_template` | Yes | No |
| Default batch size | 1 | 1 |
| Default concurrency | 5 | 5 |

---

## Available Tasks

### Generation tasks (`generate_until`) — both endpoints

| Task | Description | Recommended n-shot |
|---|---|---|
| `gsm8k` | Grade school math word problems | 5 |
| `triviaqa` | Open-domain trivia QA | 3 |
| `drop` | Discrete reasoning over paragraphs | 3 |
| `humaneval` | Python code generation | 0 |
| `mbpp` | Python programming problems | 3 |
| `hendrycks_math` | Competition-level math | 4 |
| `aime24`, `aime25` | AIME math olympiad | 0 |
| `truthfulqa_gen` | Truthfulness (generative) | 0 |
| `coqa` | Conversational QA | 0 |
| `cnn_dailymail` | News summarization | 1 |

> **Note:** `humaneval` and `mbpp` execute model-generated code locally. You must add `--confirm_run_unsafe_code` and set `HF_ALLOW_CODE_EVAL=1`:
> ```bash
> HF_ALLOW_CODE_EVAL=1 lm_eval --model cerebras-chat-completions \
>   --model_args model=llama3.1-8b \
>   --tasks humaneval,mbpp \
>   --apply_chat_template --num_fewshot 0 \
>   --confirm_run_unsafe_code
> ```

### Multiple-choice tasks (`multiple_choice`) — completions endpoint only

| Task | Description | Recommended n-shot |
|---|---|---|
| `mmlu` | 57-subject knowledge benchmark | 5 |
| `hellaswag` | Commonsense sentence completion | 10 |
| `arc_easy` | Elementary science questions | 0 |
| `arc_challenge` | Harder science questions | 25 |
| `winogrande` | Pronoun coreference resolution | 5 |
| `piqa` | Physical intuition QA | 0 |
| `boolq` | Boolean yes/no questions | 0 |
| `copa` | Causal reasoning | 0 |
| `rte` | Textual entailment | 0 |
| `anli_r1`, `anli_r2`, `anli_r3` | Adversarial NLI | 0 |
| `commonsense_qa` | Commonsense reasoning | 7 |
| `headqa_en` | Medical exam QA (English) | 0 |

### Perplexity tasks (`loglikelihood_rolling`) — completions endpoint only

| Task | Description |
|---|---|
| `wikitext` | Wikipedia perplexity |
| `pile_10k` | The Pile (10k subset) |
| `paloma_c4_en` | C4 corpus perplexity |

### Word prediction tasks (`loglikelihood`) — completions endpoint only

| Task | Description | Recommended n-shot |
|---|---|---|
| `lambada_openai` | Predict final word of passage | 0 |
| `lambada_standard` | LAMBADA standard variant | 0 |

---

## Key Flags

| Flag | Description |
|---|---|
| `--model_args model=<name>` | Model to evaluate (see Available Models) |
| `--model_args tokenizer=<hf_id>` | HF tokenizer for loglikelihood tasks |
| `--tasks <task1,task2>` | Comma-separated list of tasks |
| `--apply_chat_template` | Required for `cerebras-chat-completions` |
| `--num_fewshot <n>` | Number of few-shot examples (0 = zero-shot) |
| `--limit <n>` | Cap samples per task (for quick testing) |
| `--output_path <dir>` | Save results JSON to directory |
| `--log_samples` | Save per-sample predictions |
| `--confirm_run_unsafe_code` | Required for `humaneval` and `mbpp` (executes generated code) |

---

## Example: Standard Benchmark Suite

```bash
# Generation benchmarks via chat completions
lm_eval --model cerebras-chat-completions \
  --model_args model=gpt-oss-120b \
  --tasks gsm8k,triviaqa,drop \
  --apply_chat_template \
  --num_fewshot 0 \
  --output_path ./results

# Knowledge + reasoning benchmarks via completions
# Note: use a tokenizer matching your model's family.
# For Llama-based models (llama3.1-8b, gpt-oss-120b) use NousResearch/Meta-Llama-3.1-8B.
# For other models, find a matching HF tokenizer.
lm_eval --model cerebras-completions \
  --model_args model=gpt-oss-120b,tokenizer=NousResearch/Meta-Llama-3.1-8B \
  --tasks mmlu,hellaswag,arc_challenge,winogrande \
  --num_fewshot 0 \
  --output_path ./results
```

---

## Listing All Available Tasks

```bash
lm_eval ls tasks       # all tasks, groups, and tags
lm_eval ls subtasks    # individual task names only
lm_eval ls groups      # group names (e.g. mmlu contains 57 subtasks)
```

Or from Python:

```python
from lm_eval.tasks import TaskManager
tm = TaskManager()
print(tm.all_subtasks)
```
