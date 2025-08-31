### Environment setup for Multi-Turn RL Agent

These steps follow the upstream README while keeping our repo unchanged.

1) Create a dedicated Python env (names and versions are examples, not enforced):

```bash
conda create -n verifier_env python=3.11 -y
conda activate verifier_env
```

2) Install the submodule in editable mode and optional flash-attn:

```bash
pip install -e external/Multi-Turn-RL-Agent
pip install flash-attn --no-build-isolation
```

3) Java (only if using local wiki search):

```bash
sudo apt-get remove --purge openjdk* -y || true
sudo apt-get update || true
sudo apt-get install -y openjdk-21-jdk || true
java -version
```

4) Build local wiki index (optional):

```bash
python external/Multi-Turn-RL-Agent/verifiers/tools/local_wiki_search.py
```

5) Accelerate config (if needed):
6) Prepare a JSONL dataset to drive outcome rewards (fields: prompt, answer):

```json
{"prompt": "Who discovered penicillin?", "answer": ["Alexander Fleming", "Fleming"]}
{"prompt": "Capital of France?", "answer": "Paris"}
```

Run with:

```bash
DATASET_JSONL=path/to/data.jsonl MODEL_NAME="Qwen/Qwen2.5-7B" NUM_GPUS=1 ./scripts/run_mt_rl_jsonl.sh
```

```bash
accelerate config
```

Notes:
- Do not hardcode credentials or model names in code. Pass via CLI/ENV.
- The user will provide outcome rewards later. Turn-level reward flags can be toggled in the runner.

