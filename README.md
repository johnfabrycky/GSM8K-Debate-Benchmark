# gsm8k-debate (MATH-500)

Async multi-agent debate benchmark on `HuggingFaceH4/MATH-500` (`"test"` split) using Python 3.10+, `sympy` symbolic & LaTeX equivalence verification (`\boxed{...}`), and local Ollama (`http://localhost:11434/v1`, `qwen2.5:7b`) via the OpenAI Python SDK (`openai.AsyncOpenAI`).

## Setup

### 1. Activate the virtual environment

In PowerShell from inside `gsm8k-debate/`:

```powershell
.\.venv\Scripts\Activate.ps1
```

(If dependencies are not yet installed in a fresh environment, run `pip install -e ".[dev]"` after activating `.venv`. Note that `asyncio` is part of the Python standard library and does not need to be installed separately.)

### 2. Configure local Ollama endpoint

By default, all scripts connect via `AsyncOpenAI` to your local Ollama OpenAI-compatible server:

- **Base URL**: `http://localhost:11434/v1` (override via `--base-url` or `OLLAMA_BASE_URL`)
- **API Key**: `"ollama"` (override via `--api-key` or `OLLAMA_API_KEY`)
- **Model**: `"qwen2.5:7b"` (override via `--model`)

Ensure Ollama is running locally with `qwen2.5:7b` pulled:

```powershell
ollama pull qwen2.5:7b
```

You can also set `OLLAMA_BASE_URL` / `OLLAMA_API_KEY` in a local `.env` file (see `.env.example`) or pass `--base-url`, `--api-key`, and `--model` directly to any script.

### 3. Verify installation

With `.venv` activated:

```powershell
python -c "import openai, datasets, pandas, sympy; print('Ready')"
```

Run the test suite:

```powershell
pytest
```

### 4. Run single-agent MATH-500 baseline

Evaluate `qwen2.5:7b` on a subset of `HuggingFaceH4/MATH-500` (default: 15 samples):

```powershell
python scripts/run_single_agent.py --num-samples 15
```

You can also filter by competition subject or difficulty level (`1`–`5`):

```powershell
python scripts/run_single_agent.py --num-samples 20 --subject "Number Theory" --level 5
```

### 5. Run multi-agent debate (Du et al.)

Run the 3-agent, 3-round asynchronous debate loop (`Round 1` independent competition derivation + `Rounds 2 & 3` peer-debate rounds) and inspect intermediate logs (`logs/qwen_math500/debate_logs.json`):

```powershell
python scripts/run_debate.py --num-samples 5 --show-logs 2
```

If running on limited local GPU VRAM/slots, reduce concurrency with `--max-concurrency 1`:

```powershell
python scripts/run_debate.py --num-samples 2 --max-concurrency 1 --show-logs 2
```

### 6. Preemption & Resume Support

Both `run_single_agent.py` and `run_debate.py` flush incremental JSONL checkpoints (`logs/qwen_math500/single_agent_checkpoint.jsonl` and round-level `logs/qwen_math500/debate_checkpoint.jsonl`) and handle `SIGINT`/`SIGTERM` gracefully. If interrupted mid-run, resume without repeating completed rounds or samples:

```powershell
python scripts/run_debate.py --num-samples 5 --resume
```

### 7. Inspect intermediate round logs

Inspect `logs/qwen_math500/debate_logs.json` (completed runs) or `logs/qwen_math500/debate_checkpoint.jsonl` (in-progress or preempted runs) to see every round's peer inputs (`--show-prompts`), critiques, and answer changes:

```powershell
python scripts/inspect_logs.py --log-file logs/qwen_math500/debate_logs.json --show-prompts
```

### 8. Side-by-side Single-Agent vs. Debate benchmark

Compare Single-Agent baseline against 3-Agent Multi-Round Debate across 100 `HuggingFaceH4/MATH-500` test problems using `qwen2.5:7b` on local Ollama (`http://localhost:11434/v1`), logging accuracy, LaTeX/SymPy answer changes, error types (`arithmetic_slip` vs. `conceptual_logic_error`), subject/level breakdowns, and execution time to `results/qwen_math500/` (isolated from earlier Gemini runs in `results/math500/`):

```powershell
python scripts/run_benchmark.py --num-samples 100 --max-concurrency 3 --checkpoint-dir logs/qwen_math500 --output-dir results/qwen_math500
```

To resume an interrupted run:

```powershell
python scripts/run_benchmark.py --num-samples 100 --max-concurrency 3 --checkpoint-dir logs/qwen_math500 --output-dir results/qwen_math500 --resume
```

To point any script at another OpenAI-compatible server or model, pass `--base-url`, `--api-key`, and `--model` (or set `OLLAMA_BASE_URL` / `OLLAMA_API_KEY`):

```powershell
python scripts/run_benchmark.py --base-url http://localhost:11434/v1 --api-key ollama --model qwen2.5:7b --num-samples 100
```
