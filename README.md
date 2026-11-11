# math-debate (AIME Olympiad Benchmark)

Async multi-agent debate benchmark on Hugging Face AIME datasets (default: `AI-MO/aimo-validation-aime`, filtered to the hardest problems `11`–`15` per contest/year; also supports `di-zhang-fdu/AIME_1983_2024` and `HuggingFaceH4/aime_2024`) using Python 3.10+, 3-digit integer answer normalization (`000`–`999`), and the `google-genai` SDK.

## Setup

### 1. Activate the virtual environment

In PowerShell from inside `gsm8k-debate/`:

```powershell
.\.venv\Scripts\Activate.ps1
```

(If dependencies are not yet installed in a fresh environment, run `pip install -e ".[dev]"` after activating `.venv`. Note that `asyncio` is part of the Python standard library and does not need to be installed separately.)

### 2. Configure your Gemini API key

Never hardcode API credentials in source files or commit them to Git. Set `GEMINI_API_KEY` via one of the following methods:

**Option A: Local `.env` file (recommended for repo-scoped development)**

```powershell
Copy-Item .env.example .env
# Edit .env and replace your_gemini_api_key_here with your actual key
```

**Option B: PowerShell session environment variable**

```powershell
$env:GEMINI_API_KEY = "your-actual-api-key"
```

**Option C: Persistent Windows user environment variable**

```powershell
setx GEMINI_API_KEY "your-actual-api-key"
```

### 3. Verify installation

With `.venv` activated:

```powershell
python -c "import google.genai, sympy; print('Ready')"
```

Run the test suite:

```powershell
pytest
```

### 4. Run single-agent AIME baseline

By default, all evaluation scripts filter to **only the hardest AIME problems (`11`–`15` per contest/year)** — 30 problems across 2022–2024 AIME I & II in `AI-MO/aimo-validation-aime` (10 per year), or 310 historical problems in `di-zhang-fdu/AIME_1983_2024`.

Evaluate Gemini Flash on 15 hard AIME problems (`11`–`15`) from `AI-MO/aimo-validation-aime`:

```powershell
python scripts/run_single_agent.py --num-samples 15
```

You can also filter by competition year (e.g., all 10 problems `11`–`15` in 2024 AIME I & II), override the problem range (`--min-problem 1 --max-problem 15`), or switch datasets (e.g., `di-zhang-fdu/AIME_1983_2024`):

```powershell
python scripts/run_single_agent.py --year 2024 --num-samples 10
```

### 5. Run multi-agent debate (Du et al.)

Run the 3-agent async debate loop on hard AIME problems (`11`–`15`; Round 1 independent Olympiad proof + Rounds 2 & 3 peer-debate rounds) and inspect the intermediate logs (`logs/aime/debate_logs.json`):

```powershell
python scripts/run_debate.py --num-samples 5 --show-logs 2
```

For Free Tier quotas (`--rpm 4`), start with 1–3 samples (3 agents × 3 turns = 9 API calls per sample):

```powershell
python scripts/run_debate.py --num-samples 2 --rpm 4 --show-logs 2
```

### 6. Preemption & Resume Support

Both `run_single_agent.py` and `run_debate.py` flush incremental JSONL checkpoints (`logs/aime/single_agent_checkpoint.jsonl` and round-level `logs/aime/debate_checkpoint.jsonl`) and handle `SIGINT`/`SIGTERM` gracefully. If interrupted mid-run, resume without repeating completed rounds or samples:

```powershell
python scripts/run_debate.py --num-samples 5 --resume
```

### 7. Inspect intermediate round logs

Inspect `logs/aime/debate_logs.json` (completed runs) or `logs/aime/debate_checkpoint.jsonl` (in-progress or preempted runs) to see every round's peer inputs (`--show-prompts`), proof critiques, and answer changes:

```powershell
python scripts/inspect_logs.py --log-file logs/aime/debate_logs.json --show-prompts
```

### 8. Side-by-side Single-Agent vs. Debate benchmark

Compare Single-Agent baseline against 3-Agent Multi-Round Debate across all **30** hard AIME problems (`11`–`15` across 2022–2024 in `AI-MO/aimo-validation-aime`; `--resume` reuses checkpoints in `logs/aime`), logging 3-digit `000`–`999` accuracy, answer changes, error types (`arithmetic_slip` vs. `conceptual_logic_error`), year/difficulty-tier breakdowns, and execution time to `results/aime/`:

```powershell
python scripts/run_benchmark.py --num-samples 30 --max-concurrency 15 --resume
```

To benchmark 100 hard historical AIME problems (`11`–`15` across `1983`–`2024`):

```powershell
python scripts/run_benchmark.py --dataset di-zhang-fdu/AIME_1983_2024 --num-samples 100 --shuffle --max-concurrency 15 --resume
```

For Free Tier quotas (e.g., 5 RPM), pass `--rpm 4` so requests are spaced smoothly without `429` bursts:

```powershell
python scripts/run_benchmark.py --num-samples 30 --rpm 4 --resume
```
