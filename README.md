# math-debate (HuggingFaceH4/MATH-500)

Async multi-agent debate benchmark on `HuggingFaceH4/MATH-500` using Python 3.10+, `sympy` symbolic equivalence verification, and the `google-genai` SDK.

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

Run the smoke and unit test suite:

```powershell
pytest
```

### 4. Run single-agent MATH-500 baseline

Evaluate Gemini Flash on a subset of `HuggingFaceH4/MATH-500` (default: 15 samples):

```powershell
python scripts/run_single_agent.py --num-samples 15
```

You can also filter by competition subject or difficulty level (`1`–`5`):

```powershell
python scripts/run_single_agent.py --num-samples 20 --subject "Number Theory" --level 5
```

### 5. Run multi-agent debate (Du et al.)

Run the 3-agent async debate loop (Round 1 independent competition proof + Rounds 2 & 3 peer-debate rounds) and inspect the intermediate logs (`logs/math500/debate_logs.json`):

```powershell
python scripts/run_debate.py --num-samples 5 --show-logs 2
```

For Free Tier quotas (`--rpm 4`), start with 1–3 samples (3 agents × 3 turns = 9 API calls per sample):

```powershell
python scripts/run_debate.py --num-samples 2 --rpm 4 --show-logs 2
```

### 6. Preemption & Resume Support

Both `run_single_agent.py` and `run_debate.py` flush incremental JSONL checkpoints (`logs/math500/single_agent_checkpoint.jsonl` and round-level `logs/math500/debate_checkpoint.jsonl`) and handle `SIGINT`/`SIGTERM` gracefully. If interrupted mid-run, resume from where it left off without repeating completed rounds or samples:

```powershell
python scripts/run_debate.py --num-samples 5 --resume
```

### 7. Inspect intermediate round logs

Inspect `logs/math500/debate_logs.json` (completed runs) or `logs/math500/debate_checkpoint.jsonl` (in-progress or preempted runs) to see every round's peer inputs (`--show-prompts`), proof critiques, and answer changes:

```powershell
python scripts/inspect_logs.py --log-file logs/math500/debate_logs.json --show-prompts
```

### 8. Side-by-side Single-Agent vs. Debate benchmark

Compare Single-Agent baseline against 3-Agent Multi-Round Debate on 100 `HuggingFaceH4/MATH-500` test instances (`--resume` reuses checkpoints in `logs/math500`), logging accuracy, LaTeX/SymPy answer changes, error types (`arithmetic_slip` vs. `conceptual_logic_error`), subject/level breakdowns, and execution time to `results/math500/`:

```powershell
python scripts/run_benchmark.py --num-samples 100 --max-concurrency 15 --resume
```

For Free Tier quotas (e.g., 5 RPM), pass `--rpm 4` so requests are spaced smoothly without `429` bursts:

```powershell
python scripts/run_benchmark.py --num-samples 100 --rpm 4 --resume
```
