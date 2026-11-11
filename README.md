# gsm8k-debate

Async multi-agent debate benchmark on GSM8K using Python 3.10+ and the `google-genai` SDK.

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
python -c "import google.genai; print('Ready')"
```

Run the smoke and unit test suite:

```powershell
pytest
```

### 4. Run single-agent GSM8K baseline

Evaluate Gemini Flash on a small subset of the `openai/gsm8k` test split (default: 15 samples):

```powershell
python scripts/run_single_agent.py --num-samples 15
```

If you are on the Gemini API Free Tier (e.g., 5 requests/minute), pass `--rpm 4` to pace requests cleanly alongside automatic `429`/`503` retries:

```powershell
python scripts/run_single_agent.py --num-samples 10 --rpm 4
```

### 5. Run multi-agent debate (Du et al.)

Run the 3-agent, 3-round asynchronous debate loop and inspect the intermediate logs (default saves full JSON traces to `logs/debate_logs.json` and prints a preview trace):

```powershell
python scripts/run_debate.py --num-samples 5 --show-logs 2
```

For Free Tier quotas (`--rpm 4`), start with 1–3 samples (3 agents × 3 rounds = 9 API calls per sample):

```powershell
python scripts/run_debate.py --num-samples 2 --rpm 4 --show-logs 2
```

### 6. Preemption & Resume Support

Both `run_single_agent.py` and `run_debate.py` flush incremental JSONL checkpoints (`logs/single_agent_checkpoint.jsonl` and round-level `logs/debate_checkpoint.jsonl`) and handle `SIGINT`/`SIGTERM` gracefully. If interrupted mid-run, resume from where it left off without repeating completed rounds or samples:

```powershell
python scripts/run_debate.py --num-samples 5 --rpm 4 --resume
```

