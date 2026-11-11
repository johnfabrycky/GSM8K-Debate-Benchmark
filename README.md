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

Run the smoke test suite:

```powershell
pytest
```

