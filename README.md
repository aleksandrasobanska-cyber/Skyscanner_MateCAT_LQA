# Skyscanner MateCAT LQA Pipeline

Automates the full MateCAT -> XLIFF download -> parsing -> glossary/TB enrichment -> dual-agent LQA (Gemini/Claude/GPT) -> scorecard export.

## Key Components
- `download_xliff.py`: Selenium downloader. Groups tracker rows by `target` and saves XLIFFs under `<xliff_download_dir>\<Language>`.
- `run_pipeline.py`: Main orchestrator for parse + TB + Agent1 + Agent2 + report export.
- `src/downloader.py`: MateCAT download helpers (Chrome profile, CDP download path, rename/unzip logic).
- `src/parser.py`: XLIFF parsing, context window generation, word-count audit.
- `src/tb.py`: Glossary language-code mapping and RapidFuzz-based TB matching.
- `src/lqa_pipeline.py`: Batching, payload creation, async LLM calls, and result stitching.
- `src/llm/`: Provider wrappers (`gemini.py`, `claude.py`, `gpt.py`) and JSON extraction helper (`common.py`).
- `src/reporting.py`: Formatted Excel LQA scorecard generator.
- `config.yaml`: All runtime paths, language mappings, model settings, batching limits.

## Setup (Windows PowerShell, end-to-end)

### 1. Prerequisites
- Windows 10 or 11
- Python 3.12 or newer
- Google Chrome installed
- MateCAT account with access to the job links in tracker
- Internet access for LLM APIs and MateCAT

Check Python from PowerShell:

```powershell
python --version
```

If `python` is not found, install Python and ensure "Add Python to PATH" is enabled.

### 2. Open PowerShell in the repo folder

```powershell
cd "C:\Users\<YourWindowsUser>\Desktop\Skyscanner_MateCAT_LQA"
```

### 3. Create a virtual environment

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, run this once in the same PowerShell window, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional quick import check:

```powershell
python -c "import pandas,lxml,selenium,rapidfuzz,openpyxl,tenacity,nest_asyncio,google.genai,yaml,openai,anthropic,pyarrow; print('Dependencies OK')"
```

## Selenium and Chrome Profile Setup (required for `download_xliff.py`)

The downloader uses Selenium with Chrome and a saved Chrome user profile for MateCAT authentication.

### 1. Choose a dedicated profile folder
Example folder:

```text
C:\Users\<YourWindowsUser>\Desktop\Skyscanner_chrome_profile
```

Set this exact path in `config.yaml` under:

```yaml
paths:
  chrome_profile_dir: "C:\\Users\\<YourWindowsUser>\\Desktop\\Skyscanner_chrome_profile"
```

### 2. Create the profile and log in to MateCAT manually once
Run Chrome with that profile folder:

```powershell
$chrome = "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe"
if (!(Test-Path $chrome)) { $chrome = "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe" }
& $chrome --user-data-dir="C:\Users\<YourWindowsUser>\Desktop\Skyscanner_chrome_profile"
```

In that Chrome window:
1. Open `https://www.matecat.com/revise2/Google_Ads_BF_ad_copy_request/en-GB-zh-TW/11765984-f54747794526` (or open one job link from the tracker).
2. Complete login manually (including SSO/MFA if needed).
3. Confirm you can open a job page.
4. Close all Chrome windows opened with this profile.

Important:
- Do not use your daily Chrome profile; use a dedicated folder.
- Before running the downloader, close any other Chrome process using that same `--user-data-dir`.
- First automation run should use `downloader.headless: false` for easier troubleshooting.

### 3. Chromedriver note
This code uses Selenium 4 (`webdriver.Chrome()`), which usually resolves the driver automatically.
If startup fails due to browser/driver mismatch, update Chrome first, then retry.

## Configure `config.yaml`

At minimum, confirm these keys:

- `paths.tracker`: absolute path to tracker Excel.
- `paths.glossary`: absolute path to multilingual glossary Excel.
- `paths.prompts_dir`: prompt templates directory (usually `prompts`).
- `paths.instructions_dir`: language guideline files root (usually `Langs_Instructions`).
- `paths.xliff_download_dir`: where downloaded XLIFFs are stored.
- `paths.chrome_profile_dir`: dedicated Chrome profile path used by Selenium.
- `paths.output_dir`: report output folder.
- `paths.checkpoints_dir`: per-language parquet checkpoints for resume behavior.

- `languages.process`: `["ALL"]` or explicit language list.
- `languages.mapping.<Language>.code`: glossary locale code (for example `de-DE`).
- `languages.mapping.<Language>.guidelines`: guideline filename in that language folder.

- `downloader.headless`: `false` is recommended for first run.
- `downloader.throttle_seconds`: wait between downloads.
- `parser.context_size`: neighbor context size.
- `tb.threshold` and `tb.min_len_fuzzy`: glossary matching sensitivity.

- `llm.agent1_model`, `llm.agent2_model`: model IDs.
- `llm.agent1_api_key` / `llm.agent2_api_key` or env-var references.
- `llm.batch_segments`, `llm.max_concurrency`, `llm.wait_seconds`: throughput and rate-limit controls.

- `prompts.system_1`, `prompts.user_agent_1`, `prompts.system_2`, `prompts.user_agent_2`: prompt file names.

## API Keys (recommended via environment variables)

For the current PowerShell session:

```powershell
$env:GOOGLE_API_KEY = "your_google_key"
$env:ANTHROPIC_API_KEY = "your_anthropic_key"
```

Then reference them in `config.yaml`:

```yaml
llm:
  agent1_api_key_env: "GOOGLE_API_KEY"
  agent2_api_key_env: "ANTHROPIC_API_KEY"
```

If you want persistent user-level env vars:

```powershell
[Environment]::SetEnvironmentVariable("GOOGLE_API_KEY", "your_google_key", "User")
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your_anthropic_key", "User")
```

Open a new PowerShell window after setting persistent vars.

## Required Input Structure (must match code expectations)

### 1. Tracker Excel requirements
The tracker file must contain these exact headers (case-sensitive):

| Column Header | Required By | Type/Format | Notes |
|---|---|---|---|
| `R2 Job ID` | Downloader + Parser | string/int | Used to locate XLIFF files and identify jobs |
| `project_id` | Downloader | string/int | Used in output filename stem |
| `target` | Downloader + Parser | string | Language name, must align with mapping/folders |
| `service_type` | Downloader | string | Used in output filename stem |
| `matecat_raw_words` | Downloader + Parser | numeric | Used in filename stem and audit comparison |
| `Link` | Downloader + Parser | URL | MateCAT job URL |
| `Job first segment` | Parser | integer | Scope start segment ID |
| `Job last segment` | Parser | integer | Scope end segment ID |

If these columns are missing or malformed, scripts will fail or skip rows.

### 2. Glossary Excel requirements
- Must contain an English source column: `en-GB` or `en-US`.
- Must contain one column per target locale code used in `languages.mapping` (for example `ar-SA`, `de-DE`, `es-MX`).
- Empty source/target glossary pairs are dropped.

### 3. Language mapping and guideline requirements
- `languages.mapping` keys must match tracker `target` values (for example `Arabic`, `German`).
- For each mapped language, guideline file should exist at:
  - `<instructions_dir>\<Language>\<guidelines_file>`
- Missing guideline file logs a warning and continues with empty guidelines.

### 4. Prompt file requirements
Prompt files must exist in `paths.prompts_dir` and match `config.yaml` names:
- Agent1: `system_1`, `user_agent_1`
- Agent2: `system_2`, `user_agent_2`

### 5. XLIFF folder/file requirements
- Expected folder layout:
  - `<xliff_download_dir>\<Language>\*.xlf` (or `.xliff`, `.sdlxliff`)
- Parser searches files per job using `*<R2 Job ID>*.xl*`.
- If files are missing for a job, that job is marked `File Not Found` in audit and skipped.

## Run Commands

Always run commands from repo root with the venv active.

### 1. Download XLIFFs from MateCAT

All configured languages:

```powershell
.\.venv\Scripts\python.exe download_xliff.py --config config.yaml
```

Single language:

```powershell
.\.venv\Scripts\python.exe download_xliff.py --config config.yaml --language Dutch
```

Multiple languages:

```powershell
.\.venv\Scripts\python.exe download_xliff.py --config config.yaml --language Arabic --language German
```

### 2. Run full LQA pipeline

All configured languages:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --config config.yaml
```

Single language:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --config config.yaml --language Arabic
```

Multiple languages:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --config config.yaml --language Arabic --language French
```

## Outputs and Resume Behavior

- Scorecards are written to:
  - `<output_dir>\<Language>\MosAIQ LQA_<Language>.xlsx`
- Checkpoints are written per language to:
  - `<checkpoints_dir>\<Language>\df_checkpoint.parquet`
- If a checkpoint exists, pipeline resumes from it instead of reprocessing from scratch.

## Concurrency and Rate Limits

- Agent1 and Agent2 run one warm-up batch first, then process remaining batches in concurrent windows.
- Main controls:
  - `llm.batch_segments`
  - `llm.max_concurrency`
  - `llm.wait_seconds`
- If you hit quota/rate-limit errors (429), reduce concurrency and/or increase wait time.

## Troubleshooting

- Downloader opens MateCAT but cannot continue:
  - Session likely expired. Log in again using the same profile path.
- Downloader says no XLIFF URL found:
  - Verify the `Link` points to a downloadable MateCAT job page.
- Parser logs folder or file missing:
  - Check `paths.xliff_download_dir` and language folder names.
- LQA fails with API key error:
  - Verify env vars or keys in `config.yaml`.
- LQA fails with provider quotas:
  - Lower `llm.max_concurrency`, increase `llm.wait_seconds`, or use another provider key/model.

