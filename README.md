# Skyscanner MateCAT LQA Pipeline

Automates the full MateCAT → XLIFF download → parsing → glossary/TB enrichment → dual-agent LQA (Gemini/Claude/GPT) → scorecard export. Config-driven so you can run all tracker languages or a specific language in one go.

## Key Components
- `download_xliff.py`: Selenium downloader. Groups tracker rows by `target` and saves XLIFFs under `XLIFF_Downloads/<lang>`.
- `run_pipeline.py`: Orchestrates parsing, TB matching, Agent1/Agent2 LQA, and scorecard generation.
- `src/downloader.py`: MateCAT download helpers (Chrome profile, CDP download path, renaming/unzipping).
- `src/parser.py`: XLIFF parsing, context windowing, audit of word counts.
- `src/tb.py`: Glossary language-code mapping and RapidFuzz TB matching.
- `src/llm/`: Provider-specific async callers (`gemini.py`, `claude.py`, `gpt.py`) plus shared JSON extraction (`common.py`).
- `src/lqa_pipeline.py`: Batching, payload construction (context + TB), Agent1/Agent2 execution, and result stitching.
- `src/reporting.py`: Builds formatted Excel scorecards per language.
- `config.yaml`: All paths, language mappings, LLM settings, batching limits.

## Setup
1. Python 3.12+.  
2. Create venv and install deps:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Ensure Chrome + chromedriver compatible with your Chrome profile path in `config.yaml`.

## Configuration (`config.yaml`)
- `paths`: tracker Excel, glossary Excel, prompts dir, instructions dir, XLIFF download dir, Chrome profile dir, output dir.
- `languages`: set `process: ["ALL"]` or a list; `mapping` holds glossary code + guidelines filename per language.
- `downloader`: `headless`, `throttle_seconds`.
- `parser`: `context_size`.
- `tb`: `threshold`, `min_len_fuzzy`.
- `llm`: default/agent-specific models and API keys (`api_key` / `agent1_api_key` / `agent2_api_key` or env vars), `batch_segments`, `max_concurrency`, `wait_seconds`, `source_lang_label`.
- `prompts`: filenames for system/user prompts (Agent1/Agent2).

## Running
- Download XLIFFs (all or specific language):
  ```bash
  .venv\Scripts\python download_xliff.py --config config.yaml
  .venv\Scripts\python download_xliff.py --config config.yaml --language Dutch
  ```
- Full LQA pipeline:
  ```bash
  .venv\Scripts\python run_pipeline.py --config config.yaml
  .venv\Scripts\python run_pipeline.py --config config.yaml --language Arabic
  ```
- Outputs: Excel scorecards per language under `Output/<Language>/MosAIQ LQA_<Language>.xlsx`.

## Concurrency & Warm-up
- Agent1/Agent2 each run batch #1 (warm-up), then pause `wait_seconds`, then process remaining batches in windows of `max_concurrency`, pausing between windows.
- Tune `batch_segments` (segments per call) and `max_concurrency` to balance throughput vs. rate limits.

## Notes & Troubleshooting
- Use provider-specific models/keys (Gemini/Claude/GPT) via `llm.agent1_model`, `llm.agent2_model`, and corresponding keys/env vars.
- If you hit 429/quota errors, lower `max_concurrency`, raise `batch_segments`, or switch provider/key with available quota.
- Guidelines are loaded from `Langs_Instructions/<Language>/<file>`; ensure filenames match `languages.mapping`.
- Glossary expects an English column (`en-GB`/`en-US`) plus target locale code per mapping.
