# -*- coding: utf-8 -*-
import logging
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ===================== Logging =====================
log = logging.getLogger("matecat_xliff_dl")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ===================== Helpers =====================
INVALID_FS_CHARS = r'[<>:"/\\|?*]'
ALLOWED_EXTS = (".xlf", ".xliff", ".zip", ".sdlxliff")


def safe_name(s: str, max_len: int = 120) -> str:
    if s is None or pd.isna(s):
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    s = re.sub(INVALID_FS_CHARS, "_", s)
    return s[:max_len].rstrip("._-")


def make_lang_folder(base_dir: Path, lang: str) -> Path:
    p = base_dir / (safe_name(str(lang)) or "UnknownLang")
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_stem(row: pd.Series) -> str:
    """
    Constructs the filename stem based on the NEW DataFrame columns.
    Format: <R2 Job ID>__<project_id>__<target>__<service_type>__<words>
    """
    job_id = str(row.get("R2 Job ID", "")).strip()
    proj = safe_name(row.get("project_id", ""), 80)
    lang = safe_name(row.get("target", ""), 40)
    svc = safe_name(row.get("service_type", ""), 20)
    words = str(row.get("matecat_raw_words", "")).strip()

    parts = [b for b in [job_id, proj, lang, svc, words] if b]
    return "__".join(parts) if parts else "download_unknown"


def init_chrome(user_data_dir: str, headless: bool = False) -> webdriver.Chrome:
    opts = webdriver.ChromeOptions()
    opts.add_argument(rf"--user-data-dir={user_data_dir}")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--disable-blink-features=AutomationControlled")

    if headless:
        opts.add_argument("--headless=new")

    opts.add_experimental_option(
        "prefs",
        {
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
            "safebrowsing.disable_download_protection": True,
        },
    )

    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(180)
    return driver


def check_is_logged_in(driver: webdriver.Chrome) -> bool:
    """Checks if we have been redirected to the login page."""
    try:
        if "login" in driver.current_url.lower():
            return False
        if driver.find_elements(By.NAME, "password") or driver.find_elements(By.CSS_SELECTOR, ".login-wrapper"):
            return False
        return True
    except Exception:
        return True  # Assume yes if check fails to avoid false negatives


# ===================== Download Logic =====================
def _snapshot_files(folder: Path) -> Dict[str, tuple]:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in folder.glob("*")}


def wait_for_download_complete(
    folder: Path, start_snapshot: Dict[str, tuple], start_timeout: int = 30, finish_timeout: int = 300
) -> Path:
    t0 = time.time()
    started = False

    # 1. Wait for start
    while time.time() - t0 < start_timeout:
        if list(folder.glob("*.crdownload")):
            started = True
            break
        current_files = sorted(
            [p for p in folder.glob("*") if p.suffix.lower() in ALLOWED_EXTS],
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        if current_files:
            newest = current_files[0]
            if newest.name not in start_snapshot or (newest.stat().st_size != start_snapshot[newest.name][0]):
                started = True
                break
        time.sleep(0.5)

    if not started:
        raise TimeoutError("Download failed to start within timeout.")

    # 2. Wait for finish (no .crdownload)
    t1 = time.time()
    while time.time() - t1 < finish_timeout:
        if not list(folder.glob("*.crdownload")):
            candidates = sorted(
                [p for p in folder.glob("*") if p.suffix.lower() in ALLOWED_EXTS],
                key=lambda p: p.stat().st_mtime_ns,
                reverse=True,
            )
            if candidates:
                return candidates[0]
        time.sleep(0.5)

    raise TimeoutError("Download failed to finish (crdownload persisted).")


def get_xliff_href(driver: webdriver.Chrome) -> str:
    """Resiliently finds the XLIFF download link."""
    from urllib.parse import urljoin

    wait = WebDriverWait(driver, 20)
    root = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#action-download")))

    href = ""
    try:
        elems = driver.find_elements(By.CSS_SELECTOR, "#action-download a.sdlxliff")
        if elems and elems[0].get_attribute("href"):
            return urljoin(driver.current_url, elems[0].get_attribute("href"))
    except Exception:
        pass

    try:
        driver.execute_script("arguments[0].click();", root)
        time.sleep(1)
        elems = driver.find_elements(By.CSS_SELECTOR, "ul#previewDropdown li[data-value='xlif'] a")
        if elems and elems[0].get_attribute("href"):
            return urljoin(driver.current_url, elems[0].get_attribute("href"))
    except Exception:
        pass

    if not href:
        elems = driver.find_elements(By.XPATH, "//a[contains(@href, 'format=sdlxliff')]")
        if elems:
            return urljoin(driver.current_url, elems[0].get_attribute("href"))

    raise RuntimeError("Could not find XLIFF download URL in DOM.")


def process_download(dl_path: Path, dest_dir: Path, stem: str) -> List[str]:
    """
    Handles renaming.
    If .zip -> extracts, renames parts, deletes zip.
    If .xlf -> renames.
    Returns list of saved filenames.
    """
    saved_files = []

    if dl_path.suffix.lower() == ".zip":
        log.info(f"Detected ZIP file: {dl_path.name}. Extracting...")
        extract_dir = dest_dir / f"temp_{int(time.time())}"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(dl_path, "r") as zf:
                zf.extractall(extract_dir)

            extracted_files = [f for f in extract_dir.rglob("*") if f.is_file() and f.suffix.lower() != ".zip"]
            if not extracted_files:
                log.warning("Zip file was empty or contained no valid files.")
                return []

            for i, fpath in enumerate(extracted_files):
                suffix_part = f"_part{i+1}" if len(extracted_files) > 1 else ""
                new_name = f"{stem}{suffix_part}{fpath.suffix}"
                final_path = dest_dir / new_name
                shutil.move(str(fpath), str(final_path))
                saved_files.append(new_name)

        finally:
            shutil.rmtree(extract_dir, ignore_errors=True)
            dl_path.unlink(missing_ok=True)

    else:
        final_path = dest_dir / f"{stem}{dl_path.suffix}"
        if final_path.exists():
            final_path.unlink()
        shutil.move(str(dl_path), str(final_path))
        saved_files.append(final_path.name)

    return saved_files


# ===================== Main Execution Function =====================
def download_matecat_files(
    tracker_df: pd.DataFrame,
    base_download_dir: str,
    user_data_dir: str,
    headless: bool = False,
    throttle: int = 1,
) -> Dict[str, Dict[str, int]]:
    req_cols = ["R2 Job ID", "project_id", "target", "service_type", "matecat_raw_words", "Link"]
    missing = [c for c in req_cols if c not in tracker_df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    base_path = Path(base_download_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    log.info("Initializing Chrome Driver...")
    driver = init_chrome(user_data_dir, headless)

    summary: Dict[str, Dict[str, int]] = {}

    try:
        for lang, group in tracker_df.groupby("target", dropna=False):
            lang_str = str(lang) if pd.notna(lang) else "Unknown"
            lang_dir = make_lang_folder(base_path, lang_str)

            log.info(f"--- Processing Language: {lang_str} ({len(group)} jobs) ---")
            try:
                driver.execute_cdp_cmd(
                    "Page.setDownloadBehavior",
                    {"behavior": "allow", "downloadPath": str(lang_dir.resolve())},
                )
            except Exception as e:
                log.error(f"Failed to set download path via CDP: {e}")
                continue

            stats = {"ok": 0, "failed": 0, "skipped": 0}

            for idx, row in group.iterrows():
                r2_id = str(row.get("R2 Job ID", ""))
                link = str(row.get("Link", ""))
                stem = build_stem(row)

                existing = list(lang_dir.glob(f"{stem}*.xl*"))
                if existing:
                    log.info(f"[{lang_str}] Job {r2_id} exists ({existing[0].name}). Skipping.")
                    stats["skipped"] += 1
                    continue

                if not link or "http" not in link:
                    log.warning(f"[{lang_str}] Job {r2_id}: Invalid Link.")
                    stats["failed"] += 1
                    continue

                try:
                    log.info(f"[{lang_str}] Job {r2_id}: Visiting link...")
                    driver.get(link)

                    if not check_is_logged_in(driver):
                        log.warning("SESSION EXPIRED. Please log in manually in the open window.")
                        log.warning("Script is paused for 60 seconds...")
                        time.sleep(60)
                        if not check_is_logged_in(driver):
                            raise PermissionError("User not logged in.")

                    xliff_url = get_xliff_href(driver)
                    before_snap = _snapshot_files(lang_dir)
                    driver.execute_script("window.open(arguments[0], '_self');", xliff_url)
                    dl_file = wait_for_download_complete(lang_dir, before_snap)
                    saved_names = process_download(dl_file, lang_dir, stem)
                    log.info(f"[{lang_str}] Job {r2_id}: Success -> {saved_names}")
                    stats["ok"] += 1

                    if throttle:
                        time.sleep(throttle)

                except Exception as e:
                    log.error(f"[{lang_str}] Job {r2_id}: FAILED -> {e}")
                    stats["failed"] += 1

            summary[lang_str] = stats
            log.info(f"Completed {lang_str}: {stats}")

    except KeyboardInterrupt:
        log.warning("Script stopped by user.")
    finally:
        driver.quit()
        log.info("Driver closed.")

    return summary
