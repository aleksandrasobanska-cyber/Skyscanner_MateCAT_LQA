import argparse
import logging
from pathlib import Path
from typing import List

from src.config_loader import load_config
from src.downloader import download_matecat_files
from src.utils import read_excel_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _filter_languages(df, languages: List[str]):
    if not languages or "ALL" in languages:
        return df
    filtered = df[df["target"].isin(languages)]
    if filtered.empty:
        raise ValueError(f"No rows found for languages: {languages}")
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Download MateCat XLIFF files per tracker.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Target language(s) to download (can be passed multiple times). Defaults to config languages or ALL.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    dl_cfg = cfg.get("downloader", {})
    langs_cfg = cfg.get("languages", {}).get("process", ["ALL"])

    tracker_path = Path(paths.get("tracker", "")).expanduser()
    base_download_dir = Path(paths.get("xliff_download_dir", "XLIFF_Downloads")).expanduser()
    user_data_dir = Path(paths.get("chrome_profile_dir", "")).expanduser()

    tracker_df = read_excel_file(tracker_path)
    target_languages = args.languages or langs_cfg
    tracker_df = _filter_languages(tracker_df, target_languages)

    summary = download_matecat_files(
        tracker_df=tracker_df,
        base_download_dir=str(base_download_dir),
        user_data_dir=str(user_data_dir),
        headless=bool(dl_cfg.get("headless", False)),
        throttle=int(dl_cfg.get("throttle_seconds", 1)),
    )

    print("\n=== Final Summary ===")
    for lang, stats in summary.items():
        print(f"{lang}: {stats}")


if __name__ == "__main__":
    main()
