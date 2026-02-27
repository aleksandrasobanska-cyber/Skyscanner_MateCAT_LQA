import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "[AA] 2026 Rounds.xlsx"
DEFAULT_OUTPUT = "Chillistore - February Round - Project IDs & Segment IDs.xlsx"
DEFAULT_PROJECTS_SHEET = "Projects"
DEFAULT_R2S_SHEET = "R2s"
DEFAULT_OUTPUT_SHEET = "February - Project IDs"

REQUIRED_PROJECTS_COLUMNS = [
    "project_id",
    "second pass ID",
    "revised_job_id",
    "service_type",
    "target",
    "Matecat job ID",
    "Matecat job password",
    "Job first segment",
    "Job last segment",
    "Matecat link",
]

REQUIRED_R2S_COLUMNS = [
    "id",
    "job_type",
    "matecat_raw_words",
]

CANONICAL_COLUMNS = [
    "R2 Job ID",
    "Revised (R1) job ID",
    "project_id",
    "job_type",
    "service_type",
    "target",
    "matecat_raw_words",
    "Matecat job ID",
    "Matecat job password",
    "Job first segment",
    "Job last segment",
    "Link",
]


def _missing_columns(df: pd.DataFrame, required: list[str]) -> list[str]:
    return [col for col in required if col not in df.columns]


def _validate_sheets(source_path: Path, projects_sheet: str, r2s_sheet: str) -> None:
    xls = pd.ExcelFile(source_path)
    required_sheets = [projects_sheet, r2s_sheet]
    missing_sheets = [sheet for sheet in required_sheets if sheet not in xls.sheet_names]
    if missing_sheets:
        raise ValueError(
            f"Missing required sheet(s): {missing_sheets}. "
            f"Available sheets: {xls.sheet_names}"
        )


def _validate_join_keys(projects_df: pd.DataFrame, r2s_df: pd.DataFrame) -> None:
    duplicate_projects = projects_df["_join_id"].duplicated()
    if duplicate_projects.any():
        dup_vals = projects_df.loc[duplicate_projects, "_join_id"].head(10).tolist()
        raise ValueError(
            f"Projects.second pass ID contains duplicates. Examples: {dup_vals}"
        )

    duplicate_r2s = r2s_df["_join_id"].duplicated()
    if duplicate_r2s.any():
        dup_vals = r2s_df.loc[duplicate_r2s, "_join_id"].head(10).tolist()
        raise ValueError(f"R2s.id contains duplicates. Examples: {dup_vals}")


def _validate_output(df: pd.DataFrame) -> None:
    if list(df.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            "Canonical output columns mismatch. "
            f"Expected: {CANONICAL_COLUMNS}, got: {list(df.columns)}"
        )

    missing_values = df[CANONICAL_COLUMNS].isna().sum()
    bad_nulls = {col: int(count) for col, count in missing_values.items() if count > 0}
    if bad_nulls:
        raise ValueError(f"Null values found in required canonical fields: {bad_nulls}")

    duplicate_r2_job = df["R2 Job ID"].astype(str).duplicated()
    if duplicate_r2_job.any():
        dup_vals = df.loc[duplicate_r2_job, "R2 Job ID"].head(10).tolist()
        raise ValueError(f"Duplicate R2 Job ID values found: {dup_vals}")

    invalid_segment_ranges = df["Job first segment"] > df["Job last segment"]
    if invalid_segment_ranges.any():
        raise ValueError(
            f"Found {int(invalid_segment_ranges.sum())} rows where "
            "'Job first segment' > 'Job last segment'."
        )

    link_values = df["Link"].astype(str).str.strip()
    invalid_links = (~link_values.str.startswith("http")) | (link_values == "")
    if invalid_links.any():
        raise ValueError(
            f"Found {int(invalid_links.sum())} rows with non URL-like Link values."
        )


def build_canonical_tracker(
    source_path: Path, projects_sheet: str, r2s_sheet: str
) -> pd.DataFrame:
    _validate_sheets(source_path, projects_sheet, r2s_sheet)

    projects_df = pd.read_excel(source_path, sheet_name=projects_sheet)
    r2s_df = pd.read_excel(source_path, sheet_name=r2s_sheet)

    missing_projects_cols = _missing_columns(projects_df, REQUIRED_PROJECTS_COLUMNS)
    if missing_projects_cols:
        raise ValueError(
            f"Projects sheet missing required columns: {missing_projects_cols}"
        )

    missing_r2s_cols = _missing_columns(r2s_df, REQUIRED_R2S_COLUMNS)
    if missing_r2s_cols:
        raise ValueError(f"R2s sheet missing required columns: {missing_r2s_cols}")

    projects_work = projects_df.copy()
    projects_work["_row_order"] = range(len(projects_work))
    projects_work["_join_id"] = projects_work["second pass ID"].astype(str).str.strip()

    r2s_work = r2s_df[REQUIRED_R2S_COLUMNS].copy()
    r2s_work["_join_id"] = r2s_work["id"].astype(str).str.strip()

    _validate_join_keys(projects_work, r2s_work)

    merged = projects_work.merge(
        r2s_work.drop(columns=["id"]),
        on="_join_id",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(projects_work):
        raise ValueError(
            "Row count changed after merge. "
            f"Projects rows={len(projects_work)}, merged rows={len(merged)}"
        )

    canonical_df = pd.DataFrame(
        {
            "R2 Job ID": merged["second pass ID"],
            "Revised (R1) job ID": merged["revised_job_id"],
            "project_id": merged["project_id"],
            "job_type": merged["job_type"],
            "service_type": merged["service_type"],
            "target": merged["target"],
            "matecat_raw_words": merged["matecat_raw_words"],
            "Matecat job ID": merged["Matecat job ID"],
            "Matecat job password": merged["Matecat job password"],
            "Job first segment": merged["Job first segment"],
            "Job last segment": merged["Job last segment"],
            "Link": merged["Matecat link"],
        }
    )

    _validate_output(canonical_df)
    return canonical_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert February client tracker workbook to canonical pipeline format."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Source workbook path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output workbook path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--projects-sheet",
        default=DEFAULT_PROJECTS_SHEET,
        help=f"Projects sheet name (default: {DEFAULT_PROJECTS_SHEET})",
    )
    parser.add_argument(
        "--r2s-sheet",
        default=DEFAULT_R2S_SHEET,
        help=f"R2s sheet name (default: {DEFAULT_R2S_SHEET})",
    )
    parser.add_argument(
        "--output-sheet",
        default=DEFAULT_OUTPUT_SHEET,
        help=f"Output sheet name (default: {DEFAULT_OUTPUT_SHEET})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()

    if not source_path.exists():
        raise FileNotFoundError(f"Source workbook not found: {source_path}")

    canonical_df = build_canonical_tracker(
        source_path=source_path,
        projects_sheet=args.projects_sheet,
        r2s_sheet=args.r2s_sheet,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        canonical_df.to_excel(writer, sheet_name=args.output_sheet, index=False)

    print(f"Conversion completed successfully.")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {len(canonical_df)}")
    print(f"Columns: {len(canonical_df.columns)}")


if __name__ == "__main__":
    main()
