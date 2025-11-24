import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from lxml import etree as ET

# ===================== 1. Setup & Helpers =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("MateCatParser")

NS = {
    "ns": "urn:oasis:names:tc:xliff:document:1.2",
    "mtc": "https://www.matecat.com",
}


def clean_tag_text(node: ET._Element) -> str:
    """
    Extracts text AND internal tags (like <b>, <ph>) from a node as a raw string.
    Does NOT strip distinct tags, but preserves them as text.
    """
    if node is None:
        return ""

    try:
        xml_bytes = ET.tostring(node, encoding="utf-8", method="xml", with_tail=False)
        xml_str = xml_bytes.decode("utf-8")
        xml_str = re.sub(r"^<[^>]+>", "", xml_str)
        xml_str = re.sub(r"</[^>]+>$", "", xml_str)
        return xml_str
    except Exception:
        return "".join(node.itertext())


def count_words(text: str) -> int:
    """Simple whitespace-based word count to match MateCat's rough logic."""
    if not text:
        return 0
    clean = re.sub(r"<[^>]+>", " ", text)
    return len(re.findall(r"\S+", clean))


def get_segment_note(tu: ET._Element) -> str:
    """
    Extracts the 'Context' note based on priority.
    """
    notes = tu.findall(".//ns:note", namespaces=NS)
    if not notes:
        notes = tu.findall(".//note")

    for n in notes:
        txt = n.text or ""
        if "translation_context" in txt:
            return txt.replace("translation_context|�|", "").strip()
        if n.get("from") == "id_content":
            return txt.strip()

    resname = tu.get("resname")
    if resname and "Translation!" in resname:
        return resname

    return ""


# ===================== 2. Core Parsing Logic =====================
def parse_xliff_file(file_path: Path, job_id: str) -> List[Dict]:
    """
    Parses a single XLIFF file.
    Uses MateCat's internal word count when available to minimize mismatches.
    """
    try:
        parser = ET.XMLParser(recover=True, remove_blank_text=False)
        tree = ET.parse(str(file_path), parser)
        root = tree.getroot()
    except Exception as e:
        log.error(f"Failed to parse XML {file_path.name}: {e}")
        return []

    trans_units = root.findall(".//ns:trans-unit", namespaces=NS)
    if not trans_units:
        trans_units = root.findall(".//trans-unit")

    segments = []

    for tu in trans_units:
        if tu.get("translate") == "no":
            continue

        help_id_raw = tu.get("help-id")
        max_width = tu.get("maxwidth")

        try:
            main_seg_id = int(help_id_raw) if help_id_raw else 0
        except ValueError:
            main_seg_id = 0

        note_context = get_segment_note(tu)

        matecat_count = None
        count_group = tu.find(".//ns:count-group", namespaces=NS)
        if count_group is not None:
            raw_node = count_group.find(".//ns:count[@count-type='x-matecat-raw']", namespaces=NS)
            if raw_node is not None and raw_node.text:
                try:
                    matecat_count = float(raw_node.text)
                except ValueError:
                    pass

        seg_source = tu.find(".//ns:seg-source", namespaces=NS)
        mrk_src_list = seg_source.findall(".//ns:mrk[@mtype='seg']", namespaces=NS) if seg_source is not None else []

        target_node = tu.find(".//ns:target", namespaces=NS)
        if target_node is None:
            target_node = tu.find(".//target")
        mrk_tgt_list = target_node.findall(".//ns:mrk[@mtype='seg']", namespaces=NS) if target_node is not None else []

        if mrk_src_list:
            for i, mrk_src in enumerate(mrk_src_list):
                current_seg_id = main_seg_id + i
                mrk_tgt = mrk_tgt_list[i] if i < len(mrk_tgt_list) else None

                src_txt = clean_tag_text(mrk_src)
                tgt_txt = clean_tag_text(mrk_tgt) if mrk_tgt is not None else ""
                words = count_words(src_txt)

                segments.append(
                    {
                        "Job_ID": job_id,
                        "Main_Segment": main_seg_id,
                        "Segment_ID": current_seg_id,
                        "Source": src_txt,
                        "Target": tgt_txt,
                        "Words": words,
                        "Character_Limit": max_width,
                        "Note": note_context,
                    }
                )
        else:
            source_node = tu.find(".//ns:source", namespaces=NS)
            if source_node is None:
                source_node = tu.find(".//source")

            src_txt = clean_tag_text(source_node)
            tgt_txt = clean_tag_text(target_node)
            final_words = matecat_count if matecat_count is not None else count_words(src_txt)

            segments.append(
                {
                    "Job_ID": job_id,
                    "Main_Segment": main_seg_id,
                    "Segment_ID": main_seg_id,
                    "Source": src_txt,
                    "Target": tgt_txt,
                    "Words": final_words,
                    "Character_Limit": max_width,
                    "Note": note_context,
                }
            )

    return segments


# ===================== 3. Context & Aggregation Logic =====================
def add_context_window(all_segments: List[Dict], window_size: int = 2) -> List[Dict]:
    """
    Adds 'Context_Before' and 'Context_After' columns.
    """
    if not all_segments:
        return []

    main_seg_map: Dict[int, List[Dict]] = {}
    main_ids_order: List[int] = []
    seen_main_ids = set()

    for seg in all_segments:
        m_id = seg["Main_Segment"]
        if m_id not in main_seg_map:
            main_seg_map[m_id] = []
        main_seg_map[m_id].append(seg)
        if m_id not in seen_main_ids:
            main_ids_order.append(m_id)
            seen_main_ids.add(m_id)

    main_text_map: Dict[int, str] = {}
    for m_id, segs in main_seg_map.items():
        full_text = " ".join([s["Source"] for s in segs])
        main_text_map[m_id] = full_text

    total_mains = len(main_ids_order)
    before_map: Dict[int, str] = {}
    after_map: Dict[int, str] = {}

    for i, m_id in enumerate(main_ids_order):
        start_b = max(0, i - window_size)
        end_b = i
        prev_ids = main_ids_order[start_b:end_b]
        before_map[m_id] = " || ".join([main_text_map[nid] for nid in prev_ids])

        start_a = i + 1
        end_a = min(total_mains, i + 1 + window_size)
        next_ids = main_ids_order[start_a:end_a]
        after_map[m_id] = " || ".join([main_text_map[nid] for nid in next_ids])

    for seg in all_segments:
        m_id = seg["Main_Segment"]
        seg["Context_Before"] = before_map.get(m_id, "")
        seg["Context_After"] = after_map.get(m_id, "")

    return all_segments


# ===================== 4. Processing & Orchestration =====================
def find_files_for_job(lang_dir: Path, job_id: str) -> List[Path]:
    """Finds ALL .xlf files for the job ID, sorted by name."""
    files = list(lang_dir.glob(f"*{job_id}*.xl*"))
    files.sort(key=lambda x: x.name)
    return files


def process_tracker(tracker_df: pd.DataFrame, base_xlf_dir: str, context_size: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main orchestration function.
    Takes `context_size` to control how many segments before/after are included.
    """
    base_path = Path(base_xlf_dir)
    all_rows: List[Dict] = []
    audit_rows: List[Dict] = []

    unique_langs = tracker_df["target"].unique()

    for lang in unique_langs:
        if pd.isna(lang):
            continue

        lang_dir = base_path / str(lang)
        if not lang_dir.exists():
            found = False
            for d in base_path.iterdir():
                if d.is_dir() and d.name.lower() == str(lang).lower():
                    lang_dir = d
                    found = True
                    break
            if not found:
                log.warning(f"Folder not found for language: {lang}")
                continue

        log.info(f"Processing Language: {lang}")
        lang_group = tracker_df[tracker_df["target"] == lang]

        for idx, row in lang_group.iterrows():
            job_id = str(row["R2 Job ID"]).strip()

            try:
                scope_start = int(row["Job first segment"])
                scope_last = int(row["Job last segment"])
                target_words = float(row["matecat_raw_words"])
            except (ValueError, TypeError):
                log.warning(f"Job {job_id}: Bad Tracker Data. Skipping.")
                audit_rows.append({"Job_ID": job_id, "Status": "Tracker Data Error"})
                continue

            base_link = str(row["Link"]).strip()
            files = find_files_for_job(lang_dir, job_id)
            if not files:
                log.warning(f"Job {job_id}: No XLIFF files found.")
                audit_rows.append({"Job_ID": job_id, "Status": "File Not Found"})
                continue

            merged_segments: List[Dict] = []
            for f in files:
                file_segs = parse_xliff_file(f, job_id)
                merged_segments.extend(file_segs)

            if not merged_segments:
                audit_rows.append({"Job_ID": job_id, "Status": "Empty Parse"})
                continue

            unique_seg_map = {s["Segment_ID"]: s for s in merged_segments}
            sorted_segments = sorted(unique_seg_map.values(), key=lambda x: x["Segment_ID"])
            full_context_segments = add_context_window(sorted_segments, window_size=context_size)

            job_word_sum = 0
            count_kept = 0

            for seg in full_context_segments:
                s_id = seg["Segment_ID"]
                if scope_start <= s_id <= scope_last:
                    full_link = f"{base_link}#{s_id}"

                    final_row = {
                        "Job_ID": job_id,
                        "Language": lang,
                        "Segment_ID": s_id,
                        "Main_Segment": seg["Main_Segment"],
                        "Source": seg["Source"],
                        "Target": seg["Target"],
                        "Words": seg["Words"],
                        "Character_Limit": seg["Character_Limit"],
                        "Note": seg["Note"],
                        "Context_Before": seg["Context_Before"],
                        "Context_After": seg["Context_After"],
                        "Link": full_link,
                    }
                    all_rows.append(final_row)
                    job_word_sum += seg["Words"]
                    count_kept += 1

            diff = job_word_sum - target_words
            status = "Match" if abs(diff) < 1 else "Mismatch"

            audit_rows.append(
                {
                    "Job_ID": job_id,
                    "Language": lang,
                    "Status": status,
                    "Tracker_Words": target_words,
                    "Parsed_Words": job_word_sum,
                    "Diff": diff,
                    "Segments_In_Scope": count_kept,
                }
            )

    return pd.DataFrame(all_rows), pd.DataFrame(audit_rows)
