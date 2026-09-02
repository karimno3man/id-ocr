"""Split Egyptian governorate (المحافظة) from Add2 address OCR text."""

from __future__ import annotations

import re
import unicodedata

# Canonical Arabic name -> OCR aliases (longest match wins at end of Add2).
GOVERNORATE_ALIASES: dict[str, list[str]] = {
    "القاهرة": ["القاهرة", "القاهره", "قاهرة", "قاهره"],
    "الإسكندرية": ["الإسكندرية", "الاسكندرية", "الاسكندريه", "الإسكندريه", "اسكندرية", "اسكندريه"],
    "بورسعيد": ["بورسعيد", "بور سعيد"],
    "السويس": ["السويس", "سويس"],
    "دمياط": ["دمياط"],
    "الدقهلية": ["الدقهلية", "الدقهليه", "دقهلية", "دقهليه"],
    "الشرقية": ["الشرقية", "الشرقيه", "شرقية", "شرقيه"],
    "القليوبية": ["القليوبية", "القليوبيه", "قليوبية", "قليوبيه"],
    "كفر الشيخ": ["كفر الشيخ", "كفرالشيخ"],
    "الغربية": ["الغربية", "الغربيه", "غربية", "غربيه"],
    "المنوفية": ["المنوفية", "المنوفيه", "منوفية", "منوفيه"],
    "البحيرة": ["البحيرة", "البحيره", "بحيرة", "بحيره"],
    "الإسماعيلية": ["الإسماعيلية", "الاسماعيلية", "الاسماعيليه", "اسماعيلية", "اسماعيليه"],
    "الجيزة": ["الجيزة", "الجيزه", "جيزة", "جيزه"],
    "بني سويف": ["بني سويف", "بنى سويف", "بني السويف", "بنى السويف"],
    "الفيوم": ["الفيوم", "فيوم"],
    "المنيا": ["المنيا", "المنيه", "منيا", "منيه"],
    "أسيوط": ["أسيوط", "اسيوط"],
    "سوهاج": ["سوهاج"],
    "قنا": ["قنا"],
    "أسوان": ["أسوان", "اسوان"],
    "الأقصر": ["الأقصر", "الاقصر", "أقصر", "اقصر"],
    "البحر الأحمر": ["البحر الأحمر", "البحر الاحمر", "بحر الاحمر", "البحراحمر"],
    "الوادي الجديد": ["الوادي الجديد", "الوادى الجديد", "وادي الجديد", "وادى الجديد"],
    "مطروح": ["مطروح"],
    "شمال سيناء": ["شمال سيناء", "شمال سينا", "شمال سينا"],
    "جنوب سيناء": ["جنوب سيناء", "جنوب سينا", "جنوب سينا"],
}

_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_SEPARATORS = re.compile(r"[\s\-–—،,]+")

_ALIAS_ENTRIES: list[tuple[str, str]] = []


def _build_alias_index() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for canonical, aliases in GOVERNORATE_ALIASES.items():
        for alias in aliases:
            entries.append((canonical, alias))
    entries.sort(key=lambda item: len(item[1]), reverse=True)
    return entries


_ALIAS_ENTRIES = _build_alias_index()


def normalize_arabic(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _strip_trailing_separators(text: str) -> str:
    return re.sub(r"^[\s\-–—،,]+|[\s\-–—،,]+$", "", text).strip()


def split_governorate(add2_text: str) -> tuple[str, str]:
    """Return (address_without_governorate, governorate_canonical).

    If no governorate is detected at the end of Add2, returns (original_text, "").
    """
    raw = (add2_text or "").strip()
    if not raw:
        return "", ""

    normalized_full = normalize_arabic(raw)

    for canonical, alias in _ALIAS_ENTRIES:
        normalized_alias = normalize_arabic(alias)
        if not normalized_alias:
            continue
        if not normalized_full.endswith(normalized_alias):
            continue

        prefix_len = len(normalized_full) - len(normalized_alias)
        prefix_normalized = normalized_full[:prefix_len] if prefix_len > 0 else ""

        if prefix_normalized and not prefix_normalized.endswith(" "):
            # Require word boundary: governorate must follow space or separator.
            last_char = prefix_normalized[-1]
            if last_char not in " -–—،,":
                continue

        # Map split position back to original string using normalized lengths.
        remainder = _extract_remainder(raw, alias)
        remainder = _strip_trailing_separators(remainder)
        return remainder, canonical

    return raw, ""


def _extract_remainder(original: str, matched_alias: str) -> str:
    """Remove the matched governorate suffix from the original string."""
    original_norm = normalize_arabic(original)
    alias_norm = normalize_arabic(matched_alias)
    if not original_norm.endswith(alias_norm):
        return original

    # Walk backwards through original to find where the alias starts.
    orig_chars = list(original)
    norm_pos = len(original_norm)
    orig_pos = len(orig_chars)

    target_norm_len = len(alias_norm)
    consumed_norm = 0
    while orig_pos > 0 and consumed_norm < target_norm_len:
        orig_pos -= 1
        char_norm = normalize_arabic(orig_chars[orig_pos])
        if char_norm:
            consumed_norm += len(char_norm)
        elif orig_chars[orig_pos].isspace():
            consumed_norm += 1

    remainder = original[:orig_pos]
    remainder = _strip_trailing_separators(remainder)
    return remainder


def apply_governorate_split(
    fields: dict[str, str],
    meta: dict[str, object],
) -> tuple[dict[str, str], dict[str, object]]:
    add2_raw = fields.get("Add2", "") or ""
    remainder, governorate = split_governorate(add2_raw)

    if not governorate:
        fields["Governorate"] = ""
        meta["Governorate"] = {"source": None, "det_conf": None}
        return fields, meta

    fields["Add2"] = remainder
    fields["Governorate"] = governorate

    add2_meta = dict(meta.get("Add2") or {})
    gov_meta: dict[str, object] = {
        "source": add2_meta.get("source"),
        "det_conf": add2_meta.get("det_conf"),
        "ocr_conf": add2_meta.get("ocr_conf"),
        "split_from": "Add2",
        "original_add2": add2_raw,
    }
    meta["Governorate"] = gov_meta
    return fields, meta
