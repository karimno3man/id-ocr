"""Parse a 14-digit Egyptian national ID number.

Digit layout:
  1     century (2 = 1900s, 3 = 2000s)
  2-7   birth date YYMMDD
  8-9   governorate code
  10-13 serial (digit 13 odd = male, even = female)
  14    check digit (MOI algorithm is unpublished; not verified)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date

EASTERN_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

GOVERNORATES: dict[str, str] = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Sharqia",
    "14": "Qalyubia",
    "15": "Kafr El Sheikh",
    "16": "Gharbia",
    "17": "Monufia",
    "18": "Beheira",
    "19": "Ismailia",
    "21": "Giza",
    "22": "Beni Suef",
    "23": "Fayoum",
    "24": "Minya",
    "25": "Asyut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
    "88": "Born abroad",
}

NID_PATTERN = re.compile(r"\d{14}")


def to_western_digits(text: str) -> str:
    return text.translate(EASTERN_DIGITS)


def extract_digit_strings(text: str) -> str:
    return re.sub(r"\D+", "", to_western_digits(text))


def find_nid_candidates(text: str) -> list[str]:
    western = to_western_digits(text)
    groups = re.findall(r"\d+", western)
    digits = "".join(groups)
    ranked: list[str] = []

    def add(candidate: str) -> None:
        if len(candidate) == 14 and candidate not in ranked:
            ranked.append(candidate)

    add(digits)
    for group in groups:
        add(group)
    if len(groups) >= 2:
        add("".join(groups))
        add("".join(reversed(groups)))
    if len(digits) > 14:
        for match in NID_PATTERN.findall(digits):
            add(match)
        for i in range(len(digits) - 13):
            add(digits[i : i + 14])
    ranked.sort(key=_candidate_score, reverse=True)
    return ranked


def _candidate_score(nid: str) -> tuple[int, int, int]:
    starts_ok = int(nid[0] in {"2", "3"})
    month = int(nid[3:5])
    day = int(nid[5:7])
    date_ok = int(1 <= month <= 12 and 1 <= day <= 31)
    gov_ok = int(nid[7:9] in GOVERNORATES)
    return (starts_ok, date_ok, gov_ok)


@dataclass
class NidDecode:
    nid: str
    century_digit: str
    birth_year: int | None
    birth_month: int | None
    birth_day: int | None
    birth_date: str | None
    governorate_code: str
    governorate: str | None
    serial: str
    gender: str | None
    check_digit: str
    is_valid_structure: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decode_nid(raw: str) -> NidDecode:
    nid = extract_digit_strings(raw)
    issues: list[str] = []
    if len(nid) != 14:
        issues.append(f"expected 14 digits, got {len(nid)}")
        return NidDecode(
            nid=nid,
            century_digit=nid[:1] if nid else "",
            birth_year=None,
            birth_month=None,
            birth_day=None,
            birth_date=None,
            governorate_code=nid[7:9] if len(nid) >= 9 else "",
            governorate=None,
            serial=nid[9:13] if len(nid) >= 13 else "",
            gender=None,
            check_digit=nid[13:14] if len(nid) >= 14 else "",
            is_valid_structure=False,
            issues=issues,
        )

    century_digit = nid[0]
    century = {"2": 1900, "3": 2000}.get(century_digit)
    if century is None:
        issues.append(f"century digit must be 2 or 3, got {century_digit}")

    year_yy = int(nid[1:3])
    month = int(nid[3:5])
    day = int(nid[5:7])
    birth_year = century + year_yy if century is not None else None
    birth_date: str | None = None
    if birth_year is not None:
        try:
            birth_date = date(birth_year, month, day).isoformat()
        except ValueError:
            issues.append(f"invalid birth date {birth_year:04d}-{month:02d}-{day:02d}")
            birth_year = None

    gov_code = nid[7:9]
    governorate = GOVERNORATES.get(gov_code)
    if governorate is None:
        issues.append(f"unknown governorate code {gov_code}")

    serial = nid[9:13]
    gender_digit = int(nid[12])
    gender = "male" if gender_digit % 2 == 1 else "female"

    return NidDecode(
        nid=nid,
        century_digit=century_digit,
        birth_year=birth_year,
        birth_month=month,
        birth_day=day,
        birth_date=birth_date,
        governorate_code=gov_code,
        governorate=governorate,
        serial=serial,
        gender=gender,
        check_digit=nid[13],
        is_valid_structure=not issues,
        issues=issues,
    )


def pick_best_nid(text: str) -> NidDecode | None:
    candidates = find_nid_candidates(text)
    if not candidates:
        return None
    decoded = [decode_nid(item) for item in candidates]
    decoded.sort(key=lambda item: (item.is_valid_structure, item.nid.startswith(("2", "3"))), reverse=True)
    return decoded[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode a 14-digit Egyptian NID.")
    parser.add_argument("nid", help="14-digit NID, Eastern or Western numerals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = pick_best_nid(args.nid) or decode_nid(args.nid)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
