"""Immutable, public, no-account source snapshots for research candidates.

New collection only. Historical downloads never imply observed historical PIT.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests

SOURCES = {
    "vix": (
        "Cboe VIX",
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    ),
    "vix9d": (
        "Cboe VIX9D",
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    ),
    "vvix": (
        "Cboe VVIX",
        "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    ),
    "ads": (
        "Philadelphia Fed ADS",
        "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/ads/ADS_All_Vintages-zip.zip",
    ),
    "sloos": (
        "Federal Reserve SLOOS",
        "https://www.federalreserve.gov/releases/sloos/data/FRB_SLOOS_xml.zip",
    ),
    "cmdi": (
        "New York Fed CMDI",
        "https://www.newyorkfed.org/medialibrary/research/interactives/data/cmdi/cmdi_interactive_data.xlsx",
    ),
}


def collect_sources(directory: Path, *, refresh: bool = False) -> dict:
    """Cache downloaded bytes by hash and persist first-seen availability."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    prior = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"sources": {}}
    )

    def collect(item):
        key, (name, url) = item
        old = prior["sources"].get(key)
        if old and old.get("url") == url and not refresh:
            raw = (directory / old["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != old["sha256"]:
                raise ValueError(f"{key} cached snapshot checksum mismatch")
            return key, old
        response = requests.get(
            url, timeout=(15, 90), headers={"User-Agent": "RegimePrivateResearch/1.0"}
        )
        response.raise_for_status()
        raw = response.content
        if not raw or len(raw) > 200_000_000:
            raise ValueError(f"{key} invalid snapshot size")
        digest = hashlib.sha256(raw).hexdigest()
        suffix = Path(url).suffix or ".bin"
        filename = f"{key}-{digest}{suffix}"
        target = directory / filename
        if not target.exists():
            with target.open("xb") as file:
                file.write(raw)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "name": name,
            "url": url,
            "sha256": digest,
            "bytes": len(raw),
            "file": filename,
            "fetched_at": now,
            "first_seen_at": old["first_seen_at"]
            if old and old["sha256"] == digest
            else now,
            "cost": "public_download_no_account",
            "availability_basis": "first_seen_snapshot",
        }
        return key, record

    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for key, record in pool.map(collect, SOURCES.items()):
            results[key] = record
    manifest = {"schema_version": "regime-additional-sources/1", "sources": results}
    temporary = directory / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(manifest_path)
    return manifest


def read_source(directory: Path, key: str) -> bytes:
    manifest = json.loads((directory / "manifest.json").read_text())
    record = manifest["sources"][key]
    raw = (directory / record["file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("snapshot checksum mismatch")
    return raw


def cboe_daily(raw: bytes) -> pd.Series:
    frame = pd.read_csv(io.BytesIO(raw))
    frame.columns = [str(x).strip().lower() for x in frame.columns]
    dates = pd.to_datetime(frame["date"], errors="coerce")
    column = "close" if "close" in frame else next(c for c in frame if c != "date")
    values = pd.to_numeric(frame[column], errors="coerce")
    result = pd.Series(values.to_numpy(), index=dates).dropna().sort_index()
    if result.index.has_duplicates or (result <= 0).any():
        raise ValueError("invalid Cboe daily history")
    return result


def cboe_weekly(directory: Path, cutoffs: pd.DatetimeIndex) -> pd.DataFrame:
    """Use strictly prior calendar-day finalized close, including historical research."""
    output = pd.DataFrame(index=cutoffs)
    for key in ("vix", "vix9d", "vvix"):
        series = cboe_daily(read_source(directory, key))
        local_days = (
            cutoffs.tz_convert("America/New_York").tz_localize(None).normalize()
            if cutoffs.tz is not None
            else cutoffs.normalize()
        )
        prior_days = pd.DatetimeIndex(
            [value.date() - timedelta(days=1) for value in local_days]
        )
        output[key] = series.reindex(prior_days, method="ffill").to_numpy()
    output["short_vol_ratio"] = output.vix9d / output.vix
    output["short_vol_ratio_change_1w"] = output.short_vol_ratio.diff()
    output["vvix_change_4w"] = output.vvix.diff(4)
    return output


def source_summary(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    for key, record in manifest["sources"].items():
        row = {
            k: record[k]
            for k in (
                "name",
                "url",
                "sha256",
                "bytes",
                "fetched_at",
                "first_seen_at",
                "availability_basis",
            )
        }
        row.update(
            id=key,
            status="collected",
            role={
                "vix": "30일 옵션 변동성",
                "vix9d": "단기 변동성 기간 구조",
                "vvix": "변동성 불확실성",
                "ads": "실물 경기와 발표 변화",
                "sloos": "은행 대출 기준과 수요",
                "cmdi": "회사채 시장 기능",
            }[key],
        )
        parsed = directory / (
            "ads-vintage-features.csv" if key == "ads" else f"{key}-parsed.csv"
        )
        if parsed.exists():
            frame = pd.read_csv(parsed)
            row["parsed_rows"] = len(frame)
            row["status"] = "parsed"
        elif key in ("vix", "vix9d", "vvix"):
            row["parsed_rows"] = len(cboe_daily(read_source(directory, key)))
            row["status"] = "evaluated"
        rows.append(row)
    return {"status": "collected", "sources": rows, "promotion": "research_only"}


def sloos_history(raw: bytes) -> pd.DataFrame:
    archive = zipfile.ZipFile(io.BytesIO(raw))
    root = ET.fromstring(archive.read("SLOOS_data.xml"))
    rows = []
    for series in root.iter():
        if series.tag.rsplit("}", 1)[-1] != "Series":
            continue
        if series.get("PANEL") != "DOM" or series.get("BANKSIZE") != "ALL":
            continue
        name = series.get("SERIES_NAME", "")
        descriptions = [
            "".join(x.itertext())
            for x in series.iter()
            if x.tag.endswith("AnnotationText")
        ]
        description = descriptions[0] if descriptions else ""
        if "C&I" not in description or not any(
            k in description.lower() for k in ("standards", "demand")
        ):
            continue
        for obs in series:
            if obs.tag.endswith("Obs") and obs.get("OBS_STATUS") == "A":
                rows.append(
                    {
                        "series": name,
                        "description": description,
                        "observation_date": obs.get("TIME_PERIOD"),
                        "value": float(obs.get("OBS_VALUE")),
                        "availability_basis": "current_download_first_seen_only",
                    }
                )
    return pd.DataFrame(rows)


def cmdi_history(raw: bytes) -> pd.DataFrame:
    frame = pd.read_excel(io.BytesIO(raw))
    columns = ["eow_friday", "Market CMDI", "IG CMDI", "HY CMDI"]
    if not set(columns).issubset(frame):
        raise ValueError("CMDI columns changed")
    frame = frame[columns].rename(columns={"eow_friday": "observation_date"}).dropna()
    if frame.observation_date.duplicated().any():
        raise ValueError("CMDI dates must be unique")
    frame["availability_basis"] = "monthly_publication_first_seen_only"
    return frame


def ads_vintage_features(raw: bytes, cutoffs: pd.DatetimeIndex) -> pd.DataFrame:
    """Stream the 1.5GB worksheet, retain only required past-vintage cells.

    ADS column names encode each actual vintage date. Unknown publication times
    defer usability to the next calendar day. Current vintage rewrites never
    enter an earlier cutoff. The downloaded archive itself remains retrospective.
    """
    archive = zipfile.ZipFile(io.BytesIO(raw))
    book = zipfile.ZipFile(io.BytesIO(archive.read(archive.namelist()[0])))
    strings = [
        "".join(x.itertext()) for x in ET.fromstring(book.read("xl/sharedStrings.xml"))
    ]
    vintage_cols = {}

    def column_name(i):
        name = ""
        while i:
            i, remainder = divmod(i - 1, 26)
            name = chr(65 + remainder) + name
        return name

    for i, name in enumerate(strings):
        if re.fullmatch(r"ADS_INDEX_\d{6}", name):
            vintage_cols[column_name(i + 1)] = pd.Timestamp(
                datetime.strptime(name[-6:], "%m%d%y")
            ).date()
    selected = {}
    for cutoff in cutoffs:
        local = (
            cutoff.tz_convert("America/New_York").date()
            if cutoff.tzinfo
            else cutoff.date()
        )
        eligible = [(date, col) for col, date in vintage_cols.items() if date < local]
        if eligible:
            date, col = max(eligible)
            selected.setdefault(
                col, {"vintage_date": date, "cutoffs": [], "values": []}
            )["cutoffs"].append(cutoff)
    # XML has one cell row per calendar day. Processing row bytes avoids
    # constructing tens of millions of Python cell objects.
    with book.open("xl/worksheets/sheet1.xml") as stream:
        buffer = b""
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            while b"</row>" in buffer:
                row, buffer = buffer.split(b"</row>", 1)
                match = re.search(rb'<c r="A\d+"[^>]*t="s"[^>]*><v>(\d+)</v>', row)
                if not match:
                    continue
                text = strings[int(match.group(1))]
                if not re.fullmatch(r"\d{4}:\d{2}:\d{2}", text):
                    continue
                day = pd.Timestamp(text.replace(":", "-")).date()
                eligible = {
                    col
                    for col, info in selected.items()
                    if 0 <= (info["vintage_date"] - day).days <= 105
                }
                if not eligible:
                    continue
                for match in re.finditer(
                    rb'<c r="([A-Z]+)\d+"[^>]*><v>([^<]+)</v></c>', row
                ):
                    col = match.group(1).decode()
                    if col in eligible:
                        try:
                            value = float(match.group(2))
                        except ValueError:
                            continue
                        selected[col]["values"].append((day, value))
    rows = []
    for col, info in selected.items():
        values = sorted(info["values"])
        if not values:
            continue
        observed, value = values[-1]

        def difference(days):
            earlier = [v for d, v in values if (observed - d).days >= days]
            return value - earlier[-1] if earlier else None

        for cutoff in info["cutoffs"]:
            rows.append(
                {
                    "cutoff": cutoff,
                    "ads_level": value,
                    "ads_change_1w": difference(7),
                    "ads_change_4w": difference(28),
                    "ads_change_13w": difference(91),
                    "vintage_date": info["vintage_date"].isoformat(),
                    "observation_date": observed.isoformat(),
                }
            )
    return pd.DataFrame(rows).set_index("cutoff").sort_index()
