"""Optional, isolated forecast information research. Never writes the live DB.

Historical market downloads support *reconstructed* sensitivity studies only.
Calendars and TFF have no inferred historical availability: the first stored
snapshot is the earliest admissible decision time. Calendar versions replace
the whole provider schedule, so moved/cancelled events do not accumulate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from html import unescape
import io
import json
from pathlib import Path
import re
from urllib.parse import urlparse
import zipfile

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from regime_lab.data.release_archive import ReleaseRecord
from regime_lab.schema import STATE_ORDER

SCHEMA_VERSION = "regime-forecast-new-information/1"
SELECTION_END = pd.Timestamp("2023-01-01", tz="UTC")
ET = "America/New_York"
PROBABILITY_COLUMNS = [f"p_{state}" for state in STATE_ORDER]
MAX_BYTES = 25_000_000
SOURCES = {
    "vix3m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "vix9d": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "vvix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    "fomc": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "bls": "https://www.bls.gov/schedule/news_release/bls.ics",
    "board_ebp": "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv",
}


def utc(value) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("an explicit timezone-aware timestamp is required")
    return result.tz_convert("UTC")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path: Path, value) -> None:
    """Atomic output; failed runs cannot replace the last complete summary."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False,
                                    allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Snapshot:
    source_id: str
    url: str
    sha256: str
    retrieved_at: pd.Timestamp
    raw: bytes

    def __post_init__(self):
        object.__setattr__(self, "retrieved_at", utc(self.retrieved_at))
        if hashlib.sha256(self.raw).hexdigest() != self.sha256:
            raise ValueError("snapshot checksum mismatch")

    def manifest(self) -> dict:
        return {"source_id": self.source_id, "url": self.url,
                "sha256": self.sha256, "retrieved_at": self.retrieved_at.isoformat(),
                "bytes": len(self.raw), "cost": "free_public_no_account",
                "availability_policy": "first_stored_snapshot"}


def fetch_snapshot(source_id: str, url: str, directory: Path, *,
                   refresh: bool = False, session=None) -> Snapshot:
    """Bounded official downloads, immutable timestamped manifests and blobs."""
    allowed = {"cdn.cboe.com", "www.federalreserve.gov", "www.bls.gov", "www.cftc.gov"}
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed:
        raise ValueError("source must be a registered free official HTTPS host")
    if not re.fullmatch(r"[a-z0-9_-]+", source_id):
        raise ValueError("invalid source id")
    directory.mkdir(parents=True, exist_ok=True)
    records = sorted(directory.glob(f"{source_id}-*.json"))
    for record in reversed(records):
        meta = json.loads(record.read_text())
        if meta["url"] == url and not refresh:
            return Snapshot(source_id, url, meta["sha256"], utc(meta["retrieved_at"]),
                            (directory / (meta["sha256"] + ".blob")).read_bytes())
    client = session or requests
    # No automatic retry storm; a failed optional block is reported by the runner.
    with client.get(url, timeout=(10, 40), stream=True,
                    headers={"User-Agent": "RegimePrivateResearch/1.0"}) as response:
        response.raise_for_status()
        if urlparse(response.url).hostname not in allowed:
            raise ValueError("unexpected non-official redirect")
        chunks, size = [], 0
        for chunk in response.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError("source exceeds bounded download size")
            chunks.append(chunk)
        raw = b"".join(chunks)
    if not raw:
        raise ValueError("empty source response")
    digest = hashlib.sha256(raw).hexdigest()
    now = pd.Timestamp(datetime.now(timezone.utc))
    blob = directory / (digest + ".blob")
    if not blob.exists():
        with blob.open("xb") as stream:
            stream.write(raw)
    elif hashlib.sha256(blob.read_bytes()).hexdigest() != digest:
        raise ValueError("existing source blob is corrupt")
    snapshot = Snapshot(source_id, url, digest, now, raw)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    write_json(directory / f"{source_id}-{stamp}.json", snapshot.manifest())
    return snapshot


def load_snapshots(directory: Path, source_id: str) -> list[Snapshot]:
    result = []
    for path in sorted(directory.glob(f"{source_id}-*.json")):
        meta = json.loads(path.read_text())
        result.append(Snapshot(source_id, meta["url"], meta["sha256"],
                               utc(meta["retrieved_at"]),
                               (directory / (meta["sha256"] + ".blob")).read_bytes()))
    return result


def store_observed_snapshot(source_id: str, url: str, raw: bytes, directory: Path) -> Snapshot:
    """Store an actually retrieved alternate representation at the current clock.

    Used for an official HTML bundle or a web-tool page extraction. This API
    deliberately has no historical timestamp override.
    """
    if source_id not in {"bls_html", "bls_web_extract"} or url != "https://www.bls.gov/schedule/":
        raise ValueError("only the explicit BLS alternate representation is supported")
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("invalid alternate snapshot size")
    directory.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(raw).hexdigest()
    for prior in load_snapshots(directory, source_id):
        if prior.sha256 == sha:
            return prior
    now = pd.Timestamp(datetime.now(timezone.utc))
    snapshot = Snapshot(source_id, url, sha, now, raw)
    blob = directory / (sha + ".blob")
    if not blob.exists():
        with blob.open("xb") as stream:
            stream.write(raw)
    write_json(directory / f'{source_id}-{now.strftime("%Y%m%dT%H%M%S%fZ")}.json', snapshot.manifest())
    return snapshot


def parse_cboe(snapshot: Snapshot) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(snapshot.raw))
    frame.columns = [str(c).strip().lower() for c in frame]
    value_column = "vvix" if snapshot.source_id == "vvix" and "vvix" in frame else "close"
    if not {"date", value_column} <= set(frame):
        raise ValueError("Cboe DATE/CLOSE columns required")
    dates = pd.to_datetime(frame.date, format="mixed", errors="raise").dt.normalize()
    close = pd.to_numeric(frame[value_column], errors="raise")
    if dates.isna().any() or dates.duplicated().any() or not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Cboe dates and positive closes must be unique and finite")
    result = pd.DataFrame({"observed_date": dates, "value": close})
    # The CSV supplies dates, not historical release timestamps. Do not fabricate one.
    result["source_released_at"] = pd.NaT
    result["reconstructed_available_at"] = (
        (dates + pd.Timedelta(1, unit="D")).dt.tz_localize(ET).dt.tz_convert("UTC"))
    result["available_at"] = snapshot.retrieved_at
    result["retrieved_at"] = snapshot.retrieved_at
    result["source_id"] = snapshot.source_id
    result["source_sha256"] = snapshot.sha256
    result["source_url"] = snapshot.url
    if (result.reconstructed_available_at > snapshot.retrieved_at).any():
        raise ValueError("Cboe snapshot contains a not-yet-completed observation day")
    return result.sort_values("observed_date").reset_index(drop=True)


def market_features(snapshots: dict[str, Snapshot], origins: pd.DatetimeIndex, *,
                    track: str = "first_seen") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align four quotes on the same prior day, with explicit reconstruction mode."""
    if track not in {"first_seen", "reconstructed_market_prior_day"}:
        raise ValueError("unknown evidence track")
    origins = pd.DatetimeIndex([utc(t) for t in origins])
    if not origins.is_monotonic_increasing or origins.has_duplicates:
        raise ValueError("origins must be strictly increasing")
    required = ("vix", "vix9d", "vvix", "vix3m")
    parsed = {key: parse_cboe(snapshots[key]).set_index("observed_date") for key in required}
    daily = pd.concat({key: frame.value for key, frame in parsed.items()}, axis=1).dropna()
    feature_rows, lineage = [], []
    for origin in origins:
        eligible = daily.index < origin.tz_convert(ET).tz_localize(None).normalize()
        if track == "first_seen" and any(snapshots[k].retrieved_at > origin for k in required):
            eligible[:] = False
        prior = daily.loc[eligible]
        row = {"origin_date": origin}
        if len(prior) and (origin.tz_convert(ET).date() - prior.index[-1].date()).days <= 7:
            day, values = prior.index[-1], prior.iloc[-1]
            row.update({"control_vix": values.vix,
                        "control_vix9d_ratio": values.vix9d / values.vix,
                        "control_vvix": values.vvix,
                        "vix3m_ratio": values.vix3m / values.vix})
            lag = prior.loc[prior.index <= day - pd.Timedelta(28, unit="D")]
            if len(lag) and (day - pd.Timedelta(28, unit="D") - lag.index[-1]).days <= 7:
                row["vix3m_ratio_change_4w"] = row["vix3m_ratio"] - lag.iloc[-1].vix3m / lag.iloc[-1].vix
            lineage.append({"origin_date": origin, "observed_date": day.date().isoformat(),
                            "source_released_at": None,
                            "available_at": max(snapshots[k].retrieved_at for k in required)
                            if track == "first_seen" else utc((day + pd.Timedelta(1, unit="D")).tz_localize(ET)),
                            "retrieved_at": max(snapshots[k].retrieved_at for k in required),
                            "evidence_track": track,
                            "source_sha256": {k: snapshots[k].sha256 for k in required}})
        feature_rows.append(row)
    output = pd.DataFrame(feature_rows).set_index("origin_date")
    for col in ("control_vix", "control_vix9d_ratio", "control_vvix", "vix3m_ratio", "vix3m_ratio_change_4w"):
        if col not in output:
            output[col] = np.nan
    return output, pd.DataFrame(lineage)


def _calendar_row(snapshot: Snapshot, kind: str, event_at, event_id: str,
                  precision: str = "minute") -> dict:
    return {"event_kind": kind, "event_at": utc(event_at), "event_id": event_id,
            "time_precision": precision, "known_at": snapshot.retrieved_at,
            "retrieved_at": snapshot.retrieved_at, "source_released_at": None,
            "source_url": snapshot.url, "source_sha256": snapshot.sha256}


def parse_bls_calendar(snapshot: Snapshot) -> pd.DataFrame:
    """Parse explicit BLS events; DTSTAMP never backdates schedule knowledge."""
    text = snapshot.raw.decode("utf-8-sig")
    text = re.sub(r"\r?\n[ \t]", "", text)  # RFC5545 folded lines
    if "BEGIN:VCALENDAR" not in text or "END:VCALENDAR" not in text:
        raise ValueError("BLS response is not a complete iCalendar")
    rows = []
    for block in re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, flags=re.S):
        fields = {}
        for line in block.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        summary = next((v for k, v in fields.items() if k == "SUMMARY"), "")
        kind = "cpi" if "Consumer Price Index" in summary else (
            "jobs" if "Employment Situation" in summary else None)
        if not kind:
            continue
        if "RRULE" in fields or "RDATE" in fields:
            raise ValueError("recurring BLS events need explicit expansion before use")
        if fields.get("STATUS") == "CANCELLED":
            continue
        start = [(k, v) for k, v in fields.items() if k.split(";")[0] == "DTSTART"]
        if len(start) != 1:
            raise ValueError("BLS event requires exactly one DTSTART")
        key, value = start[0]
        if value.endswith("Z"):
            event = pd.to_datetime(value, format="%Y%m%dT%H%M%SZ", utc=True)
        else:
            timezone_match = re.search(r"TZID=([^;:]+)", key)
            if not timezone_match:
                raise ValueError("BLS event timezone must be explicit")
            zone = timezone_match.group(1)
            zone = ET if zone == "US/Eastern" else zone
            event = pd.to_datetime(value, format="%Y%m%dT%H%M%S").tz_localize(zone).tz_convert("UTC")
        rows.append(_calendar_row(snapshot, kind, event, fields.get("UID", f"{kind}-{event.isoformat()}")))
    if not rows:
        raise ValueError("BLS calendar has no CPI/employment events")
    output = pd.DataFrame(rows)
    if output.duplicated("event_id").any():
        raise ValueError("duplicate BLS event IDs")
    return output.sort_values("event_at").reset_index(drop=True)


def parse_fomc_calendar(snapshot: Snapshot) -> pd.DataFrame:
    """Read meeting end dates, including month-crossing ranges; no inferred releases."""
    text = snapshot.raw.decode("utf-8-sig")
    sections = re.split(r"(20\d\d)\s+FOMC Meetings", text)
    rows = []
    months = {name.lower(): i for i, name in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December"), 1)}
    for i in range(1, len(sections), 2):
        year, section = int(sections[i]), sections[i + 1]
        pattern = (r'<div[^>]*class="[^"]*fomc-meeting__month[^\"]*"[^>]*>(.*?)</div>'
                   r'\s*<div[^>]*class="[^"]*fomc-meeting__date[^\"]*"[^>]*>(.*?)</div>')
        matches_found = re.findall(pattern, section, flags=re.S)
        if len(matches_found) != len(re.findall(r'class="[^"]*fomc-meeting__month[^\"]*"', section)):
            raise ValueError("incomplete FOMC month/date pairing")
        for month_html, day_html in matches_found:
            month_text = unescape(re.sub(r"<[^>]+>", "", month_html)).strip()
            day_text = unescape(re.sub(r"<[^>]+>", "", day_html)).strip()
            month_name = month_text.split("/")[-1].strip().lower()
            matches = [n for name, n in months.items() if name.startswith(month_name)]
            days = re.match(r"^(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?", day_text)
            if len(matches) != 1 or not days:
                raise ValueError(f"unrecognized FOMC date: {month_text} {day_text}")
            day = int(days.group(2) or days.group(1))
            # Calendar is date-granular. Midnight is an indexing anchor, not an announced release time.
            event = pd.Timestamp(year=year, month=matches[0], day=day, tz=ET)
            rows.append(_calendar_row(snapshot, "fomc", event, f"fomc-{event.date()}", "date"))
    if not rows:
        raise ValueError("FOMC meeting dates not found")
    result = pd.DataFrame(rows)
    if result.duplicated("event_id").any():
        raise ValueError("duplicate FOMC meetings")
    return result.sort_values("event_at").reset_index(drop=True)


def parse_bls_html(snapshot: Snapshot, kind: str) -> pd.DataFrame:
    """Official release-specific HTML table, also accepting saved rendered text."""
    if kind not in {"cpi", "jobs"}:
        raise ValueError("unknown BLS release calendar")
    text = unescape(re.sub(r"<[^>]+>", " ", snapshot.raw.decode("utf-8-sig")))
    text = re.sub(r"L\d+:\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    title = "Consumer Price Index" if kind == "cpi" else "Employment Situation"
    heading = f"Schedule of Releases for the {title}"
    if heading not in text:
        raise ValueError("official BLS release schedule heading missing")
    text = text[text.rfind(heading) + len(heading):].split("Subscribe to", 1)[0]
    pattern = r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},\s+\d{4})\s+(\d{1,2}:\d{2}\s+[AP]M)\b"
    rows = []
    for release_date, release_time in re.findall(pattern, text):
        normalized = re.sub(r"\s+", " ", release_date.replace(".", "") + " " + release_time)
        normalized = re.sub(r"^([A-Za-z]{3})[a-z]+", r"\1", normalized)
        event = pd.Timestamp(datetime.strptime(normalized, "%b %d, %Y %I:%M %p")).tz_localize(ET)
        rows.append(_calendar_row(snapshot, kind, event, f"{kind}-{event.date()}"))
    if not rows:
        raise ValueError("no explicit release-date/time pairs in BLS schedule")
    result = pd.DataFrame(rows)
    if result.duplicated("event_id").any():
        raise ValueError("duplicate BLS HTML event dates")
    return result.sort_values("event_at").reset_index(drop=True)


def parse_bls_alternate(snapshot: Snapshot) -> pd.DataFrame:
    if snapshot.source_id == "bls_html":
        pages = json.loads(snapshot.raw)
        frames = []
        for kind, page in pages.items():
            raw = page["text"].encode()
            part = Snapshot("bls", page["url"], hashlib.sha256(raw).hexdigest(), snapshot.retrieved_at, raw)
            frames.append(parse_bls_html(part, kind))
    elif snapshot.source_id == "bls_web_extract":
        frames = []
        for kind, name in (("cpi", "cpi"), ("jobs", "empsit")):
            url = f"https://www.bls.gov/schedule/news_release/{name}.htm"
            if url.encode() not in snapshot.raw:
                raise ValueError("official release URL missing from saved web extraction")
            part = Snapshot("bls", url, snapshot.sha256, snapshot.retrieved_at, snapshot.raw)
            frames.append(parse_bls_html(part, kind))
    else:
        raise ValueError("unknown BLS alternate format")
    if set(pd.concat(frames).event_kind) != {"cpi", "jobs"}:
        raise ValueError("alternate BLS snapshot must cover both releases")
    return pd.concat(frames, ignore_index=True).sort_values("event_at").reset_index(drop=True)


def collect_bls_html_fallback(directory: Path, *, refresh: bool = False) -> Snapshot:
    """Try the official release-specific HTML calendars when iCalendar fails."""
    pages = {}
    for kind, name in (("cpi", "cpi"), ("jobs", "empsit")):
        url = f"https://www.bls.gov/schedule/news_release/{name}.htm"
        page = fetch_snapshot(f"bls_{kind}_html", url, directory, refresh=refresh)
        parse_bls_html(page, kind)  # store a combined version only if both parse
        pages[kind] = {"url": url, "text": page.raw.decode("utf-8-sig"), "raw_sha256": page.sha256,
                       "retrieved_at": page.retrieved_at.isoformat()}
    return store_observed_snapshot("bls_html", "https://www.bls.gov/schedule/",
                                   json.dumps(pages).encode(), directory)


def calendar_features(versions: list[pd.DataFrame], origins: pd.DatetimeIndex, *,
                      maximum_snapshot_age_days: int = 7) -> pd.DataFrame:
    """Use only the latest full snapshot known then, including deletions/revisions.

    Counts mean known scheduled events, not a guarantee no unscheduled event can
    occur. NaN means unavailable/stale/insufficient future schedule coverage.
    """
    rows = []
    for origin in origins:
        origin = utc(origin)
        row = {"origin_date": origin}
        for provider, kinds in (("fomc", ("fomc",)), ("bls", ("cpi", "jobs"))):
            eligible = [v for v in versions if len(v) and v.event_kind.isin(kinds).any()
                        and utc(v.known_at.max()) <= origin]
            if not eligible:
                continue
            snapshot = max(eligible, key=lambda v: utc(v.known_at.max()))
            age = (origin - utc(snapshot.known_at.max())).total_seconds() / 86400
            if age > maximum_snapshot_age_days:
                continue
            for kind in kinds:
                events = snapshot.loc[snapshot.event_kind.eq(kind), "event_at"]
                if not len(events) or max(events) < origin + pd.Timedelta(7, unit="D"):
                    continue
                future = events[events > origin]
                row[f"{kind}_known_events_7d"] = int((future <= origin + pd.Timedelta(7, unit="D")).sum())
                row[f"{kind}_days_to_next_known"] = float((min(future) - origin).total_seconds() / 86400)
                row[f"{kind}_snapshot_at"] = snapshot.known_at.max()
                row[f"{kind}_source_sha256"] = snapshot.source_sha256.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).set_index("origin_date")


def parse_cftc_tff(snapshot: Snapshot, *, contract_codes: tuple[str, ...]) -> pd.DataFrame:
    """Parse futures-only TFF, preserving market codes and first-seen availability."""
    raw = snapshot.raw
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [m for m in archive.infolist() if m.filename.lower().endswith((".txt", ".csv"))]
            if len(members) != 1 or members[0].file_size > MAX_BYTES:
                raise ValueError("TFF ZIP requires one bounded CSV/text member")
            raw = archive.read(members[0])  # never extract paths to disk
    frame = pd.read_csv(io.BytesIO(raw), dtype=str, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame]
    date_col = next((c for c in ("Report_Date_as_YYYY-MM-DD", "Report_Date_as_MM_DD_YYYY") if c in frame), None)
    names = {"Open_Interest_All": "open_interest",
             "Asset_Mgr_Positions_Long_All": "asset_long", "Asset_Mgr_Positions_Short_All": "asset_short",
             "Lev_Money_Positions_Long_All": "leveraged_long", "Lev_Money_Positions_Short_All": "leveraged_short"}
    if not date_col or not {"CFTC_Contract_Market_Code", "Market_and_Exchange_Names", "FutOnly_or_Combined", *names} <= set(frame):
        raise ValueError("required TFF futures columns missing")
    frame["CFTC_Contract_Market_Code"] = frame.CFTC_Contract_Market_Code.str.strip()
    frame = frame.loc[frame.CFTC_Contract_Market_Code.isin(contract_codes)].copy()
    if frame.empty:
        raise ValueError("configured TFF contracts absent")
    if not frame.FutOnly_or_Combined.str.strip().str.lower().isin(["futonly", "futures only"]).all():
        raise ValueError("combined/futures report mixing is forbidden")
    result = pd.DataFrame({"observed_date": pd.to_datetime(frame[date_col], format="mixed", errors="raise"),
                           "contract_code": frame.CFTC_Contract_Market_Code,
                           "market_name": frame.Market_and_Exchange_Names})
    for source, target in names.items():
        values = pd.to_numeric(frame[source].str.replace(",", "", regex=False), errors="raise")
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError("TFF positions must be finite nonnegative integer counts")
        result[target] = values
    if (result.open_interest <= 0).any() or result.duplicated(["contract_code", "observed_date"]).any():
        raise ValueError("TFF requires positive OI and unique contract/date")
    for actor in ("asset", "leveraged"):
        if (result[f"{actor}_long"] > result.open_interest).any() or (result[f"{actor}_short"] > result.open_interest).any():
            raise ValueError("TFF category position exceeds open interest")
        result[f"{actor}_net_oi"] = (result[f"{actor}_long"] - result[f"{actor}_short"]) / result.open_interest
    if result.observed_date.isna().any() or (result.observed_date.dt.date > snapshot.retrieved_at.tz_convert(ET).date()).any():
        raise ValueError("TFF contains invalid/future observed dates")
    result["source_released_at"] = pd.NaT  # Tuesday+3 is not evidence of actual publication.
    result["available_at"] = snapshot.retrieved_at
    result["retrieved_at"] = snapshot.retrieved_at
    result["source_sha256"] = snapshot.sha256
    result["source_url"] = snapshot.url
    return result.sort_values(["contract_code", "observed_date"]).reset_index(drop=True)


def positioning_features(records: pd.DataFrame, origins: pd.DatetimeIndex, *,
                         maximum_observation_age_days: int = 21) -> pd.DataFrame:
    rows = []
    for origin in origins:
        origin = utc(origin)
        row = {"origin_date": origin}
        eligible = records.loc[(records.available_at <= origin) &
                               (records.observed_date.dt.date < origin.tz_convert(ET).date())]
        for contract, frame in eligible.groupby("contract_code"):
            frame = frame.sort_values(["observed_date", "retrieved_at"]).drop_duplicates("observed_date", keep="last")
            last = frame.iloc[-1]
            age = (origin.tz_convert(ET).date() - last.observed_date.date()).days
            if age > maximum_observation_age_days:
                continue
            prefix = f"tff_{contract}"
            row[f"{prefix}_observed_date"] = last.observed_date.date().isoformat()
            row[f"{prefix}_available_at"] = last.available_at
            row[f"{prefix}_source_sha256"] = last.source_sha256
            for actor in ("asset", "leveraged"):
                field = f"{actor}_net_oi"
                value = float(last[field])
                row[f"{prefix}_{field}"] = value
                lag_date = last.observed_date - pd.Timedelta(28, unit="D")
                lag = frame.loc[frame.observed_date.eq(lag_date), field]
                row[f"{prefix}_{actor}_change_4w"] = value - float(lag.iloc[-1]) if len(lag) else np.nan
                past = frame.loc[(frame.observed_date < last.observed_date) &
                                 (frame.observed_date >= last.observed_date - pd.Timedelta(364, unit="D")), field]
                row[f"{prefix}_{actor}_percentile_52w"] = float((past <= value).mean()) if len(past) >= 26 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("origin_date")


def ebp_features(records: list[ReleaseRecord], origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Adapter to the already-planned release interface; revisions stay first-seen."""
    if any(record.source_id != "board_ebp" for record in records):
        raise ValueError("only board_ebp release records accepted")
    rows = []
    for origin in origins:
        origin = utc(origin)
        eligible = [r for r in records if r.value is not None and
                    max(utc(r.source_released_at), utc(r.provider_first_seen_at), utc(r.system_retrieved_at)) <= origin
                    and r.observed_period_end < origin.date()]
        row = {"origin_date": origin}
        if eligible:
            latest = max(eligible, key=lambda r: (r.observed_period_end, r.revision_seq, r.system_retrieved_at))
            if (origin.date() - latest.observed_period_end).days <= 100:
                row.update({"ebp": latest.value, "ebp_observed_date": str(latest.observed_period_end),
                            "ebp_available_at": max(latest.source_released_at, latest.provider_first_seen_at, latest.system_retrieved_at),
                            "ebp_source_sha256": latest.raw_sha256})
        rows.append(row)
    return pd.DataFrame(rows).set_index("origin_date")


def parse_ebp(snapshot: Snapshot) -> list[ReleaseRecord]:
    """Monthly official EBP CSV adapted to the existing first-seen interface."""
    frame = pd.read_csv(io.BytesIO(snapshot.raw))
    frame.columns = [str(c).strip().lower() for c in frame]
    if not {"date", "ebp"} <= set(frame):
        raise ValueError("official EBP CSV requires date and ebp")
    dates = pd.to_datetime(frame.date, format="mixed", errors="raise")
    if dates.isna().any() or not dates.dt.day.eq(1).all() or dates.duplicated().any():
        raise ValueError("EBP rows must identify unique monthly periods")
    values = pd.to_numeric(frame.ebp, errors="raise")
    if np.isinf(values).any():
        raise ValueError("EBP must be finite or explicitly missing")
    month_ends = dates + pd.offsets.MonthEnd(0)
    if (month_ends.dt.date > snapshot.retrieved_at.date()).any():
        raise ValueError("EBP contains an unfinished future month")
    return [ReleaseRecord(
        source_id="board_ebp", series_id="ebp", observed_period_end=period.date(),
        value=None if pd.isna(value) else float(value),
        source_released_at=snapshot.retrieved_at.to_pydatetime(),
        provider_first_seen_at=snapshot.retrieved_at.to_pydatetime(),
        system_retrieved_at=snapshot.retrieved_at.to_pydatetime(), revision_seq=0,
        raw_sha256=snapshot.sha256, units="percentage_points",
        metadata={"source_url": snapshot.url, "timestamp_semantics": "first_seen_only",
                  "actual_source_released_at": None, "revision_policy": "entire_history_may_revise",
                  "file_month_label": observed.date().isoformat()})
        for observed, period, value in zip(dates, month_ends, values)]


def load_boundary_history(payload: dict) -> pd.DataFrame:
    models = payload["research"]["forecast_improvement"]["models"]
    history = next(m["history"] for m in models if m["id"] == "boundary_filtered_history")
    rows = [{**{k: item[k] for k in ("origin_date", "target_date", "current_state", "actual", "evaluation_split")},
             **{f"p_{s}": item["probabilities"][s] for s in STATE_ORDER}} for item in history if item.get("actual") in STATE_ORDER]
    frame = pd.DataFrame(rows)
    frame["origin_date"] = pd.to_datetime(frame.origin_date, utc=True)
    frame["target_date"] = pd.to_datetime(frame.target_date, utc=True)
    return validate_history(frame)


def validate_history(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy(deep=True)
    frame["origin_date"] = [utc(t) for t in frame.origin_date]
    frame["target_date"] = [utc(t) for t in frame.target_date]
    if frame.origin_date.duplicated().any() or not (frame.target_date > frame.origin_date).all():
        raise ValueError("history must have unique origins and future targets")
    if not frame.current_state.isin(STATE_ORDER).all() or not frame.actual.isin(STATE_ORDER).all():
        raise ValueError("invalid state label")
    probability = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any() or not np.allclose(probability.sum(axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError("invalid baseline probability simplex")
    split = np.where((frame.origin_date < SELECTION_END) & (frame.target_date < SELECTION_END),
                     "selection", np.where(frame.origin_date >= SELECTION_END, "holdout", "boundary_excluded"))
    if not (frame.evaluation_split.to_numpy() == split).all():
        raise ValueError("split disagrees with frozen origin/target cutoff")
    return frame.sort_values("origin_date").reset_index(drop=True)


def probability_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict:
    if len(frame) == 0:
        return {"n": 0}
    p = np.asarray(probability, dtype=float)
    if p.shape != (len(frame), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-7, rtol=0):
        raise ValueError("scoring requires aligned probability simplex")
    ranks = {s: i for i, s in enumerate(STATE_ORDER)}
    actual = frame.actual.map(ranks).to_numpy()
    current = frame.current_state.map(ranks).to_numpy()
    predicted = p.argmax(axis=1)
    changed, alarm = actual != current, predicted != current
    result = {"n": len(frame), "log_loss": float(-np.log(np.maximum(p[np.arange(len(p)), actual], 1e-12)).mean()),
              "brier": float(np.square(p - np.eye(3)[actual]).sum(axis=1).mean()),
              "accuracy": float((predicted == actual).mean()),
              "transition_events": int(changed.sum()),
              "transition_captured": int((changed & alarm & (predicted == actual)).sum()),
              "false_alarms": int((alarm & ~changed).sum())}
    for kind, mask in (("worsening", np.arange(3)[None, :] > current[:, None]),
                       ("recovery", np.arange(3)[None, :] < current[:, None])):
        y = (actual > current) if kind == "worsening" else (actual < current)
        direction_p = (p * mask).sum(axis=1)
        # Structural zeros in states that cannot move in this direction are
        # reported separately from at-risk origins, not allowed to inflate skill.
        at_risk = mask.any(axis=1)
        yp, pp = y[at_risk], direction_p[at_risk]
        result[kind] = {"at_risk_origins": int(at_risk.sum()), "events": int(y.sum()),
                        "captured": int((y & (predicted == actual) & alarm).sum()),
                        "false_alarms": int((((predicted > current) if kind == "worsening" else (predicted < current)) & ~y).sum()),
                        "brier": float(np.square(pp - yp).mean()) if len(pp) else None,
                        "log_loss": float(-(yp * np.log(np.maximum(pp, 1e-12)) + (~yp) * np.log(np.maximum(1 - pp, 1e-12))).mean()) if len(pp) else None,
                        "average_precision": float(average_precision_score(yp, pp)) if yp.any() else None}
    return result


def run_same_origin_ablation(history: pd.DataFrame, features: pd.DataFrame, *,
                             control_columns: list[str], extra_columns: list[str],
                             minimum_training_rows: int = 104,
                             regularization_c: float = 0.1,
                             mixture_weight: float = 0.25) -> tuple[pd.DataFrame, dict]:
    """Fixed prequential logistic augmentation, paired eligible training/test rows.

    Both models receive baseline log probabilities/current state; the control
    adds the already-available information, the candidate adds the new block.
    A fixed 25% augmentation/75% boundary mixture limits model disturbance.
    Neither hyperparameters nor direction thresholds are selected on diagnostics.
    """
    history = validate_history(history)
    if not 0 < mixture_weight <= 1 or regularization_c <= 0 or minimum_training_rows < 3:
        raise ValueError("invalid frozen experiment settings")
    if features.index.has_duplicates or features.index.tz is None:
        raise ValueError("features require unique timezone-aware origins")
    columns = control_columns + extra_columns
    if len(set(columns)) != len(columns) or not columns or not set(columns) <= set(features):
        raise ValueError("missing or overlapping feature columns")
    aligned = features.reindex(pd.DatetimeIndex(history.origin_date))[columns].astype(float)
    usable = np.isfinite(aligned.to_numpy()).all(axis=1)
    p = history[PROBABILITY_COLUMNS].to_numpy()
    base = np.column_stack([np.log(np.maximum(p, 1e-8)),
                            np.eye(3)[history.current_state.map({s: i for i, s in enumerate(STATE_ORDER)})]])
    x = {"control": np.column_stack([base, aligned[control_columns].to_numpy()]),
         "candidate": np.column_stack([base, aligned.to_numpy()])}
    y = history.actual.map({s: i for i, s in enumerate(STATE_ORDER)}).to_numpy()
    rows, excluded = [], {"missing_features": int((~usable).sum()), "insufficient_past_training": 0,
                         "boundary_excluded": 0, "fit_failure": 0}
    failures = []
    for i, row in history.iterrows():
        if not usable[i]:
            continue
        if row.evaluation_split == "boundary_excluded":
            excluded["boundary_excluded"] += 1
            continue
        train = usable & (history.target_date < row.origin_date).to_numpy()
        # Completed diagnostic history may enter later rolling fits; this is a
        # frozen updating rule, never evidence of a newly untouched holdout.
        if train.sum() < minimum_training_rows or len(np.unique(y[train])) < 3:
            excluded["insufficient_past_training"] += 1
            continue
        predictions = {"boundary_reference": p[i]}
        try:
            for model, matrix in x.items():
                fitted = make_pipeline(StandardScaler(), LogisticRegression(C=regularization_c, max_iter=2000,
                                                                            random_state=0))
                fitted.fit(matrix[train], y[train])
                proposal = fitted.predict_proba(matrix[[i]])[0]
                predictions[model] = (1 - mixture_weight) * p[i] + mixture_weight * proposal
        except (ValueError, FloatingPointError) as error:
            excluded["fit_failure"] += 1
            failures.append({"origin_date": row.origin_date, "reason": str(error)})
            continue  # drop both arms, never substitute a successful-looking baseline
        last_target = history.loc[train, "target_date"].max()
        for model, probability in predictions.items():
            rows.append({"origin_date": row.origin_date, "target_date": row.target_date,
                         "current_state": row.current_state, "actual": row.actual,
                         "evaluation_split": row.evaluation_split, "model": model,
                         "train_rows": int(train.sum()), "last_train_target": last_target,
                         **dict(zip(PROBABILITY_COLUMNS, probability))})
    output = pd.DataFrame(rows)
    summary = {"status": "evaluated" if rows else "not_evaluable", "excluded": excluded,
               "fit_failures": failures, "metrics": {},
               "settings": {"minimum_training_rows": minimum_training_rows, "regularization_c": regularization_c,
                            "mixture_weight": mixture_weight, "control_columns": control_columns,
                            "extra_columns": extra_columns},
               "evidence": "reconstructed diagnostic; fixed updating rule; no automatic promotion"}
    if rows:
        for split, group in output.groupby("evaluation_split"):
            summary["metrics"][split] = {model: probability_metrics(part, part[PROBABILITY_COLUMNS].to_numpy())
                                         for model, part in group.groupby("model")}
            control = group.loc[group.model.eq("control")].sort_values("origin_date")
            candidate = group.loc[group.model.eq("candidate")].sort_values("origin_date")
            summary["metrics"][split]["paired_delta"] = {
                "candidate_minus_control_log_loss": summary["metrics"][split]["candidate"]["log_loss"] - summary["metrics"][split]["control"]["log_loss"],
                "candidate_minus_control_brier": summary["metrics"][split]["candidate"]["brier"] - summary["metrics"][split]["control"]["brier"],
                "same_origins": control.origin_date.tolist() == candidate.origin_date.tolist(),
                "same_training_counts": control.train_rows.tolist() == candidate.train_rows.tolist(),
                "origin_start": control.origin_date.min(), "origin_end": control.origin_date.max()}
        summary["all_train_targets_strictly_before_origin"] = bool((output.last_train_target < output.origin_date).all())
        recent_start = output.origin_date.max() - pd.DateOffset(weeks=51)
        recent = output.loc[output.origin_date >= recent_start]
        summary["recent_52_origin_weeks"] = {
            model: probability_metrics(part, part[PROBABILITY_COLUMNS].to_numpy())
            for model, part in recent.groupby("model")}
    return output, summary


def build_forecast_information(summary: dict) -> dict:
    """Small optional UI adapter; never edits a payload or promotes a candidate.

    Candidate comparisons use both the identically fitted control and the
    existing boundary probabilities, always rescored on the very same origins.
    """
    rows = []
    experiment = summary.get("experiments", {}).get("vix3m", {})
    for split, metrics in experiment.get("metrics", {}).items():
        if not {"candidate", "control", "boundary_reference", "paired_delta"} <= set(metrics):
            raise ValueError("incomplete paired information experiment")
        pair = metrics["paired_delta"]
        if not pair["same_origins"] or not pair["same_training_counts"]:
            raise ValueError("information comparison must use matched origins and training")
        candidate = metrics["candidate"]
        for baseline_key, model in (("control", "boundary_existing_vol_control"),
                                    ("boundary_reference", "boundary_filtered_history")):
            baseline = metrics[baseline_key]
            if candidate["n"] != baseline["n"]:
                raise ValueError("information comparison populations differ")
            direction_delta = None
            if candidate["worsening"]["brier"] is not None and baseline["worsening"]["brier"] is not None:
                direction_delta = candidate["worsening"]["brier"] - baseline["worsening"]["brier"]
            rows.append({"feature_block": "vix3m_term_structure", "model": "boundary_vix3m_augmented",
                         "baseline_model": model, "evaluation_split": split, "matched_n": candidate["n"],
                         "baseline_log_loss": baseline["log_loss"], "candidate_log_loss": candidate["log_loss"],
                         "delta_log_loss": candidate["log_loss"] - baseline["log_loss"],
                         "delta_brier": candidate["brier"] - baseline["brier"],
                         "delta_worsening_brier": direction_delta})
    notes = ["모든 비교는 같은 origin/target에서 재계산한 기준선과 후보를 사용합니다.",
             "VIX3M은 이전 관측일 종가를 이용한 시장자료 재구성 연구입니다. 실제 과거 발행 성과가 아닙니다.",
             "2023년 이후 holdout 표기는 기존 아티팩트의 구분명이며 이미 검토된 진단 구간입니다.",
             "확률 점수 차이는 후보-기준선이며 음수가 개선입니다. 운영 모델은 변경하지 않습니다."]
    for key in ("fomc", "bls", "cftc_tff", "board_ebp"):
        block = summary.get("blocks", {}).get(key, {})
        notes.append(f'{key}: {block.get("status", "not_supplied")} — {block.get("reason", "")}')
    return json_safe({"schema_version": "regime-forecast-information/1", "data_as_of": summary["data_as_of"],
                      "status": "research_only" if rows else "unavailable", "rows": rows, "notes": notes})
