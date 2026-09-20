"""Corpus ingest (CLAUDE.md §5.2).

Real public data only. Two sources, neither of which needs an API key:

* **Reddit** — `HuggingFaceGECLM/REDDIT_comments`, the r/SkincareAddiction shards.
  A public dump, which is the second option §5.2 allows. Permalinks are
  reconstructed from `link_id` + comment `id`, so every URL resolves.
* **Amazon** — `McAuley-Lab/Amazon-Reviews-2023`, the All_Beauty review file, with
  product titles joined from the matching metadata file. URLs are `/dp/<asin>`.

Two deliberate choices worth defending:

1. **Nothing is truncated.** §5.1 permits truncating at 400 chars, but a comment cut
   mid-sentence reads like fabricated data to anyone who opens the expander. We
   select comments that are *naturally* under the limit instead, so every `text` is
   a complete, verbatim thought.
2. **Selection is deterministic.** Candidates are sorted by a fixed key, never
   sampled randomly. Re-running ingest on the same input yields the same corpus in
   the same order, which is what lets §5.1's "IDs are assigned once and frozen"
   actually hold.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from src.themes import ALL_THEMES, classify, score_themes

USER_AGENT = "receipts-ingest/0.1 (CREWASIS hackathon; contact via repo)"

HF_BASE = "https://huggingface.co/datasets"
REDDIT_REPO = f"{HF_BASE}/HuggingFaceGECLM/REDDIT_comments/resolve/main/data"
AMAZON_REPO = f"{HF_BASE}/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"

SUBREDDIT = "SkincareAddiction"

# Complete thoughts only: long enough to carry a claim, short enough to quote whole.
MIN_CHARS = 90
MAX_CHARS = 380

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_BOT_AUTHOR = re.compile(r"(bot|automoderator)$", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[a-z/][^>]{0,20}>", re.IGNORECASE)

# The theme keywords are deliberately broad, which lets a few off-topic comments in
# ("working out adds an hour to your life" matches `sweat`). Every signal must also
# be about skin — otherwise it is noise in an evidence set a judge will read.
_SKIN_DOMAIN = re.compile(
    r"\b(skin|face|facial|complexion|pore|pores|cheek|cheeks|forehead|chin|nose|"
    r"acne|breakout|pimple|rosacea|eczema|derm|dermatologist|moisturi[sz]|"
    r"cleanser|serum|sunscreen|spf|retinol|retinoid|niacinamide|hyaluronic|"
    r"salicylic|glycolic|azelaic|toner|exfoliat|cream|lotion|routine|"
    r"blackhead|whitehead|sebum|oily|t-zone|makeup|moisture barrier)\b",
    re.IGNORECASE,
)

# Amazon All_Beauty covers hair, fragrance and tools too. Keep skincare only (§5.2).
_SKINCARE_PRODUCT = re.compile(
    r"\b(serum|moisturi[sz]er|cleanser|face wash|facial|sunscreen|spf|toner|"
    r"retinol|retinoid|niacinamide|hyaluronic|salicylic|glycolic|vitamin c|"
    r"eye cream|face cream|face lotion|face oil|exfoliant|peel|acne|blemish|"
    r"moisture cream|skin care|skincare|day cream|night cream|face mask|"
    r"face moisturizer|spot treatment|ceramide)\b",
    re.IGNORECASE,
)

# Things that look like skincare words but are not facial skincare products.
_NOT_SKINCARE = re.compile(
    r"\b(shampoo|conditioner|hair (mask|oil|spray|dye|color|colour)|nail|"
    r"perfume|cologne|eau de|lipstick|mascara|foundation|brush set|"
    r"body wash|deodorant|toothpaste|wig|extensions)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RawSignal:
    """A candidate signal, before an ID is minted for it."""

    native_key: str  # stable upstream identity, e.g. "reddit:hx8k2ab"
    text: str
    source: str
    source_detail: str
    url: str
    timestamp: date
    theme: str
    rank_score: int  # upstream popularity — used only to order candidates


# --------------------------------------------------------------------------------------
# Shared text hygiene
# --------------------------------------------------------------------------------------


def normalise(text: str) -> str:
    """Whitespace normalisation only. No other cleaning — `text` stays verbatim (§5.1)."""
    return _WS.sub(" ", text or "").strip()


def is_quotable(text: str) -> bool:
    """Would a judge reading this in the citation expander believe it is real?"""
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if _URL.search(text):
        return False
    if text.startswith(">") or "&gt;" in text:  # quoted reply, missing its context
        return False
    if text.count("*") > 4 or "](" in text:  # heavy markdown reads badly raw
        return False
    # Amazon review bodies carry raw HTML. Markup is not the author's words, and
    # §5.1 forbids cleaning, so drop these rather than strip them.
    if _HTML_TAG.search(text):
        return False
    # U+FFFD means the upstream dump already mangled this text. It is not the
    # author's words any more, and §5.1 forbids us from repairing it.
    if "�" in text:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / len(text)
    if ascii_ratio < 0.98:
        return False
    if not _SKIN_DOMAIN.search(text):
        return False
    # Needs some actual prose, not just an emoji reaction or a product list.
    return len(text.split()) >= 18


# --------------------------------------------------------------------------------------
# Reddit
# --------------------------------------------------------------------------------------


def reddit_permalink(link_id: str, comment_id: str) -> str:
    """Canonical comment permalink. `link_id` arrives as "t3_<submission_id>"."""
    submission = link_id[3:] if link_id.startswith("t3_") else link_id
    return f"https://www.reddit.com/r/{SUBREDDIT}/comments/{submission}/comment/{comment_id}/"


def extract_reddit(parquet_path: Path, min_score: int = 5) -> list[RawSignal]:
    """Pull themed, quotable comments out of a downloaded r/SkincareAddiction shard."""
    import pyarrow.parquet as pq

    wanted = ["body", "id", "link_id", "created_utc", "score", "author"]
    out: list[RawSignal] = []
    seen: set[str] = set()

    pf = pq.ParquetFile(str(parquet_path))
    for batch in pf.iter_batches(batch_size=20_000, columns=wanted):
        for row in batch.to_pylist():
            body = normalise(row.get("body") or "")
            if body in ("[deleted]", "[removed]", ""):
                continue
            author = row.get("author") or ""
            if author in ("[deleted]", "AutoModerator") or _BOT_AUTHOR.search(author):
                continue
            try:
                score = int(row.get("score") or 0)
            except (TypeError, ValueError):
                continue
            if score < min_score:
                continue
            if not is_quotable(body):
                continue
            theme = classify(body)
            if theme is None:
                continue

            key = body.lower()
            if key in seen:
                continue
            seen.add(key)

            try:
                created = int(row["created_utc"])
            except (TypeError, ValueError, KeyError):
                continue

            out.append(
                RawSignal(
                    native_key=f"reddit:{row['id']}",
                    text=body,
                    source="reddit",
                    source_detail=f"r/{SUBREDDIT}",
                    url=reddit_permalink(row.get("link_id") or "", row["id"]),
                    timestamp=datetime.fromtimestamp(created, timezone.utc).date(),
                    theme=theme,
                    rank_score=score,
                )
            )
    return out


# --------------------------------------------------------------------------------------
# Amazon
# --------------------------------------------------------------------------------------


def _open_remote(url: str, byte_range: tuple[int, int] | None = None):
    headers = {"User-Agent": USER_AGENT}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=600
    )


def _stream_jsonl(url: str, byte_range: tuple[int, int] | None = None) -> Iterator[dict]:
    """Yield JSON objects from a (possibly gzipped, possibly partial) remote JSONL file.

    Range requests land mid-line at both ends, so the first and last fragments are
    dropped rather than guessed at.
    """
    with _open_remote(url, byte_range) as resp:
        stream = gzip.GzipFile(fileobj=resp) if url.endswith(".gz") else resp
        first = True
        pending: bytes | None = None
        for raw in stream:
            if first and byte_range and byte_range[0] > 0:
                first = False
                continue  # partial first line
            first = False
            if pending is not None:
                yield from _maybe_parse(pending)
            pending = raw
        if pending is not None and not byte_range:
            yield from _maybe_parse(pending)


def _maybe_parse(raw: bytes) -> Iterator[dict]:
    try:
        yield json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return


def download_amazon_reviews(dest: Path, megabytes: int = 140) -> Path:
    """Fetch the leading slice of All_Beauty reviews. 140 MB is ~150k reviews."""
    url = f"{AMAZON_REPO}/review_categories/All_Beauty.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > megabytes * 1_000_000 * 0.9:
        return dest
    with _open_remote(url, (0, megabytes * 1_000_000)) as resp, dest.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def _amazon_titles(asins: set[str]) -> dict[str, str]:
    """Stream the metadata file, keeping titles only for the ASINs we selected."""
    url = f"{AMAZON_REPO}/meta_categories/meta_All_Beauty.jsonl"
    titles: dict[str, str] = {}
    for record in _stream_jsonl(url):
        asin = record.get("parent_asin")
        if asin in asins and asin not in titles:
            title = normalise(record.get("title") or "")
            if title:
                titles[asin] = title
            if len(titles) == len(asins):
                break
    return titles


DEAD_ASINS_PATH = Path(__file__).resolve().parents[1] / "data" / "dead_asins.json"


def load_dead_asins(path: Path | None = None) -> set[str]:
    """ASINs whose product page now 404s.

    §5.1 requires every URL to resolve — "a judge will click one" — and the
    Amazon dataset is a 2023 snapshot, so delisted products rot out of it. About
    a third of the ASINs the first corpus selected were already dead. They are
    denylisted here rather than filtered at read time so the list is reviewable,
    diffable and re-checkable before demo day.

    Note this cannot be checked with curl: Amazon serves an identical bot-block
    page for live and dead ASINs alike. See scripts/check_amazon_urls.md.
    """
    target = path or DEAD_ASINS_PATH
    if not target.exists():
        return set()
    return set(json.loads(target.read_text(encoding="utf-8")).get("dead", []))


def extract_amazon(
    reviews_path: Path,
    since_year: int = 2015,
    dead_asins: set[str] | None = None,
) -> list[RawSignal]:
    """Pull themed, quotable skincare reviews out of a downloaded All_Beauty slice."""
    dead = load_dead_asins() if dead_asins is None else dead_asins
    candidates: list[dict] = []
    seen: set[str] = set()

    with reviews_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("verified_purchase"):
                continue
            text = normalise(rec.get("text") or "")
            if not is_quotable(text):
                continue
            asin = rec.get("parent_asin")
            if not asin or asin in dead:
                continue
            try:
                when = datetime.fromtimestamp(rec["timestamp"] / 1000, timezone.utc).date()
            except (KeyError, TypeError, ValueError, OSError, OverflowError):
                continue
            if when.year < since_year:
                continue
            theme = classify(text)
            if theme is None:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "asin": asin,
                    "text": text,
                    "when": when,
                    "theme": theme,
                    "votes": int(rec.get("helpful_vote") or 0),
                    "review_title": normalise(rec.get("title") or ""),
                }
            )

    titles = _amazon_titles({c["asin"] for c in candidates})

    out: list[RawSignal] = []
    for c in candidates:
        product = titles.get(c["asin"])
        if not product:
            continue
        if _NOT_SKINCARE.search(product) or not _SKINCARE_PRODUCT.search(product):
            continue
        out.append(
            RawSignal(
                # A product can have many reviews, so the ASIN alone is not unique.
                # hashlib, not hash() — the builtin is salted per process, which would
                # silently break ID stability across runs.
                native_key=f"amazon:{c['asin']}:{_text_fingerprint(c['text'])}",
                text=c["text"],
                source="amazon",
                source_detail=product[:90],
                url=f"https://www.amazon.com/dp/{c['asin']}",
                timestamp=c["when"],
                theme=c["theme"],
                rank_score=c["votes"],
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Selection and ID assignment (§5.1)
# --------------------------------------------------------------------------------------


def _text_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _sort_key(sig: RawSignal) -> tuple:
    """Deterministic ordering: strongest theme match, then popularity, then recency."""
    return (
        -score_themes(sig.text)[sig.theme],
        -sig.rank_score,
        -sig.timestamp.toordinal(),
        sig.native_key,
    )


def balance(
    reddit: list[RawSignal],
    amazon: list[RawSignal],
    target: int = 400,
    reddit_share: float = 0.55,
) -> list[RawSignal]:
    """Pick `target` signals spread across themes and both sources.

    Themes that one source cannot fill are topped up from the other, so a thin
    bucket degrades to a single-source bucket rather than to an empty one.
    """
    per_theme = max(1, round(target / len(ALL_THEMES)))
    reddit_quota = round(per_theme * reddit_share)

    by_theme: dict[str, dict[str, list[RawSignal]]] = {
        theme: {"reddit": [], "amazon": []} for theme in ALL_THEMES
    }
    for sig in reddit:
        by_theme[sig.theme]["reddit"].append(sig)
    for sig in amazon:
        by_theme[sig.theme]["amazon"].append(sig)

    chosen: list[RawSignal] = []
    for theme in ALL_THEMES:
        pools = by_theme[theme]
        r = sorted(pools["reddit"], key=_sort_key)
        a = sorted(pools["amazon"], key=_sort_key)
        take_r = r[:reddit_quota]
        take_a = a[: per_theme - len(take_r)]
        shortfall = per_theme - len(take_r) - len(take_a)
        if shortfall > 0:
            take_r += r[len(take_r) : len(take_r) + shortfall]
        shortfall = per_theme - len(take_r) - len(take_a)
        if shortfall > 0:
            take_a += a[len(take_a) : len(take_a) + shortfall]
        chosen.extend(take_r + take_a)

    # Stable final order so IDs run in a readable sequence.
    chosen.sort(key=lambda s: (ALL_THEMES.index(s.theme), s.source, _sort_key(s)))
    return chosen[:target]


def assign_ids(
    selected: list[RawSignal], ledger_path: Path
) -> tuple[list[dict], dict[str, str]]:
    """Mint `sig_NNNN` IDs, reusing any that a previous run already assigned.

    §5.1: IDs are assigned once and frozen. The ledger is what makes that true
    across re-runs and across four people's machines — it is committed to the repo.
    Nothing downstream of this function may mint an ID.
    """
    ledger: dict[str, str] = {}
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    used = {int(v.split("_")[1]) for v in ledger.values()} if ledger else set()
    next_n = 1

    records: list[dict] = []
    for sig in selected:
        sid = ledger.get(sig.native_key)
        if sid is None:
            while next_n in used:
                next_n += 1
            sid = f"sig_{next_n:04d}"
            used.add(next_n)
            ledger[sig.native_key] = sid
        records.append(
            {
                "id": sid,
                "text": sig.text,
                "source": sig.source,
                "source_detail": sig.source_detail,
                "url": sig.url,
                "timestamp": sig.timestamp.isoformat(),
                "theme": sig.theme,
            }
        )

    records.sort(key=lambda r: r["id"])
    return records, ledger
