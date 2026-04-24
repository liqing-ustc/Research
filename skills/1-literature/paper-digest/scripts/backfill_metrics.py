#!/usr/bin/env python3
"""
Batch-backfill influence metrics (S2 citation, HF upvotes, GitHub) for existing
Papers/ notes that were created before the Step 2.4 metrics step was added.

Design:
- Anonymous S2 batch API (up to 20 ids per call, sleep 5s between calls)
- Idempotent via Workbench/metrics_cache.jsonl — re-runs skip already-fetched short_titles
- Does NOT modify paper notes; writes cache for a separate merge step
- Conservative on shared-IP rate limits: exponential backoff, Scholar fallback left to merge step
- Uses only stdlib

Usage:
    python backfill_metrics.py                 # scan Papers/, fetch missing, append cache
    python backfill_metrics.py --dry-run       # list what would be fetched
    python backfill_metrics.py --limit 20      # cap this run (shared-IP safety valve)
    python backfill_metrics.py --force 2504-UWM,2202-DUET  # force refresh specific notes

Cache entry schema (one JSON per line):
    {
      "short_title": "2504-UWM",
      "measured_at": "YYYY-MM-DD",
      "arxiv_id": "2504.02792",
      "gh_slug": "owner/repo" | null,
      "date_publish": "YYYY-MM-DD" | null,
      "paper":  {"citation_count": N, "influential_citations": N,
                 "citation_velocity_per_month": N.NN, "months_since_publish": N.N,
                 "citation_source": "s2" | "s2_unavailable"} | null,
      "hf":     {"upvotes": N} | null,
      "github": {"stars": N, "forks": N, "open_issues": N,
                 "pushed_days_ago": N, "commits_last_90d": "N" | "100+",
                 "is_stale": bool} | null,
      "issues": ["..."]
    }
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

# --- paths -------------------------------------------------------------------
VAULT = Path(__file__).resolve().parents[4]  # skills/1-literature/paper-digest/scripts -> vault root
PAPERS_DIR = VAULT / "Papers"
CACHE = VAULT / "Workbench" / "metrics_cache.jsonl"

# --- tuning ------------------------------------------------------------------
S2_BATCH_SIZE = 20          # S2 accepts up to 500, stay polite on shared IP
S2_SLEEP_SEC = 5.0          # between S2 batch calls
HF_SLEEP_SEC = 0.5          # between HF single calls (HF is lenient)
GH_SLEEP_SEC = 0.2          # gh api is 5000/hr authed, this is just courtesy
MAX_S2_RETRIES = 3
BACKOFF_BASE = 10.0         # seconds; 10, 30, 90 for retries
HTTP_TIMEOUT = 20.0

TODAY = date.today().isoformat()

# --- regex for parsing -------------------------------------------------------
RE_ARXIV = re.compile(r"arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,6})")
RE_GH = re.compile(r'github\.com/([^/\s"\']+/[^/\s"\']+?)(?:\.git(?=[/\s"\'?#]|$)|[/#?"\']|$)')
RE_METRICS_LINE = re.compile(r"^\*\*Metrics\*\* \(as of ", re.MULTILINE)
RE_FRONTMATTER = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


@dataclass
class PaperInput:
    short_title: str          # "2504-UWM"
    path: Path
    arxiv_id: str | None
    gh_slug: str | None
    date_publish: str | None  # YYYY-MM-DD or None
    already_has_metrics: bool


@dataclass
class PaperOutput:
    short_title: str
    measured_at: str = TODAY
    arxiv_id: str | None = None
    gh_slug: str | None = None
    date_publish: str | None = None
    paper: dict[str, Any] | None = None
    hf: dict[str, Any] | None = None
    github: dict[str, Any] | None = None
    issues: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "short_title": self.short_title,
                "measured_at": self.measured_at,
                "arxiv_id": self.arxiv_id,
                "gh_slug": self.gh_slug,
                "date_publish": self.date_publish,
                "paper": self.paper,
                "hf": self.hf,
                "github": self.github,
                "issues": self.issues,
            },
            ensure_ascii=False,
        )


# --- parsing -----------------------------------------------------------------
def parse_paper(path: Path) -> PaperInput:
    text = path.read_text(encoding="utf-8", errors="replace")
    already = bool(RE_METRICS_LINE.search(text))
    fm_match = RE_FRONTMATTER.search(text)
    fm = fm_match.group(1) if fm_match else ""

    arxiv_id = None
    gh_slug = None
    date_publish = None
    for line in fm.splitlines():
        line = line.rstrip()
        if line.startswith("paper:"):
            m = RE_ARXIV.search(line)
            if m:
                arxiv_id = m.group(1)
        elif line.startswith("github:"):
            m = RE_GH.search(line)
            if m:
                gh_slug = m.group(1)
        elif line.startswith("date_publish:"):
            v = line.split(":", 1)[1].strip()
            # tolerate YYYY-MM-DD or YYYY-MM
            if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                date_publish = v
            elif re.match(r"^\d{4}-\d{2}$", v):
                date_publish = v + "-01"

    # NOTE: no whole-doc fallback — body hits match *cited* papers' arxiv IDs
    # (bug found on 2604-GEN1: empty frontmatter `paper:` → body fallback grabbed
    # GPT-3's 2005.14165 from a reference). Treat empty/non-arxiv frontmatter as
    # "no arxiv" and let the cache record `no_arxiv_id_in_note`.

    return PaperInput(
        short_title=path.stem,
        path=path,
        arxiv_id=arxiv_id,
        gh_slug=gh_slug,
        date_publish=date_publish,
        already_has_metrics=already,
    )


# --- cache -------------------------------------------------------------------
def load_cache() -> dict[str, dict]:
    """Return {short_title: last_entry_dict} — most-recent wins on duplicates."""
    if not CACHE.exists():
        return {}
    out: dict[str, dict] = {}
    with CACHE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
                out[e["short_title"]] = e
            except json.JSONDecodeError:
                continue
    return out


def append_cache(entry: PaperOutput) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as f:
        f.write(entry.to_json() + "\n")


# --- HTTP --------------------------------------------------------------------
def http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method=method, data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except (urllib.error.URLError, TimeoutError) as e:
        return 0, str(e).encode()


# --- S2 fetch ----------------------------------------------------------------
def s2_batch(arxiv_ids: list[str]) -> dict[str, dict] | None:
    """Return {arxiv_id: s2_record}, or None on fatal failure."""
    if not arxiv_ids:
        return {}
    body = json.dumps({"ids": [f"arXiv:{a}" for a in arxiv_ids]}).encode()
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/batch"
        "?fields=citationCount,influentialCitationCount,publicationDate"
    )
    headers = {"Content-Type": "application/json"}
    for attempt in range(MAX_S2_RETRIES):
        code, payload = http_request(url, method="POST", headers=headers, body=body)
        if code == 200:
            try:
                arr = json.loads(payload)
            except json.JSONDecodeError:
                return None
            out: dict[str, dict] = {}
            # response is a list aligned with request order; nulls for not-found
            for req_id, rec in zip(arxiv_ids, arr):
                if rec:
                    out[req_id] = rec
            return out
        if code in (429, 500, 502, 503, 504) and attempt < MAX_S2_RETRIES - 1:
            wait = BACKOFF_BASE * (3 ** attempt)  # 10, 30, 90
            print(f"  S2 {code}, backoff {wait:.0f}s (attempt {attempt + 1}/{MAX_S2_RETRIES})",
                  file=sys.stderr)
            time.sleep(wait)
            continue
        print(f"  S2 batch failed: HTTP {code}", file=sys.stderr)
        return None
    return None


def months_between(d_pub: str, d_now: str = TODAY) -> float:
    a = datetime.fromisoformat(d_pub).date()
    b = datetime.fromisoformat(d_now).date()
    return (b - a).days / 30.4375


def shape_s2(arxiv_id: str, rec: dict | None, fm_date: str | None) -> tuple[dict | None, str | None]:
    """(paper_block, effective_date_publish)"""
    if rec is None:
        return None, fm_date
    cc = rec.get("citationCount")
    ic = rec.get("influentialCitationCount")
    pub = rec.get("publicationDate") or fm_date
    months = round(months_between(pub), 1) if pub else None
    vel = round(cc / max(months, 1.0), 2) if (cc is not None and months is not None) else None
    return {
        "citation_count": cc,
        "influential_citations": ic,
        "citation_velocity_per_month": vel,
        "months_since_publish": months,
        "citation_source": "s2",
    }, pub


# --- HF fetch ----------------------------------------------------------------
def hf_upvotes(arxiv_id: str) -> dict | None:
    code, payload = http_request(f"https://huggingface.co/api/papers/{arxiv_id}")
    if code == 200:
        try:
            data = json.loads(payload)
            return {"upvotes": data.get("upvotes", 0)}
        except json.JSONDecodeError:
            return None
    if code == 404:
        return None  # paper not on HF — legit absence, not a fetch failure
    return None


# --- GitHub fetch ------------------------------------------------------------
def have_gh_cli() -> bool:
    try:
        r = subprocess.run(["gh", "auth", "status"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def gh_metrics(slug: str) -> dict | None:
    """Returns github block or None. Requires `gh` CLI authed."""
    try:
        r = subprocess.run(
            ["gh", "api", f"repos/{slug}"],
            capture_output=True, timeout=15, text=True,
        )
        if r.returncode != 0:
            return None
        repo = json.loads(r.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None

    since = (datetime.now(timezone.utc).date().toordinal() - 90)
    since_iso = date.fromordinal(since).isoformat() + "T00:00:00Z"
    try:
        r = subprocess.run(
            ["gh", "api", f"repos/{slug}/commits?since={since_iso}&per_page=100"],
            capture_output=True, timeout=20, text=True,
        )
        commits_90d: str
        if r.returncode == 0:
            arr = json.loads(r.stdout)
            commits_90d = "100+" if len(arr) >= 100 else str(len(arr))
        else:
            commits_90d = "N/A"
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        commits_90d = "N/A"

    pushed_at = repo.get("pushed_at")
    if pushed_at:
        pushed_dt = datetime.fromisoformat(pushed_at.replace("Z", "+00:00"))
        pushed_days = (datetime.now(timezone.utc) - pushed_dt).days
    else:
        pushed_days = None

    is_stale = bool(
        pushed_days is not None and pushed_days > 180 and commits_90d == "0"
    )
    return {
        "stars": repo.get("stargazers_count"),
        "forks": repo.get("forks_count"),
        "open_issues": repo.get("open_issues_count"),
        "pushed_days_ago": pushed_days,
        "commits_last_90d": commits_90d,
        "is_stale": is_stale,
    }


# --- main --------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list targets, no fetching")
    ap.add_argument("--limit", type=int, default=0, help="cap papers fetched this run (0 = no cap)")
    ap.add_argument("--force", default="", help="comma-separated short_titles to refresh even if cached")
    ap.add_argument("--include-with-metrics", action="store_true",
                    help="also process notes that already have a **Metrics** line")
    args = ap.parse_args()

    force_set = {s.strip() for s in args.force.split(",") if s.strip()}
    cache = load_cache()

    # Scan targets
    targets: list[PaperInput] = []
    no_arxiv: list[str] = []
    skipped_has_metrics = 0
    skipped_cached = 0

    for path in sorted(PAPERS_DIR.glob("*.md")):
        p = parse_paper(path)
        if p.already_has_metrics and not args.include_with_metrics and p.short_title not in force_set:
            skipped_has_metrics += 1
            continue
        if p.short_title in cache and p.short_title not in force_set:
            skipped_cached += 1
            continue
        if not p.arxiv_id:
            no_arxiv.append(p.short_title)
            # still cache a stub so we don't re-probe forever
            targets.append(p)
            continue
        targets.append(p)

    print(f"scan: {len(list(PAPERS_DIR.glob('*.md')))} notes "
          f"| skipped(has_metrics)={skipped_has_metrics} "
          f"| skipped(cached)={skipped_cached} "
          f"| to_fetch={len(targets)} "
          f"| no_arxiv={len(no_arxiv)}")
    if no_arxiv:
        print(f"  no_arxiv_ids: {', '.join(no_arxiv[:10])}"
              f"{' ...' if len(no_arxiv) > 10 else ''}")

    if args.limit > 0 and len(targets) > args.limit:
        print(f"  limit applied: truncating to first {args.limit}")
        targets = targets[: args.limit]

    if args.dry_run:
        for t in targets:
            print(f"  {t.short_title}  arxiv={t.arxiv_id}  gh={t.gh_slug}")
        return 0

    if not targets:
        print("nothing to do.")
        return 0

    gh_available = have_gh_cli()
    if not gh_available:
        print("warning: gh CLI not authed — github block will be null", file=sys.stderr)

    # --- S2 in batches ---
    with_arxiv = [t for t in targets if t.arxiv_id]
    s2_by_id: dict[str, dict | None] = {}
    for i in range(0, len(with_arxiv), S2_BATCH_SIZE):
        batch = with_arxiv[i : i + S2_BATCH_SIZE]
        ids = [t.arxiv_id for t in batch]
        print(f"S2 batch {i // S2_BATCH_SIZE + 1}/{(len(with_arxiv) - 1) // S2_BATCH_SIZE + 1}  "
              f"({len(ids)} ids)")
        recs = s2_batch(ids)
        if recs is None:
            for t in batch:
                s2_by_id[t.arxiv_id] = None
        else:
            for t in batch:
                s2_by_id[t.arxiv_id] = recs.get(t.arxiv_id)
        if i + S2_BATCH_SIZE < len(with_arxiv):
            time.sleep(S2_SLEEP_SEC)

    # --- per-paper HF + GH + emit ---
    n_ok = n_s2_miss = n_hf_miss = n_gh_miss = 0
    for t in targets:
        out = PaperOutput(
            short_title=t.short_title,
            arxiv_id=t.arxiv_id,
            gh_slug=t.gh_slug,
            date_publish=t.date_publish,
        )
        if not t.arxiv_id:
            out.issues.append("no_arxiv_id_in_note")
        else:
            rec = s2_by_id.get(t.arxiv_id)
            paper_block, eff_date = shape_s2(t.arxiv_id, rec, t.date_publish)
            out.paper = paper_block
            out.date_publish = eff_date
            if paper_block is None:
                out.issues.append("s2_unavailable")
                n_s2_miss += 1

            out.hf = hf_upvotes(t.arxiv_id)
            if out.hf is None:
                n_hf_miss += 1
            time.sleep(HF_SLEEP_SEC)

        if gh_available and t.gh_slug:
            out.github = gh_metrics(t.gh_slug)
            if out.github is None:
                out.issues.append(f"github_fetch_failed:{t.gh_slug}")
                n_gh_miss += 1
            time.sleep(GH_SLEEP_SEC)
        elif t.gh_slug:
            out.issues.append("gh_cli_unavailable")
            n_gh_miss += 1

        append_cache(out)
        tag_s2 = "✓" if out.paper else "✗"
        tag_hf = "✓" if out.hf else "·"
        tag_gh = "✓" if out.github else ("·" if not t.gh_slug else "✗")
        print(f"  [{tag_s2}{tag_hf}{tag_gh}] {t.short_title}  "
              f"cite={(out.paper or {}).get('citation_count', '—')}  "
              f"upvotes={(out.hf or {}).get('upvotes', '—')}  "
              f"stars={(out.github or {}).get('stars', '—')}")
        n_ok += 1

    print(
        f"\ndone: {n_ok} written to {CACHE.relative_to(VAULT)}  "
        f"| s2_miss={n_s2_miss} hf_miss={n_hf_miss} gh_miss={n_gh_miss}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
