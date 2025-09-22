"""
Streamlit PoC — Venture Signal from GitHub Trending & arXiv (RSS)
-----------------------------------------------------------------
A quick proof‑of‑concept that pulls items from GitHub Trending and arXiv RSS feeds,
then uses an LLM to summarize and produce a heuristic "Venture Signal" score.

How to run locally
------------------
1) Create & activate a virtualenv (Python 3.10+ recommended).
2) Install deps: `pip install -r requirements.txt` (see inline list below) or copy the minimal set:
   feedparser, requests, pydantic, python-dateutil, pandas, streamlit, openai
3) Run: `streamlit run app.py`

Minimal requirements.txt
------------------------
feedparser
requests
pydantic
python-dateutil
pandas
streamlit
openai>=1.0.0

Notes
-----
- The user will input their OpenAI API key directly from the sidebar (no need for env var).
- Feeds used by default:
  • GitHub Trending (daily, all languages): https://mshibanami.github.io/GitHubTrendingRSS/daily/all.xml
  • arXiv: quant-ph, q-bio.QM, stat.ML
- You can add more feeds in the sidebar.
"""

import re
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any

import feedparser
import pandas as pd
import requests
from dateutil import parser as dateparser
from pydantic import BaseModel, Field
import streamlit as st

# --- LLM Client (OpenAI) -----------------------------------------------------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


def get_openai_client(api_key: str):
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. `pip install openai>=1.0.0`.")
    if not api_key:
        raise RuntimeError("Please enter your OpenAI API key in the sidebar.")
    return OpenAI(api_key=api_key)


# --- Models ------------------------------------------------------------------
class Item(BaseModel):
    id: str
    title: str
    url: str
    source: str
    published: Optional[datetime] = None
    summary_raw: str = ""
    tags: List[str] = Field(default_factory=list)

    # LLM-enriched fields
    summary_llm: Optional[str] = None
    venture_score: Optional[int] = None  # 0..100
    reasons: Optional[str] = None


# --- Utilities ----------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_feed(url: str, timeout: int = 15) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    return {"href": url, "feed": parsed}


def _first_nonempty(*vals):
    for v in vals:
        if v:
            return v
    return ""


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dateparser.parse(s)
        if dt is None:
            return None
        # Normalize to timezone-aware UTC
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None
    try:
        return dateparser.parse(s)
    except Exception:
        return None


def normalize_entry(entry: Dict[str, Any], source: str) -> Item:
    link = entry.get("link") or entry.get("id") or ""
    title = entry.get("title", "(no title)")
    published = _parse_dt(_first_nonempty(entry.get("published"), entry.get("updated")))
    summary = entry.get("summary", "")

    tags = []
    if "tags" in entry and entry["tags"]:
        tags = [t.get("term") or t.get("label") or "" for t in entry["tags"]]
        tags = [t for t in tags if t]

    return Item(
        id=link or f"{source}:{hash(title)}",
        title=title.strip(),
        url=link,
        source=source,
        published=published,
        summary_raw=summary,
        tags=tags,
    )


def collect_items(feed_urls: Dict[str, List[str]], max_items_per_feed: int = 30) -> List[Item]:
    jobs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for source, urls in feed_urls.items():
            for u in urls:
                jobs.append(ex.submit(fetch_feed, u))

        items: List[Item] = []
        for fut in as_completed(jobs):
            try:
                res = fut.result()
                href = res["href"]
                parsed = res["feed"]
                src_label = next((s for s, lst in feed_urls.items() if href in lst), "feed")
                entries = parsed.get("entries", [])[:max_items_per_feed]
                for e in entries:
                    items.append(normalize_entry(e, src_label))
            except Exception as e:
                st.warning(f"Feed error: {e}")

    dedup: Dict[str, Item] = {}
    for it in items:
        key = it.url or f"{it.source}:{it.title}"
        if key not in dedup:
            dedup[key] = it
    return list(dedup.values())


# --- LLM prompts --------------------------------------------------------------
VENTURE_PROMPT = (
    "You are a venture analyst. Read the item (title, blurb) and produce:\n"
    "1) a crisp 2-3 sentence summary for investors;\n"
    "2) a Venture Signal score 0-100 (higher = more interesting for early-stage VC);\n"
    "3) 2-4 short bullet reasons (market, timing, team/signal, traction, novelty).\n"
    "Respond in JSON with keys: summary, score (int), reasons (array of strings)."
)


def call_llm_annotate(client, model: str, item: Item, temperature: float = 0.2) -> Item:
    content = (
        f"TITLE: {item.title}\n"
        f"BLURB: {item.summary_raw[:1200]}\n"
        f"TAGS: {', '.join(item.tags) if item.tags else '-'}\n"
        f"URL: {item.url}"
    )
    try:
        chat = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": VENTURE_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        text = chat.choices[0].message.content or "{}"
        import json
        data = json.loads(text)
        item.summary_llm = data.get("summary")
        try:
            item.venture_score = int(data.get("score"))
        except Exception:
            item.venture_score = None
        reasons = data.get("reasons")
        if isinstance(reasons, list):
            item.reasons = "\n".join(f"• {r}" for r in reasons)
    except Exception as e:
        item.summary_llm = f"LLM error: {e}"
    return item


# --- Streamlit UI -------------------------------------------------------------
st.set_page_config(page_title="Venture Signal — GitHub & arXiv", layout="wide")

st.image('logo.png', width=200)
st.title("🚀 Venture Signal — GitHub Trending & arXiv (PoC)")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    model = st.text_input("OpenAI Model", value="gpt-4o-mini")

    st.header("Sources")
    feed_urls = {
        "github_trending": ["https://mshibanami.github.io/GitHubTrendingRSS/daily/all.xml"],
        "arxiv": [
            "https://rss.arxiv.org/atom/quant-ph",
            "https://rss.arxiv.org/atom/q-bio.QM",
            "https://rss.arxiv.org/atom/stat.ML",
        ],
    }

    custom_feeds = st.text_area("Extra RSS/Atom feed URLs (one per line)", value="")
    extra_urls = [u.strip() for u in custom_feeds.splitlines() if u.strip()]
    if extra_urls:
        feed_urls["custom"] = extra_urls

    st.header("Filters")
    days_back = st.slider("Lookback window (days)", 1, 30, 7)
    kw = st.text_input("Keyword filter (optional, regex supported)", value="")

    st.header("LLM")
    llm_enabled = st.toggle("Use LLM to summarize & score", value=True)
    max_items = st.slider("Max items per feed", 5, 50, 20)
    temp = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.1)

# Fetch feeds
items = collect_items(feed_urls, max_items_per_feed=max_items)
lookback_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
items = [it for it in items if (it.published is None or it.published >= lookback_dt)]

if kw:
    try:
        regex = re.compile(kw, re.IGNORECASE)
        items = [it for it in items if regex.search(it.title) or regex.search(it.summary_raw)]
    except re.error:
        st.warning("Invalid regex; skipping keyword filter.")

st.write(f"Fetched {len(items)} items after filters.")
items.sort(key=lambda x: x.published or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

if llm_enabled and items and api_key:
    try:
        client = get_openai_client(api_key)
    except Exception as e:
        st.error(str(e))
        client = None

    if client:
        annotate_progress = st.progress(0.0, text="Annotating with LLM…")
        out: List[Item] = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(call_llm_annotate, client, model, it, temp): it for it in items}
            total = len(futs)
            done = 0
            for fut in as_completed(futs):
                result = fut.result()
                out.append(result)
                done += 1
                annotate_progress.progress(done / max(1, total), text=f"Annotating… {done}/{total}")
        items = out
        annotate_progress.empty()

items.sort(
    key=lambda x: (
        x.venture_score if x.venture_score is not None else -1,
        x.published or datetime.min.replace(tzinfo=timezone.utc)
    ),
    reverse=True
)

st.subheader("📊 Results")
cols = st.columns([3, 2])

with cols[0]:
    for it in items:
        with st.container(border=True):
            st.markdown(f"### [{it.title}]({it.url})")
            meta = []
            if it.source:
                meta.append(f"**Source:** {it.source}")
            if it.published:
                meta.append(f"**Published:** {it.published.strftime('%Y-%m-%d %H:%M UTC')}")
            if it.tags:
                meta.append(f"**Tags:** {', '.join(it.tags[:8])}")
            st.caption(" · ".join(meta))

            if it.summary_llm:
                st.write(it.summary_llm)
            else:
                st.write(it.summary_raw[:600] + ("…" if len(it.summary_raw) > 600 else ""))

            if it.venture_score is not None:
                st.metric(label="Venture Signal", value=it.venture_score)
            if it.reasons:
                with st.expander("Why it matters"):
                    st.write(it.reasons)

with cols[1]:
    df = pd.DataFrame([
        {
            "title": it.title,
            "source": it.source,
            "published": it.published,
            "score": it.venture_score,
            "url": it.url,
        }
        for it in items
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
