# main.py
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Iterable

import feedparser
import pandas as pd
import requests
from dateutil import parser as dateparser
from pydantic import BaseModel, Field
import streamlit as st

# -------------------- LLM client (OpenAI) --------------------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


def get_openai_client(api_key: Optional[str]):
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package non installato. `pip install openai>=1.0.0`.")
    if not api_key:
        raise RuntimeError("Inserisci la tua OpenAI API key nella sidebar o in secrets.")
    return OpenAI(api_key=api_key)


# -------------------- Models --------------------
class Item(BaseModel):
    id: str
    title: str
    url: str
    source: str
    published: Optional[datetime] = None  # UTC aware
    summary_raw: str = ""
    tags: List[str] = Field(default_factory=list)

    # LLM-enriched fields
    summary_llm: Optional[str] = None
    venture_score: Optional[int] = None  # 0..100
    reasons: Optional[str] = None


# -------------------- Time helpers --------------------
AWARE_MIN = datetime.min.replace(tzinfo=timezone.utc)


def to_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_dt_aware_utc(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = dateparser.parse(s)
        return to_aware_utc(dt) if dt else None
    except Exception:
        return None


# -------------------- Fetching & parsing --------------------
@st.cache_data(show_spinner=False)
def fetch_feed(url: str, timeout: int = 15) -> Dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    return {"href": url, "feed": parsed}


def normalize_entry(entry: Dict[str, Any], source: str) -> Item:
    link = entry.get("link") or entry.get("id") or ""
    title = (entry.get("title") or "(no title)").strip()
    published = parse_dt_aware_utc(entry.get("published") or entry.get("updated"))
    summary = entry.get("summary") or ""

    tags: List[str] = []
    if entry.get("tags"):
        tags = [t.get("term") or t.get("label") or "" for t in entry["tags"]]
        tags = [t for t in tags if t]

    return Item(
        id=link or f"{source}:{hash(title)}",
        title=title,
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

    # de-dup by url or title
    dedup: Dict[str, Item] = {}
    for it in items:
        key = it.url or f"{it.source}:{it.title}"
        if key not in dedup:
            dedup[key] = it
    return list(dedup.values())


# -------------------- LLM prompts --------------------
VENTURE_PROMPT = (
   """You are a venture analyst. Read the item (title, blurb) and produce:
1) a crisp 2-3 sentence summary for investors;
2) a Venture Signal score 0-100 (higher = more interesting for early-stage VC);
3) 2-4 short bullet reasons (market, timing, team/signal, traction, novelty).
Respond in JSON with keys: summary, score (int), reasons (array of strings)."""
)


def call_llm_annotate(client, model: str, item: Item) -> Item:
    content = (
        f"TITLE: {item.title}\n"
        f"BLURB: {item.summary_raw[:1200]}\n"
        f"TAGS: {', '.join(item.tags) if item.tags else '-'}\n"
        f"URL: {item.url}"
    )
    try:
        chat = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VENTURE_PROMPT},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        text = chat.choices[0].message.content or "{}"
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


# -------------------- Global insights batching --------------------
def chunked(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def llm_batch_insights(client, model: str, items: List[Item], batch_size: int = 60) -> Optional[dict]:
    if not items:
        return None

    prompt = (
        "You are a venture analyst. Read the JSON list of items (title, blurb, source, tags) and GENERATE data: "
        "1) topics: array of {name, count} (3-8 items); "
        "2) buzzwords: array of {term, count} (5-15 items); "
        "3) insights: array of 3-6 bullets (<=20 words each); "
        "4) notable_projects: array of {title, reason}; "
        "5) notable_papers: array of {title, reason}; "
        "Only use the provided content. Respond as strict JSON."
    )

    batches = []
    for group in chunked(items[:240], batch_size):
        rows = [
            {
                "title": it.title,
                "blurb": (it.summary_llm or it.summary_raw)[:400],
                "source": it.source,
                "tags": it.tags[:6],
            }
            for it in group
        ]
        corpus = json.dumps(rows, ensure_ascii=False)
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": corpus},
                ],
                response_format={"type": "json_object"},
            )
            d = json.loads(chat.choices[0].message.content or "{}")
        except Exception as e:
            st.warning(f"AI insights batch error: {e}")
            d = {}
        for k in ["topics", "buzzwords", "insights", "notable_projects", "notable_papers"]:
            d.setdefault(k, [])
        batches.append(d)

    topics = defaultdict(int)
    buzz = defaultdict(int)
    insights: List[str] = []
    projects: List[Dict[str, str]] = []
    papers: List[Dict[str, str]] = []

    for b in batches:
        for t in b.get("topics", []):
            name = str(t.get("name", "")).strip()
            cnt = int(t.get("count", 0) or 0)
            if name:
                topics[name] += cnt
        for z in b.get("buzzwords", []):
            term = str(z.get("term", "")).strip().lower()
            cnt = int(z.get("count", 0) or 0)
            if term:
                buzz[term] += cnt
        insights.extend([str(x) for x in b.get("insights", [])])
        projects.extend([{"title": str(x.get("title", "")), "reason": str(x.get("reason", ""))} for x in b.get("notable_projects", [])])
        papers.extend([{"title": str(x.get("title", "")), "reason": str(x.get("reason", ""))} for x in b.get("notable_papers", [])])

    def dedupe_keep_order(seq: Iterable[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen and s:
                seen.add(s)
                out.append(s)
        return out

    insights = dedupe_keep_order(insights)[:8]

    topics_out = sorted(({"name": k, "count": v} for k, v in topics.items()), key=lambda x: x["count"], reverse=True)[:12]
    buzz_out = sorted(({"term": k, "count": v} for k, v in buzz.items()), key=lambda x: x["count"], reverse=True)[:20]
    projects = projects[:10]
    papers = papers[:10]

    return {
        "topics": topics_out,
        "buzzwords": buzz_out,
        "insights": insights,
        "notable_projects": projects,
        "notable_papers": papers,
    }


# -------------------- UI --------------------
st.set_page_config(page_title="Venture Signal — GitHub & arXiv", layout="wide", page_icon="🚀")

# --- Background video + CSS theme ---


# --- Header / Hero ---
try:
    col_logo, col_title = st.columns([2, 10], vertical_alignment="center")
    with col_logo:
        st.image("logo.png", use_container_width=True)
    with col_title:
        st.markdown("<h1>Venture Signal — GitHub Trending & arXiv</h1>", unsafe_allow_html=True)
except Exception:
    st.title("🚀 Venture Signal — GitHub Trending & arXiv")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    # Prefer secrets, fallback a input manuale
    #api_key = st.secrets.get("api_key", None)
    api_key = st.text_input("OpenAI API Key (opzionale)", type="password", value="api_key")

    model = st.selectbox(
        "OpenAI Model",
        options=[
            "gpt-5",         # default
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo"
        ],
        index=0  # di default seleziona gpt-5
    )

    st.header("Sources")
    feed_urls: Dict[str, List[str]] = {
        "github_trending": ["https://mshibanami.github.io/GitHubTrendingRSS/daily/all.xml"],
        "arxiv": [
            "https://rss.arxiv.org/atom/quant-ph",
            "https://rss.arxiv.org/atom/q-bio.QM",
            "https://rss.arxiv.org/atom/stat.ML",
        ],
    }
    custom_feeds = st.text_area("Extra RSS/Atom (una per riga)")
    extra_urls = [u.strip() for u in custom_feeds.splitlines() if u.strip()]
    if extra_urls:
        feed_urls["custom"] = extra_urls

    st.header("Filters")
    days_back = st.slider("Lookback (days)", 1, 30, 7)
    kw = st.text_input("Filtro keyword/regex", value="")

    st.header("LLM")
    colA, colB = st.columns(2)
    with colA:
        llm_enabled = st.toggle("Annota per item", value=True)
    with colB:
        insights_enabled = st.toggle("Insights globali", value=True)
    max_items = st.slider("Max items per feed", 5, 50, 20)
    

# --- Main glass container ---
with st.container():
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    # Fetch + filter
    items = collect_items(feed_urls, max_items_per_feed=max_items)

    lookback_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    items = [it for it in items if (it.published is None or it.published >= lookback_dt)]

    if kw:
        try:
            regex = re.compile(kw, re.IGNORECASE)
            items = [it for it in items if regex.search(it.title) or regex.search(it.summary_raw)]
        except re.error:
            st.warning("Regex non valida; filtro ignorato.")

    # Sort by recency first (safe)
    items.sort(key=lambda x: (x.published or AWARE_MIN).timestamp(), reverse=True)

    # Per-item LLM
    client = None
    if llm_enabled and items and api_key:
        try:
            client = get_openai_client(api_key)
            annotate_progress = st.progress(0.0, text="Annotazione LLM…")
            out: List[Item] = []
            with ThreadPoolExecutor(max_workers=4) as ex:
                futs = {ex.submit(call_llm_annotate, client, model, it): it for it in items}
                total = len(futs)
                done = 0
                for fut in as_completed(futs):
                    result = fut.result()
                    out.append(result)
                    done += 1
                    annotate_progress.progress(done / max(1, total), text=f"Annotazione… {done}/{total}")
            items = out
            annotate_progress.empty()
        except Exception as e:
            st.error(str(e))

    # Ranking finale: score desc poi recency
    items.sort(
        key=lambda x: (
            x.venture_score if x.venture_score is not None else -1,
            (x.published or AWARE_MIN).timestamp(),
        ),
        reverse=True,
    )

    # ---------------- Tabs Layout ----------------
    tab_feed, tab_leaderboard, tab_insights = st.tabs(["📚 Feed", "🏆 Leaderboard", "🧠 Insights"])

    # FEED: card grid
    with tab_feed:
        st.caption(f"Items dopo filtri: **{len(items)}**")
        # griglia a 3 colonne su desktop, 1-2 su mobile
        # Griglia senza API interne: scegli il numero di colonne in base a quanti item hai
        def pick_cols(n_items: int) -> int:
            if n_items >= 12:
                return 3
            if n_items >= 4:
                return 2
            return 1

        ncols = pick_cols(len(items))
        cols = st.columns(ncols)

        for idx, it in enumerate(items):
            with cols[idx % len(cols)]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                title_md = f'<a class="title-link" href="{it.url}" target="_blank" rel="noopener">{it.title}</a>'
                top_row = st.columns([8, 4])
                with top_row[0]:
                    st.markdown(f"### {title_md}", unsafe_allow_html=True)
                with top_row[1]:
                    if it.venture_score is not None:
                        st.markdown(f'<div class="score-badge">Signal {it.venture_score}</div>', unsafe_allow_html=True)
                meta_bits = []
                if it.source:
                    meta_bits.append(f"**Source:** {it.source}")
                if it.published:
                    meta_bits.append(f"**UTC:** {it.published.strftime('%Y-%m-%d %H:%M')}")
                if it.tags:
                    meta_bits.append(f"**Tags:** {', '.join(it.tags[:6])}")
                st.markdown(f'<div class="meta">{" · ".join(meta_bits)}</div>', unsafe_allow_html=True)

                st.write((it.summary_llm or it.summary_raw)[:650] + ("…" if len((it.summary_llm or it.summary_raw)) > 650 else ""))

                if it.reasons:
                    with st.expander("Why it matters"):
                        st.write(it.reasons)

                st.markdown('</div>', unsafe_allow_html=True)

    # LEADERBOARD: tabella compatta
    with tab_leaderboard:
        df = pd.DataFrame([
            {
                "Title": it.title,
                "Score": it.venture_score,
                "Source": it.source,
                "Published (UTC)": it.published,
                "URL": it.url,
            }
            for it in items
        ])
        # ordina per score disc poi data
        df = df.sort_values(by=["Score", "Published (UTC)"], ascending=[False, False], na_position="last")
        st.dataframe(df, use_container_width=True, hide_index=True)

    # INSIGHTS: grafici + liste
    with tab_insights:
        if (insights_enabled and api_key and items):
            try:
                client = client or get_openai_client(api_key)
                ai = llm_batch_insights(client, model, items)
                if ai:
                    colA, colB = st.columns(2)
                    with colA:
                        if ai.get("topics"):
                            st.markdown("#### Topics")
                            topics_df = pd.DataFrame(ai["topics"]).rename(columns={"name": "Topic", "count": "Count"})
                            if not topics_df.empty:
                                topics_df = topics_df.sort_values("Count", ascending=False)
                                st.bar_chart(topics_df.set_index("Topic")["Count"], use_container_width=True)
                    with colB:
                        if ai.get("buzzwords"):
                            st.markdown("#### Buzzwords")
                            buzz_df = pd.DataFrame(ai["buzzwords"]).rename(columns={"term": "Term", "count": "Count"})
                            if not buzz_df.empty:
                                buzz_df = buzz_df.sort_values("Count", ascending=False)
                                st.bar_chart(buzz_df.set_index("Term")["Count"], use_container_width=True)

                    if ai.get("insights"):
                        st.markdown("#### Key insights")
                        for b in ai["insights"][:8]:
                            st.write(f"• {b}")

                    col_p, col_r = st.columns(2)
                    with col_p:
                        if ai.get("notable_projects"):
                            st.markdown("#### Notable projects")
                            for n in ai["notable_projects"][:10]:
                                st.write(f"- **{n.get('title','(untitled)')}** — {n.get('reason','')}")
                    with col_r:
                        if ai.get("notable_papers"):
                            st.markdown("#### Notable papers")
                            for n in ai["notable_papers"][:10]:
                                st.write(f"- **{n.get('title','(untitled)')}** — {n.get('reason','')}")
                else:
                    st.caption("(Nessun insight generato)")
            except Exception as e:
                st.warning(f"Insights error: {e}")
        else:
            st.caption("Abilita LLM + API key per generare gli insights.")

    st.markdown('</div>', unsafe_allow_html=True)