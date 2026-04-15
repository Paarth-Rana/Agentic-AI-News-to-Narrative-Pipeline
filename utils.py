import re
import json
import requests
from urllib.parse import quote

BASE_OUT = "outputs"

def clean_one_line(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def normalize_query(q: str) -> str:
    q = clean_one_line(q)
    q = re.sub(r'^(search query:|query:)\s*', '', q, flags=re.I).strip()
    q = q.replace('"', '').replace("“", "").replace("”", "").strip()
    q = re.split(r"\b(SEARCH_QUERY|SEARCH QUERY|WIKI|WIKIPEDIA|LINK|URL)\b\s*:?", q, flags=re.I)[0].strip()
    return q

def normalize_title_or_url(x: str) -> str:
    x = clean_one_line(x)
    m = re.search(r"wikipedia\.org\/wiki\/([^#\?]+)", x, flags=re.I)
    if m:
        return requests.utils.unquote(m.group(1)).replace("_", " ").strip()
    x = re.sub(r'^(title:)\s*', '', x, flags=re.I).strip()
    x = x.replace('"', '').replace("“", "").replace("”", "").strip()
    return x

def wiki_search(query: str, limit: int = 12):
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": limit},
        headers={"User-Agent": "WikiLangGraphColab"},
        timeout=30
    )
    r.raise_for_status()
    return r.json().get("query", {}).get("search", [])

def wiki_summary(title_or_url: str):
    title = normalize_title_or_url(title_or_url)
    if not title:
        return {"title": "", "extract": "", "url": ""}

    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(title, safe="")
        r = requests.get(url, headers={"User-Agent": "WikiLangGraphColab"}, timeout=30)
        if r.status_code == 200:
            j = r.json()
            return {
                "title": j.get("title", title),
                "extract": (j.get("extract") or "").strip(),
                "url": (j.get("content_urls", {}).get("desktop", {}) or {}).get("page", "") or "",
            }
    except Exception:
        pass

    return {"title": title, "extract": "", "url": ""}

def extract_first_json_object(s: str):
    if not s:
        return None
    start = s.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s[start:])
        return obj
    except Exception:
        return None

def fallback_split(script: str, n: int):
    words = (script or "").split()
    if not words:
        return [{"section_text": ""} for _ in range(n)]
    per = max(70, len(words) // n)
    out = []
    for i in range(n):
        chunk = " ".join(words[i * per:(i + 1) * per]).strip()
        out.append({"section_text": chunk})
    return out
