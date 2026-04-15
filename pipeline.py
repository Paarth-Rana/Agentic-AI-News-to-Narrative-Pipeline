import os
import re
import json
import time
import torch
import numpy as np
import soundfile as sf

from typing import Dict, Any, TypedDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoProcessor,
    BarkModel
)
from diffusers import StableDiffusionXLPipeline
from langgraph.graph import StateGraph, END

from utils import (
    BASE_OUT,
    clean_one_line,
    normalize_query,
    normalize_title_or_url,
    wiki_search,
    wiki_summary,
    extract_first_json_object,
    fallback_split
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TTS_NAME = "suno/bark-small"
IMG_MODEL = "stabilityai/sdxl-turbo"
N_SCENES = 8
NEG = "text, watermark, caption, logo, poster, collage"

os.makedirs(BASE_OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(LLM_NAME, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=(torch.float16 if torch.cuda.is_available() else torch.float32),
    device_map="auto" if torch.cuda.is_available() else None,
).eval()

tts_processor = AutoProcessor.from_pretrained(TTS_NAME)
tts_model = BarkModel.from_pretrained(
    TTS_NAME,
    torch_dtype=dtype
).to(device).eval()

pipe = StableDiffusionXLPipeline.from_pretrained(
    IMG_MODEL,
    torch_dtype=dtype,
    variant="fp16" if device == "cuda" else None,
    use_safetensors=True
).to(device)
pipe.set_progress_bar_config(disable=True)

@torch.inference_mode()
def llm(prompt: str, max_new_tokens: int = 300, temperature: float = 0.0, top_p: float = 0.9, top_k: int = 50) -> str:
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        attention_mask=attention_mask,
    )

    if temperature and float(temperature) > 0:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )
    else:
        gen_kwargs.update(do_sample=False)

    out = model.generate(input_ids=input_ids, **gen_kwargs)
    gen = out[0][input_ids.shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

def tts(text, path):
    inputs = tts_processor(text, voice_preset="v2/en_speaker_6", return_tensors="pt").to(device)
    audio = tts_model.generate(**inputs)
    audio_np = audio.cpu().numpy().squeeze()
    sf.write(path, audio_np.astype(np.float32), 24000)

@torch.inference_mode()
def gen_image(prompt, path):
    img = pipe(
        prompt=prompt,
        negative_prompt=NEG,
        num_inference_steps=4,
        guidance_scale=0.0
    ).images[0]
    img.save(path)

BAD_PATTERNS = re.compile(
    r"\b("
    r"history of|timeline of|list of|overview|in the united states|in the u\.s\.|"
    r"in (the )?united states|in america|by country|by state|"
    r"prohibition|crime in|corruption in|politics of|"
    r"mass incarceration|organized crime|gangs in|"
    r"era|period|movement|"
    r"category|outline"
    r")\b",
    re.I
)

GOOD_PATTERNS = re.compile(
    r"\b("
    r"murder|kidnapping|abduction|disappearance|heist|robbery|fraud|scandal|"
    r"bombing|shooting|massacre|assassination|poisoning|conspiracy|"
    r"trial|case|affair|incident|attack|plot|"
    r"killer|serial killer|murderer|con man|"
    r"cult|hostage|terror|"
    r"watergate|enron|theranos|"
    r")\b",
    re.I
)

def looks_like_specific(title: str) -> bool:
    t = (title or "").strip()
    if not t:
        return False
    if BAD_PATTERNS.search(t):
        return False
    return True

def node_make_query(state: Dict[str, Any]):
    seed = clean_one_line(state.get("query") or "")
    prompt = f"""Pick ONE random compelling true-crime / mystery / criminal / scandal / heist topic
that has a dedicated Wikipedia page and is mostly 1900s+ (OK: late 1800s).
Return ONE Wikipedia search query (4–10 words), one line only, no quotes.

User input (optional): {seed if seed else "(none)"}
"""
    q = normalize_query(llm(prompt, max_new_tokens=32, temperature=0.7))
    if not q:
        q = seed if seed else "famous 20th century heist"
    return {"query": q}

def node_search(state: Dict[str, Any]):
    q = clean_one_line(state.get("query", ""))
    hits = wiki_search(q, limit=12)
    cands = [
        {
            "title": (h.get("title") or "").strip(),
            "snippet": clean_one_line(re.sub("<.*?>", "", h.get("snippet", "")))
        }
        for h in hits
    ]
    cands = [c for c in cands if c["title"]]
    return {"candidates": cands}

def node_pick_source(state: Dict[str, Any]):
    cands = state.get("candidates") or []
    if not cands:
        return {"source_title": "D. B. Cooper"}

    filtered = []
    for c in cands:
        title = normalize_title_or_url(c.get("title", ""))
        if not title:
            continue
        if looks_like_specific(title):
            filtered.append({"title": title, "snippet": c.get("snippet", "")})

    if not filtered:
        filtered = [
            {"title": normalize_title_or_url(c.get("title", "")), "snippet": c.get("snippet", "")}
            for c in cands if c.get("title")
        ]

    scored = []
    for c in filtered:
        t = c["title"]
        sn = c.get("snippet") or ""
        score = 0
        if GOOD_PATTERNS.search(t):
            score += 3
        if GOOD_PATTERNS.search(sn):
            score += 2
        if re.search(r"\bin the\b", t, re.I):
            score -= 2
        if re.search(r"\bof the\b", t, re.I):
            score -= 1
        scored.append((score, t, sn))

    scored.sort(reverse=True)
    top = scored[:8]

    options = "\n".join([f"- {t}: {sn}" for _, t, sn in top])
    prompt = f"""Pick ONE Wikipedia page title that is ONE specific person OR ONE specific case/event.
Constraints:
- Prefer 1900+ (late 1800s ok)
- Avoid broad eras/categories (no “in the United States”, no “history of”, no “timeline of”, no “list of”)
Return ONLY JSON: {{"title":"..."}}
Do NOT return a URL.

Options:
{options}
"""
    try:
        raw = llm(prompt, max_new_tokens=120, temperature=0.2)
        obj = extract_first_json_object(raw) or {}
        picked = normalize_title_or_url(obj.get("title", ""))
        if picked and looks_like_specific(picked):
            return {"source_title": picked}
    except Exception:
        pass

    for _, t, _ in top:
        if looks_like_specific(t):
            return {"source_title": t}

    return {"source_title": top[0][1]}

def node_fetch_summary(state: Dict[str, Any]):
    info = wiki_summary(state.get("source_title", ""))
    return {
        "source_title": info["title"],
        "source_url": info["url"],
        "source_text": info["extract"]
    }

def node_write_script(state: Dict[str, Any]):
    title = clean_one_line(state.get("source_title", ""))
    src = (state.get("source_text") or "").strip()
    if not src:
        return {"script": f"Not enough Wikipedia summary text for {title}. Try a different run."}

    prompt = f"""Write a compelling, factual true-crime/mystery narration about: {title}

Rules:
- Use ONLY the SOURCE text. No inventions.
- 500–700 words.
- Chronological, documentary tone.
- Output narration only.

SOURCE:
{src[:3500]}
"""
    script = clean_one_line(llm(prompt, max_new_tokens=1000, temperature=0.0))
    return {"script": script}

def node_split_sections(state: Dict[str, Any]):
    script = (state.get("script") or "").strip()
    prompt = f"""Split the narration into exactly {N_SCENES} sections.
Return ONLY JSON:
{{"sections":[{{"section_text":"..."}}, ...]}}

Rules:
- Exactly {N_SCENES} items
- Each 60–95 words
- Keep order
- No extra text outside JSON

NARRATION:
{script}
"""
    raw = llm(prompt, max_new_tokens=1200, temperature=0.0)
    obj = extract_first_json_object(raw)

    if isinstance(obj, dict) and isinstance(obj.get("sections"), list):
        secs = []
        for s in obj["sections"][:N_SCENES]:
            t = clean_one_line(s.get("section_text", ""))
            if t:
                secs.append({"section_text": t})
        if len(secs) == N_SCENES:
            return {"sections": secs}

    return {"sections": fallback_split(script, N_SCENES)}

def node_make_prompts(state: Dict[str, Any]):
    title = clean_one_line(state.get("source_title", ""))
    sections = state.get("sections") or []
    out = []

    for sec in sections:
        st = clean_one_line(sec.get("section_text", ""))

        prompt = f"""Return ONLY JSON:
{{
  "image_prompt": "realistic documentary photo prompt (<=240 chars)",
  "narration": "60–95 words, only based on the section text"
}}

Must-follow:
- image_prompt MUST explicitly mention "{title}"
- keep image content consistent with the story (no nature/travel/random rainforest)
- realistic documentary photo, natural lighting
- no text/logos/watermarks

SECTION TEXT:
{st}
"""
        raw = llm(prompt, max_new_tokens=260, temperature=0.0)
        obj = extract_first_json_object(raw) or {}
        ip = clean_one_line(obj.get("image_prompt", ""))
        nar = clean_one_line(obj.get("narration", ""))

        if title.lower() not in ip.lower():
            ip = f"{title}. {ip}" if ip else f"Documentary photo about {title}, realistic, natural lighting, no text"
        if not nar:
            nar = st

        out.append({
            "section_text": st,
            "image_prompt": ip,
            "narration": nar
        })

    return {"sections": out}

def node_generate_assets(state):
    title = clean_one_line(state.get("source_title", "story")) or "story"
    run_id = str(int(time.time()))
    folder = os.path.join(
        BASE_OUT,
        re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60] + "_" + run_id
    )
    os.makedirs(folder, exist_ok=True)

    story_path = os.path.join(folder, "story.txt")
    sections_path = os.path.join(folder, "sections.json")
    prompts_path = os.path.join(folder, "image_prompts.txt")
    manifest_path = os.path.join(folder, "manifest.json")

    with open(story_path, "w", encoding="utf-8") as f:
        f.write(state.get("script", "") or "")

    with open(sections_path, "w", encoding="utf-8") as f:
        json.dump(state.get("sections", []), f, ensure_ascii=False, indent=2)

    with open(prompts_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(state.get("sections", []), 1):
            f.write(f"{i}. {s.get('image_prompt', '')}\n")

    items = []
    for i, s in enumerate(state.get("sections", []), 1):
        img_path = os.path.join(folder, f"img_{i:02d}.png")
        wav_path = os.path.join(folder, f"audio_{i:02d}.wav")

        gen_image(s.get("image_prompt", ""), img_path)
        tts(s.get("narration", ""), wav_path)

        items.append({
            "index": i,
            "image_prompt": s.get("image_prompt", ""),
            "narration": s.get("narration", ""),
            "image_path": img_path,
            "audio_path": wav_path,
        })

    manifest = {
        "query": state.get("query", ""),
        "source_title": state.get("source_title", ""),
        "source_url": state.get("source_url", ""),
        "out_dir": folder,
        "items": items
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "out_dir": folder,
        "manifest_path": manifest_path,
        "story_path": story_path,
        "sections_path": sections_path,
        "prompts_path": prompts_path,
    }

class StoryState(TypedDict, total=False):
    query: str
    candidates: list
    source_title: str
    source_url: str
    source_text: str
    script: str
    sections: list
    out_dir: str
    manifest_path: str
    story_path: str
    sections_path: str
    prompts_path: str

def build_app():
    g = StateGraph(StoryState)
    g.add_node("make_query", node_make_query)
    g.add_node("search", node_search)
    g.add_node("pick_source", node_pick_source)
    g.add_node("fetch_summary", node_fetch_summary)
    g.add_node("write_script", node_write_script)
    g.add_node("split_sections", node_split_sections)
    g.add_node("make_prompts", node_make_prompts)
    g.add_node("generate_assets", node_generate_assets)

    g.set_entry_point("make_query")
    g.add_edge("make_query", "search")
    g.add_edge("search", "pick_source")
    g.add_edge("pick_source", "fetch_summary")
    g.add_edge("fetch_summary", "write_script")
    g.add_edge("write_script", "split_sections")
    g.add_edge("split_sections", "make_prompts")
    g.add_edge("make_prompts", "generate_assets")
    g.add_edge("generate_assets", END)

    return g.compile()

app = build_app()

def run_pipeline(query: str = ""):
    return app.invoke({"query": query})
