#!/usr/bin/env python3
"""
Topic -> ~10 minute English YouTube video.

Gemini (script) -> edge-tts (voice) -> images (AI + real photos) -> FFmpeg
(slow zoom/pan, crossfades, section titles, burned-in subtitles, optional music)

Usage:
    python generate_video.py "The history of coffee" --minutes 10 --style documentary
Styles: documentary, stickman, whiteboard, cartoon-kids, anime, pixel-art, 3d-cartoon
Env vars:
    GEMINI_API_KEY   (required)
    PEXELS_API_KEY / PIXABAY_API_KEY   (optional extra photo sources)
Optional: put a royalty-free `music.mp3` next to this script for background music.
"""
import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts
import requests

W, H, FPS = 1920, 1080, 30
XFADE = 0.6            # crossfade seconds between scenes
PAD = 0.35             # silence after each narrated scene
SCENES_PER_SECTION = 8
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
STYLES = {
    "documentary": dict(
        mode="mix", text="keywords", voice="en-US-AndrewNeural", rate="-3%", words="17-22", character=None,
        suffix="cinematic documentary photograph, realistic, natural lighting, high detail",
        tone="serious, vivid documentary storytelling like the best educational YouTube channels",
        bg="0x0B1220", fg="white", accent="0xB91C1C"),
    "stickman": dict(
        mode="ai_only", text="keywords", voice="en-US-GuyNeural", rate="+0%", words="17-22",
        character="a friendly black stick figure with a round head and simple dot eyes",
        suffix="simple stick figure cartoon, thick black hand-drawn lines, minimalist doodle, plain white background, "
               "a few flat accent colors, clean composition, no text",
        tone="witty, conversational explainer with punchy short sentences and light humor, like popular stick-figure explainer channels",
        bg="0xFFFFFF", fg="0x111111", accent="0x2563EB"),
    "whiteboard": dict(
        mode="ai_only", text="cards", voice="en-US-AriaNeural", rate="-2%", words="17-22", character=None,
        suffix="hand-drawn whiteboard sketch, black marker drawing with blue and red accents, white board background, "
               "simple diagrams and icons, no text",
        tone="clear, friendly teacher explaining step by step at a whiteboard",
        bg="0xF8FAFC", fg="0x111827", accent="0xDC2626"),
    "cartoon-kids": dict(
        mode="ai_only", text="keywords", voice="en-US-AnaNeural", rate="-8%", words="13-17",
        character="a cheerful cute animal character with big eyes",
        suffix="cute colorful children's cartoon illustration, bright cheerful colors, soft rounded shapes, "
               "friendly characters, simple background, no text",
        tone="gentle, playful storytelling for children aged 4 to 8, very simple words, short sentences, warm and positive, nothing scary",
        bg="0xFEF3C7", fg="0x1E3A8A", accent="0xF97316"),
    "anime": dict(
        mode="ai_only", text="keywords", voice="en-GB-RyanNeural", rate="-3%", words="17-22",
        character="a determined young protagonist with expressive eyes",
        suffix="anime style illustration, vibrant colors, detailed background, dramatic lighting, studio quality, no text",
        tone="dramatic, emotional, epic storytelling",
        bg="0x1E1B4B", fg="white", accent="0xDB2777"),
    "pixel-art": dict(
        mode="ai_only", text="keywords", voice="en-US-GuyNeural", rate="+0%", words="17-22",
        character="a small pixel-art hero",
        suffix="16-bit pixel art, retro video game style, limited color palette, crisp pixels, no text",
        tone="fun, energetic retro-gamer storytelling",
        bg="0x111827", fg="0x86EFAC", accent="0x16A34A"),
    "3d-cartoon": dict(
        mode="ai_only", text="keywords", voice="en-US-AriaNeural", rate="-3%", words="17-22",
        character="a charming expressive main character",
        suffix="3D animated movie style, soft lighting, expressive characters, rich colors, high detail, no text",
        tone="warm, vivid, entertaining storytelling",
        bg="0x0F766E", fg="white", accent="0xF59E0B"),
}
UA = {"User-Agent": "yt-video-gen/1.0 (personal educational project)"}
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.8-flash,gemini-3.8-flash-lite,gemini-flash-latest"
).split(",") if m.strip()]


# =====================================================================
# 1. Script (Gemini)
# =====================================================================
_MODELS_CACHE = []


def candidate_models(key):
    """Ask Google which models this key can really use; prefer flash, then pro."""
    if _MODELS_CACHE:
        return _MODELS_CACHE
    found = []
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"key": key, "pageSize": 100}, timeout=30)
        skip = ("image", "tts", "embed", "live", "audio", "robotics", "computer",
                "vision", "aqa", "learnlm", "gemma")
        for m in r.json().get("models", []):
            name = m["name"].split("/", 1)[1]
            if "generateContent" in m.get("supportedGenerationMethods", []) \
                    and not any(s in name for s in skip):
                found.append(name)
    except Exception as e:
        print("Could not list models:", e)
    print("Available models:", found)
    ordered = [m for m in GEMINI_MODELS if m in found]
    ordered += [m for m in found if "flash" in m and m not in ordered]
    ordered += [m for m in found if "pro" in m and m not in ordered]
    if not ordered:
        ordered = list(GEMINI_MODELS)
    _MODELS_CACHE.extend(ordered[:6])
    print("Will try:", _MODELS_CACHE)
    return _MODELS_CACHE


def gemini_json(prompt, rounds=4):
    key = os.environ["GEMINI_API_KEY"]
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.8},
    }
    for rnd in range(rounds):
        for model in candidate_models(key):
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
            try:
                r = requests.post(url, json=body, timeout=180)
                if r.status_code == 200:
                    try:
                        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                        return json.loads(text)
                    except Exception as e:
                        print(f"Gemini {model}: bad JSON ({e})")
                else:
                    print(f"Gemini {model} -> {r.status_code}: {r.text[:120]}")
            except Exception as e:
                print(f"Gemini {model} error: {e}")
            time.sleep(3)
        time.sleep(20 * (rnd + 1))
    sys.exit("Gemini failed after retries")


def build_script(topic, minutes, st, per_section):
    char_hint = st["character"]
    char_field = (f'"character": a 15-25 word visual description of ONE recurring main character that fits the story '
                  f'and this art style (start from: {char_hint}),') if char_hint else ""
    outline = gemini_json(f"""You are a YouTube scriptwriter. Narration style: {st['tone']}.
Topic: "{topic}". Target length: {minutes} minutes.
Return JSON: {{"title": catchy YouTube title (max 70 chars), "description": 2-3 paragraph YouTube description,
"tags": [10 tags], {char_field}
"sections": [exactly {minutes} objects {{"heading": short chapter title (max 6 words), "summary": 1-2 sentences,
"points": [exactly 3 key takeaways, each max 7 words]}}]}}.
The sections must tell a coherent story: strong hook, logical progression, satisfying conclusion.""")
    sections = outline["sections"][:minutes]
    scenes = []
    prev = "None"
    for idx, sec in enumerate(sections):
        print(f"Writing section {idx + 1}/{len(sections)}: {sec['heading']}")
        role = ""
        if idx == 0:
            role = "This is the opening: start with a gripping hook, no greetings like 'Welcome back'."
        if idx == len(sections) - 1:
            role = "This is the final section: wrap up with a memorable conclusion and a short call to subscribe."
        data = gemini_json(f"""Video topic: "{topic}". Narration style: {st['tone']}.
Full outline: {json.dumps([s['heading'] for s in sections])}.
Write section {idx + 1} of {len(sections)}: "{sec['heading']}" ({sec['summary']}). Previous section: {prev}.
{role}
Return JSON: {{"scenes": [exactly {per_section} objects
{{"narration": str, "image_query": str, "ai_prompt": str, "on_screen": str}}]}}.
Rules:
- narration: {st['words']} words of natural spoken English (plain text, no stage directions, no emojis, no markdown),
  flowing continuously from one scene to the next.
- image_query: 2-4 English words for a photo search. It MUST include the main subject word of the topic so
  results stay on-topic (e.g. "ottoman empire coffeehouse"), concrete and photographable.
- ai_prompt: vivid 15-25 word description of the exact scene for an AI image generator. Describe the scene only
  (art style is added automatically). No text or letters in the image, no named real people.
- on_screen: 2-4 words to show as a big on-screen callout (a key term, date or number from this scene),
  or "" if nothing deserves highlighting. Use it in about half of the scenes.""")
        for s in data["scenes"]:
            s["section"] = sec["heading"]
            s["sec_idx"] = idx
            scenes.append(s)
        prev = sec["heading"]
        time.sleep(4)  # free-tier rate limit
    return outline, scenes


# =====================================================================
# 2. Voice (edge-tts)
# =====================================================================
def audio_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


async def make_audio(scenes, voice, rate, outdir):
    for i, s in enumerate(scenes):
        mp3 = outdir / f"a{i:03}.mp3"
        wav = outdir / f"a{i:03}.wav"
        for attempt in range(4):
            try:
                await edge_tts.Communicate(s["narration"], voice, rate=rate).save(str(mp3))
                break
            except Exception as e:
                print(f"TTS retry scene {i}: {e}")
                await asyncio.sleep(3)
        else:
            sys.exit(f"TTS failed for scene {i}")
        # normalise to 44.1k stereo wav and add a short pause after the scene
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), "-af", f"apad=pad_dur={PAD}",
             "-ar", "44100", "-ac", "2", str(wav)], check=True)
        s["audio"] = str(wav)
        s["dur"] = audio_duration(wav)
    return scenes


# =====================================================================
# 3. Images (AI + real photos, never a blank screen)
# =====================================================================
def valid_image(b):
    if len(b) < 10_000:
        return False
    return b[:2] == b"\xff\xd8" or b[:4] == b"\x89PNG" or (b[:4] == b"RIFF" and b[8:12] == b"WEBP")


def src_pexels(q, dest, used):
    key = os.getenv("PEXELS_API_KEY")
    if not key:
        return False
    r = requests.get("https://api.pexels.com/v1/search", headers={"Authorization": key},
                     params={"query": q, "orientation": "landscape", "per_page": 15}, timeout=30)
    photos = [p for p in r.json().get("photos", []) if f"px{p['id']}" not in used]
    if not photos:
        return False
    p = random.choice(photos[:8])
    img = requests.get(p["src"]["large2x"], timeout=60)
    if valid_image(img.content):
        dest.write_bytes(img.content)
        used.add(f"px{p['id']}")
        return True
    return False


def src_pixabay(q, dest, used):
    key = os.getenv("PIXABAY_API_KEY")
    if not key:
        return False
    r = requests.get("https://pixabay.com/api/", params={
        "key": key, "q": q[:100], "image_type": "photo", "orientation": "horizontal",
        "per_page": 20, "min_width": 1280, "safesearch": "true"}, timeout=30)
    hits = [h for h in r.json().get("hits", []) if f"pb{h['id']}" not in used]
    if not hits:
        return False
    h = random.choice(hits[:10])
    img = requests.get(h.get("fullHDURL") or h["largeImageURL"], timeout=60)
    if valid_image(img.content):
        dest.write_bytes(img.content)
        used.add(f"pb{h['id']}")
        return True
    return False


BAD_TITLE = ("drawing", "plan", "map", "diagram", "scan", "habs", "historic american", "sheet", "chart", "logo",
             "flag", "coat of arms", "blueprint", "schematic", "floor", "elevation", "section", "poster", "cover",
             "page", "manuscript", "stamp", "banknote", "coin", "screenshot", "graph", "table", "list")


def src_wikimedia(q, dest, used):
    """Real photos from Wikimedia Commons: free, no key, great for history & science."""
    for query in (q, " ".join(q.split()[:2])):
        r = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=30, params={
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": 30, "prop": "imageinfo",
            "iiprop": "url|size|mime", "iiurlwidth": 1920})
        pages = ((r.json().get("query") or {}).get("pages") or {}).values()
        cands = []
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            w, h = ii.get("width", 0), ii.get("height", 1)
            title = p.get("title", "").lower()
            key = f"wm{p.get('pageid')}"
            if (ii.get("mime") == "image/jpeg" and w >= 1200 and w >= h * 1.2 and key not in used
                    and not any(b in title for b in BAD_TITLE)):
                cands.append((key, ii.get("thumburl") or ii.get("url")))
        random.shuffle(cands)
        for key, url in cands[:4]:
            img = requests.get(url, headers=UA, timeout=60)
            if img.status_code == 200 and valid_image(img.content):
                dest.write_bytes(img.content)
                used.add(key)
                return True
        time.sleep(0.5)
    return False


AI_STATE = {"fails": 0, "seed": None}


def src_ai(prompt, dest, used, patient=False):
    """Pollinations AI image (free). In 'mix' mode gives up after 3 failures in a row;
    in AI-only styles it waits and keeps trying so the style never breaks."""
    if AI_STATE["fails"] >= 3:
        if not patient:
            return False
        time.sleep(40)
        AI_STATE["fails"] = 2
    for attempt in range(3):
        try:
            seed = AI_STATE["seed"] or random.randint(1, 999999)
            url = (f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
                   f"?width={W}&height={H}&nologo=true&seed={seed + attempt}")
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code == 200 and valid_image(r.content):
                dest.write_bytes(r.content)
                AI_STATE["fails"] = 0
                time.sleep(2)
                return True
            print(f"  AI image status {r.status_code}")
            if r.status_code == 429:
                time.sleep(15)
        except Exception as e:
            print("  AI image error:", e)
            time.sleep(5)
    AI_STATE["fails"] += 1
    return False


def full_prompt(s, st, character):
    p = (s.get("ai_prompt") or s["image_query"]).rstrip(". ")
    if character:
        p += f". Recurring main character: {character}"
    return f"{p}. Art style: {st['suffix']}"


def get_image(i, s, dest, used, mode, st, character):
    q = s["image_query"]
    photo = [("pexels", src_pexels), ("pixabay", src_pixabay), ("wikimedia", src_wikimedia)]
    if mode == "photos":
        order = photo
    elif mode == "ai_only":
        order = [("ai", None)]
    elif mode == "ai":
        order = [("ai", None)] + photo
    else:  # mix: alternate between AI art and real photos, each is the other's backup
        order = ([("ai", None)] + photo) if i % 2 == 0 else (photo + [("ai", None)])
    for name, fn in order:
        try:
            ok = (src_ai(full_prompt(s, st, character), dest, used, patient=(mode == "ai_only"))
                  if name == "ai" else fn(q, dest, used))
        except Exception as e:
            print(f"  {name} error: {e}")
            ok = False
        if ok:
            return name
    return None


def make_card(title, points, dest, st, outdir, tag):
    """Explainer card: chapter title + 3 key points drawn on a solid background (no network)."""
    (outdir / f"card{tag}_t.txt").write_text(textwrap.fill(title, 22))
    (outdir / f"card{tag}_b.txt").write_text(
        "\n".join("•  " + textwrap.fill(p, 36, subsequent_indent="    ") for p in points[:3]))
    vf = (f"drawtext=fontfile={FONT_BOLD}:textfile=card{tag}_t.txt:fontsize=84:fontcolor={st['fg']}:x=130:y=120:line_spacing=14,"
          f"drawbox=x=130:y=385:w=300:h=12:color={st['accent']}:t=fill,"
          f"drawtext=fontfile={FONT_BOLD}:textfile=card{tag}_b.txt:fontsize=52:fontcolor={st['fg']}:x=130:y=450:line_spacing=26")
    run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                f"color=c={st['bg']}:s={W}x{H}", "-vf", vf, "-frames:v", "1", dest.name], outdir)


def src_archive_clip(query, dest, length, outdir):
    """EXPERIMENTAL: public-domain Prelinger Archives film from archive.org (old US footage)."""
    q = f"({query}) AND collection:prelinger AND mediatype:movies"
    r = requests.get("https://archive.org/advancedsearch.php", headers=UA, timeout=30,
                     params={"q": q, "fl[]": "identifier", "rows": 15, "output": "json"})
    docs = r.json().get("response", {}).get("docs", [])
    random.shuffle(docs)
    tmp = outdir / "dl.mp4"
    for d in docs[:4]:
        meta = requests.get(f"https://archive.org/metadata/{d['identifier']}", headers=UA, timeout=30).json()
        files = [f for f in meta.get("files", []) if f.get("name", "").lower().endswith(".mp4")
                 and 3e6 < int(f.get("size") or 0) < 70e6]
        if not files:
            continue
        f = random.choice(files)
        url = f"https://archive.org/download/{d['identifier']}/{urllib.parse.quote(f['name'])}"
        with requests.get(url, headers=UA, stream=True, timeout=120) as resp:
            if resp.status_code != 200:
                continue
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(1 << 20):
                    fh.write(chunk)
        dur = audio_duration(tmp)
        if dur < length + 3:
            continue
        start = random.uniform(1, dur - length - 1)
        run_ffmpeg(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", "dl.mp4",
                    "-t", f"{length:.2f}", "-an", "-vf",
                    f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},fps={FPS},format=yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", dest.name], outdir)
        return True
    return False


# =====================================================================
# 4. Edit & render
# =====================================================================
def run_ffmpeg(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise RuntimeError("ffmpeg failed")


def motion(k, frames, static=False):
    """Alternate slow zoom-in / zoom-out / pan-left / pan-right (cards stay still)."""
    cy = "ih/2-(ih/zoom/2)"
    if static:
        return "1", "iw/2-(iw/zoom/2)", cy
    p = k % 4
    if p == 0:
        return f"1+0.15*on/{frames}", "iw/2-(iw/zoom/2)", cy
    if p == 1:
        return f"1.15-0.15*on/{frames}", "iw/2-(iw/zoom/2)", cy
    if p == 2:
        return "1.12", f"(iw-iw/zoom)*(0.1+0.8*on/{frames})", cy
    return "1.12", f"(iw-iw/zoom)*(0.9-0.8*on/{frames})", cy


TRANSITIONS = ["fade", "dissolve", "smoothleft", "fade", "smoothright", "dissolve"]


def render_section(idx, heading, items, outdir, st, with_title):
    """items: list of (filename, duration, kind, callout). One ffmpeg call per chapter:
    zoom/pan + crossfades + on-screen callouts + chapter title."""
    n = len(items)
    total = sum(it[1] for it in items)
    nframes = int(round(total * FPS))
    (outdir / f"sec{idx}.txt").write_text(textwrap.fill(heading, 30))

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    parts = []
    for k, (img, d, kind, _kw) in enumerate(items):
        cmd += ["-i", img]
        length = d + (XFADE if k < n - 1 else 0)
        frames = max(2, int(round(length * FPS)))
        if kind == "clip":
            parts.append(f"[{k}:v]fps={FPS},setsar=1,format=yuv420p,trim=duration={length:.3f},"
                         f"setpts=PTS-STARTPTS,settb=1/{FPS}[v{k}]")
            continue
        z, x, y = motion(k + idx, frames, static=(kind == "card"))
        parts.append(
            f"[{k}:v]scale=2880:1620:force_original_aspect_ratio=increase,crop=2880:1620,"
            f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},setsar=1,format=yuv420p,"
            f"settb=1/{FPS}[v{k}]"
        )
    cur, acc = "v0", items[0][1]
    for k in range(1, n):
        out = f"x{k}"
        parts.append(f"[{cur}][v{k}]xfade=transition={TRANSITIONS[(k + idx) % len(TRANSITIONS)]}"
                     f":duration={XFADE}:offset={acc:.3f}[{out}]")
        cur, acc = out, acc + items[k][1]

    # big on-screen callouts (key term / date / number)
    start = 0.0
    for k, (_img, d, kind, kw) in enumerate(items):
        if kw and kind == "img":
            a, b = start + 0.8, start + min(4.2, d - 0.4)
            if b > a + 1.0:
                (outdir / f"kw{idx}_{k}.txt").write_text(textwrap.fill(kw, 26))
                parts.append(
                    f"[{cur}]drawtext=fontfile={FONT_BOLD}:textfile=kw{idx}_{k}.txt:fontsize=64:fontcolor=white:"
                    f"box=1:boxcolor={st['accent']}@0.9:boxborderw=26:x=(w-text_w)/2:y=h-380:"
                    f"enable='between(t,{a:.2f},{b:.2f})'[kw{k}]")
                cur = f"kw{k}"
        start += d

    if with_title:
        alpha = "if(lt(t,0.8),t/0.8,if(lt(t,4),1,max(0,(4.8-t)/0.8)))"
        parts.append(
            f"[{cur}]drawtext=fontfile={FONT_BOLD}:textfile=sec{idx}.txt:fontsize=58:fontcolor=white:"
            f"borderw=3:bordercolor=black@0.85:shadowx=3:shadowy=3:x=90:y=120:"
            f"alpha='{alpha}':enable='lt(t,4.8)'[ttl]")
        cur = "ttl"
    parts.append(f"[{cur}]tpad=stop_mode=clone:stop_duration=2,format=yuv420p[vout]")
    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]", "-frames:v", str(nframes),
            "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-threads", "2", f"sec{idx}.mp4"]
    run_ffmpeg(cmd, outdir)
    return f"sec{idx}.mp4"


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02}:{ms % 3600000 // 60000:02}:{ms % 60000 // 1000:02},{ms % 1000:03}"


def chunk_words(text, max_words=9, min_words=5):
    chunks, cur = [], []
    for w in text.split():
        cur.append(w)
        if (len(cur) >= min_words and re.search(r"[,.;:!?]$", w)) or len(cur) >= max_words:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        if chunks and len(cur) < 3:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


def write_srt(scenes, path):
    lines, t, n = [], 0.0, 1
    for s in scenes:
        speech = max(0.5, s["dur"] - PAD)
        chunks = chunk_words(s["narration"])
        weights = [len(c) for c in chunks]
        tot = sum(weights)
        pos = t
        for c, wgt in zip(chunks, weights):
            span = speech * wgt / tot
            lines += [str(n), f"{srt_time(pos)} --> {srt_time(pos + span)}", textwrap.fill(c, 44), ""]
            pos += span
            n += 1
        t += s["dur"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def assemble(scenes, outdir, st, music=None, callouts=True):
    sections = {}
    for s in scenes:
        kind = s.get("kind", "img")
        kw = (s.get("on_screen") or "").strip() if callouts else ""
        sections.setdefault(s["sec_idx"], (s["section"], []))[1].append(
            (Path(s["image"]).name, s["dur"], kind, kw))

    print("  rendering chapters in parallel...")
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {idx: ex.submit(render_section, idx, head, items, outdir, st, items[0][2] != "card")
                for idx, (head, items) in sorted(sections.items())}
        secfiles = [futs[i].result() for i in sorted(futs)]
    print(f"  {len(secfiles)} chapters rendered")

    (outdir / "vlist.txt").write_text("".join(f"file '{f}'\n" for f in secfiles))
    run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "vlist.txt",
                "-c", "copy", "video_nosub.mp4"], outdir)

    (outdir / "alist.txt").write_text("".join(f"file '{Path(s['audio']).name}'\n" for s in scenes))
    run_ffmpeg(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", "alist.txt",
                "-c", "copy", "narration.wav"], outdir)

    write_srt(scenes, outdir / "subs.srt")
    style = ("FontName=DejaVu Sans,FontSize=12,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
             "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=45")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "video_nosub.mp4", "-i", "narration.wav"]
    if music and Path(music).exists():
        cmd += ["-stream_loop", "-1", "-i", str(Path(music).resolve())]
        fc = (f"[0:v]subtitles=subs.srt:force_style='{style}'[v];"
              "[1:a]volume=1.0[n];[2:a]volume=0.10[m];"
              "[n][m]amix=inputs=2:duration=first:dropout_transition=0,volume=2[a]")
    else:
        fc = f"[0:v]subtitles=subs.srt:force_style='{style}'[v];[1:a]anull[a]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
            "-movflags", "+faststart", "video.mp4"]
    run_ffmpeg(cmd, outdir)
    return outdir / "video.mp4"


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--style", default="documentary", choices=list(STYLES))
    ap.add_argument("--voice", default="auto", help="'auto' = the style's default voice")
    ap.add_argument("--images", default="auto", choices=["auto", "mix", "ai", "photos"])
    ap.add_argument("--text", default="auto", choices=["auto", "off", "keywords", "cards", "both"])
    ap.add_argument("--clips", type=int, default=0, help="experimental: number of archive.org film clips")
    ap.add_argument("--music", default="music.mp3")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    st = STYLES[args.style]
    voice = st["voice"] if args.voice == "auto" else args.voice
    mode = st["mode"] if args.images == "auto" else args.images
    if st["mode"] == "ai_only" and args.images == "auto":
        mode = "ai_only"
    text = st["text"] if args.text == "auto" else args.text
    use_cards = text in ("cards", "both")
    use_kw = text in ("keywords", "both")

    slug = re.sub(r"[^a-z0-9]+", "-", args.topic.lower()).strip("-")[:50]
    outdir = Path(args.out) / slug
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Style: {args.style} | images: {mode} | text: {text} | voice: {voice}")

    print("== 1/4 Writing script ==")
    outline, scenes = build_script(args.topic, args.minutes, st, 7 if use_cards else SCENES_PER_SECTION)
    character = (outline.get("character") or st["character"]) if st["character"] else None
    if character:
        AI_STATE["seed"] = random.randint(1, 900000)  # same seed helps keep the character consistent

    if use_cards:  # one narrated explainer card at the start of each chapter
        new, seen = [], set()
        for s in scenes:
            if s["sec_idx"] not in seen:
                seen.add(s["sec_idx"])
                sec = outline["sections"][s["sec_idx"]]
                if sec.get("points") and sec.get("summary"):
                    new.append(dict(narration=sec["summary"], section=sec["heading"], sec_idx=s["sec_idx"],
                                    kind="card", points=sec["points"], on_screen=""))
            new.append(s)
        scenes = new
    (outdir / "script.json").write_text(json.dumps(scenes, indent=2, ensure_ascii=False))

    print("== 2/4 Generating voice ==")
    scenes = asyncio.run(make_audio(scenes, voice, st["rate"], outdir))

    print("== 3/4 Getting images ==")
    used, counts, last_good = set(), {}, []
    clip_slots = set()
    if args.clips > 0:
        cand = [i for i, s in enumerate(scenes) if s.get("kind") != "card" and s["dur"] >= 4]
        step = max(1, len(cand) // args.clips)
        clip_slots = set(cand[step // 2::step][:args.clips])
    for i, s in enumerate(scenes):
        if s.get("kind") == "card":
            img = outdir / f"card{i:03}.png"
            make_card(s["section"], s["points"], img, st, outdir, f"{i:03}")
            s["image"] = str(img)
            counts["card"] = counts.get("card", 0) + 1
            print(f"  scene {i + 1}/{len(scenes)}: card")
            continue
        src = None
        if i in clip_slots:
            clip = outdir / f"clip{i:03}.mp4"
            try:
                if src_archive_clip(s["image_query"], clip, s["dur"] + XFADE + 0.3, outdir):
                    s["kind"], s["image"], src = "clip", str(clip), "archive-clip"
            except Exception as e:
                print("  archive clip error:", e)
        if src is None:
            img = outdir / f"i{i:03}.jpg"
            src = get_image(i, s, img, used, mode, st, character)
            if src is None:  # never show a blank screen: reuse an earlier image
                if last_good:
                    img.write_bytes(Path(random.choice(last_good)).read_bytes())
                    src = "reused"
                else:
                    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                                    f"color=c=0x101820:s={W}x{H}", "-frames:v", "1", str(img)], check=True)
                    src = "blank"
            else:
                last_good.append(str(img))
            s["image"] = str(img)
        counts[src] = counts.get(src, 0) + 1
        print(f"  scene {i + 1}/{len(scenes)}: {src}")
    print("Image sources:", counts)

    print("== 4/4 Editing & rendering ==")
    final = assemble(scenes, outdir, st, music=args.music, callouts=use_kw)

    meta = {k: outline[k] for k in ("title", "description", "tags")}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nDone: {final}  ({audio_duration(final) / 60:.1f} min)")
    print(f"Title: {meta['title']}")


if __name__ == "__main__":
    main()
