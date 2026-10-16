#!/usr/bin/env python3
"""
Topic -> ~10 minute English YouTube video.

Gemini (script) -> edge-tts (voice) -> images (AI + real photos) -> FFmpeg
(slow zoom/pan, crossfades, section titles, burned-in subtitles, optional music)

Usage:
    python generate_video.py "The history of coffee" --minutes 10 --images mix
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


def build_script(topic, minutes):
    outline = gemini_json(f"""You are a YouTube scriptwriter for educational documentary-style channels.
Topic: "{topic}". Target length: {minutes} minutes.
Return JSON: {{"title": catchy YouTube title (max 70 chars), "description": 2-3 paragraph YouTube description,
"tags": [10 tags], "sections": [exactly {minutes} objects {{"heading": short chapter title (max 6 words), "summary": 1-2 sentences}}]}}.
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
        data = gemini_json(f"""Video topic: "{topic}". Full outline: {json.dumps([s['heading'] for s in sections])}.
Write section {idx + 1} of {len(sections)}: "{sec['heading']}" ({sec['summary']}). Previous section: {prev}.
{role}
Return JSON: {{"scenes": [exactly {SCENES_PER_SECTION} objects
{{"narration": str, "image_query": str, "ai_prompt": str}}]}}.
Rules: each narration is 17-22 words of natural spoken English (plain text, no stage directions, no emojis,
no markdown), flowing continuously from one scene to the next. image_query is 2-4 English words for a stock/
archive photo search (concrete and photographable, e.g. "ancient roman ruins"). ai_prompt is a vivid 15-25 word
description of the exact scene for an AI image generator: cinematic, documentary style, rich lighting,
no text or letters in the image, no named real people.""")
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


async def make_audio(scenes, voice, outdir):
    for i, s in enumerate(scenes):
        mp3 = outdir / f"a{i:03}.mp3"
        wav = outdir / f"a{i:03}.wav"
        for attempt in range(4):
            try:
                await edge_tts.Communicate(s["narration"], voice, rate="-3%").save(str(mp3))
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


def src_wikimedia(q, dest, used):
    """Real photos from Wikimedia Commons: free, no key, great for history & science."""
    for query in (q, " ".join(q.split()[:2])):
        r = requests.get("https://commons.wikimedia.org/w/api.php", headers=UA, timeout=30, params={
            "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
            "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": 25, "prop": "imageinfo",
            "iiprop": "url|size|mime", "iiurlwidth": 1920})
        pages = ((r.json().get("query") or {}).get("pages") or {}).values()
        cands = []
        for p in pages:
            ii = (p.get("imageinfo") or [{}])[0]
            w, h = ii.get("width", 0), ii.get("height", 1)
            key = f"wm{p.get('pageid')}"
            if ii.get("mime") in ("image/jpeg", "image/png") and w >= 1200 and w >= h * 1.2 and key not in used:
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


AI_STATE = {"fails": 0}


def src_ai(prompt, dest, used):
    """Pollinations AI image (free). Gives up after 3 failures in a row to save time."""
    if AI_STATE["fails"] >= 3:
        return False
    for attempt in range(3):
        try:
            url = (f"https://image.pollinations.ai/prompt/{urllib.parse.quote(prompt)}"
                   f"?width={W}&height={H}&nologo=true&seed={random.randint(1, 999999)}")
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


def get_image(i, s, dest, used, mode):
    q = s["image_query"]
    ai_prompt = (s.get("ai_prompt") or q) + ", cinematic documentary photo, 16:9"
    photo = [("pexels", src_pexels), ("pixabay", src_pixabay), ("wikimedia", src_wikimedia)]
    ai = [("ai", None)]
    if mode == "photos":
        order = photo
    elif mode == "ai":
        order = ai + photo
    else:  # mix: alternate between AI art and real photos, each is the other's backup
        order = (ai + photo) if i % 2 == 0 else (photo + ai)
    for name, fn in order:
        try:
            ok = src_ai(ai_prompt, dest, used) if name == "ai" else fn(q, dest, used)
        except Exception as e:
            print(f"  {name} error: {e}")
            ok = False
        if ok:
            return name
    return None


# =====================================================================
# 4. Edit & render
# =====================================================================
def run_ffmpeg(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise RuntimeError("ffmpeg failed")


def motion(k, frames):
    """Alternate slow zoom-in / zoom-out / pan-left / pan-right."""
    p = k % 4
    cy = "ih/2-(ih/zoom/2)"
    if p == 0:
        return f"1+0.15*on/{frames}", "iw/2-(iw/zoom/2)", cy
    if p == 1:
        return f"1.15-0.15*on/{frames}", "iw/2-(iw/zoom/2)", cy
    if p == 2:
        return f"1.12", f"(iw-iw/zoom)*(0.1+0.8*on/{frames})", cy
    return f"1.12", f"(iw-iw/zoom)*(0.9-0.8*on/{frames})", cy


TRANSITIONS = ["fade", "dissolve", "smoothleft", "fade", "smoothright", "dissolve"]


def render_section(idx, heading, items, outdir):
    """items: list of (image_filename, duration). One ffmpeg call: zoom/pan + crossfades + title."""
    n = len(items)
    total = sum(d for _, d in items)
    nframes = int(round(total * FPS))
    (outdir / f"sec{idx}.txt").write_text(textwrap.fill(heading, 30))

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    parts = []
    for k, (img, d) in enumerate(items):
        cmd += ["-i", img]
        length = d + (XFADE if k < n - 1 else 0)
        frames = max(2, int(round(length * FPS)))
        z, x, y = motion(k + idx, frames)
        parts.append(
            f"[{k}:v]scale=2880:1620:force_original_aspect_ratio=increase,crop=2880:1620,"
            f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},setsar=1,format=yuv420p[v{k}]"
        )
    cur, acc = "v0", items[0][1]
    for k in range(1, n):
        out = f"x{k}"
        parts.append(f"[{cur}][v{k}]xfade=transition={TRANSITIONS[(k + idx) % len(TRANSITIONS)]}"
                     f":duration={XFADE}:offset={acc:.3f}[{out}]")
        cur, acc = out, acc + items[k][1]
    alpha = "if(lt(t,0.8),t/0.8,if(lt(t,4),1,max(0,(4.8-t)/0.8)))"
    parts.append(
        f"[{cur}]drawtext=fontfile={FONT_BOLD}:textfile=sec{idx}.txt:fontsize=58:fontcolor=white:"
        f"borderw=3:bordercolor=black@0.85:shadowx=3:shadowy=3:x=90:y=120:"
        f"alpha='{alpha}':enable='lt(t,4.8)',tpad=stop_mode=clone:stop_duration=2,format=yuv420p[vout]"
    )
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


def assemble(scenes, outdir, music=None):
    sections = {}
    for s in scenes:
        sections.setdefault(s["sec_idx"], (s["section"], []))[1].append((Path(s["image"]).name, s["dur"]))

    print("  rendering sections in parallel...")
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {idx: ex.submit(render_section, idx, head, items, outdir)
                for idx, (head, items) in sorted(sections.items())}
        secfiles = [futs[i].result() for i in sorted(futs)]
        for i in sorted(futs):
            print(f"  section {i + 1}/{len(futs)} done")

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
    ap.add_argument("--voice", default="en-US-AndrewNeural")
    ap.add_argument("--images", default="mix", choices=["mix", "ai", "photos"])
    ap.add_argument("--music", default="music.mp3")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    slug = re.sub(r"[^a-z0-9]+", "-", args.topic.lower()).strip("-")[:50]
    outdir = Path(args.out) / slug
    outdir.mkdir(parents=True, exist_ok=True)

    print("== 1/4 Writing script ==")
    outline, scenes = build_script(args.topic, args.minutes)
    (outdir / "script.json").write_text(json.dumps(scenes, indent=2, ensure_ascii=False))

    print("== 2/4 Generating voice ==")
    scenes = asyncio.run(make_audio(scenes, args.voice, outdir))

    print("== 3/4 Getting images ==")
    used, counts, last_good = set(), {}, []
    for i, s in enumerate(scenes):
        img = outdir / f"i{i:03}.jpg"
        src = get_image(i, s, img, used, args.images)
        if src is None:  # never show a blank screen: reuse an earlier image
            if last_good:
                shutil_src = random.choice(last_good)
                img.write_bytes(Path(shutil_src).read_bytes())
                src = "reused"
            else:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                                f"color=c=0x101820:s={W}x{H}", "-frames:v", "1", str(img)], check=True)
                src = "blank"
        else:
            last_good.append(str(img))
        counts[src] = counts.get(src, 0) + 1
        s["image"] = str(img)
        print(f"  scene {i + 1}/{len(scenes)}: {src}")
    print("Image sources:", counts)

    print("== 4/4 Editing & rendering ==")
    final = assemble(scenes, outdir, music=args.music)

    meta = {k: outline[k] for k in ("title", "description", "tags")}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nDone: {final}  ({audio_duration(final) / 60:.1f} min)")
    print(f"Title: {meta['title']}")


if __name__ == "__main__":
    main()
