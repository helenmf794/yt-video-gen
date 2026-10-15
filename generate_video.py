#!/usr/bin/env python3
"""
Topic -> ~10 minute English YouTube video (slideshow + AI voice).

Pipeline: Gemini (script) -> edge-tts (voice) -> Pexels/Pollinations (images) -> FFmpeg (render)

Usage:
    python generate_video.py "The history of the Roman Empire" --minutes 10
Env vars:
    GEMINI_API_KEY   (required)  https://aistudio.google.com/apikey
    PEXELS_API_KEY   (optional)  https://www.pexels.com/api/
    GEMINI_MODEL     (optional, default gemini-3.8-flash)
"""
import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path

import edge_tts
import requests

W, H, FPS = 1920, 1080, 30
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
SCENES_PER_SECTION = 5


# ---------- 1. Script (Gemini) ----------
def gemini_json(prompt, retries=5):
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.8},
    }
    for i in range(retries):
        try:
            r = requests.post(url, json=body, timeout=120)
            if r.status_code == 200:
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                return json.loads(text)
            print(f"Gemini {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print("Gemini error:", e)
        time.sleep(10 * (i + 1))
    sys.exit("Gemini failed after retries")


def build_script(topic, minutes):
    outline = gemini_json(f"""You are a YouTube scriptwriter for educational documentary-style channels.
Topic: "{topic}". Target length: {minutes} minutes.
Return JSON: {{"title": catchy YouTube title (max 70 chars), "description": 2-3 paragraph YouTube description,
"tags": [10 tags], "sections": [exactly {minutes} objects {{"heading": str, "summary": 1-2 sentences}}]}}.
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
Return JSON: {{"scenes": [exactly {SCENES_PER_SECTION} objects {{"narration": str, "image_query": str}}]}}.
Rules: each narration is 28-35 words of natural spoken English (plain text, no stage directions, no emojis,
no markdown), flowing continuously across scenes. image_query is 2-4 English words describing a concrete,
photographable visual for a stock photo site (e.g. "ancient roman ruins", not abstract ideas).""")
        for s in data["scenes"]:
            s["section"] = sec["heading"]
            scenes.append(s)
        prev = sec["heading"]
        time.sleep(4)  # free-tier rate limit
    return outline, scenes


# ---------- 2. Voice (edge-tts) ----------
async def make_audio(scenes, voice, outdir):
    for i, s in enumerate(scenes):
        path = outdir / f"a{i:03}.mp3"
        for attempt in range(4):
            try:
                await edge_tts.Communicate(s["narration"], voice, rate="-3%").save(str(path))
                break
            except Exception as e:
                print(f"TTS retry scene {i}: {e}")
                await asyncio.sleep(3)
        else:
            sys.exit(f"TTS failed for scene {i}")
        s["audio"] = str(path)
    return scenes


def audio_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


# ---------- 3. Images ----------
def fetch_image(query, dest, used):
    key = os.getenv("PEXELS_API_KEY")
    if key:
        try:
            r = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": key},
                params={"query": query, "orientation": "landscape", "per_page": 15},
                timeout=30,
            )
            photos = [p for p in r.json().get("photos", []) if p["id"] not in used]
            if photos:
                p = random.choice(photos[:8])
                used.add(p["id"])
                dest.write_bytes(requests.get(p["src"]["large2x"], timeout=60).content)
                return True
        except Exception as e:
            print("Pexels error:", e)
    pkey = os.getenv("PIXABAY_API_KEY")
    if pkey:
        try:
            r = requests.get(
                "https://pixabay.com/api/",
                params={"key": pkey, "q": query[:100], "image_type": "photo",
                        "orientation": "horizontal", "per_page": 20,
                        "min_width": 1280, "safesearch": "true"},
                timeout=30,
            )
            hits = [h for h in r.json().get("hits", []) if h["id"] not in used]
            if hits:
                h = random.choice(hits[:10])
                used.add(h["id"])
                url = h.get("fullHDURL") or h["largeImageURL"]
                dest.write_bytes(requests.get(url, timeout=60).content)
                return True
        except Exception as e:
            print("Pixabay error:", e)
    try:  # fallback: AI-generated image, no key needed
        url = (f"https://image.pollinations.ai/prompt/{urllib.parse.quote(query + ', cinematic, high quality')}"
               f"?width={W}&height={H}&nologo=true&seed={random.randint(1, 99999)}")
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and len(r.content) > 5000:
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        print("Pollinations error:", e)
    return False


def make_fallback_image(dest):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x101820:s={W}x{H}", "-frames:v", "1", str(dest)],
        capture_output=True, check=True,
    )


# ---------- 4. Render (FFmpeg) ----------
def render_scene(img, audio, dur, out, zoom_in):
    frames = int((dur + 0.4) * FPS)
    z = f"1+0.15*on/{frames}" if zoom_in else f"1.15-0.15*on/{frames}"
    vf = (
        f"scale=2880:1620:force_original_aspect_ratio=increase,crop=2880:1620,"
        f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(img), "-i", str(audio),
        "-vf", vf, "-af", "apad=pad_dur=0.4",
        "-t", f"{dur + 0.4:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("topic")
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--voice", default="en-US-AndrewNeural")
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

    print("== 3/4 Fetching images ==")
    used = set()
    for i, s in enumerate(scenes):
        img = outdir / f"i{i:03}.jpg"
        if not fetch_image(s["image_query"], img, used):
            make_fallback_image(img)
        s["image"] = str(img)
        time.sleep(0.3)

    print("== 4/4 Rendering ==")
    clips = []
    for i, s in enumerate(scenes):
        clip = outdir / f"c{i:03}.mp4"
        render_scene(s["image"], s["audio"], audio_duration(s["audio"]), clip, zoom_in=(i % 2 == 0))
        clips.append(clip)
        print(f"  scene {i + 1}/{len(scenes)}")

    listfile = outdir / "list.txt"
    listfile.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    final = outdir / "video.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", "-movflags", "+faststart", str(final)],
        capture_output=True, check=True,
    )

    meta = {k: outline[k] for k in ("title", "description", "tags")}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nDone: {final}  ({audio_duration(str(final)) / 60:.1f} min)")
    print(f"Title: {meta['title']}")


if __name__ == "__main__":
    main()
