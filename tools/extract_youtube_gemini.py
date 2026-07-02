#!/usr/bin/env python3
"""
Extract PCA exam questions from YouTube videos using the Gemini API.

Gemini can ingest a YouTube URL directly and analyze the video content
(Claude/this hub cannot access YouTube transcripts, but Gemini can).

SETUP
  pip install google-genai
  # optional, to expand a whole playlist into video URLs:
  pip install yt-dlp
  # get a key at https://aistudio.google.com/apikey
  export GEMINI_API_KEY=YOUR_KEY

RUN on individual videos:
  python extract_youtube_gemini.py https://youtu.be/VIDEO_ID ... > q-video.json

RUN on a whole playlist (expand first):
  yt-dlp --flat-playlist -j "https://www.youtube.com/playlist?list=PLB574eEmT4ofIr8LFjiKJfTjjg7StIAQm" \
    | python -c "import sys,json;[print('https://youtu.be/'+json.loads(l)['id']) for l in sys.stdin]" > urls.txt
  python extract_youtube_gemini.py $(cat urls.txt) > q-video.json

The output q-video.json uses the SAME schema as this study hub's quiz, so it can be
merged in later (each object: c, q, o[4], a, e, oe[4]).
"""
import os, sys, json, time

try:
    from google import genai
    from google.genai import types
except ImportError:
    sys.exit("Run: pip install google-genai")

KEY = os.environ.get("GEMINI_API_KEY")
if not KEY:
    sys.exit("Set GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)")

client = genai.Client(api_key=KEY)
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # adjust to a current Gemini model if needed

PROMPT = """This is a Google Cloud Professional Cloud Architect (PCA) exam-prep video.
Extract EVERY distinct multiple-choice exam question discussed in it.
Return ONLY a JSON array. Each element must be exactly:
{"c":"<compute|data|net|sec|ops|ai|mig|case>","q":"<full question text>","o":["A","B","C","D"],"a":<0-based index of correct option>,"e":"<one-line rationale>","oe":["why A is right/wrong","why B","why C","why D"]}
oe[a] must explain why the correct option is right; the others why they are wrong.
If the video contains no exam questions, return []. Output JSON only, no prose, no code fences."""

def extract(url):
    resp = client.models.generate_content(
        model=MODEL,
        contents=types.Content(parts=[
            types.Part(file_data=types.FileData(file_uri=url)),
            types.Part(text=PROMPT),
        ]),
    )
    txt = (resp.text or "").strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        if txt.lstrip().lower().startswith("json"):
            txt = txt.lstrip()[4:]
        txt = txt.strip().rstrip("`").strip()
    try:
        return json.loads(txt)
    except Exception as e:
        sys.stderr.write(f"  ! parse failed for {url}: {e}\n")
        return []

def main():
    urls = sys.argv[1:]
    if not urls:
        sys.exit("Pass one or more YouTube video URLs (see header for playlist expansion).")
    out = []
    for i, url in enumerate(urls, 1):
        sys.stderr.write(f"[{i}/{len(urls)}] {url}\n")
        try:
            out += extract(url)
        except Exception as e:
            sys.stderr.write(f"  ! error: {e}\n")
            time.sleep(2)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    sys.stderr.write(f"Done. Extracted {len(out)} questions.\n")

if __name__ == "__main__":
    main()
