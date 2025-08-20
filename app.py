import os
import io
import re
import tempfile
import subprocess
from pathlib import Path
from typing import List
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from gtts import gTTS
try:
    from openai import OpenAI
    OPENAI_SDK = True
except ImportError:
    OPENAI_SDK = False
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder="templates")
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_SDK and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Failed to initialize OpenAI: {str(e)}")

OPENAI_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "ash", "sage"]
VOICE_META = {
    "alloy": {"gender": "male", "accent": "american"},
    "echo": {"gender": "neutral", "accent": "american"},
    "fable": {"gender": "female", "accent": "american"},
    "onyx": {"gender": "male", "accent": "british"},
    "nova": {"gender": "female", "accent": "american"},
    "shimmer": {"gender": "female", "accent": "american"},
    "coral": {"gender": "female", "accent": "british"},
    "ash": {"gender": "male", "accent": "indian"},
    "sage": {"gender": "female", "accent": "indian"},
}
ACCENT_ORDER = ["american", "british", "indian", "australian", "italian", "filipino", "arabic", "asian", "russian"]
INDIAN_STATES = [
    "west_bengal", "tamil_nadu", "maharashtra", "karnataka", "delhi",
    "uttar_pradesh", "kerala", "punjab", "gujarat", "andhra_pradesh"
]
ACCENT_SYNONYMS = {
    "phillipino": "filipino", "philippino": "filipino", "philippine": "filipino",
    "russia": "russian", "rus": "russian", "uae": "arabic", "saudi": "arabic",
}
LANG_TLD_MAP = {
    "indian": "co.in", "british": "co.uk", "australian": "com.au", "american": "com",
    "filipino": "com.ph", "italian": "it", "arabic": "com.sa", "russian": "ru",
}
STATE_TLD_MAP = {
    "west_bengal": "co.in",
    "tamil_nadu": "co.in",
    "maharashtra": "co.in",
    "karnataka": "co.in",
    "delhi": "co.in",
    "uttar_pradesh": "co.in",
    "kerala": "co.in",
    "punjab": "co.in",
    "gujarat": "co.in",
    "andhra_pradesh": "co.in"
}

def normalize_accent(name: str) -> str:
    if not name:
        return "auto"
    name = name.strip().lower()
    return ACCENT_SYNONYMS.get(name, name)

def pick_openai_voice(gender: str, accent: str, state: str = None) -> str:
    gender = (gender or "auto").lower()
    accent = normalize_accent(accent)
    if accent == "indian":
        return "ash" if gender in ("male", "auto") else "sage"
    for v, m in VOICE_META.items():
        if gender != "auto" and m["gender"] == gender and (accent in ("auto", m["accent"])):
            return v
    for v, m in VOICE_META.items():
        if gender != "auto" and m["gender"] == gender:
            return v
    for v, m in VOICE_META.items():
        if accent != "auto" and m["accent"] == accent:
            return v
    return "alloy" if gender in ("male", "auto") else "nova"

def have_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def apply_speed_ffmpeg(in_path: str, out_path: str, speed_pct: int) -> bool:
    sp = max(-100, min(100, int(speed_pct)))
    factor = 1.0 + (sp / 100.0)
    factor = max(0.5, min(3.0, factor))
    cmd = ["ffmpeg", "-y", "-i", in_path, "-filter:a", f"atempo={factor}", "-c:a", "mp3", out_path]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print(f"FFmpeg error: {str(e)}")
        return False

def concat_mp3(parts: List[str], out_path: str) -> bool:
    if not parts:
        return False
    if len(parts) == 1:
        try:
            with open(parts[0], "rb") as f_in, open(out_path, "wb") as f_out:
                f_out.write(f_in.read())
            return True
        except Exception as e:
            print(f"Concat single file error: {str(e)}")
            return False
    if not have_ffmpeg():
        try:
            with open(out_path, "wb") as f_out:
                for p in parts:
                    with open(p, "rb") as f_in:
                        f_out.write(f_in.read())
            return True
        except Exception as e:
            print(f"Concat fallback error: {str(e)}")
            return False
    tmp_list = out_path + ".txt"
    try:
        with open(tmp_list, "w", encoding="utf-8") as f:
            for p in parts:
                f.write(f"file '{p}'\n")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", tmp_list, "-c", "copy", out_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception as e:
        print(f"Concat FFmpeg error: {str(e)}")
        return False
    finally:
        try:
            os.remove(tmp_list)
        except Exception:
            pass

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
def chunk_text(text: str, max_len: int = 480) -> List[str]:
    sentences = _SENT_SPLIT.split(text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) + 1 <= max_len:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s.strip()
    if cur:
        chunks.append(cur)
    final = []
    for c in chunks:
        if len(c) <= max_len:
            final.append(c)
        else:
            words = c.split()
            buf = []
            lng = 0
            for w in words:
                if lng + len(w) + 1 > max_len:
                    final.append(" ".join(buf))
                    buf, lng = [w], len(w)
                else:
                    buf.append(w)
                    lng += len(w) + 1
            if buf:
                final.append(" ".join(buf))
    return final

def punctuate_for_speed(text: str, speed_pct: int) -> str:
    if speed_pct >= 0:
        return text
    words = text.split()
    if not words:
        return text
    mag = min(100, max(0, -int(speed_pct)))
    every_k = max(2, int(10 - (mag / 10)))
    pause_token = " … " if mag >= 50 else ", "
    out, cnt = [], 0
    for w in words:
        out.append(w)
        cnt += 1
        if cnt >= every_k:
            out.append(pause_token)
            cnt = 0
    out_text = " ".join(out)
    out_text = re.sub(r",\s*", ", … ", out_text)
    return out_text

def expressive_rewrite(text: str, level: int, accent: str, state: str = None) -> str:
    if not openai_client or level <= 0 or not text.strip():
        return text
    level = max(0, min(100, int(level)))
    style = "subtle" if level < 30 else "natural" if level < 70 else "expressive"
    accent = normalize_accent(accent)
    accent_note = ""
    if accent == "indian" and state:
        state = state.lower()
        state_phrasing = {
            "west_bengal": (
                "Use Bengali-influenced Indian English with a highly melodic, articulate intonation, "
                "slightly slower pacing (e.g., 0.9x speed), clear syllable-timed pronunciation (e.g., 'in-no-va-tive' with equal stress), "
                "non-rhotic sounds (e.g., 'car' as 'kaa', 'start' as 'staa-t'), and crisp vowel articulation (e.g., 'face' as 'fess'). "
                "Emulate the polished, broadcaster-like tone of native Bengali English speakers, ensuring a distinctly Indian cadence from the first word."
            ),
            "tamil_nadu": (
                "Use Tamil-influenced Indian English with rhythmic, precise pauses, clear vowel articulation (e.g., 'face' as 'fess'), "
                "and a formal, polished tone, reflecting the structured, deliberate cadence of Tamil speakers from the first word."
            ),
            "maharashtra": (
                "Use Marathi-influenced Indian English with clear, articulate enunciation, moderate pacing, "
                "slight retroflex stress (e.g., 't' as 'ṭ', 'data' as 'daa-ṭa'), and a professional conversational tone from the first word."
            ),
            "karnataka": (
                "Use Kannada-influenced Indian English with steady, balanced pacing, neutral yet clear intonation, "
                "and highly articulate pronunciation (e.g., 'in-no-va-tive' with equal stress), reflecting Kannada’s precise phonetics from the first word."
            ),
            "delhi": (
                "Use Hindi-influenced Indian English with dynamic, articulate intonation, moderate pacing, "
                "clear syllables, and slight retroflex sounds (e.g., 't' as 'ṭ', 'better' as 'beṭ-ṭer'), with a polished urban tone from the first word."
            ),
            "uttar_pradesh": (
                "Use Hindi-influenced Indian English with a conversational yet polished North Indian tone, "
                "clear syllable-timed rhythm (e.g., 'pro-ject' with equal stress), and precise stress on key words from the first word."
            ),
            "kerala": (
                "Use Malayalam-influenced Indian English with soft, flowing intonation, gentle yet clear rhythm, "
                "and polished vowel pronunciation (e.g., 'face' as 'fess'), reflecting Malayalam’s smooth, melodic phonetics from the first word."
            ),
            "punjab": (
                "Use Punjabi-influenced Indian English with confident, articulate intonation, "
                "clear enunciation, and a slightly faster-paced rhythm (e.g., 1.1x speed), ensuring a distinctly Indian cadence from the first word."
            ),
            "gujarat": (
                "Use Gujarati-influenced Indian English with concise, highly clear delivery, "
                "neutral intonation, and precise syllable stress (e.g., 'ser-vice' with equal stress), maintaining a professional tone from the first word."
            ),
            "andhra_pradesh": (
                "Use Telugu-influenced Indian English with expressive yet precise pauses, formal tone, "
                "and clear, articulate delivery, emphasizing a steady, rhythmic flow with distinct syllable articulation from the first word."
            )
        }
        accent_note = (
            f"Match a distinctly authentic, professional Indian English accent with {state_phrasing.get(state, 'generic Indian English')} "
            "Prioritize syllable-timed rhythm (e.g., 'in-no-va-tive' with equal stress), crisp vowel articulation (e.g., 'face' as 'fess'), "
            "non-rhotic pronunciation (e.g., 'car' as 'kaa', 'start' as 'staa-t'), and clear, melodic intonation patterns typical of native Indian English speakers from this region. "
            "Ensure a polished, broadcaster-like tone from the start, distinctly different from American or British accents, avoiding colloquial or informal phrases."
        )
    elif accent == "indian":
        accent_note = (
            "Match a distinctly authentic, professional Indian English accent with a clear syllable-timed rhythm (e.g., 'pro-ject' with equal stress), "
            "crisp vowel articulation (e.g., 'face' as 'fess'), non-rhotic pronunciation (e.g., 'car' as 'kaa', 'start' as 'staa-t'), "
            "and a melodic, articulate intonation. "
            "Emulate the polished, broadcaster-like tone of native Indian English speakers, "
            "ensuring the accent is strong and distinctly Indian from the first word, avoiding colloquial or informal phrases."
        )
    elif accent == "british":
        accent_note = (
            "Match a natural, professional British English accent with clear, articulate intonation, "
            "crisp enunciation, and a refined, formal tone typical of native British speakers (e.g., BBC presenters). "
            "Use rhotic pronunciation where appropriate and a smooth, measured cadence, avoiding overly casual or regional slang."
        )
    elif accent == "american":
        accent_note = (
            "Match a natural, professional American English accent with clear, articulate intonation, "
            "standard pronunciation, and a neutral, professional tone typical of American broadcasters. "
            "Ensure a smooth, confident delivery, avoiding regional slang."
        )
    else:
        accent_note = (
            f"Match a natural, professional {accent} English accent with clear, articulate intonation, "
            "balanced rhythm, and precise tone, ensuring a polished, native-like delivery from the first word."
        )
    prompt = (
        "Rewrite for natural text-to-speech, keeping meaning intact. "
        "Do NOT prepend any priming phrase; start directly with the input text. "
        "Add punctuation (commas, em-dashes, ellipses) for precise pauses and rhythm, "
        "and use *asterisks* for light emphasis on key words to enhance clarity. "
        "Ensure the tone and phrasing suit a native speaker's polished, professional delivery for the specified accent. "
        "For Indian English, prioritize syllable-timed rhythm (e.g., 'in-no-va-tive' with equal stress), crisp vowel articulation (e.g., 'face' as 'fess'), "
        "non-rhotic sounds (e.g., 'car' as 'kaa'), and articulate, broadcaster-like intonation patterns, distinctly different from American or British accents, tailored to regional nuances if specified. "
        f"Style: {style}. {accent_note}\n\n---\n{text}\n---\n"
        "Return ONLY the rewritten script, starting directly with the input text."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You refine text for natural TTS, preserving meaning and enhancing delivery for the specified accent. "
                        "For Indian English, prioritize syllable-timed rhythm (e.g., 'in-no-va-tive'), crisp vowel articulation (e.g., 'face' as 'fess'), "
                        "non-rhotic pronunciation (e.g., 'car' as 'kaa', 'start' as 'staa-t'), and polished, broadcaster-like intonation patterns from the first word. "
                        "For British English, use clear, articulate intonation and a refined, formal tone (e.g., BBC presenter style). "
                        "For American English, use standard, confident pronunciation with a neutral, professional tone. "
                        "Ensure a clear, professional, and distinctly appropriate tone for each accent, avoiding colloquial phrases."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.03,
            max_tokens=1000,
        )
        rewritten = (resp.choices[0].message.content or "").strip() or text
        return rewritten
    except Exception as e:
        print(f"OpenAI rewrite error: {str(e)}")
        return text

def tts_openai_to_mp3_files(text: str, gender: str, accent: str, state: str = None) -> List[str]:
    chosen_voice = pick_openai_voice(gender, accent, state)
    if accent == "indian":
        chosen_voice = "ash" if gender in ("male", "auto") else "sage"
    parts = []
    for chunk in chunk_text(text):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as t:
            out_path = t.name
        try:
            with openai_client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=chosen_voice,
                input=chunk,
                response_format="mp3",
                speed=1.0
            ) as resp:
                resp.stream_to_file(out_path)
        except Exception:
            try:
                r = openai_client.audio.speech.create(
                    model="tts-1",
                    voice=chosen_voice,
                    input=chunk,
                    response_format="mp3",
                    speed=1.0
                )
                data = getattr(r, "content", None) or r.read()
                with open(out_path, "wb") as f:
                    f.write(data)
            except Exception as e:
                print(f"OpenAI TTS error: {str(e)}")
                try:
                    os.remove(out_path)
                except Exception:
                    pass
                raise e
        parts.append(out_path)
    return parts

def tts_gtts_to_mp3_files(text: str, language: str, accent: str, state: str = None) -> List[str]:
    lang_map = {"en-us": "en", "en-uk": "en", "en-au": "en", "zh-cn": "zh-CN"}
    lang = lang_map.get((language or "en").lower(), (language or "en").lower())
    tld = STATE_TLD_MAP.get(state, "co.in") if accent == "indian" and state else LANG_TLD_MAP.get(accent, "co.in" if lang.startswith("hi") else "com")
    parts = []
    for chunk in chunk_text(text, max_len=350):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as t:
            out_path = t.name
        try:
            tts = gTTS(chunk, lang=lang, tld=tld, slow=False)
            with open(out_path, "wb") as f:
                tts.write_to_fp(f)
            parts.append(out_path)
        except Exception as e:
            print(f"gTTS error: {str(e)}")
            try:
                os.remove(out_path)
            except Exception:
                pass
            raise e
    return parts

@app.route("/")
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        print(f"Template render error: {str(e)}")
        return jsonify({"error": f"Failed to render template: {str(e)}"}), 500

@app.route("/providers", methods=["GET"])
def providers():
    try:
        data = {
            "providers": [
                {
                    "id": "gtts",
                    "name": "Google gTTS",
                    "voices": [{"id": "standard", "label": "Standard"}],
                    "languages": [
                        "en", "en-us", "en-uk", "en-au", "hi", "bn", "ta",
                        "te", "gu", "kn", "ml", "mr", "fr", "de", "es",
                        "it", "pt", "ru", "ja", "ko", "zh-CN", "ar"
                    ],
                    "supports": {"gender": False, "accent": True, "speed": True, "exaggeration": True, "state": True}
                }
            ],
            "default_provider": "gtts",
            "gender_options": ["auto", "male", "female", "neutral"],
            "accent_options": ["auto"] + ACCENT_ORDER,
            "indian_states": [
                {"id": s, "label": s.replace("_", " ").title()}
                for s in INDIAN_STATES
            ]
        }
        if openai_client:
            data["providers"].append({
                "id": "openai",
                "name": "OpenAI TTS",
                "voices": [{"id": v, "label": v[0].upper() + v[1:]} for v in OPENAI_VOICES],
                "languages": ["auto"],
                "supports": {"gender": True, "accent": True, "speed": True, "exaggeration": True, "state": True}
            })
        return jsonify(data)
    except Exception as e:
        print(f"Providers endpoint error: {str(e)}")
        return jsonify({"error": f"Failed to load providers: {str(e)}"}), 500

@app.route("/generate", methods=["POST"])
def generate():
    try:
        d = request.get_json(force=True)
        provider = (d.get("provider") or "gtts").lower()
        voice = (d.get("voice") or "auto").lower()
        language = d.get("language") or "en"
        text = (d.get("prompt") or "").strip()
        gender = (d.get("gender") or "auto").lower()
        accent = normalize_accent(d.get("accent") or "auto")
        state = (d.get("state") or "").lower() if accent == "indian" else None
        exaggeration = int(d.get("exaggeration") or 0)
        speed_pct = int(d.get("speed") or 0)

        if not text:
            return jsonify({"error": "Please provide text/prompt"}), 400

        paced = punctuate_for_speed(text, speed_pct)
        styled_text = expressive_rewrite(paced, exaggeration, accent, state)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "final.mp3")
            if provider == "openai":
                if not openai_client:
                    return jsonify({"error": "OpenAI not configured."}), 400
                chosen_voice = voice if (voice != "auto" and voice in OPENAI_VOICES) else pick_openai_voice(gender, accent, state)
                parts = tts_openai_to_mp3_files(styled_text, gender, accent, state)
            else:
                parts = tts_gtts_to_mp3_files(styled_text, language, accent, state)
            ok_concat = concat_mp3(parts, out_path)
            if not ok_concat:
                out_path = parts[0]
            if speed_pct != 0:
                sped_path = str(Path(tmpdir) / "sped.mp3")
                if apply_speed_ffmpeg(out_path, sped_path, speed_pct):
                    out_path = sped_path
            with open(out_path, "rb") as f:
                audio_bytes = f.read()
        buf = io.BytesIO(audio_bytes)
        buf.seek(0)
        return send_file(buf, mimetype="audio/mpeg", as_attachment=True, download_name="output.mp3")
    except Exception as e:
        print(f"Generate error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
