import os
import io
import tempfile
import subprocess
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from gtts import gTTS

# Optional OpenAI
try:
    from openai import OpenAI
    OPENAI_SDK = True
except Exception:
    OPENAI_SDK = False

load_dotenv()

app = Flask(__name__, template_folder="templates")
CORS(app)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_SDK and OPENAI_API_KEY) else None

# ---------- Voice metadata & helpers ----------
OPENAI_VOICES = [
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage"
]

# Heuristic mapping (tweak to your taste).
# Only a few OpenAI voices have clear regional vibes; others are neutral/American.
VOICE_META = {
    "alloy":   {"gender": "male",    "accent": "american"},
    "echo":    {"gender": "neutral", "accent": "american"},
    "fable":   {"gender": "female",  "accent": "american"},
    "onyx":    {"gender": "male",    "accent": "american"},
    "nova":    {"gender": "female",  "accent": "american"},
    "shimmer": {"gender": "female",  "accent": "american"},

    "coral":   {"gender": "female",  "accent": "british"},
    "verse":   {"gender": "male",    "accent": "british"},
    "ballad":  {"gender": "neutral", "accent": "british"},

    "ash":     {"gender": "male",    "accent": "indian"},
    "sage":    {"gender": "female",  "accent": "indian"},
}

# New accents added here
ACCENT_ORDER = [
    "american",
    "british",
    "indian",
    "australian",
    "italian",
    "filipino",  # normalize "phillipino"/"philippino" -> "filipino"
    "arabic",
    "asian",
    "russian"
]

# Synonym normalization for accents
ACCENT_SYNONYMS = {
    "phillipino": "filipino",
    "philippino": "filipino",
    "philipino": "filipino",
    "phillipine": "filipino",
    "phillipines": "filipino",
    "filipine": "filipino",
    "russia": "russian",
    "rus": "russian",
    "uae": "arabic",
    "saudi": "arabic",
}

def normalize_accent(name: str) -> str:
    if not name:
        return "auto"
    n = name.strip().lower()
    return ACCENT_SYNONYMS.get(n, n)

def pick_openai_voice(gender: str, accent: str) -> str:
    """Pick a reasonable OpenAI voice for requested gender/accent.
       If accent is outside our mapping, fall back to a sensible default."""
    gender = (gender or "auto").lower()
    accent = normalize_accent(accent)

    # Exact match first
    for v, meta in VOICE_META.items():
        if (gender in ("auto", meta["gender"])) and (accent in ("auto", meta["accent"])):
            return v

    # If a known accent was requested, prefer anything with that accent
    known_accents = {m["accent"] for m in VOICE_META.values()}
    if accent in known_accents:
        for v, meta in VOICE_META.items():
            if meta["accent"] == accent:
                return v

    # If accent is one of our new unsupported ones, pick closest:
    # - italian/filipino/arabic/asian/russian -> choose a neutral/american base
    #   (We still pass an accent hint to the exaggeration step to influence delivery.)
    # Prefer neutral voice when gender=neutral, else alloy (male) or fable/nova (female)
    if gender == "female":
        return "fable"
    if gender == "neutral":
        return "echo"
    return "alloy"

def apply_speed_ffmpeg(in_path: str, out_path: str, speed_pct: int):
    """Pitch-preserving speed using ffmpeg atempo. speed_pct in [-50, +50]."""
    try:
        sp = max(-50, min(50, int(speed_pct)))
        factor = 1.0 + (sp / 100.0)  # -50% -> 0.5, +50% -> 1.5
        # atempo supports 0.5–2.0 range (we're within this)
        cmd = ["ffmpeg", "-y", "-i", in_path, "-filter:a", f"atempo={factor}", "-c:a", "mp3", out_path]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True, None
    except Exception as e:
        return False, str(e)

def exaggerate_script(text: str, level: int, accent_hint: str) -> str:
    """Use LLM to add expressive cues/pauses based on exaggeration level (0-100)."""
    if not openai_client or level <= 0:
        return text

    level = max(0, min(100, int(level)))
    style = "subtle" if level < 30 else "lively" if level < 70 else "dramatic"
    accent_hint = normalize_accent(accent_hint)
    accent_note = f"Use an {accent_hint} English delivery." if accent_hint and accent_hint not in ("auto",) else ""

    prompt = (
        "Rewrite the following lines to sound more expressive for text-to-speech. "
        "Keep meaning the same; add natural pauses like [pause], light emphasis with *asterisks*, "
        "and short sentence breaks for clarity. Do NOT add more than 10% extra words. "
        f"Exaggeration level: {style}. {accent_note}\n\n---\n{ text }\n---\n"
        "Return ONLY the rewritten script."
    )
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You polish lines for expressive TTS without changing the message."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=400
        )
        out = resp.choices[0].message.content.strip()
        return out or text
    except Exception:
        return text

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/providers", methods=["GET"])
def providers():
    data = {
        "providers": [
            {
                "id": "gtts",
                "name": "Google gTTS (free)",
                "voices": [{"id": "standard", "label": "Standard"}],
                "languages": [
                    "en", "en-uk", "en-us", "en-au",
                    "hi", "bn", "ta", "te", "gu", "kn", "ml", "mr",
                    "fr", "de", "es", "it", "pt", "ru", "ja", "ko", "zh-CN", "ar"
                ],
                "supports": {"gender": False, "accent": True, "speed": True, "exaggeration": True}
            }
        ],
        "default_provider": "gtts",
        "gender_options": ["auto", "male", "female", "neutral"],
        "accent_options": ["auto"] + ACCENT_ORDER
    }

    if openai_client:
        data["providers"].append({
            "id": "openai",
            "name": "OpenAI TTS (gpt-4o-mini-tts)",
            "voices": [{"id": v, "label": v.capitalize()} for v in OPENAI_VOICES],
            "languages": ["auto"],  # OpenAI handles multilingual text
            "supports": {"gender": True, "accent": True, "speed": True, "exaggeration": True}
        })

    return jsonify(data)

@app.route("/generate", methods=["POST"])
def generate():
    """
    JSON body:
    {
      "provider": "gtts" | "openai",
      "voice": "auto" | valid voice id,
      "language": "en" | "hi" | "auto",
      "prompt": "text",
      "gender": "auto|male|female|neutral",
      "accent": "auto|american|british|indian|australian|italian|filipino|arabic|asian|russian",
      "exaggeration": 0-100,
      "speed": -50..+50   (percent)
    }
    """
    try:
        d = request.get_json(force=True)
        provider = (d.get("provider") or "gtts").lower()
        voice = (d.get("voice") or "auto").lower()
        language = d.get("language") or "en"
        text = (d.get("prompt") or "").strip()
        gender = (d.get("gender") or "auto").lower()
        accent = normalize_accent(d.get("accent") or "auto")
        exaggeration = int(d.get("exaggeration") or 0)
        speed_pct = int(d.get("speed") or 0)

        if not text:
            return jsonify({"error": "Please provide text/prompt"}), 400

        # 1) Style enhancement per Exaggeration (hint with accent)
        styled_text = exaggerate_script(text, exaggeration, accent)

        # 2) TTS synthesis
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_out:
            out_path = tmp_out.name

        if provider == "openai":
            if not openai_client:
                return jsonify({"error": "OpenAI not configured. Set OPENAI_API_KEY or choose gTTS."}), 400

            chosen_voice = voice if (voice != "auto" and voice in OPENAI_VOICES) else pick_openai_voice(gender, accent)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_raw:
                raw_path = tmp_raw.name

            try:
                # Prefer streaming API
                try:
                    with openai_client.audio.speech.with_streaming_response.create(
                        model="gpt-4o-mini-tts",
                        voice=chosen_voice,
                        input=styled_text,
                        response_format="mp3"
                    ) as resp, open(raw_path, "wb") as f:
                        f.write(resp.read())
                except AttributeError:
                    # Fallback (SDK variants)
                    tts = openai_client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice=chosen_voice,
                        input=styled_text,
                        response_format="mp3"
                    )
                    content = getattr(tts, "content", None)
                    with open(raw_path, "wb") as f:
                        f.write(content if content else tts.read())

                # Speed adjust if requested
                if speed_pct != 0:
                    ok, _ = apply_speed_ffmpeg(raw_path, out_path, speed_pct)
                    if not ok:
                        out_path = raw_path
                else:
                    out_path = raw_path

            except Exception as e:
                return jsonify({"error": f"OpenAI TTS failed: {e}"}), 500

        else:
            # gTTS path (accent affects choice of language code only if you wire it;
            # here we leave it to the user via the language dropdown)
            lang = (language or "en").lower()
            lang_map = {"en-us": "en", "en-uk": "en", "en-au": "en", "zh-cn": "zh-CN", "ar": "ar", "ru": "ru", "it": "it"}
            lang = lang_map.get(lang, lang)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_raw:
                raw_path = tmp_raw.name

            tts = gTTS(styled_text, lang=lang, slow=False)
            with open(raw_path, "wb") as f:
                tts.write_to_fp(f)

            if speed_pct != 0:
                ok, _ = apply_speed_ffmpeg(raw_path, out_path, speed_pct)
                if not ok:
                    out_path = raw_path
            else:
                out_path = raw_path

        # 3) Return audio
        with open(out_path, "rb") as f:
            audio_bytes = f.read()
        buf = io.BytesIO(audio_bytes)
        buf.seek(0)
        return send_file(buf, mimetype="audio/mpeg", as_attachment=True, download_name="output.mp3")

    except Exception as e:
        print("Error:", repr(e))
        return jsonify({"error": str(e)}), 500

# ---------- Runner ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
