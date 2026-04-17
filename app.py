"""
Stern Stewart Case Interview — Production Server
Uses Groq API (free) + edge-tts (Microsoft Neural voices).
"""

import os
import json
import asyncio
import tempfile
import base64
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import edge_tts

app = Flask(__name__, static_folder="static")
CORS(app)

VOICE = "de-DE-FlorianMultilingualNeural"
MODEL = "llama-3.3-70b-versatile"

sessions = {}

SYSTEM_PROMPT = """You are Dr. Keller, Partner at Stern Stewart & Co. You are conducting a live case interview.

LANGUAGE RULES:
- Speak in GERMAN at all times
- When presenting data, numbers, KPIs, or frameworks, also state them in English in parentheses for clarity
- Example: "Die EBITDA-Marge liegt bei 12 Prozent (EBITDA margin: 12%)"

CANDIDATE PROFILE:
- Advanced candidate with strong fundamentals
- Preparing for real consulting interviews
- Pressure level: medium to high

YOUR BEHAVIOR:
- Act as a REAL interviewer — never break character
- Be concise, direct, professional, and slightly intimidating
- Do NOT teach, coach, or over-explain
- Let the candidate drive the case — intervene only to challenge or redirect
- Expect hypothesis-driven thinking from the start
- Challenge vague answers immediately
- Push for precision in language, logic, and numbers
- Interrupt if reasoning is unclear or inefficient
- Apply realistic time pressure
- CRITICAL: Keep responses SHORT — max 3-4 sentences. This is a spoken conversation, not a written one.
- Do not use markdown formatting, asterisks, bullet points, or special characters. Write plain spoken German as if you are talking.

CASE:
A mid-sized European industrial valve manufacturer with 380 Millionen Euro Umsatz. Growth stagnated at 1 bis 2 Prozent per year for three years. CEO asks: Should we enter the North American market, and if so, how? Budget: up to 50 Millionen Euro.

CASE DATA (reveal progressively, only when relevant):
- Current markets: DACH 55 Prozent, Western Europe 35 Prozent, Asia 10 Prozent
- Product lines: Standard valves 60 Prozent revenue 15 Prozent margin, Specialty high-pressure valves 30 Prozent revenue 28 Prozent margin, Services aftermarket 10 Prozent revenue 45 Prozent margin
- NA market: roughly 12 Milliarden Dollar total, growing 3 bis 4 Prozent per year
- Top 3 NA competitors hold roughly 40 Prozent share, rest fragmented
- Client has zero NA presence, no brand recognition
- Acquisition target: Texas-based valve company, 80 Millionen Dollar revenue, 8 Prozent EBITDA margin, asking price roughly 120 Millionen Dollar
- Client current ROIC: 14 Prozent, WACC: 9 Prozent
- CEO wants decision within 6 months

IMPORTANT: Write numbers as words or with digits, never use special symbols. Write Euro not the symbol. Write Dollar not the symbol. Write Prozent not the symbol. No bullet points or dashes. This text will be read aloud by a voice system."""


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def call_llm(messages):
    client = get_client()
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = client.chat.completions.create(
        model=MODEL,
        messages=chat_messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_audio(text):
    clean = text.replace('"', '').replace("'", "").replace("*", "").replace("#", "").replace("_", "")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        mp3_path = tmp.name

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _gen():
            communicate = edge_tts.Communicate(clean, voice=VOICE, rate="-3%", pitch="-2Hz")
            await communicate.save(mp3_path)

        loop.run_until_complete(_gen())
        loop.close()

        with open(mp3_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"TTS error: {e}")
        return None
    finally:
        try:
            os.unlink(mp3_path)
        except:
            pass


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/start", methods=["POST"])
def start():
    session_id = str(uuid.uuid4())

    messages = [
        {"role": "user", "content": "Bitte stellen Sie sich kurz vor und präsentieren Sie den Case. Maximal 4 bis 5 Sätze, gesprochen, direkt."}
    ]

    reply = call_llm(messages)
    messages.append({"role": "assistant", "content": reply})
    sessions[session_id] = messages

    audio = generate_audio(reply)
    return jsonify({"reply": reply, "audio": audio, "session_id": session_id})


@app.route("/api/message", methods=["POST"])
def message():
    data = request.json
    text = data.get("text", "").strip()
    session_id = data.get("session_id", "")

    if not text:
        return jsonify({"error": "No text"}), 400

    messages = sessions.get(session_id, [])
    messages.append({"role": "user", "content": text})

    reply = call_llm(messages)
    messages.append({"role": "assistant", "content": reply})
    sessions[session_id] = messages

    audio = generate_audio(reply)
    return jsonify({"reply": reply, "audio": audio})


@app.route("/api/end", methods=["POST"])
def end():
    data = request.json
    session_id = data.get("session_id", "")

    messages = sessions.get(session_id, [])
    messages.append({
        "role": "user",
        "content": "Das Interview ist zu Ende. Gib jetzt das strukturierte Feedback auf Deutsch. Bewerte Structure, Analysis, Quantitative Performance, Communication jeweils 1 bis 5. Gib eine Overall Recommendation: PASS, BORDERLINE, oder FAIL. Beschreibe wie ein Top 10 Prozent Kandidat den Case angegangen wäre. Schreib es als gesprochenen Text ohne Sonderzeichen."
    })

    reply = call_llm(messages)
    messages.append({"role": "assistant", "content": reply})
    sessions[session_id] = messages

    audio = generate_audio(reply)

    if session_id in sessions:
        del sessions[session_id]

    return jsonify({"reply": reply, "audio": audio})


@app.route("/api/health", methods=["GET"])
def health():
    has_key = bool(os.environ.get("GROQ_API_KEY"))
    return jsonify({"status": "ok", "api_key_configured": has_key})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8181))
    app.run(host="0.0.0.0", port=port, debug=False)
