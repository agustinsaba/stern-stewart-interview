"""
Stern Stewart Case Interview — Production Server
Groq API (free) for AI + edge-tts server-side for natural voice.
Stateless: client sends full conversation history each request.
"""

import os
import json
import asyncio
import tempfile
import base64
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from openai import OpenAI
import edge_tts

app = Flask(__name__, static_folder="static")
CORS(app)

MODEL = "llama-3.3-70b-versatile"
TTS_VOICE = "de-DE-FlorianMultilingualNeural"

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
- CRITICAL: Keep responses SHORT — max 3-4 sentences. This is a spoken conversation.
- Do not use markdown formatting, asterisks, bullet points, or special characters. Write plain spoken German.

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

IMPORTANT: Write numbers as words or digits, never symbols. Write Euro not the symbol. Write Dollar not the symbol. Write Prozent not the symbol. No bullet points or dashes. This text will be read aloud."""


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


def generate_tts(text):
    """Generate MP3 audio bytes using edge-tts."""
    clean = text.replace('"', '').replace("'", "").replace("*", "").replace("#", "").replace("_", "")

    async def _generate():
        communicate = edge_tts.Communicate(clean, voice=TTS_VOICE, rate="-3%", pitch="-2Hz")
        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    try:
        loop = asyncio.new_event_loop()
        audio_bytes = loop.run_until_complete(_generate())
        loop.close()
        return audio_bytes
    except Exception as e:
        print(f"TTS error: {e}")
        return None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        messages = data.get("messages", [])
        if not messages:
            return jsonify({"error": "No messages"}), 400
        reply = call_llm(messages)
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts", methods=["POST"])
def tts():
    """Generate audio from text. Returns MP3 binary."""
    try:
        data = request.json
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text"}), 400

        audio = generate_tts(text)
        if audio and len(audio) > 0:
            return Response(audio, mimetype="audio/mpeg",
                            headers={"Content-Length": str(len(audio)),
                                     "Cache-Control": "no-cache"})
        else:
            return jsonify({"error": "TTS generation failed"}), 500
    except Exception as e:
        print(f"TTS endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    has_key = bool(os.environ.get("GROQ_API_KEY"))
    return jsonify({"status": "ok", "api_key_configured": has_key})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8181))
    app.run(host="0.0.0.0", port=port, debug=False)
