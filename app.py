"""
Stern Stewart Case Interview — Production Server
Groq API (free) for AI + Groq Orpheus TTS + Groq Whisper STT.
Stateless: client sends full conversation history each request.
Random case selection for unique sessions.
"""

import os
import io
import json
import random
import tempfile
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__, static_folder="static")
CORS(app)

MODEL = "llama-3.3-70b-versatile"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "daniel"

# ═══════════════════════════════════════
# CASE LIBRARY — random case each session
# ═══════════════════════════════════════
CASES = [
    {
        "id": "valves",
        "brief": "Ein mittelstaendischer europaeischer Hersteller von Industrieventilen mit 380 Millionen Euro Umsatz. Wachstum stagniert bei 1 bis 2 Prozent pro Jahr seit drei Jahren. Der CEO fragt: Sollen wir in den nordamerikanischen Markt eintreten, und wenn ja, wie? Budget: bis zu 50 Millionen Euro.",
        "data": """CASE DATA (reveal progressively):
- Current markets: DACH 55 Prozent, Western Europe 35 Prozent, Asia 10 Prozent
- Product lines: Standard valves 60 Prozent revenue 15 Prozent margin, Specialty high-pressure valves 30 Prozent revenue 28 Prozent margin, Services aftermarket 10 Prozent revenue 45 Prozent margin
- NA market: roughly 12 Milliarden Dollar total, growing 3 bis 4 Prozent per year
- Top 3 NA competitors hold roughly 40 Prozent share, rest fragmented
- Client has zero NA presence, no brand recognition
- Acquisition target: Texas-based company, 80 Millionen Dollar revenue, 8 Prozent EBITDA margin, asking price roughly 120 Millionen Dollar
- Client current ROIC: 14 Prozent, WACC: 9 Prozent
- CEO wants decision within 6 months"""
    },
    {
        "id": "pharma",
        "brief": "Ein mittelgrosses europaeisches Pharmaunternehmen mit 620 Millionen Euro Umsatz, spezialisiert auf Generika. Die Margen sinken seit zwei Jahren wegen zunehmendem Preisdruck. Der CEO fragt: Sollen wir in den Bereich Biosimilars einsteigen, und wie? Investitionsbudget: 200 Millionen Euro ueber drei Jahre.",
        "data": """CASE DATA (reveal progressively):
- Portfolio: 45 Generika-Produkte, 3 in der Pipeline
- Margins dropping from 22 Prozent to 16 Prozent EBITDA over 2 years
- Biosimilar market growing 25 Prozent per year, expected 80 Milliarden Dollar by 2030
- Key competitor launched 3 biosimilars last year, captured 12 Prozent market share
- Client has no biologics manufacturing capability
- Potential partnership with Korean CDMO, would reduce capex by 40 Prozent
- R und D cost per biosimilar: roughly 100 bis 200 Millionen Dollar, 5 bis 8 year timeline
- Regulatory pathway: abbreviated in EU, more complex in US
- Client ROIC: 18 Prozent, WACC: 11 Prozent"""
    },
    {
        "id": "logistics",
        "brief": "Ein europaeischer Logistikkonzern mit 1,2 Milliarden Euro Umsatz. Die E-Commerce-Sparte waechst 15 Prozent pro Jahr, aber die traditionelle B2B-Sparte schrumpft um 3 Prozent. Der CEO fragt: Wie sollen wir das Portfolio restrukturieren? Sollen wir die B2B-Sparte verkaufen?",
        "data": """CASE DATA (reveal progressively):
- E-Commerce division: 400 Millionen Euro revenue, 8 Prozent EBIT margin, growing 15 Prozent per year
- B2B division: 800 Millionen Euro revenue, 12 Prozent EBIT margin, declining 3 Prozent per year
- B2B has 3 large warehouse facilities, partially underutilized at 65 Prozent capacity
- E-Commerce needs 2 new fulfillment centers, estimated 80 Millionen Euro each
- Private equity buyer offered 900 Millionen Euro for B2B division, roughly 7x EBIT
- Last-mile delivery costs increasing 8 Prozent per year
- 3 key B2B customers represent 35 Prozent of B2B revenue, contracts expire in 18 months
- Company net debt: 400 Millionen Euro, leverage ratio 2.1x
- Industry average PE for logistics: 14x"""
    },
    {
        "id": "energy",
        "brief": "Ein deutsches Energieunternehmen mit 950 Millionen Euro Umsatz, hauptsaechlich im Bereich konventionelle Stromerzeugung. Der Regulierungsdruck steigt, CO2-Kosten verdreifachen sich bis 2028. Der CEO fragt: Wie transformieren wir unser Geschaeftsmodell Richtung erneuerbare Energien? Budget: 500 Millionen Euro ueber fuenf Jahre.",
        "data": """CASE DATA (reveal progressively):
- Revenue split: conventional power 70 Prozent, renewables 20 Prozent, grid services 10 Prozent
- Conventional assets: 4 gas plants, 2 coal plants, average age 18 years
- Coal plants must close by 2030 per regulation, represent 25 Prozent of revenue
- Current renewable portfolio: 800 MW wind, 200 MW solar
- Pipeline opportunities: 2 GW offshore wind project, requires 350 Millionen Euro equity
- CO2 cost currently 45 Euro per ton, expected 120 Euro per ton by 2028
- Green hydrogen pilot possible, estimated 80 Millionen Euro investment
- Current EBITDA margin: 19 Prozent conventional, 35 Prozent renewables
- WACC: 8 Prozent, renewable projects qualify for 3 Prozent green financing"""
    },
    {
        "id": "retail",
        "brief": "Eine europaeische Modekette mit 280 Filialen und 1,8 Milliarden Euro Umsatz. Der Online-Anteil liegt bei nur 12 Prozent, waehrend der Branchendurchschnitt bei 30 Prozent liegt. Die Profitabilitaet sinkt seit drei Jahren. Der CEO fragt: Sollen wir 80 Filialen schliessen und voll auf Digital setzen, oder gibt es einen besseren Weg?",
        "data": """CASE DATA (reveal progressively):
- 280 stores across 8 European countries, average store size 450 sqm
- Store EBIT contribution varies wildly: top 30 Prozent generate 85 Prozent of store profit
- Online channel: 12 Prozent of revenue, growing 22 Prozent per year, 14 Prozent EBIT margin
- Store channel: 88 Prozent of revenue, declining 4 Prozent per year, 3 Prozent EBIT margin
- Average lease duration: 7 years, 40 Prozent of leases expire within 2 years
- Closing cost per store: roughly 800.000 Euro including severance and lease penalties
- Customer data shows 60 Prozent of online orders involve prior store visit
- Omnichannel customers spend 2.4x more than pure online or pure store customers
- Competitor closed 100 stores last year, lost 15 Prozent of total revenue"""
    }
]

BASE_SYSTEM = """You are Dr. Keller, Partner at Stern Stewart & Co. You are conducting a live case interview.

LANGUAGE RULES:
- Speak in GERMAN at all times
- When presenting data, numbers, KPIs, or frameworks, also state them in English in parentheses for clarity
- Example: Die EBITDA-Marge liegt bei 12 Prozent (EBITDA margin: 12 percent)

CANDIDATE PROFILE:
- Advanced candidate with strong fundamentals
- Preparing for real consulting interviews
- Pressure level: medium to high

YOUR BEHAVIOR:
- Act as a REAL interviewer. Never break character
- Be concise, direct, professional, and slightly intimidating
- Do NOT teach, coach, or over-explain
- Let the candidate drive the case. Intervene only to challenge or redirect
- Expect hypothesis-driven thinking from the start
- Challenge vague answers immediately
- Push for precision in language, logic, and numbers
- Interrupt if reasoning is unclear or inefficient
- Apply realistic time pressure
- CRITICAL: Keep responses SHORT. Max 3 to 4 sentences. This is a spoken voice conversation
- Do not use markdown, asterisks, bullet points, dashes, or special characters
- Write plain spoken German as if you are talking to someone in person
- Write numbers as words or digits. Write Euro, Dollar, Prozent as words, never symbols

CASE:
{case_brief}

{case_data}

FLOW:
1. Start with a brief professional introduction and present the case clearly
2. Let the candidate drive. Ask sharp follow-ups. Introduce data only when relevant
3. After enough discussion, push for a final recommendation as a 60 second CEO pitch
4. When asked for debrief, give structured feedback with scores 1 to 5 for Structure, Analysis, Quantitative Performance, Communication, plus an overall PASS, BORDERLINE, or FAIL, plus how a top 10 Prozent candidate would have approached it"""


def get_system_prompt(case_id=None):
    if case_id:
        case = next((c for c in CASES if c["id"] == case_id), random.choice(CASES))
    else:
        case = random.choice(CASES)
    return BASE_SYSTEM.format(case_brief=case["brief"], case_data=case["data"]), case["id"]


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def call_llm(messages, system_prompt):
    client = get_client()
    chat_messages = [{"role": "system", "content": system_prompt}] + messages
    response = client.chat.completions.create(
        model=MODEL,
        messages=chat_messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_tts(text):
    """Generate audio using Groq Orpheus TTS. Returns WAV bytes."""
    client = get_client()
    try:
        response = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            input=text,
            response_format="wav"
        )
        return response.content
    except Exception as e:
        print(f"TTS error: {e}")
        return None


# ═══════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════

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
        case_id = data.get("case_id", None)
        if not messages:
            return jsonify({"error": "No messages"}), 400
        system_prompt, used_case_id = get_system_prompt(case_id)
        reply = call_llm(messages, system_prompt)
        return jsonify({"reply": reply, "case_id": used_case_id})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts", methods=["POST"])
def tts():
    try:
        data = request.json
        text = data.get("text", "")
        if not text:
            return jsonify({"error": "No text"}), 400
        audio = generate_tts(text)
        if audio and len(audio) > 100:
            return Response(audio, mimetype="audio/wav",
                            headers={"Content-Length": str(len(audio))})
        else:
            return jsonify({"error": "TTS failed"}), 500
    except Exception as e:
        print(f"TTS error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stt", methods=["POST"])
def stt():
    """Transcribe audio using Groq Whisper. Accepts audio file upload."""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file"}), 400

        audio_file = request.files["audio"]
        client = get_client()

        # Save to temp file (Whisper needs a file-like with name)
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            audio_file.save(tmp)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    language="de",
                )
            text = transcription.text.strip()
            return jsonify({"text": text})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        print(f"STT error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    has_key = bool(os.environ.get("GROQ_API_KEY"))
    return jsonify({"status": "ok", "api_key_configured": has_key, "cases": len(CASES)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8181))
    app.run(host="0.0.0.0", port=port, debug=False)
