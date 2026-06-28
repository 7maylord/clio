import os
import re
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
if os.path.exists("../museum-companion/.env"):
    load_dotenv("../museum-companion/.env")
else:
    load_dotenv()

# Google Cloud Configuration
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")
agent_runtime_id = os.environ.get("AGENT_RUNTIME_ID")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("museum_dashboard")

# Parse runtime ID
agent_engine_id = None
if agent_runtime_id:
    match = re.search(r"projects/([^/]+)/locations/([^/]+)/reasoningEngines/(\d+)", agent_runtime_id)
    if match:
        if not project_id:
            project_id = match.group(1)
        location = match.group(2)
        agent_engine_id = match.group(3)
    else:
        agent_engine_id = agent_runtime_id.split("/")[-1]

logger.info(f"Initialized with Project: {project_id}, Location: {location}, Agent Engine ID: {agent_engine_id}")

import vertexai
from vertexai.preview.reasoning_engines import ReasoningEngine
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.cloud import aiplatform_v1beta1 as aip_types
from vertexai.reasoning_engines import _utils

# Initialize Vertex AI
if project_id and location:
    vertexai.init(project=project_id, location=location)

app = FastAPI(title="Clio — Museum Companion Dashboard")

# ── Request Models ────────────────────────────────────────────────────────────

class BeaconEvent(BaseModel):
    user_profile: str
    beacon_id: str


class ResumeEvent(BaseModel):
    session_id: str
    interrupt_id: str
    answer: str


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/beacon")
async def trigger_beacon(event: BeaconEvent):
    """Creates a new session and sends a beacon event to the deployed agent."""
    if not project_id or not agent_runtime_id:
        raise HTTPException(status_code=500, detail="Agent Runtime not configured.")

    try:
        engine = ReasoningEngine(agent_runtime_id)

        message_text = (
            f"Visitor at beacon_id: {event.beacon_id}. "
            f"Visitor profile: {event.user_profile}"
        )

        message = {
            "role": "user",
            "parts": [{"text": message_text}]
        }

        logger.info(f"Sending beacon event: {event.beacon_id}")

        def invoke_query():
            return engine.execution_api_client.stream_query_reasoning_engine(
                request=aip_types.types.StreamQueryReasoningEngineRequest(
                    name=engine.resource_name,
                    input={
                        "user_id": "museum-visitor",
                        "message": message,
                    },
                    class_method="stream_query",
                )
            )

        response = await asyncio.to_thread(invoke_query)

        # Parse response events
        events = []
        for chunk in response:
            for parsed in _utils.yield_parsed_json(chunk):
                if parsed:
                    events.append(parsed)

        # Extract text content and session_id
        responses = []
        session_id = None
        interrupt = None

        for evt in events:
            # Grab session_id
            if "session_id" in evt:
                session_id = evt["session_id"]

            # Check for content text
            content = evt.get("content")
            if content and "parts" in content:
                text = "".join(p.get("text", "") for p in content["parts"] if p.get("text"))
                if text:
                    try:
                        import json
                        parsed_data = json.loads(text)
                        if isinstance(parsed_data, dict):
                            if "script" in parsed_data:
                                responses.append(parsed_data["script"])
                        else:
                            responses.append(text)
                    except Exception:
                        responses.append(text)

            # Check for HITL interrupt
            for fc in evt.get("content", {}).get("parts", []) if evt.get("content") else []:
                if "function_call" in fc and fc["function_call"].get("name") == "adk_request_input":
                    args = fc["function_call"].get("args", {})
                    interrupt = {
                        "interrupt_id": fc["function_call"].get("id", ""),
                        "message": args.get("message", "What would you like to do next?")
                    }

        if not session_id and agent_engine_id:
            try:
                session_service = VertexAiSessionService(
                    project=project_id,
                    location=location,
                    agent_engine_id=agent_engine_id
                )
                list_resp = await session_service.list_sessions(app_name="app", user_id="museum-visitor")
                if list_resp.sessions:
                    session_id = list_resp.sessions[0].id
                    logger.info(f"Retrieved session_id from list_sessions: {session_id}")
            except Exception as se:
                logger.error(f"Error listing sessions: {se}")

        return {
            "session_id": session_id,
            "responses": responses,
            "interrupt": interrupt
        }

    except Exception as e:
        logger.error(f"Error triggering beacon: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/resume")
async def resume_agent(event: ResumeEvent):
    """Resumes the paused agent session with the visitor's answer."""
    if not project_id or not agent_runtime_id:
        raise HTTPException(status_code=500, detail="Agent Runtime not configured.")

    try:
        engine = ReasoningEngine(agent_runtime_id)

        resume_message = {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "id": event.interrupt_id,
                        "name": "adk_request_input",
                        "response": {"text": event.answer}
                    }
                }
            ]
        }

        logger.info(f"Resuming session {event.session_id}")

        def invoke_resume():
            return engine.execution_api_client.stream_query_reasoning_engine(
                request=aip_types.types.StreamQueryReasoningEngineRequest(
                    name=engine.resource_name,
                    input={
                        "user_id": "museum-visitor",
                        "session_id": event.session_id,
                        "message": resume_message,
                    },
                    class_method="stream_query",
                )
            )

        response = await asyncio.to_thread(invoke_resume)

        events = []
        for chunk in response:
            for parsed in _utils.yield_parsed_json(chunk):
                if parsed:
                    events.append(parsed)

        responses = []
        for evt in events:
            content = evt.get("content")
            if content and "parts" in content:
                text = "".join(p.get("text", "") for p in content["parts"] if p.get("text"))
                if text:
                    try:
                        import json
                        parsed_data = json.loads(text)
                        if isinstance(parsed_data, dict):
                            if "script" in parsed_data:
                                responses.append(parsed_data["script"])
                        else:
                            responses.append(text)
                    except Exception:
                        responses.append(text)

        return {
            "responses": responses,
            "interrupt": None
        }

    except Exception as e:
        logger.error(f"Error resuming session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── HTML Dashboard ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the Museum Companion Simulator Dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clio • Accessible Art Guide</title>
    <meta name="description" content="Clio is an AI-powered museum companion — named after the Greek muse of history — that provides personalized audio descriptions of artwork for visually impaired visitors and art students.">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-deep: #060a14;
            --bg-surface: rgba(16, 22, 40, 0.78);
            --bg-surface-raised: rgba(22, 30, 52, 0.85);
            --glass-border: rgba(255, 255, 255, 0.06);
            --glass-border-hover: rgba(255, 255, 255, 0.14);
            --text-primary: #eaeff6;
            --text-secondary: #7b86a0;
            --text-muted: #505a72;
            --accent-indigo: #6366f1;
            --accent-violet: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-cyan: #22d3ee;
            --accent-amber: #f59e0b;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background orbs */
        body::before, body::after {
            content: '';
            position: fixed;
            border-radius: 50%;
            filter: blur(140px);
            opacity: 0.35;
            z-index: 0;
            animation: float 25s ease-in-out infinite;
        }
        body::before {
            width: 650px; height: 650px;
            background: radial-gradient(circle, var(--accent-indigo) 0%, transparent 70%);
            top: -220px; left: -120px;
        }
        body::after {
            width: 550px; height: 550px;
            background: radial-gradient(circle, var(--accent-pink) 0%, transparent 70%);
            bottom: -180px; right: -120px;
            animation-delay: -12s;
        }

        /* Third orb for depth */
        .orb-extra {
            position: fixed;
            width: 400px; height: 400px;
            border-radius: 50%;
            filter: blur(130px);
            opacity: 0.2;
            background: radial-gradient(circle, var(--accent-cyan) 0%, transparent 70%);
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            z-index: 0;
            animation: float 30s ease-in-out infinite reverse;
        }

        @keyframes float {
            0%, 100% { transform: translate(0, 0) scale(1); }
            33% { transform: translate(30px, -30px) scale(1.05); }
            66% { transform: translate(-20px, 20px) scale(0.95); }
        }

        .app-wrapper {
            position: relative;
            z-index: 1;
            max-width: 960px;
            margin: 0 auto;
            padding: 48px 24px;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 44px;
        }
        .header-badges {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 18px;
            flex-wrap: wrap;
        }
        .header-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 100px;
            font-size: 0.75rem;
            font-weight: 500;
            color: var(--accent-cyan);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .header-badge.track {
            background: rgba(16, 185, 129, 0.12);
            border-color: rgba(16, 185, 129, 0.3);
            color: var(--accent-emerald);
        }
        h1 {
            font-size: 3rem;
            font-weight: 700;
            line-height: 1.1;
            background: linear-gradient(135deg, #f0f4ff 0%, var(--accent-indigo) 40%, var(--accent-violet) 65%, var(--accent-pink) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 6px;
        }
        .header-subtitle {
            font-size: 1.05rem;
            color: var(--text-secondary);
            margin-top: 8px;
            font-weight: 300;
            max-width: 560px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.5;
        }

        /* Cards */
        .card {
            background: var(--bg-surface);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--glass-border);
            border-radius: 22px;
            padding: 28px;
            margin-bottom: 22px;
            transition: border-color 0.35s ease, box-shadow 0.35s ease;
        }
        .card:hover {
            border-color: var(--glass-border-hover);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        }

        .card-label {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 18px;
        }
        .card-label .step-num {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 30px;
            height: 30px;
            border-radius: 9px;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet));
            font-size: 0.8rem;
            font-weight: 700;
            color: white;
            flex-shrink: 0;
        }
        .card-label span:last-child {
            font-weight: 600;
            font-size: 1rem;
            color: var(--text-secondary);
        }

        /* Select */
        .custom-select {
            position: relative;
        }
        .custom-select select {
            width: 100%;
            padding: 14px 18px;
            padding-right: 42px;
            background: rgba(8, 12, 24, 0.65);
            border: 1px solid var(--glass-border);
            border-radius: 14px;
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
            font-weight: 400;
            outline: none;
            appearance: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .custom-select select:focus {
            border-color: var(--accent-indigo);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }
        .custom-select::after {
            content: '▾';
            position: absolute;
            right: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-secondary);
            pointer-events: none;
            font-size: 1rem;
        }

        .profile-voice-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        @media (max-width: 640px) {
            .profile-voice-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Section labels inside cards */
        .section-label {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }
        .section-label .section-icon {
            font-size: 1rem;
        }
        .section-label .section-title {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: var(--text-muted);
        }
        .section-label.paintings .section-title { color: var(--accent-amber); }
        .section-label.sculptures .section-title { color: var(--accent-emerald); }
        .artwork-section + .section-label { margin-top: 20px; }

        /* Artwork buttons */
        .artwork-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }
        .artwork-btn {
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            padding: 20px 12px 16px;
            background: rgba(8, 12, 24, 0.55);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            overflow: hidden;
        }
        .artwork-btn::before {
            content: '';
            position: absolute;
            inset: 0;
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        .artwork-btn:hover::before { opacity: 1; }
        .artwork-btn:hover {
            border-color: var(--glass-border-hover);
            transform: translateY(-3px);
            box-shadow: 0 14px 28px rgba(0, 0, 0, 0.3);
        }
        .artwork-btn:active { transform: translateY(-1px); }

        /* Painting-specific hover gradients */
        .artwork-btn.starry::before { background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(245, 158, 11, 0.08)); }
        .artwork-btn.mona::before { background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(245, 158, 11, 0.08)); }
        .artwork-btn.pearl::before { background: linear-gradient(135deg, rgba(34, 211, 238, 0.1), rgba(245, 158, 11, 0.08)); }
        .artwork-btn.lilies::before { background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(99, 102, 241, 0.08)); }
        .artwork-btn.kiss::before { background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(236, 72, 153, 0.08)); }
        .artwork-btn.wave::before { background: linear-gradient(135deg, rgba(34, 211, 238, 0.12), rgba(99, 102, 241, 0.08)); }
        /* Sculpture hover gradients */
        .artwork-btn.david::before { background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(139, 92, 246, 0.08)); }
        .artwork-btn.thinker::before { background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(16, 185, 129, 0.08)); }
        .artwork-btn.venus::before { background: linear-gradient(135deg, rgba(236, 72, 153, 0.1), rgba(16, 185, 129, 0.08)); }

        .artwork-btn .emoji {
            font-size: 2rem;
            position: relative;
            z-index: 1;
        }
        .artwork-btn .label {
            position: relative;
            z-index: 1;
            text-align: center;
            line-height: 1.25;
        }
        .artwork-btn .label small {
            color: var(--text-secondary);
            font-weight: 400;
        }
        .artwork-btn .type-badge {
            position: absolute;
            top: 8px;
            right: 8px;
            font-size: 0.6rem;
            padding: 2px 7px;
            border-radius: 6px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            z-index: 2;
        }
        .type-badge.painting {
            background: rgba(245, 158, 11, 0.15);
            color: var(--accent-amber);
            border: 1px solid rgba(245, 158, 11, 0.25);
        }
        .type-badge.sculpture {
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-emerald);
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        /* Response area */
        .response-card {
            position: relative;
            min-height: 180px;
        }
        .response-inner {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 16px;
            padding: 26px;
            min-height: 160px;
            font-size: 1.02rem;
            line-height: 1.75;
            color: #c8cdd8;
            position: relative;
        }
        .response-inner .placeholder-text {
            color: var(--text-secondary);
            font-style: italic;
        }

        /* Speaker icon */
        .speaker-btn {
            position: absolute;
            top: 16px;
            right: 16px;
            background: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 10px;
            width: 42px;
            height: 42px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            cursor: pointer;
            transition: all 0.3s;
            color: var(--text-secondary);
        }
        .speaker-btn:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: rgba(99, 102, 241, 0.4);
        }
        .speaker-btn.playing {
            color: var(--accent-cyan);
            border-color: var(--accent-cyan);
            background: rgba(34, 211, 238, 0.1);
            animation: pulse-glow 2s ease-in-out infinite;
        }
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 0 0 rgba(34, 211, 238, 0); }
            50% { box-shadow: 0 0 14px 5px rgba(34, 211, 238, 0.18); }
        }

        /* Loader */
        .loader-overlay {
            position: absolute;
            inset: 0;
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 14px;
            background: rgba(6, 10, 20, 0.75);
            border-radius: 16px;
            z-index: 10;
        }
        .loader-overlay.active { display: flex; }
        .loader-dots {
            display: flex;
            gap: 8px;
        }
        .loader-dots span {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent-indigo);
            animation: dot-bounce 1.4s ease-in-out infinite;
        }
        .loader-dots span:nth-child(2) { animation-delay: 0.16s; background: var(--accent-violet); }
        .loader-dots span:nth-child(3) { animation-delay: 0.32s; background: var(--accent-pink); }
        .loader-text {
            font-size: 0.82rem;
            color: var(--text-secondary);
            font-weight: 400;
        }
        @keyframes dot-bounce {
            0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
            40% { transform: scale(1.2); opacity: 1; }
        }

        /* HITL area */
        .hitl-area {
            display: none;
            margin-top: 20px;
            padding: 22px;
            background: rgba(99, 102, 241, 0.06);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 16px;
        }
        .hitl-area.visible { display: block; animation: fadeIn 0.4s ease; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .hitl-question {
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 14px;
            line-height: 1.5;
        }
        .hitl-row {
            display: flex;
            gap: 12px;
        }
        .hitl-row input {
            flex: 1;
            padding: 13px 16px;
            background: rgba(8, 12, 24, 0.65);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-primary);
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s;
        }
        .hitl-row input:focus {
            border-color: var(--accent-indigo);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
        }
        .reply-btn {
            padding: 13px 28px;
            background: linear-gradient(135deg, var(--accent-indigo), var(--accent-violet));
            border: none;
            border-radius: 12px;
            color: white;
            font-family: 'Outfit', sans-serif;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            white-space: nowrap;
        }
        .reply-btn:hover {
            box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
            transform: translateY(-1px);
        }

        /* Footer */
        .footer {
            text-align: center;
            margin-top: 36px;
            padding-top: 24px;
            border-top: 1px solid var(--glass-border);
            color: var(--text-muted);
            font-size: 0.78rem;
            line-height: 1.8;
        }
        .footer a {
            color: var(--accent-indigo);
            text-decoration: none;
            transition: color 0.2s;
        }
        .footer a:hover { color: var(--accent-violet); }
        .footer .tech-stack {
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        @media (max-width: 700px) {
            h1 { font-size: 2.2rem; }
            .artwork-grid { grid-template-columns: repeat(2, 1fr); }
            .app-wrapper { padding: 28px 16px; }
        }
        @media (max-width: 440px) {
            .artwork-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<div class="orb-extra"></div>

<div class="app-wrapper">
    <!-- Header -->
    <div class="header">
        <div class="header-badges">
            <div class="header-badge">🏛️ Google ADK + MCP</div>
            <div class="header-badge track">🌍 Agents for Good</div>
        </div>
        <h1>Clio</h1>
        <p class="header-subtitle">Named after the Greek muse of history — an AI-powered audio guide that adapts to every visitor with vivid spatial descriptions and deep art history.</p>
    </div>

    <!-- Step 1: Profile & Voice Selection -->
    <div class="card" id="card-profile">
        <div class="card-label">
            <div class="step-num">1</div>
            <span>Configure Visitor Profile & Voice Settings</span>
        </div>
        <div class="profile-voice-grid">
            <div>
                <label style="display: block; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">Sensory Profile</label>
                <div class="custom-select">
                    <select id="persona">
                        <option value="Visually impaired user. Needs highly literal, vivid visual descriptions focusing on colors, shapes, spatial layout, textures, and physical dimensions of the artwork before discussing history. Speak warmly and inclusively.">👁️ Visually Impaired — Vivid Visuals & Spatial Orientation</option>
                        <option value="Graduate student in Art History. Already knows what paintings look like. Needs deep dives into the artist's technique, historical context, socio-political influences, provenance, and conservation history. Speak academically but enthusiastically.">🎓 Art History Student — Technique, Context & Provenance</option>
                    </select>
                </div>
            </div>
            <div>
                <label style="display: block; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;">Text-to-Speech Voice Selection</label>
                <div class="custom-select">
                    <select id="voice-select">
                        <option value="">Loading available system voices...</option>
                    </select>
                </div>
            </div>
        </div>
    </div>

    <!-- Step 2: Gallery -->
    <div class="card" id="card-beacon">
        <div class="card-label">
            <div class="step-num">2</div>
            <span>Walk to Artwork (Beacon Trigger)</span>
        </div>

        <!-- Paintings Section -->
        <div class="section-label paintings">
            <span class="section-icon">🎨</span>
            <span class="section-title">Paintings</span>
        </div>
        <div class="artwork-grid artwork-section">
            <button class="artwork-btn starry" onclick="triggerBeacon('vangogh_starry_night')">
                <span class="type-badge painting">Painting</span>
                <span class="emoji">🌙</span>
                <span class="label">The Starry Night<br><small>Van Gogh · 1889</small></span>
            </button>
            <button class="artwork-btn mona" onclick="triggerBeacon('davinci_mona_lisa')">
                <span class="type-badge painting">Painting</span>
                <span class="emoji">🖼️</span>
                <span class="label">Mona Lisa<br><small>Da Vinci · 1503</small></span>
            </button>
            <button class="artwork-btn pearl" onclick="triggerBeacon('vermeer_pearl_earring')">
                <span class="type-badge painting">Painting</span>
                <span class="emoji">✨</span>
                <span class="label">Girl with a<br>Pearl Earring<br><small>Vermeer · 1665</small></span>
            </button>
            <button class="artwork-btn lilies" onclick="triggerBeacon('monet_water_lilies')">
                <span class="type-badge painting">Painting</span>
                <span class="emoji">🪷</span>
                <span class="label">Water Lilies<br><small>Monet · 1906</small></span>
            </button>
            <button class="artwork-btn kiss" onclick="triggerBeacon('klimt_the_kiss')">
                <span class="type-badge painting">Painting</span>
                <span class="emoji">💛</span>
                <span class="label">The Kiss<br><small>Klimt · 1908</small></span>
            </button>
            <button class="artwork-btn wave" onclick="triggerBeacon('hokusai_great_wave')">
                <span class="type-badge painting">Painting</span>
                <span class="emoji">🌊</span>
                <span class="label">The Great Wave<br><small>Hokusai · 1831</small></span>
            </button>
        </div>

        <!-- Sculptures Section -->
        <div class="section-label sculptures">
            <span class="section-icon">🗿</span>
            <span class="section-title">Sculptures</span>
        </div>
        <div class="artwork-grid artwork-section">
            <button class="artwork-btn david" onclick="triggerBeacon('michelangelo_david')">
                <span class="type-badge sculpture">Sculpture</span>
                <span class="emoji">🏛️</span>
                <span class="label">David<br><small>Michelangelo · 1504</small></span>
            </button>
            <button class="artwork-btn thinker" onclick="triggerBeacon('rodin_the_thinker')">
                <span class="type-badge sculpture">Sculpture</span>
                <span class="emoji">🤔</span>
                <span class="label">The Thinker<br><small>Rodin · 1904</small></span>
            </button>
            <button class="artwork-btn venus" onclick="triggerBeacon('venus_de_milo')">
                <span class="type-badge sculpture">Sculpture</span>
                <span class="emoji">🏺</span>
                <span class="label">Venus de Milo<br><small>c. 130 BC</small></span>
            </button>
        </div>
    </div>

    <!-- Step 3: Response -->
    <div class="card response-card" id="card-response">
        <div class="card-label">
            <div class="step-num">3</div>
            <span>Agent Response & Audio Narration</span>
        </div>
        <div class="response-inner" id="response-container">
            <button class="speaker-btn" id="speaker-icon" onclick="toggleSpeech()" title="Toggle speech">🔊</button>
            <div class="loader-overlay" id="loader">
                <div class="loader-dots"><span></span><span></span><span></span></div>
                <span class="loader-text">Agent is curating your experience…</span>
            </div>
            <div id="response-text"><span class="placeholder-text">Stand near an artwork to begin your personalized tour…</span></div>
        </div>

        <div class="hitl-area" id="hitl-area">
            <div class="hitl-question" id="hitl-question"></div>
            <div class="hitl-row">
                <input type="text" id="hitl-answer" placeholder="Type your response…">
                <button class="reply-btn" onclick="resumeAgent()">Reply</button>
            </div>
        </div>
    </div>

    <div class="footer">
        <div class="tech-stack">Clio — Built with Google ADK &bull; Vertex AI Agent Runtime &bull; MCP Server &bull; Cloud Run</div>
        <div>Kaggle AI Agents Capstone &bull; <a href="https://github.com" target="_blank">View on GitHub</a></div>
    </div>
</div>

<script>
    let currentSessionId = null;
    let currentInterruptId = null;

    // ── Speech Synthesis ──────────────────────────────────────────
    const synth = window.speechSynthesis;
    const voiceSelect = document.getElementById('voice-select');
    let voices = [];
    let isSpeaking = false;

    function populateVoiceList() {
        voices = synth.getVoices();
        if (!voiceSelect) return;
        
        voiceSelect.innerHTML = '';
        const englishVoices = voices.filter(v => v.lang.startsWith('en'));
        
        if (englishVoices.length === 0) {
            const option = document.createElement('option');
            option.textContent = 'No English voices detected';
            option.value = '';
            voiceSelect.appendChild(option);
            return;
        }

        const voiceNames = [
            'Microsoft',
            'Google',
            'Samantha',
            'Alex',
            'Siri',
            'Daniel',
            'Karen',
            'Moira',
            'Rishi',
            'Tessa',
            'Fiona',
            'Veena',
            'Serena',
            'Stephanie',
            'Victoria',
            'Hazel',
            'Susan',
            'Zira',
            'David'
        ];
        let defaultIndex = 0;
        let foundPreferred = false;

        englishVoices.forEach((voice, index) => {
            const option = document.createElement('option');
            option.textContent = `${voice.name} (${voice.lang})`;
            if (voice.default) option.textContent += ' [Default]';
            option.value = voice.name;
            voiceSelect.appendChild(option);

            if (!foundPreferred) {
                for (const name of voiceNames) {
                    if (voice.name.toLowerCase().includes(name.toLowerCase())) {
                        defaultIndex = index;
                        foundPreferred = true;
                        break;
                    }
                }
            }
        });

        voiceSelect.selectedIndex = defaultIndex;
    }
    
    populateVoiceList();
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = populateVoiceList;
    }

    function speak(text) {
        synth.cancel();
        if (!text) return;

        const utterance = new SpeechSynthesisUtterance(text);
        
        if (voiceSelect && voiceSelect.value) {
            const selectedVoice = voices.find(v => v.name === voiceSelect.value);
            if (selectedVoice) {
                utterance.voice = selectedVoice;
            }
        }

        utterance.pitch = 1.0;
        utterance.rate = 0.88;

        const icon = document.getElementById('speaker-icon');
        utterance.onstart = () => { icon.classList.add('playing'); isSpeaking = true; };
        utterance.onend = () => { icon.classList.remove('playing'); isSpeaking = false; };

        synth.speak(utterance);
    }

    function toggleSpeech() {
        if (isSpeaking) {
            synth.cancel();
            document.getElementById('speaker-icon').classList.remove('playing');
            isSpeaking = false;
        } else {
            const text = document.getElementById('response-text').innerText;
            if (text && !text.includes('Stand near')) speak(text);
        }
    }

    // ── Beacon Trigger ────────────────────────────────────────────
    async function triggerBeacon(beaconId) {
        const profile = document.getElementById('persona').value;
        const responseText = document.getElementById('response-text');
        const loader = document.getElementById('loader');
        const hitlArea = document.getElementById('hitl-area');

        // Reset
        hitlArea.classList.remove('visible');
        responseText.innerHTML = '';
        loader.classList.add('active');
        synth.cancel();
        document.getElementById('speaker-icon').classList.remove('playing');

        try {
            const res = await fetch('/api/beacon', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_profile: profile, beacon_id: beaconId })
            });
            const data = await res.json();

            if (res.ok) {
                currentSessionId = data.session_id;
                handleAgentResponse(data);
            } else {
                responseText.innerHTML = `<span style="color: #ef4444;">Error: ${data.detail || 'Unknown error'}</span>`;
            }
        } catch (err) {
            responseText.innerHTML = `<span style="color: #ef4444;">Connection error: ${err.message}</span>`;
        } finally {
            loader.classList.remove('active');
        }
    }

    // ── Resume (HITL) ─────────────────────────────────────────────
    async function resumeAgent() {
        const answer = document.getElementById('hitl-answer').value;
        if (!answer || !currentInterruptId || !currentSessionId) return;

        const loader = document.getElementById('loader');
        const hitlArea = document.getElementById('hitl-area');

        hitlArea.classList.remove('visible');
        loader.classList.add('active');
        synth.cancel();

        try {
            const res = await fetch('/api/resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    interrupt_id: currentInterruptId,
                    answer: answer
                })
            });
            const data = await res.json();
            if (res.ok) {
                handleAgentResponse(data);
            } else {
                document.getElementById('response-text').innerHTML +=
                    `<br><span style="color: #ef4444;">Error: ${data.detail}</span>`;
            }
        } catch (err) {
            document.getElementById('response-text').innerHTML +=
                `<br><span style="color: #ef4444;">Connection error: ${err.message}</span>`;
        } finally {
            loader.classList.remove('active');
            document.getElementById('hitl-answer').value = '';
        }
    }

    // ── Handle Agent Response ─────────────────────────────────────
    function handleAgentResponse(data) {
        const responseText = document.getElementById('response-text');

        if (data.responses && data.responses.length > 0) {
            const fullHtml = data.responses.join('<br><br>');
            const cleanText = data.responses.join(' ').replace(/<[^>]*>?/gm, '');
            responseText.innerHTML = fullHtml;
            speak(cleanText);
        }

        if (data.interrupt) {
            currentInterruptId = data.interrupt.interrupt_id;
            const hitlArea = document.getElementById('hitl-area');
            document.getElementById('hitl-question').innerText = data.interrupt.message;
            hitlArea.classList.add('visible');
        }
    }

    // Allow Enter key to submit HITL answer
    document.getElementById('hitl-answer').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') resumeAgent();
    });
</script>

</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
