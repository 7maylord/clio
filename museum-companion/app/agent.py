# ruff: noqa
import os
import json
import google.auth
from google.adk.workflow import Workflow, node, START
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.apps import App, ResumabilityConfig
from google.genai import types

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

credentials, project_id = google.auth.default()
if not project_id:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

if not project_id:
    raise ValueError(
        "GOOGLE_CLOUD_PROJECT must be set in your environment or .env file."
    )

os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.environ.get(
    "GOOGLE_GENAI_USE_VERTEXAI", "True"
)


# ── Artwork Catalog Tool (reads from catalog.json inside the app package) ──────
CATALOG_FILE = os.path.join(os.path.dirname(__file__), "catalog.json")

def _load_catalog():
    if not os.path.exists(CATALOG_FILE):
        return {"artworks": []}
    with open(CATALOG_FILE, "r") as f:
        return json.load(f)

CATALOG = _load_catalog()


def get_artwork_details(beacon_id: str) -> str:
    """Get detailed information about an artwork based on its physical location beacon ID.

    Args:
        beacon_id: The unique identifier broadcasted by the bluetooth beacon near
                   the artwork (e.g., 'vangogh_starry_night', 'davinci_mona_lisa',
                   'vermeer_pearl_earring').

    Returns:
        A JSON string containing the artwork's title, artist, year, medium, history,
        visual description, and technique. Returns an error if the beacon_id is not found.
    """
    for artwork in CATALOG.get("artworks", []):
        if artwork.get("beacon_id") == beacon_id:
            return json.dumps(artwork, indent=2)
    return json.dumps({"error": f"No artwork found for beacon_id '{beacon_id}'"})


# ── LLM Model ────────────────────────────────────────────────────────────────
model = Gemini(
    model="gemini-2.5-flash",
    retry_options=types.HttpRetryOptions(attempts=3),
)


# ── Unified Clio Guide Agent ─────────────────────────────────────────────────
# Merges the former Curator + Narrator into a single LLM pass.
# No structured output schema → allows streaming tokens immediately.
clio_guide = LlmAgent(
    name="clio_guide",
    model=model,
    instruction=(
        "You are Clio, an AI museum companion named after the Greek muse of history. "
        "A visitor has arrived at an artwork. Their message contains the beacon_id "
        "and their visitor profile.\n\n"
        "Follow these steps:\n"
        "1. Extract the beacon_id from the visitor's message.\n"
        "2. Use the get_artwork_details tool to fetch the artwork's metadata.\n"
        "3. Analyze the visitor's profile to decide what to emphasize:\n"
        "   - For 'visually-impaired' visitors: Focus on spatial orientation, colors, "
        "shapes, brushstroke texture, physical dimensions, and sensory-rich descriptions "
        "that paint a vivid mental picture.\n"
        "   - For 'art-history-student' visitors: Focus on historical context, "
        "socio-political influences, artistic technique, provenance, conservation "
        "history, and comparative analysis with the artist's other works.\n"
        "4. Write a warm, engaging narration script to be read aloud via Text-to-Speech.\n\n"
        "CRITICAL OUTPUT RULES:\n"
        "- Output ONLY the narration text. Nothing else.\n"
        "- Do NOT include markdown formatting, bold text, headers, or bullet points.\n"
        "- Do NOT wrap your response in JSON or any structured format.\n"
        "- Write pure conversational prose suitable for audio playback.\n"
        "- Start speaking directly to the visitor (e.g., 'Welcome! Let's explore...')."
    ),
    tools=[get_artwork_details],
    output_key="script",
)


# ── Human-in-the-Loop Node ───────────────────────────────────────────────────
@node(name="deliver_and_ask", rerun_on_resume=True)
async def deliver_and_ask(ctx: Context, node_input: dict):
    """Delivers the narrator script and asks what the visitor wants to do next."""
    interrupt_id = "ask_next"

    if not ctx.resume_inputs or interrupt_id not in ctx.resume_inputs:
        # First run: output the script and pause
        msg = "Would you like to explore another painting, or head to the sculpture garden next?"
        yield RequestInput(interrupt_id=interrupt_id, message=msg)
        return

    # Resumed: the visitor answered
    answer = ctx.resume_inputs[interrupt_id]
    if isinstance(answer, dict):
        answer_text = answer.get("text", str(answer))
    else:
        answer_text = str(answer)

    reply = f"Great choice! You said: '{answer_text}'. Let me guide you there."
    yield Event(
        output={"status": "completed", "visitor_reply": answer_text},
        content=types.Content(role="model", parts=[types.Part.from_text(text=reply)]),
    )


# ── Workflow ──────────────────────────────────────────────────────────────────
root_agent = Workflow(
    name="clio",
    edges=[
        (START, clio_guide),
        (clio_guide, deliver_and_ask),
    ],
)


# ── App ───────────────────────────────────────────────────────────────────────
app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
