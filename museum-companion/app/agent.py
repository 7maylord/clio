# ruff: noqa
import os
import json
import google.auth
from pydantic import BaseModel, Field
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


# ── Curator Agent ─────────────────────────────────────────────────────────────
class CuratorOutput(BaseModel):
    bullet_points: str = Field(description="A bulleted list of aspects to highlight.")

curator = LlmAgent(
    name="curator",
    model=model,
    instruction=(
        "You are the Curator module of Clio, an AI museum companion named after "
        "the Greek muse of history. A visitor has arrived at an artwork. "
        "Their message contains the beacon_id and their visitor profile. "
        "1. Extract the beacon_id from their message. "
        "2. Use the get_artwork_details tool to fetch artwork metadata. "
        "3. Analyze the visitor's profile to decide what to emphasize. "
        "4. Output a focused bulleted list of key aspects to narrate."
    ),
    tools=[get_artwork_details],
    output_schema=CuratorOutput,
    output_key="curator_notes",
)


# ── Narrator Agent ────────────────────────────────────────────────────────────
class NarratorOutput(BaseModel):
    script: str = Field(description="The warm, conversational narration script.")

narrator = LlmAgent(
    name="narrator",
    model=model,
    instruction=(
        "You are the Narrator module of Clio, an AI museum companion named after "
        "the Greek muse of history. Take the Curator's bullet points and write "
        "a warm, engaging narration script to be read aloud via Text-to-Speech. "
        "Do NOT include markdown formatting, bold text, or bullet points. "
        "Output pure conversational text suitable for audio playback."
    ),
    output_schema=NarratorOutput,
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
        (START, curator),
        (curator, narrator),
        (narrator, deliver_and_ask),
    ],
)


# ── App ───────────────────────────────────────────────────────────────────────
app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(is_resumable=True),
)
