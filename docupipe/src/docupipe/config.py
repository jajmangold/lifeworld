"""Central config — everything env-overridable so we can flip cloud<->local."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("DOCUPIPE_DATA", os.path.join(ROOT, "data"))
CACHE_DIR = os.path.join(DATA, "cache")
ASSETS_DIR = os.path.join(DATA, "assets")
RENDERS_DIR = os.path.join(DATA, "renders")
EPISODES_DIR = os.path.join(DATA, "episodes")
for _d in (CACHE_DIR, ASSETS_DIR, RENDERS_DIR, EPISODES_DIR):
    os.makedirs(_d, exist_ok=True)

# ── LLM (OpenAI-compatible). Default writer path is OpenCode Go DeepSeek V4 Flash.
def _opencode_go_key():
    k = os.environ.get("OPENCODE_GO_API_KEY", "")
    token_file = os.environ.get("DOCUPIPE_OPENCODE_GO_TOKEN_FILE", "")
    if not k and token_file:
        with open(token_file, encoding="utf-8") as source:
            k = source.read().strip()
    auth_file = os.path.expanduser(
        os.environ.get("DOCUPIPE_OPENCODE_AUTH_FILE", "~/.local/share/opencode/auth.json")
    )
    if not k and os.path.isfile(auth_file):
        import json
        with open(auth_file, encoding="utf-8") as source:
            auth = json.load(source)
        k = (auth.get("opencode-go") or {}).get("key", "")
    return k


def _tier_key(name):
    return os.environ.get(name, "") or _opencode_go_key()

# "bulk" tier (research/outline) and "writer" tier (script).
LLM = {
    "bulk":   {"base_url": os.environ.get("DOCUPIPE_BULK_URL",   "https://opencode.ai/zen/go/v1"),
               "model":    os.environ.get("DOCUPIPE_BULK_MODEL", "deepseek-v4-flash"),
               "key":      _tier_key("DOCUPIPE_BULK_KEY")},
    "writer": {"base_url": os.environ.get("DOCUPIPE_WRITER_URL",   "https://opencode.ai/zen/go/v1"),
               "model":    os.environ.get("DOCUPIPE_WRITER_MODEL", "deepseek-v4-flash"),
               "key":      _tier_key("DOCUPIPE_WRITER_KEY")},
}
# local example (once lm-stack up): DOCUPIPE_BULK_URL=http://localhost:8047/v1 BULK_MODEL=qwen3.5-9b KEY=x

# ── On-box services
SEARXNG_URL = os.environ.get("DOCUPIPE_SEARXNG_URL", "http://localhost:8088")  # grounded web search (JSON)
TTS_URL    = os.environ.get("DOCUPIPE_TTS_URL", "http://amd1:8064")           # qwen3-tts voxserver on amd1
ESRGAN_URL = os.environ.get("DOCUPIPE_ESRGAN_URL", "http://127.0.0.1:8197")   # ComfyUI upscale
ACESTEP_URL = os.environ.get("DOCUPIPE_ACESTEP_URL", "http://localhost:7861")
# Vision judge / chat: Qwen3.5-9B on amd0 (OpenAI /v1) — the ONE qwen9b for all vision-QA + chat.
VISION_URL = os.environ.get("DOCUPIPE_VISION_URL", "https://amd0.python-bull.ts.net/v1")
VISION_MODEL = os.environ.get("DOCUPIPE_VISION_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf")
# Z-Image Turbo (A1111 sdapi) for generative period illustrations (no-image beats).
ZIMAGE_URL = os.environ.get("DOCUPIPE_ZIMAGE_URL", "http://172.30.0.14:9000")
# FlashVSR premium superscale — resident server on rtx0 (warm pipeline, HTTP).
RTX0_SSH = os.environ.get("DOCUPIPE_RTX0_SSH", "josh@rtx0")
FLASHVSR_URL = os.environ.get("DOCUPIPE_FLASHVSR_URL", "http://rtx0:8800")  # async queue coordinator
PREMIUM_UPSCALE = os.environ.get("DOCUPIPE_PREMIUM_UPSCALE", "0") == "1"  # FlashVSR vs fast ESRGAN

# ── Render spec (docuseries craft defaults)
FPS = 30
W, H = 1920, 1080
NARRATION_WPM = 140          # documentary VO target
KENBURNS_ZOOM_PER_FRAME = 0.0015
KENBURNS_MAX_ZOOM = 1.5
STILL_MIN_SEC = 4.0
XFADE_SEC = 0.8
MUSIC_DUCK_DB = -12          # music under narration
LOUDNESS_LUFS = -14          # YouTube target
HTTP_UA = "docupipe/0.1 (documentary research; contact: local)"

# ── Original score (resident ACE-Step 1.5, --enable-api). Instrumental bed, ducked under VO.
ACESTEP_URL = os.environ.get("DOCUPIPE_ACESTEP_URL", "http://localhost:7861")
MUSIC_ENABLE = os.environ.get("DOCUPIPE_MUSIC", "1") == "1"
MUSIC_SECONDS = float(os.environ.get("DOCUPIPE_MUSIC_SECONDS", "150"))   # bed length (looped under VO)

# ── Writers' room (multi-stage script generation)
TARGET_MINUTES = float(os.environ.get("DOCUPIPE_TARGET_MINUTES", "11"))  # episode runtime target
MAX_ENTITIES = int(os.environ.get("DOCUPIPE_MAX_ENTITIES", "8"))         # deep-research dossiers (cap)
EDITOR_PASSES = int(os.environ.get("DOCUPIPE_EDITOR_PASSES", "1"))       # showrunner revise loops
WORDS_PER_MIN = 150                                                       # narration pacing

# ── Narrator voice (continuity with the news anchor — deep male 3000)
NARRATOR_REF = os.environ.get("DOCUPIPE_NARRATOR_REF", "/work/ref1.wav")  # path inside the amd1 qwen3-tts /work
# researcher-interview backdrop: a warm historian's study plate composited behind the studio-lit
# subject (replaces the newsroom set so the expert reads as documentary, not TV anchor)
INTERVIEW_BG = os.environ.get("DOCUPIPE_INTERVIEW_BG",
                              os.path.join(os.path.dirname(ROOT), "output", "study_pano.png"))
DISCLOSURE = ("This program uses AI-assisted narration and restoration of public-domain archival "
              "material. Enhanced images are interpretations, not original artifacts.")
