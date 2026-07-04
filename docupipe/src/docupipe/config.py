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

# ── LLM (OpenAI-compatible). Default cloud DeepSeek; flip to local Qwen3.5-9B when lm-stack is up.
def _deepseek_key():
    k = os.environ.get("DEEPSEEK_API_KEY", "")
    if not k and os.path.isfile("/srv/nvme-data/containers/.env"):
        for line in open("/srv/nvme-data/containers/.env"):
            if line.startswith("DEEPSEEK_API_KEY"):
                k = line.split("=", 1)[1].strip().strip('"').strip("'")
    return k

# "bulk" tier (research/outline) and "writer" tier (script). Hybrid: bulk=local later, writer=deepseek-pro.
LLM = {
    "bulk":   {"base_url": os.environ.get("DOCUPIPE_BULK_URL",   "https://api.deepseek.com/v1"),
               "model":    os.environ.get("DOCUPIPE_BULK_MODEL", "deepseek-v4-flash"),
               "key":      os.environ.get("DOCUPIPE_BULK_KEY",   _deepseek_key())},
    "writer": {"base_url": os.environ.get("DOCUPIPE_WRITER_URL",   "https://api.deepseek.com/v1"),
               "model":    os.environ.get("DOCUPIPE_WRITER_MODEL", "deepseek-v4-pro"),
               "key":      os.environ.get("DOCUPIPE_WRITER_KEY",   _deepseek_key())},
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
