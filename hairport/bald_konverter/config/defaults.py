"""Default constants for the bald-konverter pipeline."""

# --------------------------------------------------------------------------- #
# Hugging Face Hub
# --------------------------------------------------------------------------- #
HF_REPO_ID = "deepmancer/bald_konverter"
BASE_MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

# Checkpoint filenames on the Hub
LORA_FILENAME_WO_SEG = "bald_konvertor_wo_seg_000003400.safetensors"
LORA_FILENAME_W_SEG = "bald_konvertor_w_seg_000004900.safetensors"

# --------------------------------------------------------------------------- #
# Image sizes
# --------------------------------------------------------------------------- #
WO_SEG_IMAGE_SIZE = 768   # 2-panel mode (side-by-side → 1536×768)
W_SEG_IMAGE_SIZE = 1024   # 4-panel mode (2×2 grid → 1024×1024)

# --------------------------------------------------------------------------- #
# FLUX generation defaults
# --------------------------------------------------------------------------- #
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_NUM_INFERENCE_STEPS = 35
DEFAULT_STRENGTH = 1.0
DEFAULT_SEED = 42

# --------------------------------------------------------------------------- #
# Prompts (originally in .txt files, embedded here for packaging)
# --------------------------------------------------------------------------- #
PROMPT_WO_SEG = (
    "Two-panel image: [LEFT] is the original source. Generate [RIGHT] as the "
    "same subject but bald, preserving identity, facial features, clothing, "
    "background, lighting, and camera style."
)

PROMPT_W_SEG = (
    "This is a four-panel image grid: [TOP-LEFT] shows the segmentation mask "
    "of the original image (red = hair, green = skin/body); [TOP-RIGHT] shows "
    "the aligned bald segmentation. [BOTTOM-LEFT] is the original image with "
    "hair, and [BOTTOM-RIGHT] is its bald version, preserving identity, facial "
    "features, background, lighting, and camera style."
)

# --------------------------------------------------------------------------- #
# Segmentation model IDs
# --------------------------------------------------------------------------- #
SAM3_MODEL_ID = "facebook/sam3"
BEN2_MODEL_ID = "PramaLLC/BEN2"
FACE_PARSER_MODEL_ID = "jonathandinu/face-parsing"

# --------------------------------------------------------------------------- #
# SAM mask extraction defaults
# --------------------------------------------------------------------------- #
SAM_CONFIDENCE_THRESHOLD = 0.35
SAM_DETECTION_THRESHOLD = 0.4
SAM_HAIR_CONFIDENCE_THRESHOLD = 0.25
