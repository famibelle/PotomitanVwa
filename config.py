# Configuration du Projet POTOMITAN Whisper Fine-tuning

# Modèles
DEFAULT_BASE_MODEL = "misterkissi/whisper-small-haitian-creole"
OUTPUT_DIR = "./whisper-finetuned-potomitan"

# Datasets POTOMITAN
POTOMITAN_AUDIO_DATASET = "POTOMITAN/potomitan-gcf-transcription"
POTOMITAN_TRANSLATION_DATASET = "POTOMITAN/potomitan-gcf-fr-translation"

# Répertoires
DATA_DIR = "./data"
TEMPLATES_DIR = "./training_templates"

# Paramètres d'entraînement
TRAINING_CONFIG = {
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "max_steps": 2000,
    "learning_rate": 5e-6,
    "fp16": True,
    "evaluation_strategy": "steps",
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 10
}

# Audio
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 30  # secondes

# Hugging Face
HF_CACHE_DIR = "./.cache/huggingface"