import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Config:
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    LLM_MODELS = os.getenv("LLM_MODELS", "qwen2.5:latest,llama3.2:latest").split(",")

    BASE_DIR = Path(os.getenv("BASE_DIR", "."))

    TEST_FILE = os.getenv("TEST_FILE", "anatomy_test.csv")

    NUM_PROMPT_VARIANTS = int(os.getenv("NUM_PROMPT_VARIANTS", "10"))

    RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "results"))

    @classmethod
    def validate(cls):
        if not cls.BASE_DIR.exists():
            raise ValueError(f"Base directory {cls.BASE_DIR} does not exist")

        cls.RESULTS_DIR.mkdir(exist_ok=True)

        return True

