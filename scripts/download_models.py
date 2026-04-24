"""Download and cache models"""

import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen2.5-7B"
CACHE_DIR = "./models"


def download_models():
    """Download required models"""
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    logger.info(f"Downloading model: {MODEL_NAME}")
    logger.info(f"Cache directory: {CACHE_DIR}")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        logger.info("✓ Tokenizer downloaded")
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )
        logger.info("✓ Model downloaded")
        
        logger.info(f"\n✓ All models cached in {CACHE_DIR}")
        
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        raise


if __name__ == "__main__":
    download_models()
