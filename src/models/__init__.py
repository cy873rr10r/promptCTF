"""Model Loading and Management for PromptCTF-Env"""

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    AutoTokenizer = None
    AutoModelForCausalLM = None

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model loading"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype = None,
        max_memory: Optional[Dict[int, int]] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        if dtype is None and torch is not None:
            dtype = torch.float16

        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_memory = max_memory
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit


class ModelLoader:
    """Loads and manages LLM models"""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
    
    def load_model(
        self,
        model_name: str,
        model_id: str = None,
        config: ModelConfig = None
    ) -> tuple:
        """
        Load a model and tokenizer.

        Args:
            model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-7B")
            model_id: Local cache ID for the model
            config: ModelConfig object

        Returns:
            Tuple of (model, tokenizer)
        """
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError(
                "Transformers and Torch are required for model loading. "
                "Install with: pip install torch transformers"
            )

        if model_id is None:
            model_id = model_name.replace("/", "_")

        # Return cached model if available
        if model_id in self._models:
            logger.info(f"Returning cached model: {model_id}")
            return self._models[model_id], self._tokenizers[model_id]

        config = config or ModelConfig(model_name)

        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                pad_token='eos_token'
            )
            logger.info(f"Loaded tokenizer for {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
        
        # Prepare quantization config if needed
        quantization_config = None
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=config.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=config.dtype,
                device_map=config.device,
                quantization_config=quantization_config,
                trust_remote_code=True,
                max_memory=config.max_memory,
            )
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Cache
        self._models[model_id] = model
        self._tokenizers[model_id] = tokenizer
        
        return model, tokenizer
    
    def get_model(self, model_id: str):
        """Get cached model by ID"""
        return self._models.get(model_id)
    
    def get_tokenizer(self, model_id: str):
        """Get cached tokenizer by ID"""
        return self._tokenizers.get(model_id)
    
    def unload_model(self, model_id: str):
        """Unload and free memory for a model"""
        if model_id in self._models:
            del self._models[model_id]
        if model_id in self._tokenizers:
            del self._tokenizers[model_id]
        if torch is not None:
            torch.cuda.empty_cache()
        logger.info(f"Unloaded model: {model_id}")


# Global model loader
MODEL_LOADER = ModelLoader()


def get_default_model_loader() -> ModelLoader:
    """Get the global model loader instance"""
    return MODEL_LOADER
