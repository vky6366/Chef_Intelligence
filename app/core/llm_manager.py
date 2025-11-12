import os
import platform
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
from app.config import Config
from app.utils.logger import setup_logger
from app.utils.prompt_builder import PromptBuilder
from openai import OpenAI

logger = setup_logger(__name__)

class TinyLlamaManager:
    """
    TinyLlama Manager for generating responses.
    Fully cross-platform: Mac MPS, Windows/Linux CUDA, CPU fallback.
    Optimized for Docker on Mac M2.
    """

    def __init__(self):
        """Initialize TinyLlama Manager"""
        self.model_name = Config.LLM_MODEL_NAME
        self.prompt_builder = PromptBuilder()

        # Cross-platform device selection
        if torch.backends.mps.is_available() and platform.system() == "Darwin":
            self.device = "mps"  # Mac M1/M2 GPU
        elif torch.cuda.is_available():
            self.device = "cuda"  # Windows/Linux GPU
        else:
            self.device = "cpu"  # Fallback CPU

        self.max_new_tokens = Config.LLM_MAX_NEW_TOKENS
        self.temperature = Config.LLM_TEMPERATURE
        self.top_p = Config.LLM_TOP_P
        self.top_k = Config.LLM_TOP_K
        self.do_sample = Config.LLM_DO_SAMPLE

        logger.info(f"Initializing TinyLlama Manager (model={self.model_name}, device={self.device}, OS={platform.system()})")

        # Create cache directory
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)

        # Load tokenizer and model
        self._load_model()

    def _load_model(self):
        """Load TinyLlama model and tokenizer with proper device and quantization"""
        try:
            logger.info("Loading TinyLlama tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=Config.MODEL_CACHE_DIR,
                trust_remote_code=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Loading TinyLlama model...")

            # Determine dtype
            torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

            # Load model with quantization only on CUDA
            if (Config.LOAD_IN_8BIT or Config.LOAD_IN_4BIT) and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=Config.LOAD_IN_8BIT,
                    load_in_4bit=Config.LOAD_IN_4BIT
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=Config.MODEL_CACHE_DIR,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=Config.MODEL_CACHE_DIR,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map="auto" if self.device != "cpu" else None
                )
                self.model.to(self.device)

            self.model.eval()
            logger.info(f"✓ TinyLlama model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading TinyLlama model: {str(e)}")
            raise

    def _format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Format prompts in TinyLlama chat format"""
        return f"<|system|> {system_prompt}</s> <|user|> {user_prompt}</s> <|assistant|> "

    def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """Generate response using TinyLlama"""
        try:
            system_prompt, user_prompt = self.prompt_builder.build_base_prompt(query, context_chunks)
            formatted_prompt = self._format_chat_prompt(system_prompt, user_prompt)

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response.split("<|assistant|>")[-1].strip() if "<|assistant|>" in full_response else full_response.strip()

            logger.info("TinyLlama response generated successfully")
            return answer

        except Exception as e:
            logger.error(f"Error generating TinyLlama response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def cleanup(self):
        """Clean up model from memory"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
        if torch.cuda.is_available() or self.device == "mps":
            torch.cuda.empty_cache()
        logger.info("TinyLlama model cleaned up from memory")

class OpenAIManager:
    """
    OpenAI Manager for GPT-based models (default: gpt-4o-mini).
    Fully cross-platform, uses API key from Config.
    """

    def __init__(self):
        """Initialize OpenAI Manager"""
        self.model_name = getattr(Config, "OPENAI_MODEL_NAME", "gpt-4o-mini")
        self.api_key = os.getenv("OPENAI_API_KEY", getattr(Config, "OPENAI_API_KEY", None))
        self.prompt_builder = PromptBuilder()

        if not self.api_key:
            raise ValueError("❌ OPENAI_API_KEY not found. Set it in environment variables or Config.")

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Load generation config
        self.max_tokens = getattr(Config, "OPENAI_MAX_TOKENS", 512)
        self.temperature = getattr(Config, "OPENAI_TEMPERATURE", 0.7)
        self.top_p = getattr(Config, "OPENAI_TOP_P", 1.0)

        logger.info(f"Initializing OpenAI Manager (model={self.model_name}, OS={platform.system()})")

    def _format_chat_prompt(self, system_prompt: str, user_prompt: str):
        """Format prompt for GPT chat models"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate_response(self, query: str, context_chunks: List[str]) -> str:
        """Generate response using GPT-4o-mini"""
        try:
            system_prompt, user_prompt = self.prompt_builder.build_base_prompt(query, context_chunks)
            messages = self._format_chat_prompt(system_prompt, user_prompt)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content.strip()
            logger.info("OpenAI GPT response generated successfully")
            return answer

        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            return f"Error generating response: {str(e)}"


# Singleton instance
_llm_manager_instance = None

def get_llm_manager():
    """Get or create LLM manager singleton"""
    global _llm_manager_instance
    if _llm_manager_instance is None:
        _llm_manager_instance = TinyLlamaManager()
    return _llm_manager_instance
