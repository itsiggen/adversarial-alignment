import torch
from typing import List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

from .model_registry import VictimModelConfig, get_victim_model


logger = logging.getLogger(__name__)


class VictimModel:
    """Loads and queries an instruction-tuned model as the jailbreak target."""
    
    def __init__(
        self,
        model_name: str = None,
        model_id: str = None,
        device: str = "cuda:0",
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize victim model.
        
        Args:
            model_name: Name from model registry (e.g., 'mistral-7b')
            model_id: HuggingFace model ID (alternative to model_name)
            device: Device to load model on
            torch_dtype: Torch dtype for model weights
            load_in_8bit: Whether to use 8-bit quantization
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
        """
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Get model configuration
        if model_id:
            self.model_id = model_id
            self.model_name = model_id
        else:
            config = get_victim_model(model_name)
            if config is None:
                # Not in registry — treat as a direct HuggingFace model ID
                logger.info(f"'{model_name}' not in registry, using as direct HF model ID")
                self.model_id = model_name
                self.model_name = model_name
            else:
                self.model_id = config.model_id
                self.model_name = config.name
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left"  # For batch generation
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        logger.info(f"Loading victim model {self.model_name} on {device}")
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = {"": device}
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs
        )
        
        self.model.eval()
        logger.info(f"Victim model {self.model_name} loaded successfully")
    
    def generate_response(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
    ) -> Union[str, List[str]]:
        """
        Generate response(s) from the victim model for given prompt(s).
        
        Args:
            prompt: Single prompt string or list of prompts
            max_new_tokens: Override default max_new_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            do_sample: Whether to use sampling
            
        Returns:
            Generated response(s) - string if single prompt, list if batch
        """
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        max_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        temp = temperature if temperature is not None else self.temperature
        p = top_p if top_p is not None else self.top_p
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp,
                top_p=p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the generated part (not the prompt)
        responses = []
        for i, output in enumerate(outputs):
            # Remove the input prompt tokens
            generated_tokens = output[inputs.input_ids[i].shape[0]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses if is_batch else responses[0]
    
    def __call__(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """Shorthand for generate_response."""
        return self.generate_response(prompt, **kwargs)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "name": self.model_name,
            "model_id": self.model_id,
            "device": self.device,
            "parameters": self.model.num_parameters(),
            "dtype": str(self.model.dtype),
        }
