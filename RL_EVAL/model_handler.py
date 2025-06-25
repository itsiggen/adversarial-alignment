import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import threading

QUANTIZATION_AVAILABLE = False

# try:
#     from transformers import BitsAndBytesConfig
#     QUANTIZATION_AVAILABLE = True
# except ImportError:
#     QUANTIZATION_AVAILABLE = False
#     print("Warning: bitsandbytes not available, loading model without quantization")

class OptimizedLlamaHandler:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        
        if QUANTIZATION_AVAILABLE and torch.cuda.is_available():
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            self.quantization_config = None
        
        self._load_model()
    
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading arguments
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True
        }
        
        # Add quantization only if available
        if self.quantization_config is not None:
            model_kwargs["quantization_config"] = self.quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
    
    def generate_response(self, prompt, max_length=512, temperature=0.7):
        with self.lock:
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()

# Global model instance
model_handler = None

def get_model_handler():
    global model_handler
    if model_handler is None:
        model_handler = OptimizedLlamaHandler(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    return model_handler