import torch
from transformers import AutoTokenizer
import threading
import os

# Try to import vLLM, fall back to transformers if not available
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
    print("vLLM available, using optimized inference")
except ImportError:
    VLLM_AVAILABLE = False
    print("vLLM not available, falling back to transformers")
    from transformers import AutoModelForCausalLM

class OptimizedLlamaHandler:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", use_vllm=True):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_vllm = use_vllm and VLLM_AVAILABLE and torch.cuda.is_available()
        self.lock = threading.Lock()
        
        print(f"Initializing model handler with vLLM: {self.use_vllm}")
        self._load_model()
    
    def _load_model(self):
        print(f"Loading model: {self.model_name}")
        
        if self.use_vllm:
            self._load_vllm_model()
        else:
            self._load_transformers_model()
        
        print("Model loaded successfully")
    
    def _load_vllm_model(self):
        """Load model using vLLM for faster inference"""
        # Load tokenizer separately for vLLM
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # vLLM configuration
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=4096,  # Adjust based on your needs
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            tensor_parallel_size=1,  # Increase if you have multiple GPUs
            dtype="float16",  # Use FP16 for efficiency
            swap_space=2,  # 2GB swap space for larger batches
        )
    
    def _load_transformers_model(self):
        """Fallback to transformers if vLLM is not available"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model loading arguments
        model_kwargs = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
        
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
    
    def generate_response(self, prompt, max_length=512, temperature=0.7):
        """Generate response using either vLLM or transformers"""
        if self.use_vllm:
            return self._generate_vllm(prompt, max_length, temperature)
        else:
            return self._generate_transformers(prompt, max_length, temperature)
    
    def _generate_vllm(self, prompt, max_length=512, temperature=0.7):
        """Generate using vLLM (faster, no locking needed)"""
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_length,
            top_p=0.9,
            stop=None,  # Add stop tokens if needed
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text
        
        return response.strip()
    
    def _generate_transformers(self, prompt, max_length=512, temperature=0.7):
        """Generate using transformers (fallback)"""
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
                    use_cache=True,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
    
    def generate_batch(self, prompts, max_length=512, temperature=0.7):
        """Generate responses for multiple prompts (vLLM optimized)"""
        if self.use_vllm:
            try:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=max_length,
                    top_p=0.9,
                    stop=None,
                )
                
                # Process in smaller batches to avoid memory issues
                batch_size = 8  # Adjust based on your GPU memory
                all_responses = []
                
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i+batch_size]
                    outputs = self.llm.generate(batch_prompts, sampling_params)
                    batch_responses = [output.outputs[0].text.strip() for output in outputs]
                    all_responses.extend(batch_responses)
                
                return all_responses
                
            except Exception as e:
                print(f"vLLM batch generation failed: {e}")
                # Fallback to sequential generation
                return [self._generate_vllm(prompt, max_length, temperature) for prompt in prompts]
        else:
            # Fallback to sequential generation for transformers
            return [self._generate_transformers(prompt, max_length, temperature) for prompt in prompts]
# Global model instance
model_handler = None

def get_model_handler():
    global model_handler
    if model_handler is None:
        model_handler = OptimizedLlamaHandler(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            use_vllm=True  # Set to False to force transformers
        )
    return model_handler