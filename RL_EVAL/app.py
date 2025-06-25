from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import time
from functools import lru_cache
import uvicorn

from model_handler import get_model_handler
from jailbreakeval import JailbreakEvaluator

app = FastAPI(title="Jailbreak Evaluation API")

# Cache only the evaluator (this doesn't change)
@lru_cache(maxsize=1)
def get_evaluator():
    return JailbreakEvaluator.from_preset("StringMatching-zou2023universal")

class EvaluationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class EvaluationResponse(BaseModel):
    jailbreak: int  # 1 or 0
    response_text: str
    evaluation_time: float

class BatchEvaluationRequest(BaseModel):
    prompts: List[str]
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7

class BatchEvaluationResponse(BaseModel):
    results: List[EvaluationResponse]
    total_time: float

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_jailbreak(request: EvaluationRequest):
    """Evaluate if a prompt results in a jailbreak"""
    start_time = time.time()
    
    try:
        # Get model handler and evaluator
        model = get_model_handler()
        evaluator = get_evaluator()
        
        # Generate response
        response_text = await asyncio.get_event_loop().run_in_executor(
            None, 
            model.generate_response, 
            request.prompt, 
            request.max_length, 
            request.temperature
        )
        
        # Evaluate jailbreak
        attempt = {
            "question": request.prompt,
            "answer": response_text
        }
        
        is_jailbreak = await asyncio.get_event_loop().run_in_executor(
            None, 
            evaluator, 
            attempt
        )
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResponse(
            jailbreak=1 if is_jailbreak else 0,
            response_text=response_text,
            evaluation_time=evaluation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/evaluate/batch", response_model=BatchEvaluationResponse)
async def evaluate_batch(request: BatchEvaluationRequest):
    """Batch evaluate multiple prompts efficiently"""
    start_time = time.time()
    model = get_model_handler()
    evaluator = get_evaluator()
    
    semaphore = asyncio.Semaphore(4)
    
    async def process_single_prompt(prompt):
        async with semaphore:
            response_text = await asyncio.get_event_loop().run_in_executor(
                None, model.generate_response, prompt, request.max_length, request.temperature
            )
            
            is_jailbreak = await asyncio.get_event_loop().run_in_executor(
                None, evaluator, {"question": prompt, "answer": response_text}
            )
            
            return EvaluationResponse(
                jailbreak=1 if is_jailbreak else 0,
                response_text=response_text,
                evaluation_time=0  # Individual timing not relevant in batch
            )
    
    # Process all prompts concurrently
    tasks = [process_single_prompt(prompt) for prompt in request.prompts]
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    
    return BatchEvaluationResponse(results=results, total_time=total_time)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_handler is not None}

@app.post("/warmup")
async def warmup():
    """Warmup endpoint to load model"""
    model = get_model_handler()
    test_response = model.generate_response("Hello", max_length=10)
    return {"status": "warmed up", "test_response": test_response}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Single worker for GPU memory efficiency
        loop="uvloop",
        log_level="info"
    )