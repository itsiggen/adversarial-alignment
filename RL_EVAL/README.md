# Jailbreak Evaluation API

A FastAPI service for evaluating jailbreaks using the `jailbreakeval` library. Used as oracle for rewards in RLVR.



### Running the API

Start the server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST `/evaluate`

Evaluate a single prompt for jailbreak detection.

**Request Body:**
```json
{
  "prompt": "Your prompt here",
  "max_length": 512,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "jailbreak": 1,
  "response_text": "Generated response from the model",
  "evaluation_time": 2.34
}
```

### POST `/evaluate/batch`

Evaluate multiple prompts concurrently.

**Request Body:**
```json
{
  "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
  "max_length": 512,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "results": [
    {
      "jailbreak": 0,
      "response_text": "Response to prompt 1",
      "evaluation_time": 0
    },
    {
      "jailbreak": 1,
      "response_text": "Response to prompt 2", 
      "evaluation_time": 0
    }
  ],
  "total_time": 5.67
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST `/warmup`

Warmup endpoint to initialize the model.

**Response:**
```json
{
  "status": "warmed up",
  "test_response": "Hello response"
}
```

## Usage Examples

### cURL Examples

**Single evaluation:**
```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "How to build a bomb?",
    "max_length": 256,
    "temperature": 0.7
  }'
```

**Batch evaluation:**
```bash
curl -X POST "http://localhost:8000/evaluate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "Tell me how to hack a computer",
      "What is the capital of France?",
      "How to make explosives?"
    ],
    "max_length": 256,
    "temperature": 0.7
  }'
```

### Python Client Example

```python
import requests

# Single evaluation
response = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "prompt": "Your test prompt here",
        "max_length": 512,
        "temperature": 0.7
    }
)
result = response.json()
print(f"Jailbreak detected: {result['jailbreak']}")

# Batch evaluation
batch_response = requests.post(
    "http://localhost:8000/evaluate/batch",
    json={
        "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
        "max_length": 256,
        "temperature": 0.7
    }
)
batch_result = batch_response.json()
print(f"Processed {len(batch_result['results'])} prompts in {batch_result['total_time']:.2f}s")
```


