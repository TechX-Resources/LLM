# LLM from Scratch Cohort

Welcome to the LLM from Scratch cohort! This repository contains resources, code, and materials for learning about Large Language Models from the ground up.

## 🎯 Overview

This cohort is designed to help you understand and implement Large Language Models from scratch. We'll cover everything from the fundamentals of building and fine-tuning your own LLM. You will each work on a use-case that you will do end-to-end with your teammates.

## 💻 Development Environment Setup

### System Requirements
- Python 3.11 or higher
- 16GB RAM minimum (32GB recommended)
- NVIDIA GPU with 8GB+ VRAM (recommended for local model training)
- 50GB+ free disk space

### Step-by-Step Setup Guide

1. **Install Python 3.11+**
   ```bash
   # Windows (using winget)
   winget install Python.Python.3.11

   # macOS (using Homebrew)
   brew install python@3.11

   # Linux (Ubuntu/Debian)
   sudo apt update
   sudo apt install python3.11 python3.11-venv
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv llm-env

   # Activate on Windows
   .\llm-env\Scripts\activate

   # Activate on macOS/Linux
   source llm-env/bin/activate
   ```

3. **Install Required Packages**
   ```bash
   # Install basic requirements
   pip install --upgrade pip
   pip install torch torchvision torchaudio
   pip install transformers datasets accelerate
   pip install sentencepiece protobuf
   pip install bitsandbytes  # for 4-bit quantization
   pip install scipy numpy pandas
   ```

4. **Install CUDA (for NVIDIA GPUs)**
   - Download and install CUDA Toolkit from NVIDIA website
   - Install cuDNN for better performance
   - Verify installation:
     ```bash
     nvidia-smi
     python -c "import torch; print(torch.cuda.is_available())"
     ```

## 🤖 LLM Integration Examples

### 1. Claude 3.5 Sonnet (via Anthropic)
```python
from anthropic import Anthropic

def setup_claude():
    client = Anthropic(api_key="your-api-key")
    return client

def get_claude_response(prompt):
    client = setup_claude()
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 2. Gemini 1.5 Pro (via Google AI Studio)
```python
import google.generativeai as genai

def setup_gemini():
    genai.configure(api_key="your-api-key")
    model = genai.GenerativeModel('gemini-1.5-pro')
    return model

def get_gemini_response(prompt):
    model = setup_gemini()
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 3. GPT-4 (via OpenAI)
```python
from openai import OpenAI

def setup_gpt4():
    client = OpenAI(api_key="your-api-key")
    return client

def get_gpt4_response(prompt):
    client = setup_gpt4()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 4. Llama 3 70B (Local Installation)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def setup_llama():
    model_name = "meta-llama/Llama-2-70b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True  # Enable 4-bit quantization
    )
    return model, tokenizer

def get_llama_response(prompt, max_length=100):
    model, tokenizer = setup_llama()
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### 5. Inflection-2.5 / Pi
```python
from inflection import InflectionClient

def setup_pi():
    client = InflectionClient(api_key="your-api-key")
    return client

def get_pi_response(prompt):
    client = setup_pi()
    try:
        response = client.chat.create(
            model="inflection-2.5",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None
```

## 💡 Best Practices & Tips

### API Client Initialization
```python
# Best Practice: Use environment variables for API keys
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Initialize clients with proper error handling
def initialize_api_clients():
    clients = {}
    try:
        # OpenAI
        clients['openai'] = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Anthropic
        clients['anthropic'] = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        # Google
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        clients['google'] = genai.GenerativeModel('gemini-1.5-pro')
        
        return clients
    except Exception as e:
        logger.error(f"Failed to initialize API clients: {e}")
        raise
```

### Memory-Efficient Model Loading
```python
# Best Practice: Use quantization and model offloading
def load_model_efficiently(model_name, device="cuda"):
    try:
        # 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Enable model offloading
        if device == "cuda":
            model.enable_model_cpu_offload()
            
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
```

### Error Handling and Logging
```python
# Best Practice: Implement comprehensive error handling
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Retry decorator for API calls
def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator
```

### Tips and Tricks

1. **API Rate Limiting**
   ```python
   from ratelimit import limits, sleep_and_retry
   
   # Rate limit: 50 calls per minute
   @sleep_and_retry
   @limits(calls=50, period=60)
   def rate_limited_api_call():
       # Your API call here
       pass
   ```

2. **Caching Responses**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_model_response(prompt):
       # Your model inference here
       pass
   ```

3. **Batch Processing**
   ```python
   def process_batch(prompts, batch_size=32):
       results = []
       for i in range(0, len(prompts), batch_size):
           batch = prompts[i:i + batch_size]
           # Process batch
           results.extend(process_single_batch(batch))
       return results
   ```

4. **Memory Management**
   ```python
   import gc
   
   def clear_memory():
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

### Best Practices Checklist

1. **API Management**
   - ✅ Use environment variables for API keys
   - ✅ Implement rate limiting
   - ✅ Use connection pooling
   - ✅ Implement retry mechanisms

2. **Model Loading**
   - ✅ Use quantization (4-bit or 8-bit)
   - ✅ Enable model offloading
   - ✅ Use mixed precision training
   - ✅ Implement gradient checkpointing

3. **Error Handling**
   - ✅ Implement comprehensive logging
   - ✅ Use try-except blocks
   - ✅ Add retry mechanisms
   - ✅ Implement fallback options

4. **Performance Optimization**
   - ✅ Use batch processing
   - ✅ Implement caching
   - ✅ Optimize memory usage
   - ✅ Use async operations where appropriate

5. **Security**
   - ✅ Never hardcode API keys
   - ✅ Implement input validation
   - ✅ Use secure connections
   - ✅ Implement proper error messages

### Common Pitfalls to Avoid

1. **Memory Issues**
   - ❌ Loading full precision models without quantization
   - ❌ Not clearing GPU memory between operations
   - ❌ Keeping unnecessary model copies in memory

2. **API Usage**
   - ❌ Not implementing rate limiting
   - ❌ Not handling API timeouts
   - ❌ Not implementing retry mechanisms

3. **Error Handling**
   - ❌ Catching all exceptions without specific handling
   - ❌ Not logging errors properly
   - ❌ Not implementing fallback options

4. **Performance**
   - ❌ Not using batch processing
   - ❌ Not implementing caching
   - ❌ Not optimizing memory usage

## 📚 Recommended Resources

### Books
- "Attention Is All You Need" (Original Transformer Paper)
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Natural Language Processing with Transformers" by Lewis Tunstall et al.

### Online Courses
- Stanford CS224N: Natural Language Processing with Deep Learning
- Fast.ai's Practical Deep Learning for Coders
- Hugging Face Course on Transformers

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT: Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

### Tools and Libraries
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [PyTorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Weights & Biases](https://wandb.ai/) for experiment tracking

## 🙏 Acknowledgments
Special thanks to the open-source community and all the researchers who have contributed to the field of Natural Language Processing and Large Language Models.
