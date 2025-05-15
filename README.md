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

## ☁️ Cloud Development with Google Cloud Platform

### Getting Started with GCP (Optional)
# Note: You can also use any other cloud provider of your choice. Here is a list of cloud providers:
https://github.com/cloudcommunity/Cloud-Free-Tier-Comparison

1. **Sign Up and Free Credits**
   - Create a Google Cloud account
   - Get $300 free credits (valid for 90 days)
   - Enable billing (required even for free tier)

2. **Install Google Cloud CLI**
   ```bash
   # Windows (using winget)
   winget install Google.CloudSDK

   # macOS (using Homebrew)
   brew install google-cloud-sdk

   # Linux
   echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
   sudo apt-get install apt-transport-https ca-certificates gnupg
   sudo apt-get update && sudo apt-get install google-cloud-sdk
   ```

3. **Initialize and Configure GCP**
   ```bash
   # Login to GCP
   gcloud auth login

   # Set your project
   gcloud config set project YOUR_PROJECT_ID

   # Enable required APIs
   gcloud services enable compute.googleapis.com
   gcloud services enable aiplatform.googleapis.com
   ```

### Setting Up Cloud Environment

1. **Create a Virtual Machine**
   ```bash
   # Create a VM with GPU
   gcloud compute instances create llm-dev \
     --machine-type=n1-standard-4 \
     --zone=us-central1-a \
     --accelerator="type=nvidia-tesla-t4,count=1" \
     --maintenance-policy=TERMINATE \
     --image-family=debian-11-gpu \
     --image-project=debian-cloud \
     --boot-disk-size=100GB
   ```

2. **Connect to VM**
   ```bash
   # SSH into the VM
   gcloud compute ssh llm-dev --zone=us-central1-a
   ```

### Cloud Development Setup

1. **Install Dependencies on VM**
   ```bash
   # Update system
   sudo apt-get update
   sudo apt-get upgrade -y

   # Install Python and dependencies
   sudo apt-get install python3.11 python3.11-venv
   python3.11 -m venv llm-env
   source llm-env/bin/activate

   # Install CUDA and cuDNN
   sudo apt-get install nvidia-cuda-toolkit
   ```

2. **Configure Python Environment**
   ```bash
   # Install required packages
   pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install transformers datasets accelerate
   pip install google-cloud-aiplatform
   ```

### Using Google Cloud AI Platform

1. **Initialize Vertex AI**
   ```python
   from google.cloud import aiplatform

   def setup_vertex_ai():
       # Initialize Vertex AI
       aiplatform.init(
           project='your-project-id',
           location='us-central1',
           experiment='llm-experiment'
       )
       
       # Create or get endpoint
       endpoint = aiplatform.Endpoint(
           endpoint_name='projects/your-project-id/locations/us-central1/endpoints/your-endpoint-id'
       )
       return endpoint
   ```

2. **Deploy Model to Vertex AI**
   ```python
   def deploy_model_to_vertex(model_path):
       # Create model
       model = aiplatform.Model.upload(
           display_name='llm-model',
           artifact_uri=model_path,
           serving_container_image_uri='us-docker.pkg.dev/cloud-aiplatform/prediction/pytorch-gpu.1-10:latest'
       )
       
       # Deploy model
       endpoint = model.deploy(
           machine_type='n1-standard-4',
           accelerator_type='NVIDIA_TESLA_T4',
           accelerator_count=1
       )
       return endpoint
   ```

### Cost Optimization Tips

1. **Use Preemptible VMs**
   ```bash
   # Create preemptible VM (up to 80% cheaper)
   gcloud compute instances create llm-dev \
     --preemptible \
     --machine-type=n1-standard-4 \
     --zone=us-central1-a
   ```

2. **Auto-shutdown Script**
   ```bash
   # Create shutdown script
   echo '#!/bin/bash
   sudo shutdown -h now' > shutdown.sh
   chmod +x shutdown.sh

   # Add to VM creation
   gcloud compute instances create llm-dev \
     --metadata-from-file=shutdown-script=shutdown.sh
   ```

3. **Cost Monitoring**
   ```bash
   # Set budget alerts
   gcloud billing budgets create \
     --billing-account=YOUR_BILLING_ACCOUNT \
     --display-name="LLM Development Budget" \
     --budget-amount=100USD \
     --threshold-rule=percent=0.5 \
     --threshold-rule=percent=0.9
   ```

### Best Practices for Cloud Development

1. **Resource Management**
   - Use appropriate machine types
   - Implement auto-shutdown
   - Monitor resource usage
   - Clean up unused resources

2. **Data Management**
   - Use Cloud Storage for datasets
   - Implement proper backup strategies
   - Use version control for code
   - Store models in Cloud Storage

3. **Security**
   - Use service accounts
   - Implement IAM roles
   - Secure API keys
   - Enable audit logging

4. **Performance**
   - Use GPU instances when needed
   - Implement caching
   - Use batch processing
   - Optimize model size

### Cost Comparison (Approximate)

| Resource | Local Cost | GCP Cost (Free Tier) | GCP Cost (Paid) |
|----------|------------|---------------------|-----------------|
| GPU (T4) | Hardware   | $300 credits        | $0.35/hour      |
| Storage  | Free       | 5GB free            | $0.02/GB/month  |
| Network  | Free       | 1GB/day free        | $0.12/GB        |

Note: Prices may vary by region and time. Check Google Cloud pricing calculator for current rates.

## 🙏 Acknowledgments

We would like to express our sincere gratitude to the Open Source Community.
