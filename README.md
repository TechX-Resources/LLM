# LLM from Scratch Cohort

Welcome to the LLM from Scratch cohort! This repository contains resources, code, and materials for learning about Large Language Models from the ground up.

## 🎯 Course Overview

This cohort is designed to help you understand and implement Large Language Models from scratch. We'll cover everything from the fundamentals of neural networks to building and fine-tuning your own LLM.

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

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any questions or concerns, please open an issue in this repository or contact the course instructors.

## 🙏 Acknowledgments

Special thanks to the open-source community and all the researchers who have contributed to the field of Natural Language Processing and Large Language Models.
