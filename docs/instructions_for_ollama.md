# Instructions for Ollama Setup

Ollama allows running LLMs locally.

## Installation

Visit: https://docs.ollama.ai/getting-started

## Running Ollama

Start Ollama:
```bash
ollama serve
```

Check models:
```bash
ollama list
```

Pull a model (e.g. mistral):
```bash
ollama pull mistral
```

Ensure that OLLAMA_BASE_URL is correctly set in .env if changed.
