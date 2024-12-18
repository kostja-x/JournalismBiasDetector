
# JournalismBiasDetector

**JournalismBiasDetector** project was the practical part of my master's thesis! This tool helps you analyze journalistic texts for various quality aspects, potential manipulation tactics, and underlying bias. It’s designed to be run locally with a Streamlit interface and a local LLM (via Ollama), ensuring you have full control over your data and models..

## What Does It Do?

- **Assess Journalistic Quality:**  
  Rates texts on factual accuracy, balance, proper source attribution, and neutrality.
  
- **Identify Manipulative Techniques:**  
  Detects emotionalization, suggestive phrasing, framing, and omissions that might shape reader perception.
  
- **Uncover Bias:**  
  Highlights potential political, ideological, economic, and cultural biases.

- **Suggest Improvements:**  
  Offers recommendations to enhance the neutrality and credibility of the content.

- **Manual Feedback & Learning:**  
  Integrate your own manual scores to adjust internal weighting, making future analyses more aligned with human judgment.

- **Performance Tracking & Similarity Search:**  
  Collects metrics (processing time, memory usage) and uses ChromaDB to find similar past analyses.

## Who’s It For?

This is great for researchers, journalists, media literacy educators, or anyone curious about subtle influences in news reporting. You have everything locally — no remote dependencies — so you remain in control.

## Tested Environment

Primarily tested on **macOS** (Apple M1 Max, 32 GB RAM), but instructions for Windows and Linux are also included. Performance depends on your hardware and model size.

---

## Repository Layout

```
JournalismBiasDetector/
├─ .env.example
├─ README.md          
├─ requirements.txt
├─ docs/
│  ├─ instructions_for_ollama.md
│  ├─ attachments/
│  │  └─ sample_article.pdf
│  └─ previous_attempts/       
│     ├─ attempt_1_notes.txt
│     ├─ attempt_2_notes.txt
│     └─ ...
├─ output/
│  ├─ analyses/
│  │  ├─ analysis_YYYYMMDD_HHMMSS.json
│  └─ metrics/
│     ├─ metrics_YYYYMMDD_HHMMSS.json
└─ src/
   ├─ main.py
   └─ (additional modules if needed)
```

**Note:**  
- `docs/attachments/` for supplementary materials (images, PDFs, etc.).  
- `docs/previous_attempts/` for logging past approaches and learnings.  
- `output/analyses/` and `output/metrics/` store generated analysis and metrics data.

---

## Requirements

- **Python:** 3.10 or newer.
- **Ollama:** For running local LLMs.
- **Streamlit:** For the user interface.
- Adequate system resources, especially if using large models.

---

## Setup

### On macOS

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kostja-x/JournalismBiasDetector.git
   cd JournalismBiasDetector
   ```

2. **Virtual environment & dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Environment variables:**
   ```bash
   cp .env.example .env
   ```
   Adjust `OLLAMA_BASE_URL` if needed.

### On Windows

- **Native Python:**
  ```powershell
  git clone https://github.com/kostja-x/JournalismBiasDetector.git
  cd JournalismBiasDetector
  python -m venv venv
  venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  copy .env.example .env
  ```
  Make sure Ollama is accessible (may require WSL or another setup).

- **WSL2 (Ubuntu):**
  ```bash
  sudo apt update && sudo apt install python3.10 python3.10-venv git -y
  git clone https://github.com/kostja-x/JournalismBiasDetector.git
  cd JournalismBiasDetector
  python3.10 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  cp .env.example .env
  ```

### On Linux

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv git -y
git clone https://github.com/kostja-x/JournalismBiasDetector.git
cd JournalismBiasDetector
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

---

## Ollama Setup

1. **Install Ollama:**  
   Refer to [Ollama’s documentation](https://docs.ollama.ai/getting-started).

2. **Run Ollama:**
   ```bash
   ollama serve
   ```

3. **Check connectivity:**
   ```bash
   curl http://localhost:11434/api/tags
   ```
   If you see JSON output, Ollama is working.

4. **Model Download:**
   ```bash
   ollama pull mistral
   ollama list
   ```
   Make sure the model matches what’s used in `main.py`.

---

## Running the Application

Start Streamlit:

```bash
streamlit run src/main.py
```

Open `http://localhost:8501` in your browser. On the sidebar, select your model and input method. Then click “Run Analysis.”

---

## Using the Tool

1. **Input Methods:**
   - **URL:** Paste a news article link.  
   - **Direct Text:** Paste the article text directly.  
   - **File Upload:** TXT, DOCX, or PDF files supported.

2. **Start Analysis:**
   Click “Run Analysis” and wait for results. The tool cleans text, queries the LLM, and provides scores and explanations.

3. **Manual Evaluation:**
   Adjust sliders to provide your own scores. This feedback updates internal weights for more accurate future analyses.

4. **Exporting Results:**
   Download JSON results containing analyses and metrics. Check `output/analyses/` and `output/metrics/` for saved files.

---

## Performance and Metrics

The application tracks:
- Processing time
- Memory usage
- Token counts
- CPU and memory load

Use these metrics to guide hardware upgrades or model choice.

---

## Similarity and History

ChromaDB is used to embed and store analyses for semantic similarity queries. You can compare new texts to past analyses, helping identify patterns or measure progress over time.

---

## Security and Credentials

- Store sensitive data in `.env` (already `.gitignore`d).
- Consider using a credential manager for Git pushes.
- Never commit tokens or passwords directly.

---

## Attachments & Previous Attempts

- **`docs/attachments/`:** Add images, sample articles, or benchmark reports.
- **`docs/previous_attempts/`:** Keep notes on past strategies and lessons learned to trace the project’s evolution.

---

## Maintenance & Future Plans

- **Modularize Code:** Split `main.py` into logical modules.
- **Add Tests:** Improve reliability with unit and integration tests.
- **Experiment with Models:** Try different local LLMs, tune prompts.
- **Deployment:** Consider Docker or a reverse proxy for team deployments.

---

## Troubleshooting

- **Git Authentication:**  
  Use a Personal Access Token (PAT) instead of a password if you get invalid credentials.

- **Ollama Not Responding:**  
  Check if `ollama serve` is running and reachable via `curl`.

- **Model Issues:**  
  Verify model names with `ollama list`. Pull or rename as needed.

- **File Parsing Problems:**  
  Ensure `python-docx` and `PyPDF2` are installed and the files are supported formats.

- **Slow Performance:**  
  Try a smaller model or upgrade your hardware.

---

## License

If you plan to share this project publicly, include a LICENSE file (e.g., MIT License) and ensure all dependencies comply with it.

---

**You’re all set!** Enjoy exploring how news coverage might subtly shape narratives, and feel free to adapt the tool to suit your needs.
