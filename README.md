# JournalismBiasDetector

Welcome to the **JournalismBiasDetector**, the practical centerpiece of my master’s thesis. Throughout my research, I became increasingly curious about how subtle shifts in language, tone, and emphasis might shape our understanding of news events. This tool is my attempt to bring transparency to journalistic reporting by examining texts for quality, potential manipulations, and underlying biases. By harnessing local Large Language Models (LLMs) via Ollama—together with a Streamlit interface—you can run everything locally, retaining full control of your data and ensuring privacy.

## What This Tool Does

- **Assess Journalistic Quality:**  
  I wanted a clear, granular way to understand factual accuracy, balance, source usage, and neutrality. The JournalismBiasDetector breaks these qualities down, giving you nuanced insights rather than a single blanket score.

- **Identify Manipulative Techniques:**  
  Emotionalization, suggestive phrasing, framing, and omissions might all influence how we perceive a story. My tool flags these patterns to show how subtle narratives can shape your understanding.

- **Uncover Bias:**  
  Political, ideological, economic, and cultural biases often lurk beneath the surface. By highlighting them, I hope to encourage critical engagement and awareness of hidden frames within reporting.

- **Suggest Improvements:**  
  This isn’t just about pointing out problems. I’ve included recommendations to help journalists and editors move towards greater neutrality and credibility, and to guide readers in asking critical questions.

- **Manual Feedback & Learning:**  
  Over time, your manual evaluations can influence the internal weighting. My aim is that with your input, the system grows more aligned with genuine human editorial judgment.

- **Performance Tracking & Similarity Search:**  
  To make sense of the process and continuously refine approaches, I’ve integrated performance metrics (processing time, memory usage) and ChromaDB-based similarity search, so you can compare new texts to past analyses and see broader patterns.

## Who’s It For?

I designed this tool for anyone interested in understanding subtle media influences:

- **Researchers & Academics:**  
  Perfect for media literacy studies, as it helps visualize and quantify subtle manipulations.

- **Journalists & Editors:**  
  A self-check to ensure your reporting stands up to scrutiny, and a guide for refining editorial approaches.

- **Media Literacy Educators & Fact-Checkers:**  
  Show students and readers how narratives are constructed. Encourage them to spot manipulations and biases in real-life reporting.

- **Curious Readers & Enthusiasts:**  
  If you’ve ever felt that a news story was nudging you towards a certain perspective, this tool can help confirm and articulate that intuition.

Running locally means you stay in full control—no calls to external APIs, no data leaving your environment.

## Tested Environment

- **Platform:** Mostly tested on macOS (Apple M1 Max, 32 GB RAM).
- **Hardware & Model Size:** Results vary based on your hardware and chosen model’s complexity.

(Windows and Linux setups are possible with similar steps. Adjust as needed.)

---

## Repository Layout

```
JournalismBiasDetector/
├─ docs/
│  ├─ EVAL_Data
│  ├─ attachments/
│  │  └─ 30 x Eval_raw.json
│  └─ previous_attempts/       
└─ src/
   ├─ main.py
├─ .gitignore
├─ LICENCE
├─ README.md
├─ requirements.txt
```

**Notes:**  
- `docs/EVAL_Data/` holds evaluation JSON files showing how initial analyses turned out.
- `docs/attachments/` contains images, PDF samples, and other supplementary materials.
- `docs/previous_attempts/` includes notes and logs that illustrate my earlier experimental stages—some attempts were tedious and time-consuming, underlining the complexity of semantic multi-labeling.
- `src/main.py` is where you’ll find the main Streamlit application code.
- `requirements.txt` ensures you can reproduce the environment exactly.

---

## Requirements

- **Python:** 3.10 or newer.
- **Ollama:** A must, since we rely on local LLMs. See Ollama’s documentation for installation details.
- **Streamlit:** For the web-based interface.
- **ChromaDB:** Helps with semantic searches of past analyses.
- Adequate system resources, especially if you plan to run larger models.

---

## Setup Instructions

### On macOS

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kostja-x/JournalismBiasDetector.git
   cd JournalismBiasDetector
   ```

2. **Create a virtual environment & install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Environment Variables (Optional):**
   ```bash
   cp .env.example .env
   ```
   Adjust `OLLAMA_BASE_URL` or other settings if needed.

4. **Ollama Setup:**
   Refer to [Ollama’s documentation](https://github.com/ollama/ollama).

   - Start the server:
     ```bash
     ollama serve
     ```
   - Check connectivity:
     ```bash
     curl http://localhost:11434/api/tags
     ```
   - Pull and list models:
     ```bash
     ollama pull mistral
     ollama list
     ```
   Make sure the model matches what’s expected in `src/main.py`.

5. **Run the Application:**
   ```bash
   streamlit run src/main.py
   ```
   Then open `http://localhost:8501` in your browser.  
   Choose your model and input method in the sidebar, then select "Run Analysis."

### On Windows / Linux

Follow similar steps (e.g., `python -m venv venv`, `venv\Scripts\activate` on Windows), ensuring Python 3.10+ is installed. Ollama might require WSL on Windows.

---

## Using the Tool

1. **Input the Text:**
   You can paste a URL, directly enter text, or upload a TXT/DOCX/PDF. The interface is meant to be straightforward.

2. **Run Analysis:**
   The system cleans your text, talks to the chosen local model, and returns clarity on quality, manipulation, bias, plus improvement suggestions.

3. **Manual Evaluation:**
   If you disagree with any assessments, use the sliders to provide your own scores. Over time, your feedback refines the internal weightings.

4. **Exporting Results:**
   Grab a JSON file with all analysis details. Check `docs/EVAL_Data/` for reference outputs and `output/` directories (if any) for metrics and analyses you run.

---

## Performance and Metrics

I integrated performance metrics like processing time, memory usage, and token counts. If you experiment with different models or hardware, you can track improvements or issues over time. ChromaDB helps you find similar past analyses, so you can compare how different texts or system states produce different results.

---

## Attachments & Previous Attempts

- `docs/attachments/` holds images, PDFs, and benchmark reports.
- `docs/previous_attempts/` catalogs the less successful experiments and all the lessons learned along the way. The `30 x Eval_raw.json` files, for example, show how extensive and painstaking those initial attempts were. They highlight why I ultimately chose local LLMs and refined strategies to achieve more stable, meaningful outcomes.

---

## Maintenance & Future Plans

- **Modularize Code:** Splitting `main.py` into multiple modules would make the code easier to maintain and expand.
- **Add Tests:** Unit and integration tests can enhance reliability and confidence in the tool’s accuracy.
- **Experiment with Models:** Trying other local LLMs (e.g., Mistral or LLaMA2 variants) or optimizing embeddings could yield even better insights.
- **Containerization:** Dockerizing the app or using a reverse proxy might make deployment smoother, especially for team environments.
- **Extended Documentation:** More examples, benchmarks, and step-by-step guides for customizing or extending the tool are on my to-do list.

---

## Troubleshooting

- **Git Authentication:**  
  If you run into credential problems, use a Personal Access Token (PAT) instead of a password.
  
- **Ollama Not Responding:**  
  Check if `ollama serve` is running and `curl` to confirm connectivity.

- **Model Name Issues:**  
  Make sure `ollama list` matches the model names you’re using in `main.py`.

- **File Parsing Problems:**  
  Ensure `python-docx` and `PyPDF2` are installed. Stick to TXT, DOCX, or PDF formats.

- **Slow Performance:**  
  Switch to smaller models or upgrade your hardware. The complexity of the model and the text length can have a big impact on runtime.

---

## License

If you plan to share this project publicly, include a LICENSE file (e.g., MIT License) and ensure compliance with all dependencies’ licenses.

---

I hope you enjoy exploring how subtle manipulations and biases can shape news narratives. This tool is my contribution to a more aware media landscape, stemming from my personal journey through my master’s thesis. I encourage you to adapt, extend, and refine it as you see fit!
```
