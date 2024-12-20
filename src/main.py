import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict, Union, Optional, Any
import psutil
import numpy as np
import json
import io
import docx
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
import os
from dotenv import load_dotenv
import time
from datetime import datetime
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_document(uploaded_file) -> str:
    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'txt':
                return str(uploaded_file.read(), 'utf-8')
            elif file_extension == 'docx':
                doc = docx.Document(io.BytesIO(uploaded_file.read()))
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            elif file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                return '\n'.join([page.extract_text() for page in pdf_reader.pages])
            else:
                st.error("Format wird nicht unterstützt. Unterstützte Formate: TXT, DOCX, PDF")
                return None
        except Exception as e:
            st.error(f"Fehler beim Laden der Datei: {str(e)}")
            return None

def clean_text(raw_text: str) -> str:
    lines = raw_text.split('\n')
    cleaned_lines = []
    ignore_keywords = [
        "impressum", "anzeige", "datenschutz", "privatsphäre", "nutzungsbedingungen",
        "jobs", "presse", "kontakt", "hilfe", "abo kündigen", "bild & bams"
    ]
    for line in lines:
        line_stripped = line.strip()
        if len(line_stripped) == 0:
            continue
        if any(word in line_stripped.lower() for word in ignore_keywords):
            continue
        cleaned_lines.append(line_stripped)
    cleaned_text = "\n".join(cleaned_lines)
    return cleaned_text

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def chat_completion(self, messages, model="mistral", temperature=0.0):
        url = f"{self.base_url}/api/generate"
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "temperature": temperature,
            "stream": False
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            response_json = response.json()
            return {"response": response_json.get("response", "")}
        except Exception as e:
            logger.error(f"API Fehler: {str(e)}. Payload: {payload}")
            return {"error": f"API Fehler: {str(e)}"}

class MetricsManager:
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}
        
    def record_metric(self, category: str, name: str, value: float):
        if category not in self.current_metrics:
            self.current_metrics[category] = {}
        self.current_metrics[category][name] = value

    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for category, metrics in self.current_metrics.items():
            vals = list(metrics.values())
            if vals:
                summary[category] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals))
                }
            else:
                summary[category] = {'mean':0,'std':0,'min':0,'max':0}
        return summary

    def save_metrics(self, filepath: str):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'metrics_history': self.metrics_history,
                    'current_metrics': self.current_metrics
                }, f, ensure_ascii=False, indent=4)
            logger.info(f"Metriken gespeichert in {filepath}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Metriken: {str(e)}")

class WeightsManager:
    def __init__(self):
        self.weights = self._initialize_weights()
        self.learning_rate = 0.1
        self.weight_history = []
        
    def _initialize_weights(self) -> Dict[str, Dict[str, float]]:
        weights = {
            'neutrality': {
                'journalistic_quality': 0.3,
                'manipulation': 0.3,
                'bias': 0.4
            },
            'quality': {
                'faktentreue': 0.3,
                'ausgewogenheit': 0.25,
                'quellenangaben': 0.25,
                'neutralitaet': 0.2
            },
            'manipulation': {
                'emotionalisierung': 0.25,
                'suggestion': 0.25,
                'framing': 0.25,
                'auslassung': 0.25
            },
            'bias': {
                'politischer_bias': 0.3,
                'ideologischer_bias': 0.3,
                'wirtschaftlicher_bias': 0.2,
                'kultureller_bias': 0.2
            }
        }
        return weights
        
    def update_weights(self, category: str, manual_scores: Dict[str, float], auto_scores: Dict[str, float]):
        if category not in self.weights:
            return
        for aspect, weight in self.weights[category].items():
            if aspect in manual_scores and aspect in auto_scores:
                diff = manual_scores[aspect] - auto_scores[aspect]
                new_weight = weight * (1 + self.learning_rate * diff)
                new_weight = max(0.1, min(0.5, new_weight))
                self.weights[category][aspect] = new_weight
        total = sum(self.weights[category].values())
        if total > 0:
            self.weights[category] = {k: v / total for k, v in self.weights[category].items()}

        self.weight_history.append({
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'weights': self.weights[category].copy()
        })
        if len(self.weight_history) > 100:
            self.weight_history = self.weight_history[-100:]

class EnhancedManipulationDetector:
    def __init__(self):
        self.client = OllamaClient()
        self.metrics_manager = MetricsManager()
        self.weights_manager = WeightsManager()
        self.chroma_client = chromadb.Client()
        self.collection = self._setup_collection()
        self.sentence_model = self._setup_sentence_transformer()
        self.categories = self._setup_categories()
        self.models = self._get_available_models()

        self.ASPECT_KEY_MAP = {
            "faktentreue": "faktentreue",
            "ausgewogenheit": "ausgewogenheit",
            "quellenangaben": "quellenangaben",
            "neutralität": "neutralitaet",
            "emotionalisierung": "emotionalisierung",
            "suggestion": "suggestion",
            "framing": "framing",
            "auslassung": "auslassung",
            "politischer bias": "politischer_bias",
            "ideologische färbung": "ideologischer_bias",
            "wirtschaftliche interessen": "wirtschaftlicher_bias",
            "kulturelle voreingenommenheit": "kultureller_bias"
        }

    def _setup_categories(self):
        return {
            "journalistic_quality": ["faktentreue", "ausgewogenheit", "quellenangaben", "neutralität"],
            "manipulation_analysis": ["emotionalisierung", "suggestion", "framing", "auslassung"],
            "bias_detection": ["politischer bias", "ideologische färbung", "wirtschaftliche interessen", "kulturelle voreingenommenheit"]
        }

    def _get_available_models(self):
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = response.json().get("models", [])
            valid_models = [
                model["name"] for model in models
                if any(name in model["name"].lower() for name in ["mistral", "llama"])
            ]
            if not valid_models:
                st.warning("Keine passenden lokalen Modelle gefunden. Verwende 'mistral' oder 'llama2' als Fallback.")
                return ["mistral", "llama2"]
            return valid_models
        except:
            return ["mistral", "llama2"]

    def _setup_collection(self):
        try:
            collection_name = "journalistic_patterns"
            try:
                collection = self.chroma_client.get_collection(name=collection_name)
                return collection
            except:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Muster journalistischer Manipulation"}
                )
                return collection
        except:
            return None

    def _setup_sentence_transformer(self):
        try:
            return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        except:
            return None

    def _strict_format_instructions(self):
        return """WICHTIG:
Antworte NUR mit den geforderten Aspekten im exakt vorgegebenen Format.
Keine zusätzliche Einleitung, kein extra Text, keine Kommentare.
Format pro Aspekt:
ASPEKT: [Name des Aspekts genau wie vorgegeben]
SCORE: [Zahl zwischen 0.0 und 1.0, z.B. 0.75]
BEGRÜNDUNG: [Kurze Begründung, ein bis zwei Sätze]

Keine weiteren Zeilen, keine weiteren Sätze außerhalb dieser Abschnitte.
"""

    def _prompt_quality(self, text: str) -> str:
        return self._strict_format_instructions() + """
Aspekte: Faktentreue, Ausgewogenheit, Quellenangaben, Neutralität

Text:
""" + text

    def _prompt_manipulation(self, text: str) -> str:
        return self._strict_format_instructions() + """
Aspekte: Emotionalisierung, Suggestion, Framing, Auslassung

Text:
""" + text

    def _prompt_bias(self, text: str) -> str:
        return self._strict_format_instructions() + """
Aspekte: Politischer Bias, Ideologische Färbung, Wirtschaftliche Interessen, Kulturelle Voreingenommenheit

Text:
""" + text

    def _prompt_neutrality(self, text: str) -> str:
        return self._strict_format_instructions() + """
Aspekte: Faktentreue, Ausgewogenheit, Quellenangaben, Neutralität

Text:
""" + text

    def _prompt_recommendations(self, text: str) -> str:
        # Empfehlungen können frei formuliert werden
        return """Basierend auf der Analyse, gib konkrete Handlungsempfehlungen:

- Wie Emotionalisierung vermeiden?
- Wie fehlende Quellen ergänzen?
- Wie Ausgewogenheit verbessern?
- Wie manipulative Techniken vermeiden?

Bitte kurze, klare Liste, ohne weitere Einleitungen oder Schlusssätze.
"""

    def analyze_article(self, text: str, model: str) -> Dict[str, Any]:
        try:
            start_time = time.time()
            text = clean_text(text)

            results = {}
            results['journalistic_quality'] = self._run_analysis(self._prompt_quality(text), model)
            results['manipulation_analysis'] = self._run_analysis(self._prompt_manipulation(text), model)
            results['bias_detection'] = self._run_analysis(self._prompt_bias(text), model)
            results['neutrality_analysis'] = self._run_analysis(self._prompt_neutrality(text), model)
            results['recommendations'] = self._run_recommendations(self._prompt_recommendations(text), model)

            results['quantitative_metrics'] = self._calculate_quantitative_metrics(results)
            results['confidence_scores'] = self._calculate_confidence_scores(results)
            processing_time = time.time() - start_time
            perf_metrics = self._calculate_performance_metrics(text, processing_time)
            results['performance_metrics'] = perf_metrics
            results['input_text'] = text

            self.metrics_manager.record_metric('performance', 'processing_time', processing_time)

            return results

        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

    def _run_analysis(self, prompt: str, model: str) -> Dict[str, str]:
        for _ in range(3):
            resp = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": "Du bist ein strenger Medienanalyse-Experte. Folge exakt dem Format."},
                    {"role": "user", "content": prompt}
                ],
                model=model,
                temperature=0.0
            )
            txt = resp.get("response", "")
            if self._is_valid_format(txt):
                return {"response": txt}
        return {"response": ""}

    def _run_recommendations(self, prompt: str, model: str) -> Dict[str,str]:
        resp = self.client.chat_completion(
            messages=[
                {"role": "system", "content": "Du bist ein strenger Medienanalyse-Experte. Folge exakt dem Format."},
                {"role": "user", "content": prompt}
            ],
            model=model,
            temperature=0.0
        )
        txt = resp.get("response","")
        return {"response": txt}

    def _is_valid_format(self, analysis_text: str) -> bool:
        return ("ASPEKT:" in analysis_text and "SCORE:" in analysis_text and "BEGRÜNDUNG:" in analysis_text)

    def _calculate_quantitative_metrics(self, results: Dict) -> Dict[str, float]:
        quality_scores = self._extract_scores_from_analysis(results.get('journalistic_quality', {}).get('response', ''))
        manipulation_scores = self._extract_scores_from_analysis(results.get('manipulation_analysis', {}).get('response', ''))
        bias_scores = self._extract_scores_from_analysis(results.get('bias_detection', {}).get('response', ''))
        neutrality_scores = self._extract_scores_from_analysis(results.get('neutrality_analysis', {}).get('response', ''))

        quality_mean = np.mean([s['score'] for s in quality_scores.values()]) if quality_scores else 0.0
        manipulation_mean = np.mean([s['score'] for s in manipulation_scores.values()]) if manipulation_scores else 0.0
        bias_mean = np.mean([s['score'] for s in bias_scores.values()]) if bias_scores else 0.0
        neutrality_mean = np.mean([s['score'] for s in neutrality_scores.values()]) if neutrality_scores else 0.0

        return {
            'quality_score': quality_mean,
            'manipulation_score': manipulation_mean,
            'bias_score': bias_mean,
            'neutrality_score': neutrality_mean
        }

    def _calculate_confidence_scores(self, results: Dict) -> Dict[str, float]:
        return {}

    def _calculate_performance_metrics(self, text: str, processing_time: float) -> Dict[str, float]:
        token_count = len(text.split())
        memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
        return {
            'processing_time_seconds': processing_time,
            'memory_usage_mb': memory_usage,
            'tokens_per_second': token_count / max(processing_time, 0.001),
            'text_length': len(text),
            'token_count': token_count,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }

    def export_analysis_results(self, results: Dict, manual_scores: Optional[Dict] = None,
                                export_format: str = 'json') -> Union[str, bytes]:
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '2.0',
                'models_used': self.models
            },
            'analysis_results': results,
            'quantitative_metrics': self._calculate_quantitative_metrics(results),
            'performance_metrics': results.get('performance_metrics', {}),
            'manual_evaluation': manual_scores if manual_scores else {},
            'confidence_scores': self._calculate_confidence_scores(results)
        }

        if export_format == 'json':
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        elif export_format == 'csv':
            flat_data = self._flatten_dict(export_data)
            df = pd.DataFrame([flat_data])
            return df.to_csv(index=False)
        else:
            return json.dumps({'error': "Unbekanntes Export-Format"})

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def store_analysis_history(self, results: Dict, manual_scores: Optional[Dict] = None):
        analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'has_manual_evaluation': bool(manual_scores),
            'quantitative_metrics': json.dumps(self._calculate_quantitative_metrics(results))
        }
        if self.sentence_model and results.get('input_text'):
            embedding = self.sentence_model.encode(results['input_text']).tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[json.dumps(results)],
                metadatas=[metadata],
                ids=[analysis_id]
            )

    def get_similar_analyses(self, text: str, n_results: int = 5) -> List[Dict]:
        if not self.sentence_model:
            return []
        query_embedding = self.sentence_model.encode(text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        docs = results.get('documents', [[]])[0]
        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]
        return [
            {
                'id': doc_id,
                'similarity': dist,
                'analysis': json.loads(doc)
            }
            for doc, doc_id, dist in zip(docs, ids, distances)
        ]

    def _extract_scores_from_analysis(self, analysis_text: str) -> Dict[str, Dict[str, Any]]:
        scores = {}
        if not analysis_text.strip():
            logger.warning("Leere Analyseantwort, keine Scores extrahierbar.")
            return scores

        current_aspect = None
        current_explanation = []
        score_pattern = re.compile(r'SCORE:\s*([0-9]*\.?[0-9]+)')

        for line in analysis_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('aspekt:'):
                if current_aspect and current_explanation:
                    scores[current_aspect]['explanation'] = '\n'.join(current_explanation)
                aspect_name = line.replace('ASPEKT:', '').strip().lower()
                scores[aspect_name] = {'score': 0.0, 'explanation': ''}
                current_aspect = aspect_name
                current_explanation = []
            elif line.lower().startswith('score:') and current_aspect:
                match = score_pattern.search(line)
                if match:
                    try:
                        sc = float(match.group(1))
                        scores[current_aspect]['score'] = min(max(sc, 0.0), 1.0)
                    except:
                        pass
            elif line.lower().startswith('begründung:'):
                explanation_start = line.replace('BEGRÜNDUNG:', '').strip()
                current_explanation = [explanation_start]
            elif current_aspect and line:
                current_explanation.append(line)

        if current_aspect and current_explanation:
            scores[current_aspect]['explanation'] = '\n'.join(current_explanation)

        return scores

    def update_from_manual_feedback(self, manual_scores: Dict[str, Dict[str, float]], auto_results: Dict[str, Any]):
        if 'manual_corrections' not in st.session_state:
            st.session_state.manual_corrections = []
        if 'model_adjustments' not in st.session_state:
            st.session_state.model_adjustments = []

        auto_all_scores = {}
        for cat_key in ['journalistic_quality', 'manipulation_analysis', 'bias_detection']:
            cat_scores = self._extract_scores_from_analysis(auto_results.get(cat_key, {}).get('response', ''))
            for akey, ainfo in cat_scores.items():
                norm_key = self.ASPECT_KEY_MAP.get(akey, akey)
                auto_all_scores[norm_key] = ainfo['score']

        # Qualität
        q_keys = ['faktentreue','ausgewogenheit','quellenangaben','neutralitaet']
        q_scores_auto = {k:auto_all_scores.get(k,0.0) for k in q_keys}
        self.weights_manager.update_weights('quality', manual_scores.get('quality',{}), q_scores_auto)

        # Manipulation
        m_keys = ['emotionalisierung','suggestion','framing','auslassung']
        m_scores_auto = {k:auto_all_scores.get(k,0.0) for k in m_keys}
        self.weights_manager.update_weights('manipulation', manual_scores.get('manipulation',{}), m_scores_auto)

        # Bias
        b_keys = ['politischer_bias','ideologischer_bias','wirtschaftlicher_bias','kultureller_bias']
        b_scores_auto = {k:auto_all_scores.get(k,0.0) for k in b_keys}
        self.weights_manager.update_weights('bias', manual_scores.get('bias',{}), b_scores_auto)

        st.session_state.manual_corrections.append({
            'timestamp': datetime.now().isoformat(),
            'manual_scores': manual_scores,
            'auto_scores': auto_all_scores
        })
        st.session_state.model_adjustments.append({
            'timestamp': datetime.now().isoformat(),
            'weights': self.weights_manager.weights.copy()
        })

def create_ui_metrics(metrics: Dict[str, float], cols: int = 3) -> None:
    columns = st.columns(cols)
    idx = 0
    for metric, value in metrics.items():
        with columns[idx % cols]:
            if isinstance(value, float):
                st.metric(label=metric.replace('_', ' ').title(), value=f"{value:.2f}")
            else:
                st.metric(label=metric.replace('_', ' ').title(), value=str(value))
        idx += 1

def show_analysis_history(detector: EnhancedManipulationDetector):
    text = st.session_state.analysis_text if 'analysis_text' in st.session_state else ""
    if not text:
        return
    history = detector.get_similar_analyses(text)
    if not history:
        st.info("Keine ähnlichen Analysen gefunden")
        return
        
    st.markdown("### Ähnliche frühere Analysen")
    for entry in history:
        with st.expander(f"Analyse {entry['id']}"):
            st.write(entry['analysis'])
            st.metric("Ähnlichkeit", f"{entry['similarity']:.2%}")

def manual_evaluation_ui(detector: EnhancedManipulationDetector, text: str, results: Dict):
    st.markdown("### Manuelle Bewertung")
    with st.form("manual_evaluation"):
        st.subheader("Journalistische Qualität")
        col1, col2 = st.columns(2)
        with col1:
            faktentreue = st.slider("Faktentreue", 0.0, 1.0, 0.5)
            ausgewogenheit = st.slider("Ausgewogenheit", 0.0, 1.0, 0.5)
        with col2:
            quellenangaben = st.slider("Quellenangaben", 0.0, 1.0, 0.5)
            neutralitaet = st.slider("Neutralität", 0.0, 1.0, 0.5)

        st.subheader("Manipulationstechniken")
        col3, col4 = st.columns(2)
        with col3:
            emotionalisierung = st.slider("Emotionalisierung", 0.0, 1.0, 0.5)
            suggestion = st.slider("Suggestion", 0.0, 1.0, 0.5)
        with col4:
            framing = st.slider("Framing", 0.0, 1.0, 0.5)
            auslassung = st.slider("Auslassung", 0.0, 1.0, 0.5)

        st.subheader("Bias-Bewertung")
        col5, col6 = st.columns(2)
        with col5:
            politischer_bias = st.slider("Politischer Bias", 0.0, 1.0, 0.5)
            ideologischer_bias = st.slider("Ideologischer Bias", 0.0, 1.0, 0.5)
        with col6:
            wirtschaftlicher_bias = st.slider("Wirtschaftlicher Bias", 0.0, 1.0, 0.5)
            kultureller_bias = st.slider("Kulturelle Voreingenommenheit", 0.0, 1.0, 0.5)

        manual_scores = {
            'quality': {
                'faktentreue': faktentreue,
                'ausgewogenheit': ausgewogenheit,
                'quellenangaben': quellenangaben,
                'neutralitaet': neutralitaet
            },
            'manipulation': {
                'emotionalisierung': emotionalisierung,
                'suggestion': suggestion,
                'framing': framing,
                'auslassung': auslassung
            },
            'bias': {
                'politischer_bias': politischer_bias,
                'ideologischer_bias': ideologischer_bias,
                'wirtschaftlicher_bias': wirtschaftlicher_bias,
                'kultureller_bias': kultureller_bias
            }
        }

        if st.form_submit_button("Bewertung speichern"):
            detector.update_from_manual_feedback(manual_scores, results)
            st.success("Bewertung gespeichert und Gewichtungen angepasst.")

def create_analysis_dashboard(results: Dict, detector: EnhancedManipulationDetector) -> None:
    st.subheader("⚡ Performance")
    perf_metrics = results.get('performance_metrics', {})
    create_ui_metrics(perf_metrics)

    st.subheader("📊 Quantitative Analyse")
    quant_metrics = results.get('quantitative_metrics', {})
    create_ui_metrics(quant_metrics)

    def show_aspects(title, cat_key):
        st.write(f"**{title}**")
        txt = results.get(cat_key, {}).get('response', '')
        scores = detector._extract_scores_from_analysis(txt)
        if not scores:
            st.write("Keine oder ungültige Antwort erhalten. Bitte prüfen, ob das Modell korrekt geantwortet hat.")
        else:
            for a, info in scores.items():
                st.write(f"- {a.title()}: {info['score']:.2f} ({info['explanation']})")

    st.subheader("🎯 Detailanalysen")
    show_aspects("Journalistische Qualität", "journalistic_quality")
    show_aspects("Manipulationstechniken", "manipulation_analysis")
    show_aspects("Bias Detection", "bias_detection")
    show_aspects("Neutralitätsanalyse", "neutrality_analysis")

def main():
    st.set_page_config(
        page_title="Erweiterte Manipulationsanalyse für Nachrichtentexte",
        page_icon="📰",
        layout="wide"
    )

    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'analysis_text' not in st.session_state:
        st.session_state.analysis_text = None
    if 'manual_scores' not in st.session_state:
        st.session_state.manual_scores = {}
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None

    detector = EnhancedManipulationDetector()

    tab1, tab2, tab3 = st.tabs(["Analyse", "Manuelle Evaluation", "Historie"])

    with tab1:
        st.title("📰 Erweiterte Manipulationsanalyse für Nachrichtentexte (lokale Modelle)")
        with st.sidebar:
            st.title("Analyseeinstellungen")
            models = detector.models
            if models:
                selected_model = st.selectbox(
                    "🤖 Modell-Auswahl:",
                    options=models,
                    index=0,
                    help="Wählen Sie das zu verwendende lokale Modell"
                )
                st.session_state.selected_model = selected_model
            else:
                st.error("Keine lokalen Modelle verfügbar.")
                st.stop()

            input_method = st.radio(
                "Eingabemethode:",
                ["URL", "Direkter Text", "Datei Upload"]
            )

        text_to_analyze = None
        if input_method == "URL":
            url = st.text_input(
                "Artikel-URL eingeben:",
                placeholder="https://www.beispiel.de/artikel"
            )
            if url:
                with st.spinner("Lade Artikel..."):
                    try:
                        response = requests.get(url)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        text_to_analyze = ' '.join([p.text for p in soup.find_all('p')])
                        if text_to_analyze:
                            st.success("Artikel erfolgreich geladen!")
                            with st.expander("Extrahierter Text"):
                                st.write(text_to_analyze)
                    except Exception as e:
                        st.error(f"Fehler beim Laden der URL: {str(e)}")

        elif input_method == "Direkter Text":
            text_to_analyze = st.text_area(
                "Text zur Analyse eingeben:",
                height=200,
                placeholder="Hier den Nachrichtentext einfügen..."
            )

        else:
            uploaded_file = st.file_uploader(
                "Datei hochladen",
                type=['txt', 'docx', 'pdf'],
                help="Unterstützte Formate: TXT, DOCX, PDF"
            )
            if uploaded_file:
                text_to_analyze = load_document(uploaded_file)
                if text_to_analyze:
                    with st.expander("Extrahierter Text"):
                        st.write(text_to_analyze)

        if text_to_analyze and st.button("Analyse starten", type="primary"):
            st.session_state.analysis_text = text_to_analyze
            with st.spinner("Analyse wird durchgeführt..."):
                results = detector.analyze_article(
                    text_to_analyze,
                    st.session_state.selected_model
                )

                if results and 'error' not in results:
                    st.session_state.analysis_results = results
                    create_analysis_dashboard(results, detector)

                    st.subheader("💡 Verbesserungsvorschläge")
                    rec = results.get('recommendations', {}).get('response', '')
                    st.write(rec)

                    st.subheader("📥 Ergebnisse Exportieren")
                    json_results = detector.export_analysis_results(results)
                    st.download_button(
                        label="Als JSON herunterladen",
                        data=json_results,
                        file_name='analyse_ergebnisse.json',
                        mime='application/json'
                    )
                else:
                    st.error("Analyse fehlgeschlagen oder ungültige Ergebnisse. Bitte prüfen Sie die Modellantworten.")

    with tab2:
        if not st.session_state.analysis_results:
            st.warning("Bitte zuerst eine Analyse durchführen.")
        else:
            st.title("🎯 Manuelle Evaluation")
            with st.expander("Analysierter Text"):
                st.write(st.session_state.analysis_text)
            manual_evaluation_ui(detector, st.session_state.analysis_text, st.session_state.analysis_results)

    with tab3:
        st.title("📚 Analysehistorie")
        if st.session_state.analysis_results:
            show_analysis_history(detector)
        else:
            st.info("Noch keine Analysen durchgeführt.")

if __name__ == "__main__":
    main()>
