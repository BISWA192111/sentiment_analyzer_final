#loading all the required libraries, arranged at the top after completing the entire code 
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import io
import base64
import traceback
import uuid
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers.pipelines import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json
from sklearn.exceptions import NotFittedError

app = Flask(__name__, template_folder='templates')
app.secret_key = 'your-secret-key-here-123'  # can be changed according to the need, cool isn't it :)

# Initializing NLP tools required for natural language procssing
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# uloading the csv file for training, testing and validation
DB_FILE = "user_responses.csv"
TRAINING_DATA_FILE = "C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\output_cleaned.csv"  
os.makedirs('models', exist_ok=True)


import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "hf_home")

from transformers.pipelines import pipeline
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Setting model_dir globally so it is always defined and available for BERT model loading
model_dir = os.path.join(os.path.dirname(__file__), "local-bert-model")

class SentimentAnalyzer:
    def __init__(self, distilbert_model_dir=None):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        self.models = {
            "lr": None,
            "svm": None
        }
        self.model_accuracy = {}
        self.vader = SentimentIntensityAnalyzer()
        # Force use of fine-tuned DistilBERT model only since its accuracy is more than other models used and implied after testing
        distilbert_model_dir = './distilbert-finetuned-patient'
        if not os.path.exists(distilbert_model_dir):
            raise FileNotFoundError(f"Fine-tuned DistilBERT model not found at {distilbert_model_dir}. Please train the model first.")
        # Trying both 'text-classification' and 'sentiment-analysis' for compatibility
        try:
            self.distilbert_sentiment = pipeline("text-classification", model=distilbert_model_dir)
        except Exception:
            self.distilbert_sentiment = pipeline("sentiment-analysis", model=distilbert_model_dir)
        print(f"[INFO] DistilBERT sentiment model loaded from: {distilbert_model_dir}")
        # loading BERT sentiment model from local directory only to avoid any error
        try:
            self.bert_sentiment_tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.bert_sentiment_model = BertForSequenceClassification.from_pretrained(model_dir)
        except Exception as e:
            print(f"Could not load BERT sentiment model: {e}")
            print(f"model_dir used: {model_dir}")
            if os.path.exists(model_dir):
                print("Contents of model_dir:", os.listdir(model_dir))
            else:
                print("model_dir does not exist!")
            self.bert_sentiment_tokenizer = None
            self.bert_sentiment_model = None
        self.initialize_models()
        self.cached_metrics = None
        self.last_metrics_data_hash = None
        # Setting metrics cache file as instance variable and loading the csv file for phase checking
        self.metrics_cache_file = os.path.join('models', 'metrics_cache.json')
        self.before_kw, self.during_kw, self.after_kw = self.load_time_keywords_from_csv('C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\sentiment_phases_1000.csv')
        # Loading phase classifier and vectorizer from the ther model training, present in the directory uploaded
        import joblib
        self.phase_clf = joblib.load('phase_classifier.pkl')
        self.phase_vectorizer = joblib.load('phase_vectorizer.pkl')
    
    def initialize_models(self):
        """Initialize or train models"""
        try:
            self.load_models()
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.train_models_with_csv_data()
    
    def load_models(self):
        """Load pre-trained models if they exist"""
        self.models['lr'] = joblib.load('models/lr_model.pkl')
        self.models['svm'] = joblib.load('models/svm_model.pkl')
        self.vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        self.label_encoder = joblib.load('models/label_encoder.pkl')
    
    def save_models(self):
        """Save models to disk"""
        joblib.dump(self.models['lr'], 'models/lr_model.pkl')
        joblib.dump(self.models['svm'], 'models/svm_model.pkl')
        joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
    
    def clean_labels(self, series):
        """Map string labels to numeric and drop invalid/missing labels."""
        label_map = {'Negative': -1, 'Neutral': 0, 'Positive': 1, -1: -1, 0: 0, 1: 1}
        mapped = series.map(label_map)
        # Dropping NaNs (invalid labels)
        return mapped.dropna().astype(int)
    
    def load_training_data(self):
        """Load training data from CSV file with statement/label columns, always as DataFrame."""
        if not os.path.exists(TRAINING_DATA_FILE):
            print(f"Training data file '{TRAINING_DATA_FILE}' not found. Terminating.")
            raise SystemExit(1)
        try:
            df = pd.read_csv(TRAINING_DATA_FILE)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if 'statement' not in df.columns or 'label' not in df.columns:
                print("CSV must contain 'statement' and 'label' columns. Terminating.")
                raise SystemExit(1)
            df = df.dropna(subset=['statement', 'label'])
            df = df[df['statement'].apply(lambda x: isinstance(x, str))]
            # Cleaning labels
            df['label'] = self.clean_labels(df['label'])
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            if len(df) < 10:
                print("Insufficient training data (need at least 10 samples). Terminating.")
                raise SystemExit(1)
            return df.reset_index(drop=True)
        except Exception as e:
            print(f"Error loading training data: {e}. Terminating.")
            raise SystemExit(1)

    def split_train_val_test(self, X, y, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
        # First, split off the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Now spliting the remaining data into train and validation
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models_with_csv_data(self):
        """Train models with data from CSV file, with improved feature extraction and class balance check."""
        df = self.load_training_data()
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # Preprocess and filter out empty/short/whitespace-only statements
        df['cleaned_text'] = df['statement'].apply(self.preprocess_text)
        df = df[df['cleaned_text'].str.len() > 2]
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        # Printing class distribution
        class_counts = df['label'].value_counts().sort_index()
        print("Class distribution:")
        for k in [-1, 0, 1]:
            print(f"  {k}: {class_counts.get(k, 0)} samples")
        min_class = class_counts.min() if not class_counts.empty else 0
        max_class = class_counts.max() if not class_counts.empty else 0
        if min_class < 0.2 * max_class:
            print("WARNING: Your data is highly imbalanced. Consider collecting more samples for minority classes.")
        X = df['cleaned_text']
        y = self.label_encoder.fit_transform(df['label'])
        # Improved vectorizer: bigrams, stopwords, sublinear_tf, more features
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english', sublinear_tf=True)
        X_vec = self.vectorizer.fit_transform(X)
        # Splitting into train, val, test
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_train_val_test(X_vec, y)
        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
        self.models = {}
        # Logistic Regression
        self.models['lr'] = LogisticRegression(max_iter=1000)
        self.models['lr'].fit(X_train, y_train)
        # SVM
        self.models['svm'] = SVC(kernel='linear', probability=True)
        self.models['svm'].fit(X_train, y_train)
        # Accuracy
        self.model_accuracy = {
            'lr_val': self.models['lr'].score(X_val, y_val),
            'svm_val': self.models['svm'].score(X_val, y_val),
            'lr_test': self.models['lr'].score(X_test, y_test),
            'svm_test': self.models['svm'].score(X_test, y_test)
        }
        self.save_models()
        print(f"Models trained successfully with {len(df)} samples")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"LR Validation Accuracy: {self.model_accuracy['lr_val']:.2f}, SVM Validation Accuracy: {self.model_accuracy['svm_val']:.2f}")
        print(f"LR Test Accuracy: {self.model_accuracy['lr_test']:.2f}, SVM Test Accuracy: {self.model_accuracy['svm_test']:.2f}")
    
    def preprocess_text(self, text):
        """Advanced text cleaning: lowercase, remove punctuation/numbers, strip whitespace, remove stopwords, lemmatize."""
        text = str(text)
        if not text or text.isspace() or len(text.strip()) < 3:
            return ''
        _punct_num = re.compile(r'[^a-zA-Z\s]')
        text = text.lower()
        text = _punct_num.sub('', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)
    
    def split_statements(self, text):
        """Split input into sentences for future queries"""
        try:
            blob = TextBlob(text)
            sentences_attr = getattr(blob, 'sentences', None)
            # Checking if sentences_attr is iterable
            if sentences_attr is not None and hasattr(sentences_attr, '__iter__'):
                sentences = list(sentences_attr)
                if not sentences:
                    return [text]
                return [str(sentence) for sentence in sentences]
            else:
                return [text]
        except Exception as e:
            print(f"TextBlob sentence split error: {e}")
            return [text]

    def classify_time_period(self, sentence, current_phase=None, debug=False):
        # Use ML classifier for phase detection ( :( took a lot of time still not that accurate due to improper data)
        vec = self.phase_vectorizer.transform([sentence])
        phase = self.phase_clf.predict(vec)[0]
        if debug:
            print(f"[PHASE ML] '{sentence}' => {phase}")
        return phase, phase

    def predict_bert_sentiment(self, text):
        """
        Predict sentiment using a BERT model. Returns (label, score).
        """
        if not self.bert_sentiment_model or not self.bert_sentiment_tokenizer:
            return "Unknown", 0.0
        inputs = self.bert_sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = self.bert_sentiment_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            pred = int(torch.argmax(probs).item()) 
            label = "POSITIVE" if pred == 1 else "NEGATIVE"
            score = float(probs[int(pred)]) 
            return label, score

    # def get_metrics(self):
    #     """
    #     Calculate and cache only the accuracy for each model on the training set.
    #     Use disk cache for fast loading. Only recompute if the training data has changed.
    #     """
    #     import hashlib
    #     from sklearn.metrics import accuracy_score
    #     import warnings
    #     try:
    #         df = self.load_training_data()
    #         if not isinstance(df, pd.DataFrame):
    #             df = pd.DataFrame(df)
    #         df['cleaned_text'] = df['statement'].apply(self.preprocess_text)
    #         df = df[df['cleaned_text'].str.len() > 2]
    #         if not isinstance(df, pd.DataFrame):
    #             df = pd.DataFrame(df)
    #         # Compute a hash of the training data to detect changes (use CSV bytes for hash)
    #         data_hash = hashlib.md5(df.to_csv(index=False).encode('utf-8')).hexdigest()
    #     except Exception as e:
    #         return {}, f"Error loading training data: {e}"

    #     # Try to load metrics from disk cache
    #     if os.path.exists(self.metrics_cache_file):
    #         try:
    #             with open(self.metrics_cache_file, 'r') as f:
    #                 cache = json.load(f)
    #             if cache.get('data_hash') == data_hash:
    #                 return cache['output'], None
    #         except Exception as e:
    #             print(f"Warning: Could not load metrics cache: {e}")

    #     metrics = {}
    #     try:
    #         # Clean and filter labels
    #         labels_clean = self.clean_labels(df['label'])
    #         valid_labels = set(self.label_encoder.classes_)
    #         mask = labels_clean.isin(valid_labels)
    #         df = df.loc[mask].reset_index(drop=True)
    #         labels_clean = labels_clean[mask].reset_index(drop=True)

    #         if df.empty or len(labels_clean) == 0:
    #             return {}, "No valid samples with known labels found in training data for metrics calculation."

    #         try:
    #             X = self.vectorizer.transform(df['cleaned_text'])
    #         except NotFittedError:
    #             return {}, "Vectorizer is not fitted. Please retrain your models."
    #         except Exception as e:
    #             return {}, f"Error transforming features: {e}"

    #         try:
    #             y = self.label_encoder.transform(labels_clean)
    #         except NotFittedError:
    #             return {}, "Label encoder is not fitted. Please retrain your models."
    #         except ValueError as e:
    #             return {}, f"Label encoding error: {e}"

    #         if X.shape[0] == 0 or len(y) == 0:
    #             return {}, "No valid samples after feature extraction for metrics calculation."

    #         # Classic ML models
    #         with warnings.catch_warnings():
    #             warnings.simplefilter("ignore")
    #             for model_key, model in self.models.items():
    #                 if model is not None:
    #                     try:
    #                         y_pred = model.predict(X)
    #                         acc = accuracy_score(y, y_pred)
    #                         metrics[model_key] = {'accuracy': acc}
    #                     except Exception as e:
    #                         metrics[model_key] = {'accuracy': 0.0, 'error': f"Model prediction error: {e}"}
    #         # DistilBERT metrics (truncate to 512 tokens/256 words)
    #         def truncate_text(text):
    #             tokens = word_tokenize(text)
    #             return ' '.join(tokens[:512])
    #         distilbert_preds = []
    #         for text in df['cleaned_text']:
    #             try:
    #                 label = self.distilbert_sentiment(truncate_text(text))[0]['label']
    #             except Exception as e:
    #                 label = "ERROR"
    #             distilbert_preds.append(label)
    #         distilbert_map = {'POSITIVE': 1, 'NEGATIVE': -1}
    #         distilbert_preds_num = [distilbert_map.get(label.upper(), 0) if label != "ERROR" else 0 for label in distilbert_preds]
    #         y_true = df['label'].astype(int).to_numpy()
    #         if len(y_true) == 0:
    #             return {}, "No valid samples for DistilBERT metrics calculation."
    #         acc = accuracy_score(y_true, distilbert_preds_num)
    #         metrics['distilbert'] = {'accuracy': acc}
    #     except Exception as metric_err:
    #         return {}, f"Error calculating metrics: {metric_err}"

    #     output = {'metrics': metrics}
    #     # Save metrics and hash to disk cache
    #     try:
    #         with open(self.metrics_cache_file, 'w') as f:
    #             json.dump({'data_hash': data_hash, 'output': output}, f)
    #     except Exception as e:
    #         print(f"Warning: Could not save metrics cache: {e}")
    #     self.cached_metrics = output
    #     self.last_metrics_data_hash = data_hash
    #     return output, None

    def clear_metrics_cache(self):
        self.cached_metrics = None
        self.last_metrics_data_hash = None
    
    def analyze_sentiment(self, text):
        """Analyze text for healthcare bot: show sentence-level sentiment only (no before/during/after comparison)."""
        try:
            patient_id = str(uuid.uuid4())
            # Computing BERT sentiment for the entire paragraph only once
            try:
                bert_sentiment, bert_score = self.predict_bert_sentiment(text)
            except Exception as bert_err:
                print(f"BERT sentiment error: {bert_err}")
                bert_sentiment = "Unknown"
                bert_score = 0.0
            statements = self.split_statements(text)
            all_results = []
            # Addding a single BERT sentiment result for the whole paragraph
            all_results.append({
                "patient_id": patient_id,
                "text": text,
                "cleaned_text": self.preprocess_text(text),
                "timestamp": datetime.now().isoformat(),
                "sentiment_bert": bert_sentiment,
                "score_bert": bert_score
            })
            current_phase = None
            for idx, statement in enumerate(statements):
                statement = str(statement)
                cleaned_text = self.preprocess_text(statement)
                features = self.vectorizer.transform([cleaned_text])
                # Temporal classification (extensively for CSV matching :))
                time_period, current_phase = self.classify_time_period(statement, current_phase, debug=True)
                if time_period is None:
                    if idx == 0:
                        time_period = 'before'
                    elif idx == len(statements) - 1:
                        time_period = 'after'
                    else:
                        time_period = 'during'
                # DistilBERT sentiment
                try:
                    distilbert_result = self.distilbert_sentiment(cleaned_text)[0]
                    distilbert_sentiment = distilbert_result['label']
                    distilbert_score = float(distilbert_result['score'])
                except Exception as bert2_err:
                    print(f"DistilBERT sentiment error: {bert2_err}")
                    distilbert_sentiment = "Unknown"
                    distilbert_score = 0.0
                # TextBlob sentiment on cleaned text
                try:
                    tb_blob = TextBlob(cleaned_text)
                    tb_sent = tb_blob.sentiment
                    polarity = float(getattr(tb_sent, 'polarity', 0.0))
                    subjectivity = float(getattr(tb_sent, 'subjectivity', 0.0))
                    if polarity > 0.3:
                        tb_sentiment = "Positive"
                    elif polarity < -0.3:
                        tb_sentiment = "Negative"
                    else:
                        tb_sentiment = "Neutral"
                    tb_confidence = min(0.99, max(0.51, (abs(polarity) + subjectivity)/2))
                except Exception as tb_err:
                    print(f"TextBlob error: {tb_err}")
                    polarity = 0.0
                    subjectivity = 0.0
                    tb_sentiment = "Unknown"
                    tb_confidence = 0.0
                # VADER sentiment 
                try:
                    vader_scores = self.vader.polarity_scores(statement)
                    compound = vader_scores['compound']
                    if compound >= 0.05:
                        vader_sentiment = "Positive"
                    elif compound <= -0.05:
                        vader_sentiment = "Negative"
                    else:
                        vader_sentiment = "Neutral"
                except Exception as vader_err:
                    print(f"VADER error: {vader_err}")
                    vader_sentiment = "Unknown"
                    compound = 0.0
                results = {
                    "patient_id": patient_id,
                    "text": statement,
                    "cleaned_text": cleaned_text,
                    "timestamp": datetime.now().isoformat(),
                    "time_period": time_period,
                    "phase": time_period, 
                    # No BERT sentiment here
                    "sentiment_textblob": tb_sentiment,
                    "confidence_textblob": tb_confidence,
                    "polarity": polarity,
                    "subjectivity": subjectivity,
                    "sentiment_vader": vader_sentiment,
                    "vader_compound": compound,
                    "sentiment_distilbert": distilbert_sentiment,
                    "score_distilbert": distilbert_score
                }
                # Logistic Regression
                try:
                    if self.models['lr']:
                        proba = self.models['lr'].predict_proba(features)[0]
                        pred = self.models['lr'].predict(features)[0]
                        results["sentiment_lr_num"] = int(pred)
                        results["sentiment_lr"] = self.label_to_string(pred)
                        results["confidence_lr"] = float(max(proba))
                except Exception as lr_err:
                    print(f"Logistic Regression error: {lr_err}")
                    results["sentiment_lr_num"] = None
                    results["sentiment_lr"] = "Unknown"
                    results["confidence_lr"] = 0.0
                # SVM
                try:
                    if self.models['svm']:
                        proba = self.models['svm'].predict_proba(features)[0]
                        pred = self.models['svm'].predict(features)[0]
                        results["sentiment_svm_num"] = int(pred)
                        results["sentiment_svm"] = self.label_to_string(pred)
                        results["confidence_svm"] = float(max(proba))
                except Exception as svm_err:
                    print(f"SVM error: {svm_err}")
                    results["sentiment_svm_num"] = None
                    results["sentiment_svm"] = "Unknown"
                    results["confidence_svm"] = 0.0
                self.save_to_db(results)
                all_results.append(results)
            # Fallback: assign first to 'before', last to 'after', others to 'during' if no keyword matched and at least 3 sentences(lateron removed this)
            n = len(statements)
            for idx, r in enumerate(all_results[1:]):
                if r.get('phase') == 'during': 
                    if idx == 0:
                        r['phase'] = 'before'
                    elif idx == n - 2:
                        r['phase'] = 'after'
            # Phase sentiment aggregation
            from collections import defaultdict, Counter
            phase_sentiments = defaultdict(list)
            # Accept both string and numeric label outputs
            label_map = {
                'POSITIVE': 'Positive', 'NEGATIVE': 'Negative', 'NEUTRAL': 'Neutral',
                'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive',
                0: 'Negative', 1: 'Neutral', 2: 'Positive',
                '-1': 'Negative', '-1.0': 'Negative', '0': 'Neutral', '1': 'Positive',
                -1: 'Negative', 0: 'Neutral', 1: 'Positive'
            }
            statement_results = []
            for r in all_results[1:]:  
                phase = r.get('phase')
                label = r.get('sentiment_distilbert', '')
                label_upper = str(label).upper()
                human_sentiment = label_map.get(label_upper, label_map.get(label, str(label)))
                score = r.get('score_distilbert', None)
                if phase in ['before', 'during', 'after'] and label_upper in label_map:
                    phase_sentiments[phase].append(label_upper)
                statement_results.append({
                    'statement': r.get('text', ''),
                    'phase': phase.capitalize() if phase else 'N/A',
                    'sentiment': human_sentiment,
                    'score': f"{score:.3f}" if isinstance(score, float) else score
                })
            # Only returns the per-statement results
            return {
                "statements": statement_results
            }
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            traceback.print_exc()
            return {
                "text": text,
                "error": str(e)
            }
    
    def save_to_db(self, results):
        """Save analysis results to CSV"""
        try:
            df = pd.DataFrame([results])
            if not os.path.exists(DB_FILE):
                df.to_csv(DB_FILE, index=False)
            else:
                df.to_csv(DB_FILE, mode='a', header=False, index=False)
        except Exception as e:
            print(f"Error saving to database: {e}")
    
    def label_to_string(self, label):
        # Maps numeric label to string for display
        if label == 1 or label == '1':
            return 'Positive'
        elif label == 0 or label == '0':
            return 'Neutral'
        elif label == -1 or label == '-1':
            return 'Negative'
        else:
            return str(label)

    def load_time_keywords_from_csv(self, csv_path):
        before_kw, during_kw, after_kw = [], [], []
        try:
            df = pd.read_csv("C:\\Users\\USER\\Downloads\\sentiment-analysis-app-1\\common_synonyms_before_during_after_1000.csv")
            if 'before' in df.columns:
                before_kw = [str(x).strip().lower() for x in df['before'].dropna() if str(x).strip()]
            if 'during' in df.columns:
                during_kw = [str(x).strip().lower() for x in df['during'].dropna() if str(x).strip()]
            if 'after' in df.columns:
                after_kw = [str(x).strip().lower() for x in df['after'].dropna() if str(x).strip()]
        except Exception as e:
            print(f"Error loading time keywords from {csv_path}: {e}")
        return before_kw, during_kw, after_kw

analyzer = SentimentAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    error = None
    patient_id = None
    model_accuracy = None
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        if text:
            results = analyzer.analyze_sentiment(text)
            if 'error' in results:
                error = results['error']
                results = None
            else:
                patient_id = results.get('patient_id')
                model_accuracy = results.get('model_accuracy')
        else:
            error = "Please enter some text to analyze"
    return render_template('index.html', results=results, error=error, patient_id=patient_id, model_accuracy=model_accuracy)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()
    if not data or 'text' not in data or not data['text'].strip():
        return jsonify({"error": "No text provided"}), 400
    
    results = analyzer.analyze_sentiment(data['text'])
    if 'error' in results:
        return jsonify({"error": results['error']}), 500
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
