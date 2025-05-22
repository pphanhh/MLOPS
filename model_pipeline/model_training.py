#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import pandas as pd
import torch
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.pytorch

from tqdm import tqdm
from sqlalchemy import create_engine, text  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, TensorDataset
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification
)

import nltk
nltk.download('vader_lexicon', quiet=True)
from sqlalchemy.engine import Engine
# ===========================
# Define Utility Classes
# ===========================

class DataProcessor:
    def __init__(self, sql_query=None):
        self.db_user = os.getenv('DB_USER', 'mlops_user')
        self.db_pass = os.getenv('DB_PASSWORD', '12345678')
        self.db_host = os.getenv('DB_HOST', '172.21.80.1')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'mlops')
        self.sql_query = sql_query or os.getenv('SQL_QUERY', 'SELECT * FROM twitter_comments')
        self.df = None


    def load_data(self):
        db_uri = f"postgresql+psycopg2://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
        engine = create_engine(db_uri)
        print(f"ENGINE TYPE: {type(engine)}")
        with engine.connect() as conn:
            result = conn.execute(text(self.sql_query))
            self.df = pd.DataFrame(result.fetchall(), columns=result.keys())
        # Lowercase all column names
        self.df.columns = [col.lower() for col in self.df.columns]
        return self.df

    def clean_and_map(self, text_col='cleaned_text', label_col='sentiment'):
        self.df[text_col] = self.df[text_col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        sentiment_map = {'Positive': 1, 'Negative': 0, 'Neutral': 2}
        self.df['sentiment_num'] = self.df[label_col].map(sentiment_map)
        self.df = self.df.drop(columns=[label_col])
        self.df = self.df.rename(columns={'sentiment_num': label_col})
        return self.df

    def split(self, test_size=0.2, random_state=42, text_col='cleaned_text', label_col='sentiment'):
        X = self.df[text_col]
        y = self.df[label_col]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

class TransformerTrainer:
    def __init__(self, model_name, tokenizer_cls, model_cls, num_labels=3, lr=1e-5, batch_size=16, max_length=64):
        self.tokenizer = tokenizer_cls.from_pretrained(model_name)
        self.model = model_cls.from_pretrained(model_name, num_labels=num_labels)
        self.lr = lr
        self.batch_size = batch_size
        self.max_length = max_length

    def tokenize(self, texts):
        return self.tokenizer(
            texts.tolist(), truncation=True, padding=True,
            max_length=self.max_length, return_tensors='pt'
        )

    def create_loader(self, tokens, labels):
        dataset = TensorDataset(tokens['input_ids'], tokens['attention_mask'], labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, train_loader, epochs=3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    def evaluate(self, test_loader):
        self.model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds.tolist())
                truths.extend(labels.cpu().numpy().tolist())
        return preds, truths

    def run(self, X_train, y_train, X_test, y_test, epochs=3):
        tokens_train = self.tokenize(X_train)
        tokens_test = self.tokenize(X_test)
        train_loader = self.create_loader(tokens_train, torch.tensor(y_train.tolist()))
        test_loader = self.create_loader(tokens_test, torch.tensor(y_test.tolist()))

        self.train(train_loader, epochs)
        preds, truths = self.evaluate(test_loader)
        return preds, truths

class ClassicalTrainer:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.model = LogisticRegression(max_iter=1000)

    def run(self, X_train, y_train, X_test, y_test):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        self.model.fit(X_train_tfidf, y_train)
        preds = self.model.predict(X_test_tfidf)
        return preds

class VaderEvaluator:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def run(self, X_test, y_test):
        preds = []
        for text in X_test:
            score = self.analyzer.polarity_scores(text)["compound"]
            if score > 0.05:
                preds.append(1)
            elif score < -0.05:
                preds.append(0)
            else:
                preds.append(2)
        return preds

# ===========================
# Define MLflow Wrapper Classes
# ===========================

class HFTransformersWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model_class, tokenizer_class, model_path="model_path"):
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.model_path_key = model_path

    def load_context(self, context):
        model_path = context.artifacts[self.model_path_key]
        self.model = self.model_class.from_pretrained(model_path)
        self.tokenizer = self.tokenizer_class.from_pretrained(model_path)

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).numpy()
        return preds


class SklearnTextWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, context, model_input):
        X = self.vectorizer.transform(model_input["text"])
        return self.model.predict(X)

class VaderSentimentWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def predict(self, context, model_input):
        results = []
        for text in model_input["text"]:
            score = self.analyzer.polarity_scores(text)["compound"]
            if score > 0.05:
                results.append(1)
            elif score < -0.05:
                results.append(0)
            else:
                results.append(2)
        return results

# ===========================
# Define Logging Function
# ===========================
import json

def log_model_to_mlflow(model_name, model_wrapper, X_test, preds, truths, params=None, save_dir=None):
    with mlflow.start_run(run_name=model_name) as run:
        # Log model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_wrapper,
            input_example=pd.DataFrame({"text": X_test.tolist()[:2]}),
            artifacts={"model_path": save_dir} if save_dir else None
        )

        # Log params
        if params:
            for k, v in params.items():
                mlflow.log_param(k, v)

        # Log metrics
        acc = accuracy_score(truths, preds)
        report = classification_report(truths, preds, output_dict=True)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

        # Save run id to MLOPS/latest_runs.json
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        latest_runs_path = os.path.join(base_dir, "latest_runs.json")

        if os.path.exists(latest_runs_path):
            with open(latest_runs_path, "r") as f:
                latest_runs = json.load(f)
        else:
            latest_runs = {}

        latest_runs[model_name] = run.info.run_id
        with open(latest_runs_path, "w") as f:
            json.dump(latest_runs, f, indent=4)

        print(f"{model_name} model logged successfully. Run ID: {run.info.run_id}")
# ===========================

# === Main
if __name__ == '__main__':
    dp = DataProcessor()
    dp.load_data()
    dp.clean_and_map()
    X_train, X_test, y_train, y_test = dp.split()

    tracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../mlruns"))
    mlflow.set_tracking_uri(f"file://{tracking_path}")
    mlflow.set_experiment("sentiment-analysis")

    # Logistic Regression
    classical_trainer = ClassicalTrainer()
    class_preds = classical_trainer.run(X_train, y_train, X_test, y_test)
    log_model_to_mlflow("LogisticRegression_TFIDF", SklearnTextWrapper(classical_trainer.model, classical_trainer.vectorizer), X_test, class_preds, y_test, {"model_type": "LogisticRegression"})

    # BERT
    bert_trainer = TransformerTrainer('bert-base-uncased', BertTokenizer, BertForSequenceClassification)
    bert_preds, bert_truth = bert_trainer.run(X_train, y_train, X_test, y_test)
    bert_trainer.model.save_pretrained("bert_saved")
    bert_trainer.tokenizer.save_pretrained("bert_saved")
    log_model_to_mlflow("BERT_Transformer", HFTransformersWrapper(BertForSequenceClassification, BertTokenizer,"model_path"), X_test, bert_preds, bert_truth, {"model_type": "BERT"}, save_dir="bert_saved")

    # RoBERTa
    roberta_trainer = TransformerTrainer('roberta-base', RobertaTokenizer, RobertaForSequenceClassification)
    roberta_preds, roberta_truth = roberta_trainer.run(X_train, y_train, X_test, y_test)
    roberta_trainer.model.save_pretrained("roberta_saved")
    roberta_trainer.tokenizer.save_pretrained("roberta_saved")
    log_model_to_mlflow("RoBERTa_Transformer", HFTransformersWrapper(RobertaForSequenceClassification, RobertaTokenizer, "model_path"), X_test, roberta_preds, roberta_truth, {"model_type": "RoBERTa"}, save_dir="roberta_saved")

    # DistilBERT
    distil_trainer = TransformerTrainer('distilbert-base-uncased', DistilBertTokenizer, DistilBertForSequenceClassification)
    distil_preds, distil_truth = distil_trainer.run(X_train, y_train, X_test, y_test)
    distil_trainer.model.save_pretrained("distilbert_saved")
    distil_trainer.tokenizer.save_pretrained("distilbert_saved")
    log_model_to_mlflow("DistilBERT_Transformer", HFTransformersWrapper(DistilBertForSequenceClassification, DistilBertTokenizer, "model_path"), X_test, distil_preds, distil_truth, {"model_type": "DistilBERT"}, save_dir="distilbert_saved")

    # VADER
    vader_eval = VaderEvaluator()
    vader_preds = vader_eval.run(X_test, y_test)
    vader_model = VaderSentimentWrapper()
    log_model_to_mlflow("VADER_Sentiment", vader_model, X_test, vader_preds, y_test, {"model_type": "VADER"})

    print("All models trained and logged successfully.")
