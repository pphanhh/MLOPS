# Twitter Sentiment Analysis for Trump Election 2024

A complete end-to-end MLOps pipeline to analyze public sentiment on Twitter (now X) toward Donald Trump’s 2024 presidential campaign — from real-time data crawling to model serving, all automated and orchestrated using modern DevOps practices.

# Project Overview

This project builds a production-grade pipeline for daily sentiment analysis on Twitter posts related to Donald Trump. It uses Apache Airflow to orchestrate workflows, MLflow for model tracking and registration, FastAPI for real-time prediction, and Docker + GitHub Actions for CI/CD.

# Key Features
****Automated Workflow with Apache Airflow****

Two DAG paths:

- Training days (every 15 days): Full pipeline (crawling → training → deployment)

- Non-training days (daily): Crawling → prediction only

Smart branching for efficiency and automation

****Real-Time Sentiment Dashboard****

- Live Grafana dashboard showing positive, neutral, and negative sentiment trends

- Connected to PostgreSQL for real-time monitoring

****NLP Models****

- Pre-trained transformers: BERT, DistilBERT, RoBERTa

- Traditional ML: Logistic Regression

- Rule-based: VADER

- DistilBERT selected as the current champion model (best F1-score)

****Model Lifecycle with MLflow****

- Model training, evaluation, and registration

- Automatic champion vs challenger comparison based on F1

- Production model served via FastAPI and versioned using MLflow Registry

****CI/CD with GitHub Actions****

- Linting, testing, Docker image build, and deployment via GitHub workflows

- Push to Docker Hub on successful builds

- Containerized FastAPI application tested via Pytest inside Docker Compose
