# AI Traffic Congestion Predictor

An end-to-end Machine Learning and NLP system to predict urban traffic congestion levels.
Built for the Federal University of Technology, Owerri.

---
title: AI Traffic Congestion Predictor
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

## Features
- **Synthetic Data Engine:** Generates realistic urban traffic data with Gaussian noise, categorical flip noise, and temporal BPR (Bureau of Public Roads) speed-flow patterns.
- **Ensemble ML Model:** Combines XGBoost and Random Forest classifiers using soft-voting, tuned via Optuna.
- **Gemini NLP Predictor:** Extracts structured features from plain English descriptions to make predictions, and generates an AI summary of the analysis.
- **Premium Frontend:** A dark-themed, glassmorphic UI built with standard HTML/CSS/JS, featuring animated particle networks and Chart.js dashboards.
- **Terminal CLI:** Fully functional `rich`-powered terminal scripts for training, evaluation, and interactive predictions.

## Local Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Rename `.env.example` to `.env` and add your Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
PORT=5000
```

### 3. Generate Data & Train Model
```bash
python data/generate_dataset.py
python ml/train.py
python ml/evaluate.py
```

### 4. Run the Web Application
```bash
python app/app.py
```
Then open `http://localhost:5000` in your browser.

## Deployment to Hugging Face Spaces
This repository is configured to run as a Docker space on Hugging Face.
1. Create a new Space on Hugging Face.
2. Select **Docker** as the Space SDK.
3. Upload these files to the repository.
4. Add your `GEMINI_API_KEY` in the Space Settings under Secrets.
