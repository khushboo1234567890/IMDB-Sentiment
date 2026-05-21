# IMDb Sentiment Analysis

This project builds a sentiment classification system for IMDb movie reviews using the Hugging Face IMDb dataset.

## Setup

Install dependencies:

```powershell
"C:/Program Files/Python314/python.exe" -m pip install -r requirements.txt
```

## Usage

Train a model:

```powershell
"C:/Program Files/Python314/python.exe" sentiment_analysis.py --train
```

Evaluate the saved model:

```powershell
"C:/Program Files/Python314/python.exe" sentiment_analysis.py --evaluate
```

Predict a single review:

```powershell
"C:/Program Files/Python314/python.exe" sentiment_analysis.py --predict "This movie was fantastic and full of heart."
```

## Notes

- The script uses `TfidfVectorizer` and `LogisticRegression` for a reliable baseline.
- The trained pipeline is saved to `model.joblib` by default.
