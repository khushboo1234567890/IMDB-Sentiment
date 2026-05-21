import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from datasets import DatasetDict, load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    model_path: Path = Path("model.joblib")
    max_features: int = 20000
    ngram_range: tuple[int, int] = (1, 2)
    test_size: float = 0.2
    random_state: int = 42


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb_dataset() -> tuple[list[str], list[int], list[str], list[int]]:
    logger.info("Loading IMDb dataset from Hugging Face datasets...")
    dataset: DatasetDict = load_dataset("imdb")
    train_texts = [clean_text(item["text"]) for item in dataset["train"]]
    train_labels = [int(item["label"]) for item in dataset["train"]]
    test_texts = [clean_text(item["text"]) for item in dataset["test"]]
    test_labels = [int(item["label"]) for item in dataset["test"]]
    logger.info("Loaded %s train samples and %s test samples.", len(train_texts), len(test_texts))
    return train_texts, train_labels, test_texts, test_labels


def build_pipeline(config: ModelConfig) -> Pipeline:
    logger.info("Building text classification pipeline...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.max_features,
                    ngram_range=config.ngram_range,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    random_state=config.random_state,
                    max_iter=1000,
                ),
            ),
        ]
    )
    return pipeline


def train_model(config: ModelConfig) -> Pipeline:
    train_texts, train_labels, test_texts, test_labels = load_imdb_dataset()
    logger.info("Training model with %s samples...", len(train_texts))
    pipeline = build_pipeline(config)
    pipeline.fit(train_texts, train_labels)
    logger.info("Training complete.")

    if test_texts and test_labels:
        predictions = pipeline.predict(test_texts)
        accuracy = accuracy_score(test_labels, predictions)
        logger.info("Test accuracy: %.4f", accuracy)
        logger.info("Classification report:\n%s", classification_report(test_labels, predictions, digits=4))

    joblib.dump(pipeline, config.model_path)
    logger.info("Saved trained pipeline to %s", config.model_path)
    return pipeline


def load_model(model_path: Path) -> Pipeline:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def predict_text(pipeline: Pipeline, texts: list[str]) -> list[str]:
    clean_texts = [clean_text(text) for text in texts]
    predictions = pipeline.predict(clean_texts)
    return ["positive" if label == 1 else "negative" for label in predictions]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMDb movie review sentiment analysis with a reusable text classification pipeline."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the sentiment model using the IMDb dataset and save it to disk.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the saved sentiment model on the IMDb test split.",
    )
    parser.add_argument(
        "--predict",
        nargs=1,
        metavar="TEXT",
        help="Predict the sentiment of a single review text.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ModelConfig.model_path,
        help="Path to save or load the trained model pipeline.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=ModelConfig.max_features,
        help="Maximum number of TF-IDF features.",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=ModelConfig.ngram_range[0],
        help="Minimum n-gram size for TF-IDF.",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=ModelConfig.ngram_range[1],
        help="Maximum n-gram size for TF-IDF.",
    )
    return parser.parse_args()


def evaluate_model(config: ModelConfig) -> None:
    pipeline = load_model(config.model_path)
    _, _, test_texts, test_labels = load_imdb_dataset()
    predictions = pipeline.predict(test_texts)
    accuracy = accuracy_score(test_labels, predictions)
    logger.info("Evaluation accuracy: %.4f", accuracy)
    logger.info("Classification report:\n%s", classification_report(test_labels, predictions, digits=4))


def main() -> None:
    args = parse_args()
    config = ModelConfig(
        model_path=args.model_path,
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
    )

    if args.train:
        train_model(config)
        return

    if args.evaluate:
        evaluate_model(config)
        return

    if args.predict:
        pipeline = load_model(config.model_path)
        review_text = args.predict[0]
        sentiment = predict_text(pipeline, [review_text])[0]
        print(f"Predicted sentiment: {sentiment}")
        return

    logger.info("No action selected. Use --train, --evaluate, or --predict.")


if __name__ == "__main__":
    main()
