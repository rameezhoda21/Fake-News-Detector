# visualize.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.naive_bayes import MultinomialNB

from preprocess import load_and_prepare_data  # your existing function

def run_grid_search(X_train, y_train):
    pipeline = Pipeline([
        ("tfidf",      TfidfVectorizer(max_df=0.8)),
        ("classifier", MultinomialNB())
    ])
    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df":       [1, 5, 10],
        "classifier__alpha":   [0.5, 1.0, 1.5]
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("\n🔍 Best parameters:", grid.best_params_)
    # Save both the best estimator and the full grid object
    joblib.dump(grid.best_estimator_, "best_model.pkl")
    joblib.dump(grid,            "grid_search.pkl")
    print("📦 Saved best_model.pkl and grid_search.pkl")
    return grid

def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, cv_scores = learning_curve(
        estimator,
        X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = train_scores.mean(axis=1)
    cv_mean    = cv_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, marker='o', label="Training score")
    plt.plot(train_sizes, cv_mean,    marker='o', label="Cross-validation score")
    plt.xlabel("Training set size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_heatmap(grid):
    results = pd.DataFrame(grid.cv_results_)
    # Make ngram_range printable
    results["param_tfidf__ngram_range"] = results["param_tfidf__ngram_range"].astype(str)
    pivot = results.pivot_table(
        index="param_tfidf__ngram_range",
        columns="param_classifier__alpha",
        values="mean_test_score"
    )

    plt.figure()
    plt.imshow(pivot, aspect="auto", interpolation="nearest")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("classifier__alpha")
    plt.ylabel("tfidf__ngram_range")
    plt.title("Grid Search CV Mean Test Scores")
    plt.colorbar(label="Mean CV Accuracy")
    plt.show()

def main():
    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # 2. Run grid search (and save results)
    grid = run_grid_search(X_train, y_train)

    # 3. Plot learning curve
    plot_learning_curve(grid.best_estimator_, X_train, y_train)

    # 4. Plot hyperparameter heatmap
    plot_heatmap(grid)

if __name__ == "__main__":
    main()
