import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from preprocess import load_and_prepare_data

def train_model():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    pipeline = Pipeline([
        ("tfidf",      TfidfVectorizer(max_df=0.8)),
        ("classifier", MultinomialNB())
    ])

    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__min_df":       [1, 5, 10],
        "classifier__alpha":   [0.5, 1.0, 1.5]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=2, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("🔍 Best parameters:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print("\n🧪 Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"✅ Test Accuracy: {grid.score(X_test, y_test):.4f}")

    joblib.dump(grid.best_estimator_, "best_model.pkl")
    print("📦 Saved best_model.pkl")

if __name__ == "__main__":
    train_model()
# This script trains a model to classify news articles as "Real" or "Fake".
# It uses a pipeline with TF-IDF vectorization and a Naive Bayes classifier,   