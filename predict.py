import joblib

# load the entire pipeline
pipeline = joblib.load("best_model.pkl")

def predict(text):
    # feed raw text in—pipeline will vectorize then classify
    pred = pipeline.predict([text])[0]
    return "Real" if pred == 1 else "Fake"
    return "Real" if prediction[0] == 1 else "Fake"

if __name__ == "__main__":
    print("📢 Fake News Detector")
    print("Type a news article or headline to classify it.")
    print("(Type 'exit' or 'quit' to stop)\n")

    while True:
        user_input = input("> ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        result = predict(user_input)
        print(f"\n🧠 Prediction: {result.upper()}\n")
