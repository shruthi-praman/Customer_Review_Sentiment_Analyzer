from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)


model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html",result = None, review="", submitted_review=None)

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form.get("review", "").strip()
    if not review:
        return render_template(
            "index.html",
            result=None,
            review="",
            submitted_review="Please enter a review."
        )
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)
    result = prediction[0]
    return render_template(
        "index.html", 
        result=result,
        review = "",
        submitted_review = review
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )