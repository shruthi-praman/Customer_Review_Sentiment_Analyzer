import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv("dataset/amazon_reviews.csv")

#remove rows where cleaned_review is missing
data = data.dropna(subset = ["cleaned_review"])

# Select features and labels
X = data["cleaned_review"]
y = data["sentiments"]

#TF-IDF
vectorizer = TfidfVectorizer( #tuning TF-IDF
    ngram_range = (1,2),
    max_features = 5000,
    min_df = 2,
    max_df = 0.95
)
X = vectorizer.fit_transform(X)

#split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = 0.2,
    random_state = 42
)

model = LogisticRegression(
    max_iter= 1000,
    class_weight = "balanced", #handling class imbalance
    random_state = 42
)
model.fit(X_train, y_train)

#Predict sentiments for test data
y_pred = model.predict(X_test)

#calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

#detailed report
print(classification_report(y_test, y_pred))

#confusion matrix
print(confusion_matrix(y_test, y_pred))

#Save the trained model
joblib.dump(model, "models/sentiment_model.pkl")

#save the TF-IDF vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully")