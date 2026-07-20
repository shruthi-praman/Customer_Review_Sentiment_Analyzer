# Customer Review Sentiment Analyzer

A Machine Learning based web application that predicts the sentiment of customer reviews as **Positive**, **Negative**, or **Neutral** using Natural Language Processing (NLP) and Flask.

## Features

- Analyze customer reviews instantly.
- Predicts sentiment as:
  - Positive 😊
  - Negative 😞
  - Neutral 😐
- Simple and user-friendly web interface.
- Trained using Amazon customer review dataset.
- Built using Flask and Scikit-learn.

## Technologies Used

- Python 3
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML
- CSS

## Project Structure

```
Customer_Review_Sentiment_Analyzer/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── dataset/
│   └── amazon_reviews.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
└── myenv/ (ignored)
```

## Dataset

The project uses an Amazon customer review dataset containing review text and corresponding sentiment labels.

## Machine Learning Model

- TF-IDF Vectorizer
- Logistic Regression Classifier

The review text is converted into numerical features using TF-IDF and classified using Logistic Regression.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Customer_Review_Sentiment_Analyzer.git
```

### 2. Navigate to the project

```bash
cd Customer_Review_Sentiment_Analyzer
```

### 3. Create a virtual environment

```bash
python -m venv myenv
```

### 4. Activate the virtual environment

**Windows**

```bash
myenv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

Run

```bash
python train_model.py
```

This generates the trained model and TF-IDF vectorizer inside the **models** folder.

## Run the Application

```bash
python app.py
```

Open your browser and visit

```
http://127.0.0.1:5000/
```

Enter a customer review and click **Analyze Sentiment**.

## Example

**Input**

```
The product quality is amazing and delivery was very fast.
```

**Output**

```
Positive 😊
```

---

**Input**

```
Very poor quality. Completely disappointed.
```

**Output**

```
Negative 😞
```

## Future Enhancements

- Deep Learning (LSTM/BERT)
- Support for multiple languages
- Sentiment confidence score
- Graphical dashboard
- Deploy on Render or Heroku

## Author

**Shruthi**

B.Tech Graduate (Computer Science Engineering)

## License

This project is developed for educational purposes.
