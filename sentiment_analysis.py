import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data if not already
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv("tweets.csv")  # Update the path if needed
print("Data Shape:", df.shape)
print(df.head())

# Data cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Check class distribution
print("Sentiment value counts:\n", df['sentiment'].value_counts())

# Basic visualization
sns.countplot(data=df, x='sentiment')
plt.title("Sentiment Distribution")
plt.show()

# Text preprocessing
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)           # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)          # Remove special characters/numbers
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
print("Sample cleaned text:\n", df['clean_text'].head())

# WordCloud for positive and negative tweets
positive_words = " ".join(df[df['sentiment'] == 'positive']['clean_text'])
negative_words = " ".join(df[df['sentiment'] == 'negative']['clean_text'])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(WordCloud(width=400, height=300).generate(positive_words))
plt.title("Positive Tweets")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(WordCloud(width=400, height=300).generate(negative_words))
plt.title("Negative Tweets")
plt.axis("off")

plt.show()

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
