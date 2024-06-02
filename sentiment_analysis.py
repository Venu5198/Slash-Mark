import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Now you can use SentimentIntensityAnalyzer as intended
analyzer = SentimentIntensityAnalyzer()
# Example usage:
scores = analyzer.polarity_scores("The movie was great!")
print(scores)
import nltk

# Download NLTK data (if needed)
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score

def get_sentiment_category(score):
    if score['compound'] >= 0.05:
        return 'Positive'
    elif score['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Sample Data (You can replace this with your actual data source)
data = [
    "The food was great and the service was excellent!",
    "I had a terrible experience. The food was cold and the waiter was rude.",
    "The place is okay, nothing special.",
    "Amazing food! Will definitely come back.",
    "Not worth the price."
]

# Create a DataFrame
df = pd.DataFrame(data, columns=['Review'])

# Analyze Sentiment
df['Sentiment Score'] = df['Review'].apply(analyze_sentiment)
df['Sentiment'] = df['Sentiment Score'].apply(get_sentiment_category)

# Print Results
print(df)

# Save the results to a CSV file
df.to_csv('sentiment_analysis_results.csv', index=False)
