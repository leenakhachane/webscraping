# webscraping
python code for web scraping 
Assisment test for nonstop io.pvt.ltd  
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Function to scrape news articles from New York Timesdef scrape_nytimes_news():
    url = """http://timesofindia.indiatimes.com/world/china/chinese-expert-warns-of-troops-entering-kashmir/articleshow/59516912.cms"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    articles = soup.find_all('article')[:100]  # Limiting to 100 articles for this example

    data = {'content': [], 'section': []}

    for article in articles:
        content = article.find('p', class_='css-1gb49r8 e1voiwgp0').text
        section = article.find('a', class_='css-1gh531p').text  # Assuming the section is available in the link text

        data['content'].append(content)
        data['section'].append(section)

    return pd.DataFrame(data)

# Function to classify news using a text classification model
def classify_news(df):
    # Check if the dataset has enough samples for the split
    if len(df) == 0:
        raise ValueError("Not enough samples in the dataset for train-test split.")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['content'], df['section'], test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Make predictions
    predictions = classifier.predict(X_test_vectorized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report

# Main script
if __name__ == "__main__":
    # Scrape news data
    news_df = scrape_nytimes_news()

    # Check if the dataset has enough samples for the split
    if len(news_df) == 0:
        print("Not enough samples in the dataset for train-test split.")
    else:
        # Classify news using the model
        accuracy, report = classify_news(news_df)

        # Display results
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", report)

    accuracy, report = classify_news(news_df)

    # Display results
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:\n", report)
