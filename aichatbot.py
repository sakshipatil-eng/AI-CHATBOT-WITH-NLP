import nltk
import numpy as np
import random
import string
import warnings

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")


corpus = [
    "Hello, how can I help you?",
    "Hi there!",
    "I am an AI chatbot created using NLP.",
    "I can answer your basic questions.",
    "Python is a popular programming language.",
    "Machine learning is a part of artificial intelligence.",
    "Natural language processing helps machines understand human language.",
    "You can ask me about Python, AI, or NLP.",
    "Goodbye! Have a nice day."
]


lemmatizer = WordNetLemmatizer()

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return tokens


greeting_inputs = ("hello", "hi", "hey", "good morning", "good evening")
greeting_responses = ["Hello!", "Hi!", "Hey there!", "Greetings!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)


def chatbot_response(user_input):
    corpus.append(user_input)

    vectorizer = TfidfVectorizer(
        tokenizer=preprocess,
        lowercase=False,
        stop_words=None
    )

    tfidf = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf[-1], tfidf)
    idx = similarity.argsort()[0][-2]

    flat = similarity.flatten()
    flat.sort()

    if flat[-2] == 0:
        response = "Sorry, I didn't understand that."
    else:
        response = corpus[idx]

    corpus.pop()
    return response


print("🤖 AI Chatbot (Type 'exit' to stop)")

while True:
    user_input = input("You: ").lower()

    if user_input == "exit":
        print("Bot: Goodbye! 👋")
        break

    greet = greeting(user_input)
    if greet:
        print("Bot:", greet)
    else:
        print("Bot:", chatbot_response(user_input))
