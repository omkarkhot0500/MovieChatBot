import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL issue for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file with the correct encoding
file_path = os.path.abspath("./intents.json")
with open(file_path, "r", encoding='utf-8') as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

counter = 0

def main():
    global counter
    st.set_page_config(page_title="Movie Recommendation Chatbot", page_icon="🎬", layout="wide")  # Set page title and layout
    st.markdown("""
        <style>
            .header {
                font-size: 40px;
                color: #FF6347;
                text-align: center;
            }
            .subheader {
                font-size: 30px;
                color: #32CD32;
            }
            .sidebar .sidebar-content {
                background-color: #222222;
                color: white;
            }
            .stTextInput input {
                font-size: 16px;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #32CD32;
            }
            .stButton button {
                background-color: #32CD32;
                color: white;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 8px;
                border: none;
            }
            .stTextArea textarea {
                font-size: 18px;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid #32CD32;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Movie Recommendation Chatbot 🎬")  # App Title

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.write("### Let's Talk Movies! 🎥🍿")

        # Check if the chat_log.csv file exists, and if not, create it with column names
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        counter += 1
        user_input = st.text_input("You:", key=f"user_input_{counter}", placeholder="Ask me about a movie or genre!")

        if user_input:
            # Convert the user input to a string
            user_input_str = str(user_input)

            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            # Save the user input and chatbot response to the chat_log.csv file
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day! 😄")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        # Display the conversation history in a collapsible expander
        st.header("Conversation History 🗣️")
        with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                st.text(f"User: {row[0]}")
                st.text(f"Chatbot: {row[1]}")
                st.text(f"Timestamp: {row[2]}")
                st.markdown("---")

    # About Menu
    elif choice == "About":
        st.write("### About this Project 📚")

        st.write("""
            The Movie Recommendation Chatbot uses NLP techniques to recommend movies based on user input. 
            It was built using the following technologies:
            - **Natural Language Processing (NLP)** for understanding user input.
            - **Logistic Regression** for training the model.
            - **Streamlit** for creating the interactive interface.
        """)

        st.subheader("How it works:")

        st.write("""
            The chatbot recognizes various movie-related intents from the user (e.g., genre preferences, recent films, etc.) 
            and responds with movie recommendations from different genres and languages.
        """)

        st.subheader("Tech Stack:")

        st.write("""
            - **Python**: Programming language.
            - **Streamlit**: For building the web app interface.
            - **Scikit-learn**: Used for training the NLP model.
            - **NLTK**: For Natural Language Processing (tokenization and intent recognition).
        """)

        st.subheader("Future Enhancements:")

        st.write("""
            - Integrate with a movie database (like TMDB or IMDB) to provide real-time movie information.
            - Use deep learning models for more accurate intent recognition and recommendation.
        """)

if __name__ == '__main__':
    main()
