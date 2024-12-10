# Movie Recommendation Chatbot 🎬

## Overview
This project features a chatbot that provides personalized movie recommendations using Natural Language Processing (NLP). The chatbot identifies user intents, such as genre preferences or recent movie suggestions, and responds with appropriate movie recommendations. It combines NLP techniques with machine learning and a modern web interface built with Streamlit.

---

## Features
- Recommends movies based on user input, such as genres, language or preferences.
- Recognizes user intents through trained NLP models.
- Logs conversation history for review or analysis.
- Easy-to-use interface with a visually appealing design.
- Flexible and easily extendable to include new intents or genres.

---

## Technologies Used
- **Python**: Core language for the application.
- **NLTK**: For text tokenization and preprocessing.
- **Scikit-learn**: For training the machine learning model.
- **Streamlit**: To build the interactive web app.
- **JSON**: To define intents and responses.

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```


### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data(If required)
```python
import nltk
nltk.download('punkt')
```

## Usage
To run the chatbot application, execute the following command:
```bash
streamlit run app.py
