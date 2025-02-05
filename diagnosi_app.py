import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Scaricare risorse di NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Funzione per caricare il dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df = df.drop(columns=["Unnamed: 0"], errors='ignore')
    return df

# Funzione per la pulizia del testo
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

# Caricamento del dataset
file_path = r'C:\Users\maria\OneDrive\Desktop\ESAME DUCCESCHI\Train_data.csv'
dataset = load_dataset(file_path)

dataset['cleaned_text'] = dataset['symptoms'].apply(clean_text)
dataset['diagnosis']= dataset['diagnosis'].str.capitalize()

# Vettorizzazione con TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(dataset['cleaned_text'])

# Addestramento del modello
model = RandomForestClassifier(random_state=42)
model.fit(X_tfidf, dataset['diagnosis'])

# Impostazioni dell'interfaccia Streamlit
st.set_page_config(page_title="Diagnosi Medica Automatica", page_icon="💊", layout="centered")

# Stile CSS personalizzato per migliorare la grafica
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextArea, .stButton > button {
        border-radius: 12px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# Titolo dell'app
st.title("💊 Predizione Diagnosi Medica")
st.markdown("Inserisci i tuoi sintomi e ricevi una diagnosi automatica basata sull'intelligenza artificiale.")

# Barra laterale con informazioni
with st.sidebar:
    st.image("https://source.unsplash.com/300x200/?medical,doctor")
    st.subheader("📌 Istruzioni:")
    st.write("- Inserisci i sintomi in forma testuale.")
    st.write("- Clicca sul pulsante per ottenere una previsione.")
    st.write("- Le diagnosi sono ordinate per probabilità.")

# Input utente
user_input = st.text_area("✍️ Inserisci i tuoi sintomi:", height=150)

# Bottone di predizione
if st.button("🔍 Prevedi Diagnosi"):
    if user_input:
        cleaned_input = clean_text(user_input)
        user_input_tfidf = tfidf_vectorizer.transform([cleaned_input])

        prediction = model.predict(user_input_tfidf)
        probabilities = model.predict_proba(user_input_tfidf)[0]

        sorted_indices = probabilities.argsort()[::-1]
        sorted_diagnoses = [(model.classes_[i], probabilities[i]) for i in sorted_indices]

        # Risultati
        st.subheader("📋 Diagnosi Predette:")
        for diagnosis, prob in sorted_diagnoses:
            st.write(f"✅ **{diagnosis}**: {prob:.2f}")
    else:
        st.warning("⚠️ Per favore, inserisci dei sintomi.")

# Footer
st.markdown("""
    ---
    *⚕️ Questo strumento è a scopo informativo e non sostituisce una consulenza medica professionale.*
""")

