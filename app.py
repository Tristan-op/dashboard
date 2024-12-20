import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import subprocess

# Installation forcée de sentencepiece si nécessaire (pour compatibilité Streamlit Cloud)
try:
    import sentencepiece
except ImportError:
    subprocess.check_call(["pip", "install", "sentencepiece==0.1.99"])

# Chemin vers le modèle et chargement des composants du modèle
model_path = './model'
# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Charger le modèle
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Charger les données avec les colonnes spécifiées
data_path = 'data/data_p9.csv'
data = pd.read_csv(data_path, header=None, encoding='ISO-8859-1')
data.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Configuration de la page et du menu de navigation
st.set_page_config(page_title="Dashboard d'Analyse de Tweets", layout="wide")
st.title("Dashboard d'Analyse de Sentiment des Tweets")

st.sidebar.title("Navigation")
option = st.sidebar.radio("Sélectionnez une page :", ["Accueil", "Analyse des données", "Prédiction d'une ligne de données", "Prédiction d'un tweet personnalisé"])

# Page d'accueil
if option == "Accueil":
    st.write("Bienvenue sur le dashboard d'analyse de sentiment des tweets. Utilisez la barre de navigation pour explorer les données, tester des prédictions ou saisir un tweet personnalisé.")

# Page d'analyse des données
elif option == "Analyse des données":
    st.title("Analyse Exploratoire des Données")

    # Statistiques Descriptives
    if st.button("Afficher les statistiques descriptives"):
        st.subheader("Statistiques Descriptives")
        st.write(data[['target', 'text']].describe())
        st.write("Nombre total de tweets : ", len(data))
        sentiment_counts = data['target'].value_counts()
        st.write("Proportion de sentiments positifs et négatifs :")
        st.write(sentiment_counts)

    # Graphique de distribution des sentiments
    if st.button("Afficher la distribution des sentiments"):
        st.subheader("Distribution des Sentiments")
        sentiment_counts = data['target'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=['blue', 'red'])
        ax.set_title("Proportion de Tweets Positifs et Négatifs")
        ax.set_xlabel("Sentiment (0 = Négatif, 4 = Positif)")
        ax.set_ylabel("Nombre de Tweets")
        st.pyplot(fig)

    # Nuage de mots pour les tweets positifs et négatifs
    if st.button("Afficher les WordClouds (Positifs et Négatifs)"):
        st.subheader("Nuage de Mots : Tweets Positifs et Négatifs (limité à 50 000 chacun)")

        # Limiter les tweets positifs et négatifs à 50 000 maximum
        positive_tweets = data[data['target'] == 4]['text'].head(50000)
        negative_tweets = data[data['target'] == 0]['text'].head(50000)

        # Combiner les tweets en une seule chaîne de texte
        positive_text = " ".join(positive_tweets.astype(str))
        negative_text = " ".join(negative_tweets.astype(str))

        # Générer le WordCloud pour les tweets positifs
        positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

        # Générer le WordCloud pour les tweets négatifs
        negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

        # Afficher les WordClouds
        st.write("**WordCloud pour les Tweets Positifs :**")
        fig_positive, ax_positive = plt.subplots(figsize=(10, 5))
        ax_positive.imshow(positive_wordcloud, interpolation="bilinear")
        ax_positive.axis("off")
        st.pyplot(fig_positive)

        st.write("**WordCloud pour les Tweets Négatifs :**")
        fig_negative, ax_negative = plt.subplots(figsize=(10, 5))
        ax_negative.imshow(negative_wordcloud, interpolation="bilinear")
        ax_negative.axis("off")
        st.pyplot(fig_negative)

    # Analyse de la longueur des mots et des phrases par sentiment
    if st.button("Afficher l'analyse de la longueur des mots et des phrases"):
        st.subheader("Longueur des Mots et des Phrases par Sentiment")
       
        data['word_count'] = data['text'].apply(lambda x: len(x.split()))
        data['char_count'] = data['text'].apply(len)
       
        avg_word_count = data.groupby('target')['word_count'].mean()
        avg_char_count = data.groupby('target')['char_count'].mean()
       
        fig, ax = plt.subplots()
        avg_word_count.plot(kind='bar', color=['blue', 'red'], ax=ax)
        ax.set_title("Nombre Moyen de Mots par Sentiment")
        ax.set_xlabel("Sentiment (0 = Négatif, 4 = Positif)")
        ax.set_ylabel("Nombre moyen de mots par phrase")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        avg_char_count.plot(kind='bar', color=['blue', 'red'], ax=ax)
        ax.set_title("Longueur Moyenne des Phrases par Sentiment")
        ax.set_xlabel("Sentiment (0 = Négatif, 4 = Positif)")
        ax.set_ylabel("Nombre moyen de caractères")
        st.pyplot(fig)

    # Hashtags les plus fréquents
    if st.button("Afficher les hashtags les plus fréquents"):
        st.subheader("Analyse des Hashtags les plus Fréquents par Sentiment")

        # Extraire les hashtags pour les tweets positifs et négatifs
        positive_hashtags = data[data['target'] == 4]['text'].str.findall(r"#\w+").explode()
        negative_hashtags = data[data['target'] == 0]['text'].str.findall(r"#\w+").explode()

        # Calculer les hashtags les plus fréquents
        positive_hashtag_counts = positive_hashtags.value_counts().head(20)
        negative_hashtag_counts = negative_hashtags.value_counts().head(20)

        # Afficher les résultats pour les tweets positifs
        st.write("**Top 20 hashtags dans les tweets positifs :**")
        st.bar_chart(positive_hashtag_counts)

        # Afficher les résultats pour les tweets négatifs
        st.write("**Top 20 hashtags dans les tweets négatifs :**")
        st.bar_chart(negative_hashtag_counts)

    # Distribution des Heures de Publication
    if st.button("Afficher la distribution des heures de publication"):
        st.subheader("Distribution des Heures de Publication")
       
        data['hour'] = pd.to_datetime(data['date']).dt.hour
        hour_dist = data.groupby(['hour', 'target']).size().unstack().fillna(0)
       
        fig, ax = plt.subplots()
        hour_dist.plot(kind='bar', stacked=True, color=['blue', 'red'], ax=ax)
        ax.set_title("Distribution des Tweets par Heure et Sentiment")
        ax.set_xlabel("Heure")
        ax.set_ylabel("Nombre de Tweets")
        st.pyplot(fig)

# Fonction pour prédire le sentiment (0 = négatif, 1 = positif)
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    return "Positif" if predicted_class == 1 else "Négatif"

# Fonction pour prétraiter le texte
def preprocess_text(text):
    return re.sub(r'[^A-Za-z0-9\s]', '', text)

# Page de prédiction d'une ligne de données
if option == "Prédiction d'une ligne de données":
    st.title("Prédiction d'une Ligne de Données")

    # Entrée du numéro de ligne
    tweet_index = st.number_input("Entrez le numéro de la ligne du tweet (0 à {})".format(len(data) - 1), min_value=0, max_value=len(data) - 1, step=1)

    # Affichage du tweet sélectionné et de sa cible
    tweet_data = data.iloc[tweet_index]
    tweet_text = tweet_data['text']
    tweet_target = "Positif" if tweet_data['target'] == 4 else "Négatif"
    
    # Prétraitement du texte - suppression des caractères spéciaux
    preprocessed_text = preprocess_text(tweet_text)
    
    # Affichage du texte original, cible et texte prétraité
    st.write("Tweet Original :", tweet_text)
    st.write("Cible (target) :", tweet_target)
    st.write("Texte Prétraité :", preprocessed_text)
    
    # Prédiction du sentiment en utilisant le modèle
    sentiment_label = predict_sentiment(preprocessed_text)
    
    # Affichage des résultats
    st.subheader("Résultat de la Prédiction")
    st.write("Sentiment Prédit :", sentiment_label)

# Page de prédiction d'un tweet personnalisé
if option == "Prédiction d'un tweet personnalisé":
    st.title("Prédiction pour un Tweet Personnalisé")

    # Saisie de texte pour un tweet personnalisé
    user_input = st.text_input("Entrez un tweet pour prédiction")

    if user_input:
        # Prétraitement du texte
        preprocessed_text = preprocess_text(user_input)
        st.write("Texte Prétraité :", preprocessed_text)

        # Prédiction du sentiment
        sentiment_label = predict_sentiment(preprocessed_text)

        # Affichage des résultats
        st.subheader("Résultat de la Prédiction")
        st.write("Sentiment :", sentiment_label)
