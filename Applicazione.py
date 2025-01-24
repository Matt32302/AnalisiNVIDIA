import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------------------------------------------
# FUNZIONI DI SUPPORTO
# -------------------------------------------------------------------
def load_data():
    """
    Carica il dataset Titanic da seaborn.
    """
    df = sns.load_dataset("titanic")
    return df

def rename_columns(df):
    """
    Rinomina le colonne del dataset.
    """
    renamed_columns = {
        "survived": "Sopravvissuto",
        "pclass": "Classe Biglietto",
        "sex": "Sesso",
        "age": "Età",
        "sibsp": "Fratelli/Coniugi a Bordo",
        "parch": "Genitori/Figli a Bordo",
        "fare": "Prezzo Biglietto",
        "embarked": "Porto di Imbarco",
        "class": "Classe Cabina",
        "who": "Categoria Persona",
        "deck": "Ponte",
        "embark_town": "Città di Imbarco",
        "alive": "Sopravvissuto (Testo)",
        "alone": "Viaggiava da Solo"
    }
    df = df.rename(columns=renamed_columns)
    return df

def preprocess_data(df):
    """
    Esegue una pulizia e codifica dei dati di base:
    - Elimina righe con valori mancanti troppo critici (age o embarked)
    - Converte le variabili categoriche in numeriche
    - Ritorna il DataFrame con colonne rinominate.
    """
    # Rimuoviamo righe con NaN in colonne fondamentali (per semplicità)
    df = df.dropna(subset=["age", "embarked"])
    
    # Esempio di conversione di 'sex', 'embarked', 'class', ecc. in numerico
    label_cols = ['sex', 'embarked', 'class', 'who', 'adult_male', 'deck', 'embark_town', 'alone']
    for col in label_cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes  # Trasforma in codici numerici
    
    # Opzionale: si può decidere di rimuovere alcune colonne
    df = df.drop(columns=['alive', 'embark_town'])  # Esempio
    
    # Rinomina le colonne per una migliore leggibilità
    df = rename_columns(df)
    
    return df

def eda_section(df):
    """
    Sezione per l'Exploratory Data Analysis (EDA):
    - Mostra statistiche e visualizzazioni.
    """
    st.subheader("Statistiche descrittive")
    st.write(df.describe())
    
    st.subheader("Distribuzione delle variabili numeriche")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Permette all'utente di scegliere la variabile da visualizzare
    chosen_col = st.selectbox("Seleziona una colonna numerica per la distribuzione:", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x=chosen_col, kde=True, ax=ax, color='#FF6347', label=chosen_col)
    ax.set_title(f"Distribuzione di {chosen_col}", fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', title="Legenda")
    st.pyplot(fig)
    
    st.subheader("Heatmap di correlazione")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax, cbar_kws={'label': 'Correlazione'})
    ax.set_title("Matrice di Correlazione", fontsize=14, fontweight='bold')
    st.pyplot(fig)

def plot_categorical(df):
    """
    Visualizzazione interattiva per le variabili categoriche
    (in versione codificata), con etichette leggibili sull'asse X e nella legenda.
    """
    st.subheader("Analisi variabili categoriche (codificate)")
    
    # Trova le colonne categoriche (dopo la codifica sono numeriche, 
    # ma possiamo filtrare guardando l’originale o definendo un criterio)
    cat_cols = [c for c in df.columns if df[c].nunique() < 10 and c not in ['Sopravvissuto']]
    
    chosen_cat_col = st.selectbox("Seleziona una colonna categorica da visualizzare:", cat_cols)
    
    # Definizioni per le etichette leggibili sull'asse X e nella legenda
    x_labels = {
        "Sesso": {0: "Maschio", 1: "Femmina"},
        "Viaggiava da Solo": {0: "No", 1: "Sì"},
        "Sopravvissuto": {0: "No", 1: "Sì"},
        "Porto di Imbarco": {0: "Cherbourg (C)", 1: "Queenstown (Q)", 2: "Southampton (S)"},
    }
    
    # Sostituzione dei valori con etichette leggibili
    if chosen_cat_col in x_labels:
        df[chosen_cat_col] = df[chosen_cat_col].map(x_labels[chosen_cat_col])
        df["Sopravvissuto"] = df["Sopravvissuto"].map(x_labels["Sopravvissuto"])
    
    # Creazione del grafico
    fig = px.histogram(
        df,
        x=chosen_cat_col,
        color="Sopravvissuto",
        barmode='group',
        title=f"Distribuzione di {chosen_cat_col} per Sopravvissuto",
        labels={'Sopravvissuto': 'Sopravvivenza', chosen_cat_col: chosen_cat_col}
    )
    fig.update_layout(
        legend_title_text="Sopravvivenza",
        xaxis_title=chosen_cat_col
    )
    
    st.plotly_chart(fig, use_container_width=True)

def ml_section(df):
    """
    Sezione per il machine learning:
    - Selezione del modello (Logistic Regression o Random Forest)
    - Addestramento e valutazione
    """
    st.subheader("Modello di Classificazione")
    
    # Separazione feature-target
    X = df.drop(columns=['Sopravvissuto'])
    y = df['Sopravvissuto']
    
    # Train-test split
    test_size = st.slider("Seleziona la dimensione del test set:", 0.1, 0.9, 0.2, 0.05)
    random_state = st.number_input("Random State (riproducibilità)", min_value=0, max_value=100, value=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    
    # Selettore modello
    model_choice = st.radio("Seleziona il modello di classificazione:",
                            ("Logistic Regression", "Random Forest"))
    
    if model_choice == "Logistic Regression":
        # Parametri logistic regression
        C_value = st.number_input("C (inverso della regolarizzazione):", 0.01, 10.0, 1.0, 0.01)
        max_iter = st.slider("Numero massimo di iterazioni:", 100, 1000, 200, 50)
        
        model = LogisticRegression(C=C_value, max_iter=max_iter)
    
    else:  # Random Forest
        n_estimators = st.slider("Numero di alberi (n_estimators):", 50, 500, 100, 50)
        max_depth = st.slider("Profondità massima (max_depth):", 1, 20, 5, 1)
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       random_state=random_state)
    
    # Addestramento
    if st.button("Addestra il modello"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        st.success(f"**Accuracy sul test set**: {acc:.2f}")
        
        st.write("**Classification Report**:")
        st.text(classification_report(y_test, y_pred))
        
        # Matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax, cbar_kws={'label': 'Conteggio'})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold')
        st.pyplot(fig)

# -------------------------------------------------------------------
# STRUTTURA PRINCIPALE DELL'APP
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Progetto di Analisi Dati - Titanic",
                       layout="wide", initial_sidebar_state="expanded")
    
    st.title("Analisi Dati e Classificazione - Dataset Titanic")
    st.markdown("""
    Benvenuto/a in questa **Web App** di analisi esplorativa e **machine learning** sul
    celebre **Titanic dataset**. Esplora i grafici e prova un modello di classificazione!
    """)

    df = load_data()
    
    if st.checkbox("Mostra le prime righe del dataset"):
        st.dataframe(df.head())
    
    df_preprocessed = preprocess_data(df)
    
    st.sidebar.title("Navigazione")
    section = st.sidebar.radio("Vai alla sezione:", 
                               ("EDA (Analisi Esplorativa)",
                                "Analisi Categorie",
                                "Machine Learning"))
    
    if section == "EDA (Analisi Esplorativa)":
        eda_section(df_preprocessed)
    
    elif section == "Analisi Categorie":
        plot_categorical(df_preprocessed)
    
    else:  # "Machine Learning"
        ml_section(df_preprocessed)

if __name__ == "__main__":
    main()