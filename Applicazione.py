import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# Titolo dell'app
st.title("App Web per Visualizzare Grafici")

# Selezione del tipo di grafico
chart_type = st.sidebar.selectbox(
    "Seleziona il tipo di grafico:",
    ["Line Chart", "Bar Chart", "Scatter Plot"]
)

# Generazione di dati casuali per i grafici
def generate_data():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, len(x))
    return x, y

# Creazione di un dataset per i grafici a barre o scatter
@st.cache
def create_dataframe():
    np.random.seed(42)
    return pd.DataFrame({
        "X": np.random.rand(50) * 10,
        "Y": np.random.rand(50) * 10,
        "Category": np.random.choice(["A", "B", "C"], size=50)
    })

if chart_type == "Line Chart":
    # Generazione dei dati
    x, y = generate_data()

    # Creazione del grafico con Matplotlib
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Sinusoide con rumore")
    ax.set_title("Grafico a Linee")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    # Visualizzazione del grafico
    st.pyplot(fig)

elif chart_type == "Bar Chart":
    # Creazione di un dataframe
    df = create_dataframe()

    # Creazione di un grafico a barre con Plotly
    bar_fig = px.bar(df, x="Category", y="X", color="Category", title="Grafico a Barre")

    # Visualizzazione del grafico
    st.plotly_chart(bar_fig)

elif chart_type == "Scatter Plot":
    # Creazione di un dataframe
    df = create_dataframe()

    # Creazione di un grafico scatter con Plotly
    scatter_fig = px.scatter(df, x="X", y="Y", color="Category", title="Scatter Plot")

    # Visualizzazione del grafico
    st.plotly_chart(scatter_fig)

# Footer
st.write("App sviluppata con Streamlit e Python")
