import os

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

st.set_page_config(
    page_title="LangChain + Streamlit + PostgreSQL",
    page_icon="🐍",
    layout="centered",
)

st.title("Entorno Dev: LangChain + Streamlit + PostgreSQL")

database_url = os.getenv("DATABASE_URL")

st.subheader("1. Prueba de conexión a PostgreSQL")

try:
    engine = create_engine(database_url)

    with engine.connect() as connection:
        result = connection.execute(text("SELECT version();"))
        postgres_version = result.scalar()

    st.success("Conexión exitosa a PostgreSQL")
    st.code(postgres_version)

except Exception as error:
    st.error("No se pudo conectar a PostgreSQL")
    st.exception(error)


st.subheader("2. Probar modelo con LangChain")

provider = st.selectbox(
    "Selecciona proveedor",
    ["OpenAI", "Google Gemini"],
)

prompt = st.text_area(
    "Escribe un prompt",
    value="Explícame qué es LangChain en una frase sencilla.",
)

if st.button("Enviar"):
    try:
        if provider == "OpenAI":
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.2,
            )

        else:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.2,
            )

        response = llm.invoke(prompt)

        st.subheader("Respuesta")
        st.write(response.content)

    except Exception as error:
        st.error("Error ejecutando el modelo")
        st.exception(error)