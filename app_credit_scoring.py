import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------
# Classe usada no pipeline salvo (model_final.pkl)
# ---------------------------
class OutlierCapperNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_ = Q1 - self.multiplier * IQR
        self.upper_ = Q3 + self.multiplier * IQR
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, self.lower_, self.upper_)
        return X


# ---------------------------
# Configurações iniciais do app
# ---------------------------
st.set_page_config(
    page_title="Credit Scoring - Projeto Final Módulo 38",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Projeto Final – Credit Scoring (Módulo 38)")
st.write(
    """
    Esta aplicação carrega um arquivo CSV com a base de clientes, 
    aplica o **pipeline de pré-processamento + Regressão Logística** 
    treinado no notebook (`model_final.pkl`) e retorna a escoragem 
    (probabilidade de mau pagador).
    """
)

# Caminho do modelo treinado (Exercício 15)
MODEL_PATH = Path("artifacts/model_final.pkl")

# Colunas especiais (iguais às usadas no notebook)
TARGET_COL = "mau"
IGNORE_COLS = ["index", "data_ref"]  # não usadas como preditoras


# ---------------------------
# Funções auxiliares
# ---------------------------
@st.cache_resource
def load_model():
    """Carrega o modelo treinado (pipeline completo)."""
    if not MODEL_PATH.exists():
        st.error(f"Arquivo de modelo não encontrado em: {MODEL_PATH}")
        return None
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_data
def load_uploaded_csv(file) -> pd.DataFrame:
    """Carrega o CSV enviado pelo usuário em um DataFrame."""
    df = pd.read_csv(file)
    return df


def preparar_dados_para_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara o DataFrame de entrada para ser usado no modelo:
    - remove colunas de target e de identificação (se existirem).
    """
    cols_a_remover = [c for c in [TARGET_COL] + IGNORE_COLS if c in df.columns]
    X = df.drop(columns=cols_a_remover, errors="ignore")
    return X


# ---------------------------
# Carregar modelo
# ---------------------------
modelo = load_model()
if modelo is None:
    st.stop()

st.sidebar.header("Configurações")
st.sidebar.write("1. Faça upload de um arquivo CSV com a base de clientes.")
st.sidebar.write("2. O arquivo deve ter as mesmas colunas usadas no treino.")
st.sidebar.write("3. O modelo usado aqui é o `model_final.pkl` (Regressão Logística).")


# ---------------------------
# Upload do CSV
# ---------------------------
st.subheader("📂 Upload do arquivo CSV")

uploaded_file = st.file_uploader(
    "Envie um arquivo CSV com a base de clientes para escoragem:",
    type=["csv"]
)

if uploaded_file is not None:
    # 1) Ler dados
    df_input = load_uploaded_csv(uploaded_file)
    st.write("✅ Arquivo carregado com sucesso!")
    st.write("Prévia dos dados:")
    st.dataframe(df_input.head())

    # 2) Preparar X para o modelo
    X_input = preparar_dados_para_modelo(df_input)

    st.write("Colunas usadas pelo modelo (após remoção de target e identificadores):")
    st.write(list(X_input.columns))

    # 3) Gerar previsões
    with st.spinner("Gerando escore de crédito..."):
        try:
            # predict_proba retorna probabilidade de cada classe
            # [:, 1] -> probabilidade de mau (classe positiva)
            proba_mau = modelo.predict_proba(X_input)[:, 1]
            pred_class = modelo.predict(X_input)
        except Exception as e:
            st.error(
                "Erro ao tentar gerar previsões. "
                "Verifique se as colunas do CSV são compatíveis com o modelo treinado."
            )
            st.exception(e)
        else:
            # 4) Montar DataFrame de resultado
            df_result = df_input.copy()
            df_result["score_mau"] = proba_mau
            df_result["classe_prevista"] = pred_class

            st.subheader("📊 Resultado da Escoragem")
            st.write(
                "A coluna **`score_mau`** representa a probabilidade prevista "
                "de inadimplência (mau pagador)."
            )
            st.dataframe(df_result.head())

            # 5) Download do resultado como CSV
            csv_output = df_result.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="⬇️ Baixar arquivo escorado em CSV",
                data=csv_output,
                file_name="base_escorada_credit_scoring.csv",
                mime="text/csv"
            )
else:
    st.info("🔹 Aguarde o upload de um arquivo CSV para realizar a escoragem.")

