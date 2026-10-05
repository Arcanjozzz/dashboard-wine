import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Dashboard Wine Dataset", layout="wide")

# -----------------------------
# Carregamento dos dados
# -----------------------------
dados = pd.read_csv('Wine dataset.csv')
dados.columns = dados.columns.str.strip()  # remove espaços extras (ex: "Proline ")

st.title('FATEC COTIA')
st.header('Curso de Ciência de Dados')
st.subheader('Dashboard - Wine Dataset')

st.markdown(
    "Dataset clássico do UCI com **178 amostras** de vinhos, divididas em "
    "**3 classes (cultivares)**, descritas por **13 atributos químicos** "
    "(teor alcoólico, ácido málico, magnésio, flavonoides, etc.)."
)

# -----------------------------
# Sidebar - filtros e infos do usuário
# -----------------------------
st.sidebar.header("Filtros")

nome = st.sidebar.text_input("Digite seu nome")
idade = st.sidebar.number_input("Idade", min_value=0, step=1)

classes_disponiveis = sorted(dados["class"].unique())
classes_selecionadas = st.sidebar.multiselect(
    "Classe do vinho", classes_disponiveis, default=classes_disponiveis
)

colunas_numericas = [c for c in dados.columns if c != "class"]
atributo = st.sidebar.selectbox("Atributo para os gráficos", colunas_numericas)

faixa_alcool = st.sidebar.slider(
    "Faixa de Álcool",
    float(dados["Alcohol"].min()),
    float(dados["Alcohol"].max()),
    (float(dados["Alcohol"].min()), float(dados["Alcohol"].max())),
)

if nome:
    st.sidebar.write(f"Olá, {nome}! Idade: {idade}")

# -----------------------------
# Aplicar filtros
# -----------------------------
df_filtrado = dados[
    dados["class"].isin(classes_selecionadas)
    & dados["Alcohol"].between(*faixa_alcool)
]

# -----------------------------
# Métricas (KPIs)
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Amostras filtradas", len(df_filtrado))
col2.metric("Álcool médio", f'{df_filtrado["Alcohol"].mean():.2f}' if len(df_filtrado) else "-")
col3.metric("Flavonoides (média)", f'{df_filtrado["Flavanoids"].mean():.2f}' if len(df_filtrado) else "-")
col4.metric("Classes presentes", df_filtrado["class"].nunique())

st.divider()

# -----------------------------
# Tabela de dados
# -----------------------------
st.subheader("Tabela de dados filtrados")
st.dataframe(df_filtrado)

# -----------------------------
# Gráficos
# -----------------------------
st.subheader(f"Distribuição de '{atributo}' por classe")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**Gráfico de linha**")
    st.line_chart(df_filtrado[atributo])

with col_b:
    st.markdown("**Gráfico de barras (média por classe)**")
    media_por_classe = df_filtrado.groupby("class")[atributo].mean()
    st.bar_chart(media_por_classe)

st.subheader("Dispersão: Álcool x Flavonoides")
st.scatter_chart(df_filtrado, x="Alcohol", y="Flavanoids", color="class")

st.subheader("Matriz de correlação entre atributos")
st.dataframe(df_filtrado[colunas_numericas].corr().style.background_gradient(cmap="RdBu", axis=None))

# -----------------------------
# Modelo de Machine Learning - SVM
# -----------------------------
st.divider()
st.subheader("🤖 Classificação de vinhos com SVM")

st.markdown(
    """
    O **SVM (Support Vector Machine)** é um algoritmo de Machine Learning
    supervisionado usado para classificação. Neste projeto, ele aprende a
    diferenciar as **3 classes de vinho** usando os 13 atributos químicos
    do dataset.
    """
)

# Para o modelo, usamos todos os atributos químicos e a coluna "class" como alvo.
X = dados[colunas_numericas]
y = dados["class"]

# Separação dos dados em treinamento e teste.
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Padronização: importante para SVM porque os atributos estão em escalas diferentes.
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

# Treinamento do SVM.
modelo_svm = SVC(kernel="rbf", C=1.0, random_state=42)
modelo_svm.fit(X_treino_scaled, y_treino)

# Previsões e avaliação.
y_pred = modelo_svm.predict(X_teste_scaled)
acuracia = accuracy_score(y_teste, y_pred)

met1, met2 = st.columns(2)
met1.metric("Acurácia no teste", f"{acuracia:.2%}")
met2.metric("Amostras usadas no teste", len(X_teste))

st.markdown("**Matriz de confusão**")
classes_modelo = sorted(y.unique())
matriz = confusion_matrix(y_teste, y_pred, labels=classes_modelo)
matriz_df = pd.DataFrame(
    matriz,
    index=[f"Real {c}" for c in classes_modelo],
    columns=[f"Prevista {c}" for c in classes_modelo],
)
st.dataframe(matriz_df)

st.markdown("**Relatório de classificação**")
relatorio = classification_report(
    y_teste,
    y_pred,
    labels=classes_modelo,
    target_names=[f"Classe {c}" for c in classes_modelo],
    output_dict=True,
    zero_division=0,
)
st.dataframe(pd.DataFrame(relatorio).transpose().round(2))

st.markdown("**Teste de uma amostra do dataset**")
indice_amostra = st.number_input(
    "Escolha o número da amostra (0 a 177)",
    min_value=0,
    max_value=len(dados) - 1,
    value=0,
    step=1,
)

amostra = dados.loc[[indice_amostra], colunas_numericas]
classe_real = int(dados.loc[indice_amostra, "class"])
classe_prevista = int(modelo_svm.predict(scaler.transform(amostra))[0])

res1, res2 = st.columns(2)
res1.metric("Classe real", classe_real)
res2.metric("Classe prevista pelo SVM", classe_prevista)

if classe_real == classe_prevista:
    st.success("O SVM classificou esta amostra corretamente! ✅")
else:
    st.error("O SVM classificou esta amostra de forma diferente da classe real.")

# -----------------------------
# Upload de um CSV alternativo
# -----------------------------
st.divider()
st.subheader("Enviar outro CSV")
arquivo = st.file_uploader("Envie um CSV")
if arquivo:
    df_upload = pd.read_csv(arquivo)
    st.dataframe(df_upload)
