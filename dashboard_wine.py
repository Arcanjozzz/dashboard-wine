import streamlit as st
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np


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
# Upload de um CSV alternativo
# -----------------------------
st.divider()
st.subheader("Enviar outro CSV")
arquivo = st.file_uploader("Envie um CSV")
if arquivo:
    df_upload = pd.read_csv(arquivo)
    st.dataframe(df_upload)


# -----------------------------
# Agrupamento de vinhos com DBSCAN
# -----------------------------
st.divider()
st.header("🍷 Agrupamento de vinhos com DBSCAN")

st.write(
    "O DBSCAN identifica grupos de vinhos com características "
    "químicas semelhantes e detecta amostras isoladas."
)

# Selecionar os atributos químicos
X_dbscan = dados[colunas_numericas]

# Padronizar os dados
scaler_dbscan = StandardScaler()
X_scaled = scaler_dbscan.fit_transform(X_dbscan)

# Configurar os parâmetros
eps = st.slider(
    "Distância (eps)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1
)

min_samples = st.slider(
    "Mínimo de amostras",
    min_value=2,
    max_value=15,
    value=5
)

# Executar o DBSCAN
modelo_dbscan = DBSCAN(
    eps=eps,
    min_samples=min_samples
)

clusters = modelo_dbscan.fit_predict(X_scaled)

# Mostrar resultados
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_ruidos = int((clusters == -1).sum())

col1, col2, col3 = st.columns(3)
col1.metric("Grupos encontrados", n_clusters)
col2.metric("Amostras de ruído", n_ruidos)
col3.metric("Total de vinhos", len(dados))

# Reduzir os dados para duas dimensões
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

grafico = pd.DataFrame({
    "Componente 1": X_pca[:, 0],
    "Componente 2": X_pca[:, 1],
    "Grupo": [
        "Ruído" if c == -1 else f"Grupo {c}"
        for c in clusters
    ]
})

st.subheader("Visualização dos grupos")
st.scatter_chart(
    grafico,
    x="Componente 1",
    y="Componente 2",
    color="Grupo"
)

# Comparar com as classes originais
comparacao = pd.crosstab(
    dados["class"],
    clusters,
    rownames=["Classe original"],
    colnames=["Grupo DBSCAN"]
)

st.subheader("Comparação com as classes originais")
st.dataframe(comparacao)


# -----------------------------
# Método do Cotovelo para DBSCAN
# -----------------------------
st.divider()
st.header("📈 Método do Cotovelo para escolher o eps")

st.write(
    "O gráfico mostra as distâncias até os vizinhos mais próximos. "
    "O ponto de cotovelo ajuda a estimar um valor adequado para o eps."
)

# Calcular as distâncias dos vizinhos
k = 5

vizinhos = NearestNeighbors(n_neighbors=k)
vizinhos.fit(X_scaled)

distancias, indices = vizinhos.kneighbors(X_scaled)

# Ordenar as distâncias do k-ésimo vizinho
distancias_ordenadas = np.sort(distancias[:, k - 1])

# Exibir o gráfico
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(distancias_ordenadas)
ax.set_title("Gráfico do Cotovelo (k-distance)")
ax.set_xlabel("Amostras ordenadas")
ax.set_ylabel("Distância até o 5º vizinho")
ax.grid(True)

st.pyplot(fig)
plt.close(fig)

