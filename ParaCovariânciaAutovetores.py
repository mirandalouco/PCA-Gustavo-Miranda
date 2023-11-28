import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Carregar a base de dados a partir do arquivo CSV
df = pd.read_csv('C:/Users/gusta/Documents/faculdade/matemática/PCA/ChampionsNormalizado.csv', sep=';')

# Substituir vírgulas por pontos nas colunas numéricas
numeric_columns = df.columns[1:]  # Excluindo a coluna 'champion'
df[numeric_columns] = df[numeric_columns].replace(',', '.', regex=True).astype(float)

# Separar os dados de entrada (features)
X = df.drop(['champion'], axis=1)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar o PCA
pca = PCA()
principal_components = pca.fit_transform(X_scaled)

# Obter os autovalores e autovetores
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# Criar um DataFrame com os componentes principais
columns = [f'PC{i+1}' for i in range(len(X.columns))]
df_pca = pd.DataFrame(data=principal_components, columns=columns)

# Adicionar a coluna 'champion' ao DataFrame resultante
df_pca['champion'] = df['champion']


# Plotar o gráfico de dispersão
plt.figure(figsize=(10, 8))
plt.scatter(df_pca['PC1'], df_pca['PC2'], c='blue', alpha=0.5)
plt.title('PCA - Scatter Plot dos Dois Primeiros Componentes Principais')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Adicionar rótulos para cada ponto
for i, txt in enumerate(df_pca['champion']):
    plt.annotate(txt, (df_pca['PC1'][i], df_pca['PC2'][i]))

# Configurar limites dos eixos para incluir valores negativos
plt.xlim(left=df_pca['PC1'].min() - 1, right=df_pca['PC1'].max() + 1)
plt.ylim(bottom=df_pca['PC2'].min() - 1, top=df_pca['PC2'].max() + 1)

plt.title('PCA - Scatter Plot dos Dois Primeiros Componentes Principais')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.show()

# Imprimir os maiores autovalores e autovetores
print("Maiores autovalores:")
print(eigenvalues)
print("\nAutovetores correspondentes:")
print(eigenvectors)

# Imprimir a matriz de covariância original
cov_matrix = np.cov(X, rowvar=False)
print("\nMatriz de Covariância Original:")
print(cov_matrix)

# Extrair uma matriz quadrada reduzida (por exemplo, primeiras 5 linhas e colunas)
matriz_cov_reduzida = cov_matrix[:5, :5]

# Exibir a matriz de covariância reduzida
print("\nMatriz de Covariância Reduzida:")
print(matriz_cov_reduzida)




