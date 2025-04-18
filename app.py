import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Carregar dados
dados = pd.read_csv('dados\insurance.csv')
print(dados.info)

# Conhecendo os dados
print(dados.describe())

#Analisando algumas distribuições com histogramas:
#dados.hist(bins=50, figsize=(20,15))
#plt.show()

# Codificar variáveis categóricas (e.g., 'sex', 'smoke', 'region')
label_encoder = LabelEncoder()
# Ajustar e transformar os rótulos
dados['sex'] = label_encoder.fit_transform(dados['sex'])
dados['smoker'] = label_encoder.fit_transform(dados['smoker'])
dados['region'] = label_encoder.fit_transform(dados['region'])
print(dados.head())

# Separar variáveis independentes (X) e dependente (y)
X = dados.drop(columns=['charges'])
y = dados['charges']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando os valores
scaler = MinMaxScaler() #chamando o metodo de normalização dos dados (0-1)
X_scaled = scaler.fit_transform(X)

# Escalar os dados de treinamento e teste separadamente
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instanciar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test_scaled)

# Avaliar o modelo com métricas estatísticas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
print("R2 médio com validação cruzada:", scores.mean())

####  Testando outro modelo
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
print("R-squared (R2) com Random Forest:", r2)

# Visualizando as previsões
plt.scatter(range(len(y_test)), y_test, label='Real')
plt.scatter(range(len(y_pred)), y_pred, label='Previsto', color='red')
plt.xlabel('Pacientes')
plt.ylabel('Valor Despesa Médica')
plt.title('Previsões do Modelo de Regressão Linear')
plt.legend()
plt.show()

