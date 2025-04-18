import pandas as pd

import app

novo_dado = pd.DataFrame([[25, 'male', 24.0, 0, 'yes', 'southwest']], 
                         columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
novo_dado['sex'] = app.label_encoder.fit_transform(novo_dado['sex'])
novo_dado['smoker'] = app.label_encoder.fit_transform(novo_dado['smoker'])
novo_dado['region'] = app.label_encoder.fit_transform(novo_dado['region'])

novo_dado_scaled = app.scaler.transform(novo_dado)

previsao = app.model.predict(novo_dado_scaled)
print("Previsão de custo médico:", previsao[0])