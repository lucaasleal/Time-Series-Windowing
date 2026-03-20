import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import MinMaxScaler

#Leitura e construção do dataframe
path = 'accel'

files = glob.glob(os.path.join(path, "*.txt"))

columns = ['user', 'activity', 'timestamp', 'x', 'y', 'z']
df = []

for file in files:
    df_temp = pd.read_csv(file, header = None, names = columns, on_bad_lines='skip')
    df.append(df_temp)

df = pd.concat(df, ignore_index=True)


#Corrigir coluna z
df['z'] = df['z'].astype(str).str.replace(';', '')
df['z'] = pd.to_numeric(df['z'], errors='coerce')

#Limpeza de linhas incorretas
df = df.dropna()

print(df.head())


#-------------------------------------------------------------------
#Novo dataframe apenas com as colunas x, y, z
df_accel = df[['user', 'x', 'y', 'z']].copy()

print("\n--- DATAFRAME RECORTADO ---")

print(df_accel.head())

#Normalizar dados
scaler = MinMaxScaler(feature_range=(-1, 1))
df_accel[['x', 'y', 'z']] = scaler.fit_transform(df_accel[['x', 'y', 'z']])

print("\n--- DATAFRAME NORMALIZADO ---")
print(df_accel.head())

print(f"Shape antes do janelamento: {df_accel[['x', 'y', 'z']].shape}")

#-------------------------------------------------------------------
#Janelamento
windows_train = []
windows_test = []

window_size = 128
step = 64

for user in df_accel['user'].unique():
    user_data = df_accel[df_accel['user'] == user][['x', 'y', 'z']].values

    #Dados do usuário não completam pelo menos uma janela
    if(len(user_data) < window_size): 
        continue

    train_size = int(len(user_data)*0.8) #80% treino e 20% teste
    user_train = user_data[:train_size]
    user_test = user_data[train_size:]

    #Janelas de treino e teste (Deslizantes)
    for i in range(0, len(user_train), step):
        window = user_train[i : i + window_size]

        if(len(window) == window_size):
            windows_train.append(window)

    for i in range(0, len(user_test), step):
        window = user_test[i : i + window_size]

        if(len(window) == window_size):
            windows_test.append(window)
    

#Transforma as listas em tensor
X_train = np.array(windows_train)
X_test = np.array(windows_test)

print(f"Shape de uma janela: {X_train[0].shape}")
print("\n--- PREPARAÇÃO CONCLUÍDA ---")
print(f"Shape Final do Treino: {X_train.shape}")
print(f"Shape Final do Teste:  {X_test.shape}")
