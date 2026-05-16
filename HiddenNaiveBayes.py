'''
Implemetación de Hidden Naive Bayes basado en el Articulo:
“A novel Bayes Model: HiddenNaive Bayes” 
escrito por:
Liangxiao Jiang, Harry Zhang, and Zhihua Cai  
Año de publicación: 2008
Autor: A. Martín Ramírez Rabelo
Fecha de realizacion: Diciembre24-Enero25
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm #Barra de progreso -tqdm means "progress" in Arabic (taqadum, تقدّم) 

# Cargar los datos (asegúrate de tener tus datos en un dataframe de pandas)
df = pd.read_csv('DataHNB/labor_cleaned.csv')  # lectura del archivo .csv
#df = pd.read_excel('DataHNB/diabetes_cleaned.xlsx', sheet_name='Hoja1')

# Preprocesamiento de datos (ajusta según sea necesario)

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


##---------------2. Implementación de la clase HNB:----------------##

class HiddenNaiveBayes:
    def __init__(self):
        # Inicializar parámetros del modelo
        self.probabilidades_previas_clase = {}
        self.probabilidades_condicionales = {}

    def calcular_probabilidades_previas_clase(self, y_train):
        """
        Estima las probabilidades previas de clase a partir de los datos de entrenamiento.
        """
        clases, conteos = np.unique(y_train, return_counts=True)
        self.probabilidades_previas_clase = dict(zip(clases, conteos / len(y_train)))

    def calcular_informacion_mutua_condicional(self, X_train, y_train, caracteristica_i, caracteristica_j):
        """
        Calcula la información mutua condicional entre dos características dadas las clases.
        Utilizada para calcular los pesos (Wij) en la Ecuación (13).
        """
        imc = 0.0
        for clase in np.unique(y_train):
            for valor_i in np.unique(X_train[caracteristica_i]):
                for valor_j in np.unique(X_train[caracteristica_j]):
                    # Calcular las probabilidades conjuntas y condicionales necesarias
                    P_aij_c = len(X_train[(X_train[caracteristica_i] == valor_i) & 
                                           (X_train[caracteristica_j] == valor_j) & 
                                           (y_train == clase)]) / len(X_train)
                    P_ai_c = len(X_train[(X_train[caracteristica_i] == valor_i) & 
                                          (y_train == clase)]) / len(X_train)
                    P_aj_c = len(X_train[(X_train[caracteristica_j] == valor_j) & 
                                          (y_train == clase)]) / len(X_train)
                    P_c = len(X_train[y_train == clase]) / len(X_train)

                    # Agregar a la información mutua condicional si las probabilidades son no cero
                    if P_aij_c != 0 and P_ai_c != 0 and P_aj_c != 0 and P_c != 0:
                        imc += P_aij_c * np.log(P_aij_c / (P_ai_c * P_aj_c / P_c))
        return imc

    def calcular_probabilidades_condicionales(self, X_train, y_train):
        """
        Estima las probabilidades condicionales de cada característica
        dada la clase y sus padres ocultos utilizando la Ecuación (10).
        """
        self.probabilidades_condicionales = {}
        for clase in tqdm(np.unique(y_train)):
            self.probabilidades_condicionales[clase] = {}
            for caracteristica_i in X_train.columns:
                self.probabilidades_condicionales[clase][caracteristica_i] = {}
                for valor_i in np.unique(X_train[caracteristica_i]):
                    suma_ponderada = 0.0
                    suma_pesos = 0.0
                    for caracteristica_j in X_train.columns:
                        if caracteristica_j != caracteristica_i:
                            # Calcular el peso (Wij) utilizando la información mutua condicional (Ecuación 13)
                            Wij = self.calcular_informacion_mutua_condicional(X_train, y_train, caracteristica_i, caracteristica_j)
                            suma_pesos += Wij
                            for valor_j in np.unique(X_train[caracteristica_j]):
                                # Calcular las probabilidades conjuntas y condicionales
                                P_aij_c = len(X_train[(X_train[caracteristica_i] == valor_i) &
                                                   (X_train[caracteristica_j] == valor_j) &
                                                   (y_train == clase)]) / len(X_train)
                                P_aj_c = len(X_train[(X_train[caracteristica_j] == valor_j) &
                                                  (y_train == clase)]) / len(X_train)

                                # Agregar a la suma ponderada si la probabilidad es no cero
                                if P_aij_c != 0 and P_aj_c != 0:
                                    suma_ponderada += Wij * (P_aij_c / P_aj_c)

                    # Calcular la probabilidad condicional final (Ecuación 10)
                    self.probabilidades_condicionales[clase][caracteristica_i][valor_i] = suma_ponderada / suma_pesos

    def predecir(self, X_test):
        """
        Predice la clase de una nueva instancia utilizando las probabilidades previas de clase
        y las probabilidades condicionales calculadas.
        """
        predicciones = []
        for _, instancia in tqdm(X_test.iterrows()):
            probabilidades_posteriores = {}
            for clase in self.probabilidades_previas_clase:
                probabilidad = self.probabilidades_previas_clase[clase]
                for caracteristica in X_test.columns:
                    valor_caracteristica = instancia[caracteristica]
                    # Obtener la probabilidad condicional
                    probabilidad_condicional = self.probabilidades_condicionales[clase][caracteristica].get(valor_caracteristica, 0)
                    probabilidad *= probabilidad_condicional
                probabilidades_posteriores[clase] = probabilidad
            predicciones.append(max(probabilidades_posteriores, key=probabilidades_posteriores.get))
        return predicciones

# Crear una instancia del modelo HNB
modelo = HiddenNaiveBayes()

# Entrenar el modelo
modelo.calcular_probabilidades_previas_clase(y_train)
modelo.calcular_probabilidades_condicionales(X_train, y_train)

# Hacer predicciones
predicciones = modelo.predecir(X_test)

# Evaluar el rendimiento del modelo (utilizando la precisión como ejemplo)
precision = accuracy_score(y_test, predicciones)
print(f'Precisión del modelo: {precision}')