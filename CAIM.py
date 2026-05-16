import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def readData_selectAtribute(file):
    df = pd.read_csv(file)
    labels_df = df['class'].copy()
    df = df.drop(columns=['class'])
    features = list(df.columns)
    for i,feature in enumerate(features):
        print(f'{i}.- {feature}')
    feature = input('selecciona atributo a dicretizar: ')
    feature_df = df[feature]
    return labels_df, feature_df

def compute_caim(matrix, bins):
    """
    Calcula la métrica CAIM para una discretización dada.
    """
    q, r = matrix.shape
    caim_value = 0
    
    for j in range(r):
        max_nij = max(matrix[:, j])  # Máximo valor de frecuencia en la columna
        sum_nij = sum(matrix[:, j])  # Suma de la columna
        
        if sum_nij > 0:
            caim_value += (max_nij ** 2) / sum_nij
    
    return caim_value / bins

def caim_discretization(feature, labels):
    """
    Implementación del algoritmo CAIM para discretización supervisada.
    """

    # Convertir etiquetas de clase a números si son strings
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    unique_values = np.unique(feature)
    
    # 1.1 Encontrar los valores máximo y mínimo
    d_min, d_max = np.min(unique_values), np.max(unique_values)
    
    # 1.2 Formar el conjunto de puntos candidatos de corte
    candidate_splits = list(unique_values) + [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]
    candidate_splits = sorted(set(candidate_splits))

    # Inicialización
    D = [d_min, d_max]
    global_caim = 0
    k = 1
    num_classes = len(np.unique(labels))
    
    while True:
        best_caim = -1
        best_cut = None
        current_splits = sorted(D)

        for cut in candidate_splits:
            if cut in current_splits:
                continue  # Evitar repetir cortes ya seleccionados
            
            tentative_splits = sorted(current_splits + [cut])
            bins = len(tentative_splits) - 1

            # Crear matriz de frecuencias
            matrix = np.zeros((num_classes, bins))
            digitized = np.digitize(feature, tentative_splits, right=True) - 1
            
            for i, label in enumerate(labels):
                matrix[label, digitized[i]] += 1

            caim_score = compute_caim(matrix, bins)

            if caim_score > best_caim:
                best_caim = caim_score
                best_cut = cut

        # 2.4 Evaluar si se agrega el nuevo punto de corte
        if best_caim > global_caim or k < num_classes:
            D.append(best_cut)
            global_caim = best_caim
        else:
            break
        
        k += 1

    return sorted(D)

#--------------#-------#---------#-------#-#-#-#-#----#-------#-------------------#
#--------------#-#---#-#--------#-#----------#--------#-#-----#-------------------#
#--------------#---#---#-------#-#-#---------#--------#---#---#-------------------#
#--------------#-------#------#-----#--------#--------#-----#-#-------------------#
#--------------#-------#-----#-------#---#-#-#-#-#----#-------#-------------------#
if __name__ == "__main__":
    labels, feature = readData_selectAtribute('iris_data.csv')
    discretization_scheme = caim_discretization(feature, labels)
    print("Esquema de discretización:", discretization_scheme)