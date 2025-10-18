import pandas as pd
import numpy as np

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
import os

# Modelos
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score, auc, roc_curve, precision_recall_curve, log_loss

# Función para análizar columnas 
def analizar_columna(df: pd.DataFrame, columna: str, table : bool = True, plot : bool = False):
    '''
    Analiza una columna específica de un DataFrame y proporciona información detallada.

    Parámetros: 
    df (pd.DataFrame): Dataframe que contiene la columna a analizar.
    columna (str): Nombre de la columna a analizar.

    Retorna:
    dict: Diccionario con información de la columna
    '''
    if columna not in df.columns:
        return{'Error':'La columna no está en el DataFrame'}
    
    info = {}
    info['Nombre'] = columna
    info['Tipo de datos'] = df[columna].dtype
    info['Porcentaje de valores nulos'] = df[columna].isnull().mean() * 100

    if np.issubdtype(info['Tipo de datos'], np.number):
        # Si la columna es numérica, obtener estadísticas descriptivas
        if table == True:
            info['Estadística descriptivas'] = df[columna].describe().to_dict()

        # Gráficos necesarios
        fig, ax = plt.subplots(2, 1, figsize = (8, 6), gridspec_kw = {'height_ratios' : [1, 4]})
        # Boxplot
        sns.boxplot(df[columna].dropna(), color = '#f4fd39', ax = ax[0], orient = 'h')
        ax[0].set_title(f'Boxplot de {columna}', fontsize = 12)
        #ax[0].set_xlabel(columna, fontsize = 10)
        ax[0].grid(axis = 'x', linestyle = '--', alpha = 0.7)

        # Histograma
        sns.histplot(df[columna].dropna(), bins = 30, kde = True, color = 'blue', ax = ax[1])
        ax[1].set_title(f'Histograma de {columna}', fontsize = 12)
        ax[1].set_xlabel(columna, fontsize = 10)
        ax[1].set_ylabel('Frecuencia', fontsize = 10)
        ax[1].grid(axis = 'x', linestyle = '--', alpha = 0.7)

        plt.tight_layout()
        plt.show()
        print(info)

    else:
        # Si la columna es categórica, obtener el conteo por categoría
        #info['Categorías'] = df[columna].unique().tolist()
        #info['Conteo por categoría'] = df[columna].value_counts().to_dict()
        info['Porcentaje por categoría'] = (df[columna].value_counts(normalize=True) * 100).round(2)
        info['Porcentaje acumulado'] = info['Porcentaje por categoría'].cumsum()

        info = pd.DataFrame(info)

        if len(info) < 25:
            plot = True

        if plot == True:
            if len(info) < 5:
                # Gráfico de pastel
                plt.figure(figsize=(6, 4))
                plt.pie(info['Porcentaje por categoría'], labels = info.index, autopct = '%1.1f%%', startangle = 140, colors = sns.color_palette('viridis', len(info)))
                plt.title(f'Gráfico de pastel de {columna}', fontsize = 12)
                plt.show()
            else:
                # Gráfico de barras
                plt.figure(figsize=(6, 4))
                sns.barplot(x = df[columna]. value_counts().index, y = df[columna].value_counts().values, palette = 'viridis')
                plt.title(f'Gráfico de barras de {columna}', fontsize = 12)
                plt.xlabel(columna, fontsize = 10)
                plt.ylabel('Frecuencia', fontsize = 10)
                plt.xticks(rotation = 80, fontsize = 8)
                plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
                plt.show()
        if table == True:
            print(info)
        
'''------------------------------------------------------------------------------------------------------------------'''
### Análisis bibariado de datos numéricos
def analisis_bivariado_numerico(data : pd.DataFrame, var1 : str, var2 : str):
    """
    Realiza un análisis bivariado para variables numéricas.
    
    :param data: DataFrame con los datos
    :param var1: Nombre de la primera variable numérica
    :param var2: Nombre de la segunda variable numérica
    """
    # Diagrama de dispersión 
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=var1, y=var2, data=data, color='blue', alpha=0.7, edgecolor='black')

    # Personalización del gráfico
    plt.title(f'Diagrama de Dispersión de {var1} y {var2}', fontsize=12, fontweight='bold')
    plt.xlabel(f'{var1}', fontsize=10)
    plt.ylabel(f'{var2}', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

    plt.show()
    # Correlación de Spearman porque las variables numéricas no son normales
    corr, p_value = stats.spearmanr(data[var1], data[var2])
    print(f'Correlación de Spearman: {corr}, p-valor: {p_value}')

### Análisis bivariado de variables categóricas 
def analisis_bivariado_categorico(data : pd.DataFrame, var1 : str, var2 : str, table : bool = False):
    """
    Realiza un análisis bivariado para variables categóricas.
    
    :param data: DataFrame con los datos
    :param var1: Nombre de la primera variable categórica
    :param var2: Nombre de la segunda variable categórica
    """
    # Tabla de contingencia
    contingency_table = pd.crosstab(data[var1], data[var2])
    if table == True:
        print(contingency_table)

    # Test de Chi-cuadrado
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    k = min(contingency_table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    #if cramers_v >= 0.5:
    print(f'Prueba Chi-cuadrado de las variables {var1} y {var2}')
    print(f'Chi-cuadrado: {chi2}, p-valor: {p}, Cramer´s V: {cramers_v}')
    

### Análisis bivariado de variables numéricas y categóricas
def analisis_bivariado_mixto(data : pd.DataFrame, var_cat : str, var_num : str):
    """
    Realiza un análisis bivariado para variables mixtas (una categórica y una numérica).
    
    :param data: DataFrame con los datos
    :param var_cat: Nombre de la variable categórica
    :param var_num: Nombre de la variable numérica
    """
    # Boxplot
    plt.figure(figsize=(16, 6))
    sns.boxplot(
        x=var_cat, 
        y=var_num, 
        data=data, 
        width=0.6,  
        linewidth=1, 
        boxprops=dict(facecolor="gray", edgecolor="black"),  
        medianprops=dict(color="red", linewidth=2),  
        whiskerprops=dict(color="black", linewidth=1.2), 
        capprops=dict(color="black", linewidth=1.2),  
    )

    # Titulo 
    plt.title("Distribución de {} por {}".format(var_num, var_cat), fontsize=12, fontweight="bold", pad=15)
    plt.xlabel(var_cat, fontsize=10)
    plt.ylabel(var_num, fontsize=10)
    plt.xticks(rotation=85, fontsize=8)  # Rotar etiquetas del eje X si son largas
    plt.yticks(fontsize=8)
    plt.show()

    # Se verifica la cantidad de categorías para definir la prueba a usar 
    cats = data[var_cat].unique()
    num_cat = len(data[var_cat].unique())

    if num_cat == 2:
        # Distintos grupos del conjuntos de datos 
        gr1 = data[data[var_cat] == cats[0]][var_num]
        gr2 = data[data[var_cat] == cats[1]][var_num]
        # Prueba de U Mann-Whitney U
        u_stat, p_value = stats.mannwhitneyu(gr1, gr2)
        print(f'Prueba de Mann-Whitney U: U-valor = {u_stat}, p-valor = {p_value}')
    else:
        groups = [data[data[var_cat] == cat][var_num] for cat in cats]

        # Pureba de Kruskal-Wallis
        kruskal_result = stats.kruskal(*groups)
        print(f'Prueba de Kruskal-Wallis: H-valor = {kruskal_result.statistic}, p-valor = {kruskal_result.pvalue}')

'''------------------------------------------------------------------------------------------------------------------'''