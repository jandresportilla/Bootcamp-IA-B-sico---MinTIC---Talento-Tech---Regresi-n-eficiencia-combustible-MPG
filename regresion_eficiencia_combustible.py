# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 22:39:43 2025

@author: LOQ

MISIÓN 3: ML REGRESIÓN  ---->>>  eficiencia de combustible MPG Millas Por Galón de Kaggle

Visitar ---->> https://www.kaggle.com/datasets/jawadkhan65/auto-mpg-fuel-efficiency-prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, PassiveAggressiveRegressor, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor , RandomForestRegressor #GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
#from sklearn.svm import SVR
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, explained_variance_score
import joblib


# 0. convertir datos a DataFrame 
df = pd.read_csv("D:/JAPJ/IA BAsico Bootcamp/IA Bassico/MISION 3/Reduced-Data.csv")

print("Dataset Predicción Automática de Eficiencia de Combustible MPG - KAGGLE")
print(df.head(-1))


# 1. realizar analisis exploratorio de los datos
print("\n\n")
print("=" * 80)
print("ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 80)

# 2. Información basica
print(f"Tamaño del dataset: {df.shape}")
print("\nEstadísticas descriptivas:")
print(df.describe())
print(df.columns)
print("\n\nDistribución de la variable Objetivo/Target EFICIENCIA COMBUSTIBLE MILLAS POR GALÓN MPG:") # SI=1
distrib = df['Fuel consumption '].value_counts().sum()
print(distrib)

#saber el tipo de datos de los atributos, de las columnas
print("\n\n")
print(" los DOS Tipos de Atributos Categóricos que cuenta Dataset Kaggle:")
print(df.dtypes)

print(df['Ft'].value_counts())
print(df['Fm'].value_counts())

# filtrar X solo columnas tipo OBJECT
print("\n\n Sub Dataset con solo los DOS Atributos Categóricos para CODIFICAR:") 
columnas_objeto = df.select_dtypes(include=['object'])
print(columnas_objeto)



# 3. Preprocesamiento
# codificar variables categóricas

# copia, otro NUevo dataset para codificacion
df_codi = df.copy()

label_encoders = {}
for col in df_codi.select_dtypes(include=['object']).columns:
    funcion = LabelEncoder()
    df_codi[col] = funcion.fit_transform(df_codi[col])
    label_encoders[col] = funcion
    
    
print("\n\n")
print("=" * 80)
print("NUEVO DATAFRAME YA CON LOS DOS ATRIBUTOS CATEGÓRICOS CODIFICADOS")
print("=" * 80)
print(df_codi)


print("\n\n")
print("=" * 80)
print("PROCESAMIENTO A VALORES FALTANTES Y NULOS")
print("=" * 80)

print("\n\n Valores Faltantes o Nulos ?")
print(df_codi.isnull())
print("\n\n Suma de Valores Faltantes o Nulos ?")
print(df_codi.isnull().sum())


df1 = df_codi.dropna(subset=['Fuel consumption '])

# Imputar valores faltantes en las variables predictoras (X)

faltantes = SimpleImputer(strategy="median")  
# definir variables predictoras -X- y variable objetivo -y-
X = faltantes.fit_transform(df1.drop("Fuel consumption ", axis=1))
y = df1["Fuel consumption "].values

print("\n\n")
print("=" * 80)
print("GRÁFICOS DE LA VARIABLE OBJETIVO CONTINUA - REGRESIÓN")
print("=" * 80)

# grafico linea ---evovolucion  tendencias/estacionalidad.

plt.figure(figsize=(10,5))
plt.plot(df1["Fuel consumption "].reset_index(drop=True), color="crimson", linewidth=2)
plt.title("Evolución de Fuel Consumption", fontsize=14)
plt.xlabel("Índice / Observación")
plt.ylabel("Fuel Consumption")
plt.show()


# grafico de dispersion relaciones entre variables explicativas

plt.figure(figsize=(8,6))
plt.scatter(df1["Ft"], df1["Fuel consumption "], alpha=0.6, color="darkcyan")
plt.title("Dispersión: Fuel Type vs Fuel Consumption", fontsize=14)
plt.xlabel("Fuel Type")
plt.ylabel("Fuel Consumption")
plt.show()


# grafico Boxplot outliers y distribución.

plt.figure(figsize=(8,5))
sns.boxplot(x=df1["Fuel consumption "], color="tomato")
plt.title("Boxplot de Fuel Consumption", fontsize=14)
plt.xlabel("Fuel Consumption")
plt.show()


print("\n\n")
print("=" * 80)
print("INICIA EL ENTRENAMIENTO -TRAIN- DE LOS OCHO MODELOS ML DE REGRESIÓN")
print("=" * 80)


# 4. Incia el ML
# dividir en train ENTRENAMIENTO & test PRUEBA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# Escalado
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

"""
# 11. ONCE  algoritmos , estimadores regresores

SEIS LINEALES --->> Linear, Elastic, Ridge, Lasso, PassiveAgrre, SGD

UNO DE ARBOL -->> DT

UNO BOSQUEN EMSABLE --->> HistGradient Boosting 


open-source gradient boosting frameworks 
TRES TAN ESPECIALES QUE SON FUERA DE SCKIT-LEARN 
---> XGBoost -->> eXtreme Gradient Boosting
---> LightGBM -->> Light Gradient Boosting Machine --->> developed by Microsoft GOSS & EFB
---> CatBoost --->> Categorical Boosting

Haremos un FOR para dar una COMPARACION ENTRE LOS DICHOS ONCE ALGORITMOS/ESTIMADORES

Se elimniaron unos ya que No Convergen en TIEMPO, Imposible el Consumo Computacion TEMPORAL

"""

modelos = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    #"Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    #"SVR (Support Vector Regression)": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=.1),
    #"KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "LightGBM Regressor": LGBMRegressor(n_estimators=100, random_state=42),
    "CatBoost Regressor": CatBoostRegressor(iterations=500, learning_rate=0.1, depth=6, verbose=0, random_state=42),
    "ElasticNet Regression": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Lasso Regression": Lasso(alpha=0.01, random_state=42),
    "Passive-Aggressive Regressor": PassiveAggressiveRegressor(max_iter=1000, random_state=42),
    "Stochastic Gradient Descent Regressor": SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
    "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100, random_state=42),
    "HistGradient Boosting Regressor": HistGradientBoostingRegressor(random_state=42),
    #"Gaussian Process Regressor": GaussianProcessRegressor(kernel=RBF(), random_state=42),
    "MLP Regressor (Neural Net)": MLPRegressor(hidden_layer_sizes=(64,32),  activation="relu", solver="adam", max_iter=1000, random_state=42)
}

resultadosfinales = []
import time


for nombre, modelo in modelos.items():
    # Medir tiempo
    inicio = time.time()
    
    # TRAIN
    modelo.fit(X_train_scaled, y_train)
    
    # PREDICT
    y_pred = modelo.predict(X_test_scaled)
    
    fin = time.time()
    tiempo = fin - inicio
    
    # Guardar el modelo entrenado con su nombre
    filename = f"modelo_{nombre.replace(' ', '_')}.pkl"
    joblib.dump(modelo, filename)
    print(f"✅ Modelo guardado: {filename}")
    
    
    
    flag2 = time.time()
    #METRICAS
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred) #--->>> Coeficiente de Determinación
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1, y_test))) * 100 #-->> MAPE (Mean Absolute Percentage Error)
    medae = median_absolute_error(y_test, y_pred) # -->> Median Absolute Error
    evs = explained_variance_score(y_test, y_pred) #---> Explained Variance Score
    #rmsle = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred))) #-->>Root Mean Squared Logarithmic Error
    flag3 = time.time()
    tiempo_metricas = flag3 - flag2
    
    
    resultadosfinales.append([nombre, mae, mse, rmse, r2, mape, medae, evs, tiempo , tiempo_metricas])
    
    # Visualización Predicho vs Real
    plt.figure(figsize=(16,12))
    plt.scatter(y_test, y_pred, alpha=0.3, color="crimson", edgecolors="k")
    plt.plot(min(y_test.min(), y_test.max()), max(y_test.min(), y_test.max()), 
         "--", color="blue", linewidth=2, label="Ideal (y=x)")
    plt.xlabel("Valores Reales (y_test)")
    plt.ylabel(f"Valores Predichos (y_pred)  - {nombre}")
    plt.title(f"Regresión - {nombre}: Valores Reales vs Valores Predichos")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
    print("\n\n")
    
    
    # COMPARATIVO grafico linea ---evovolucion  tendencias/estacionalidad.

    plt.figure(figsize=(20,12))
    plt.plot(y_test, color="forestgreen", linewidth=2, alpha=0.3, label="Reales")
    plt.plot(y_pred, color="darkorange", linewidth=2, alpha=0.3, label="Predicciones")
    plt.title("Comparativo - Algoritmo {nombre} : Reales vs Predicciones", fontsize=14)
    plt.xlabel("Muestra / Observación")
    plt.ylabel("Fuel Consumption - CONSUMO DE COMBUSTIBLE en MPG")
    plt.legend()
    plt.show()

    print("\n\n")


# 5. comparacion de métricas --->> EVALUACIÓN

df_resultados = pd.DataFrame(resultadosfinales, columns=["Modelo","MAE","MSE","RMSE","R2", "MAPE", "MEDAE", "EVS" , "TIEMPO ENTRENAMEINTO", "TIEMPO EMPLEADO MÉTRICAS"])
print(df_resultados)


"""
# Gráfico comparativo de R2
df_resultados.set_index("Modelo")["R2"].plot(kind="bar", figsize=(8,5), color="firebrick")
plt.title("Comparación de Modelos - R²")
plt.ylabel("R² Score")
plt.ylim(0,1)
plt.xticks(rotation=45, ha="right")
plt.show()

"""



metricas = [
    ("MAE", "MAE - Mean Absolute Error", "skyblue"),
    ("MSE", "MSE - Mean Squared Error", "orange"),
    ("RMSE", "RMSE - Root Mean Squared Error", "green"),
    ("R2", "R² Score", "firebrick"),
    #("MAPE", "Mean Absolute Percentage Error (%)", "purple"),
    ("MEDAE", "MedAE - Median Absolute Error", "brown"),
    ("EVS", "EVS - Explained Variance Score", "teal")    
]


for metrica, titulo, color in metricas:
    df_resultados.set_index("Modelo")[metrica].plot(
        kind="bar", figsize=(16,10), color=color
    )
    plt.title(f"Comparación de Modelos - Métrica: {titulo}")
    plt.ylabel(metrica)
    plt.xticks(rotation=45, ha="right")
    
    # ajustes opcionales rango conocido
    if metrica in ["R2", "EVS"]:
        plt.ylim(0,1)
    
    plt.show()




# 6. predicción de muestra de 50 Nuevos Vehiculos mediante --->> ahora sí se usa el RANDOM, no para el TRain>>>>> sino para el TEST >>> data sintetica para evaluar, no para entrenar


import random


def generar_muestras_sinteticas_prueba(n, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)

    # Valores posibles de Ft y Fm basados en COLUNMAS , VALORES CATEGORICOS, ATRIBUTOS dataset tomado
    valores_Ft = [
        "petrol", "diesel", "electric", "petrol/electric", "lpg", "e85",
        "diesel/electric", "ng", "hydrogen", "PETROL", "PETROL/ELECTRIC",
        "DIESEL", "unknown", "ELECTRIC", "LPG"
    ]
    valores_Fm = ["M", "H", "E", "P", "B", "F"]

    # crear datos simulados
    data_GEN = {
        "r": np.random.randint(1000, 5000, size=n),  # rango entero
        "m (kg)": np.random.uniform(800, 3000, size=n),
        "Mt": np.random.uniform(50, 500, size=n),
        "Ewltp (g/km)": np.random.uniform(0, 300, size=n),
        "Ft": [random.choice(valores_Ft) for _ in range(n)],
        "Fm": [random.choice(valores_Fm) for _ in range(n)],
        "ec (cm3)": np.random.uniform(600, 5000, size=n),
        "ep (KW)": np.random.uniform(20, 400, size=n),
        "z (Wh/km)": np.random.uniform(0, 200, size=n),
        "Erwltp (g/km)": np.random.uniform(0, 300, size=n),
        "Electric range (km)": np.random.uniform(0, 600, size=n),
    }

    # Convertir a DataFrame
    df_sintetico = pd.DataFrame(data_GEN)
    return df_sintetico


# Generar 50 muestras
df_gen_test = generar_muestras_sinteticas_prueba(50)

print(df_gen_test.head(-1))
print(df_gen_test.dtypes)


etiqueta_encoders = {}
for col in df_gen_test.select_dtypes(include=['object']).columns:
    df_gen_test[col] = funcion.fit_transform(df_gen_test[col])
    etiqueta_encoders[col] = funcion
    

nuevo_scaled = scaler.transform(df_gen_test)


# Obtener sólo el Decision Tree, EL MEJOR DE TODOS EL DT
modelo_dt = modelos["Decision Tree"]

print("\n\n")
predicciones_SINTETICO = modelo_dt.predict(nuevo_scaled)
for i, val in enumerate(predicciones_SINTETICO):
        print(f"MODELO 🚗 OptiCOMBUS⛽: Predicción Consumo de COMBUSTIBLE --->> Registro {i+1} :    {val:.2f} MPG")
    
    
   
    
    
# 7. Predicción Manual, DIRECTA muestra unitarias de Vehiculo por TECLADO

"""
dia_nuevo = np.array([[26,75,1010]]) #Datos hipoteticos de temperatura, humedad y presion
prediccion=model.predict(dia_nuevo)
print(f"Prediccion de la lluvia para un dia de 26°C, humedad del 75% y una presion de 1010 HPA:{'si'if prediccion[0]==1 else 'no'}")
"""
    

def capturar_por_teclado():
    print("\n\n\n\n=== INGRESO MANUAL DE DATOS PARA PREDICCIÓN MPG ===")
    print("Por favor ingresa los valores para Cada Una de las DIEZ(10) Variables:\n")
    
    data_teclado = {}
    data_teclado["r"] = 1       
    
    # 2. m (kg) - peso
    while True:
        try:
            m_val = float(input("2. Masa Vehículo (kg) - peso (ej: 500-5000): "))
            data_teclado["m (kg)"] = m_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # 3. Mt
    while True:
        try:
            mt_val = float(input("3. Mt Emisiones de CO2 en TONS (ej: 600-5000): "))
            data_teclado["Mt"] = mt_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # 4. Ewltp (g/km)
    while True:
        try:
            ewltp_val = float(input("4. Ewltp Emisiones de CO2 medidas según el Procedimiento Mundial Armonizado de Ensayos de Vehículos Ligeros (WLTP) (g/km) (ej: 0-400): "))
            data_teclado["Ewltp (g/km)"] = ewltp_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # 5. Ft 
    valores_Ft = ["petrol", "diesel", "electric", "petrol/electric", "lpg", "e85", 
                 "diesel/electric", "ng", "hydrogen", "PETROL", "PETROL/ELECTRIC",
                 "DIESEL", "unknown", "ELECTRIC", "LPG"]
    print(f"5. Ft - Elige el Tipo de Combustible). Las Opciones: {', '.join(valores_Ft)}")
    while True:
        ft_val = input("Selecciona Ft: ").strip()
        if ft_val in valores_Ft:
            data_teclado["Ft"] = ft_val
            break
        else:
            print("Valor no válido. Por favor Elige de las Opciones Mostradas.")
    
    # 6. Fm
    valores_Fm = ["M", "H", "E", "P", "B", "F"]
    print(f"6. Fm. Elige la Composición de la mezcla de combustible utilizada por el vehículo. lAs Opciones: {', '.join(valores_Fm)}")
    while True:
        fm_val = input("Selecciona Fm: ").strip().upper()
        if fm_val in valores_Fm:
            data_teclado["Fm"] = fm_val
            break
        else:
            print("Valor no válido. Por favor Elige de las opciones mostradas.")
    
    # 7. ec (cm3)
    while True:
        try:
            ec_val = float(input("7. ec Capacidad del motor en (cm3) (ej: 600-8000): "))
            data_teclado["ec (cm3)"] = ec_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # 8. ep (KW)
    while True:
        try:
            ep_val = float(input("8. ep Potencia del motor en (KW) (ej: 6-1200): "))
            data_teclado["ep (KW)"] = ep_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # 9. z (Wh/km)
    while True:
        try:
            z_val = float(input("9. z Consumo de energía en (Wh/km) (ej: 8-500): "))
            data_teclado["z (Wh/km)"] = z_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # 10. Erwltp (g/km)
    while True:
        try:
            erwltp_val = float(input("10. Erwltp Reducción de emisiones de CO2 en gramos por kilómetro medida bajo WLTP (g/km) (ej: 1-7): "))
            data_teclado["Erwltp (g/km)"] = erwltp_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # 11. Electric range (km)
    while True:
        try:
            electric_range_val = float(input("11. Electric range Autonomía eléctrica (km): La distancia máxima que puede recorrer el vehículo únicamente con energía eléctrica (km) (ej: 0-300): "))
            data_teclado["Electric range (km)"] = electric_range_val
            break
        except ValueError:
            print("Por favor ingresa un número Válido decimal.")
    
    # convertir a DataFrame vector
    df_teclado = pd.DataFrame([data_teclado])
    
    print("\n✓ Datos ingresados Manualmente Correctamente:")
    for col, val in data_teclado.items():
        print(f"   {col}: {val}")
    
    return df_teclado


def prediccion_teclado():
    try:
        # Obtener datos
        df_teclado = capturar_por_teclado()
        
        # cargar LOAD modelo y preprocesadores
        modelo_dt = modelos["Decision Tree"]
        
        print("\n ✓ Modelo 🚗 OptiCOMBUS⛽: Cargado Exitosamente\n")
        
                
    except FileNotFoundError:
        print("\n ❌ Error: No se encontraron los archivos del Modelo.")
        print("\nAsegúrate de tener cargado correctament el Modelo el arhivo pkl quizas... '??.pkl'")
        return None
    
    
    try:
        
        etiqueta_encoders = {}
        for col in df_teclado.select_dtypes(include=['object']).columns:
            df_teclado[col] = funcion.fit_transform(df_teclado[col])
            etiqueta_encoders[col] = funcion
            
        #teclado_scaled = scaler.transform(df_teclado)
        
        print("\nDatos preparados para predicción:")
        print(df_teclado)
        # PREDICT REGRESIÓN
        predicciones_teclado = modelo_dt.predict(df_teclado.values)[0] ## pasar el Nunmpy, no el panda con nombres de colunma, para evitar el WARNING...
        print(f"\n\nMODELO 🚗 OptiCOMBUS⛽: Predicción Consumo de COMBUSTIBLE --->> Registro Ingresado Manualmnete por Teclado :    { predicciones_teclado:.2f} MPG")
        
        return predicciones_teclado, df_teclado
        
    except Exception as e:
        print(f"\n ❌ Error durante la predicción: {e}")
        return None, None

print("\n\n")





# MAIN
if __name__ == "__main__":
    # Ejecutar la predicción por TECLADO
    print("🔧 Iniciando sistema de Predicción de REGRESIÓN de Eficiencia de Consumo de Combustible MPG...")
    prediccion_manual, df_teclado = prediccion_teclado()
    
    
"""    
Características:
r: Autonomía, posiblemente indicando la autonomía total de conducción del vehículo.
m (kg): Masa del vehículo en kilogramos.
Monte: Emisiones de CO2 en toneladas métricas.
Ewltp (g/km): Emisiones de CO2 medidas según el Procedimiento Mundial Armonizado de Ensayos de Vehículos Ligeros (WLTP) en gramos por kilómetro.
----->>>Ft: Tipo de combustible utilizado por el vehículo (p. ej., gasolina, diésel, eléctrico).
------>>>>Fm: Composición de la mezcla de combustible utilizada por el vehículo.
ec (cm3): Capacidad del motor en centímetros cúbicos.
ep (KW): Potencia del motor en kilovatios.
z (Wh/km): Consumo de energía en vatios-hora por kilómetro.
Erwltp (g/km): Reducción de emisiones de CO2 en gramos por kilómetro medida bajo WLTP.
Consumo de combustible: La variable objetivo que representa el consumo de combustible del vehículo.
Autonomía eléctrica (km): La distancia máxima que puede recorrer el vehículo únicamente con energía eléctrica.  


"""
  
# Cargar el JOBlib Recomendado por la propia documentación de scikit-learn. # pickle → si quieres algo estándar y compatible fuera de ML.
#modelo_cargado = joblib.load("modelo_rf.pkl")
