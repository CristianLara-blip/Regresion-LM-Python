# Importar librerías
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Importar datos
df = pd.read_excel("datos_producción-oficial.xlsx")

# Mostrar las primeras filas para verificar la carga de datos
print(df.head(2))

# Eliminar el símbolo "°C" de la columna "Temperatura Ambiente" y convertir a flotantes
df['Temperatura Ambiente'] = df['Temperatura Ambiente'].str.replace('°C', '').astype(float)

# Seleccionar variables
variables_x = ["Horas Trabajadas", "Horas Descanso", "Temperatura Ambiente", "Cantidad de Trabajadores"]
variable_y = "Productos Terminados"

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(df[variables_x], df[variable_y])

# Mostrar los coeficientes del modelo
print('Coeficientes: ', modelo.coef_)
print('Intercepción: ', modelo.intercept_)

# Imprimir la ecuación del modelo
ecuacion = 'y = ' + ' + '.join([f'{round(coef, 3)} * {var}' for coef, var in zip(modelo.coef_, variables_x)]) + f' + {round(modelo.intercept_, 3)}'
print('Ecuación del modelo: ', ecuacion)

# Calcular y mostrar el coeficiente de determinación (R^2)
r2 = r2_score(df[variable_y], modelo.predict(df[variables_x]))
print('Coeficiente de determinación (R^2): ', round(r2, 3))

# Gráfico de Importancia de las Características (Coeficientes)
importancia = pd.Series(modelo.coef_, index=variables_x).sort_values()
plt.figure(figsize=(10, 6))
importancia.plot(kind='barh')
plt.xlabel('Importancia (Coeficientes)')
plt.ylabel('Variables')
plt.title('Importancia de cada variable en el modelo de regresión')
plt.grid(True)
plt.show()

# Mapa de Calor de Correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(df[variables_x + [variable_y]].corr(), annot=True, cmap='coolwarm')
plt.title('Mapa de Calor de Correlaciones')
plt.show()

# Generar predicción con nuevas variables
nuevas_variables = {
    "Horas Trabajadas": 12,
    "Horas Descanso": 2,
    "Temperatura Ambiente": 30,
    "Cantidad de Trabajadores": 5
}

# Crear un DataFrame para las nuevas variables
prediccion_nueva = pd.DataFrame(nuevas_variables, index=[0])

# Realizar la predicción
autos_producidos_prediccion = modelo.predict(prediccion_nueva)

# Mostrar la predicción
print(f'La predicción de productos terminados para las nuevas variables es: {round(autos_producidos_prediccion[0], 3)}')