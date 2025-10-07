import pandas as pd
import matplotlib.pyplot as plt
from fitter import Fitter

# Carga los datos directamente desde tu archivo CSV
# Asegúrate de que el archivo CSV esté en la misma carpeta que tu script de Python.
df = pd.read_csv('Coffe_sales.csv')

# --- El resto del código es exactamente el mismo ---

# 4. Histograma
plt.figure() # Crea una nueva figura para el histograma
plt.hist(df['money'], bins=5, edgecolor='black')
plt.title('Histograma de Frecuencia de la variable "money"')
plt.xlabel('Money')
plt.ylabel('Frecuencia')
plt.savefig('money_histogram.png') # Guarda el histograma como imagen
print("Histograma 'money_histogram.png' generado.")


# 5. Ajuste de distribución con Fitter
f = Fitter(df['money'])
f.fit()

# Muestra un resumen de las mejores distribuciones
plt.figure() # Crea una nueva figura para el resumen de fitter
f.summary()
plt.savefig('fitter_summary.png') # Guarda el resumen como imagen
print("Gráfico de resumen 'fitter_summary.png' generado.")


# Obtén la mejor distribución
best_dist = f.get_best(method='sumsquare_error')
print(f"La mejor distribución ajustada es: {best_dist}")

# Si quieres ver los gráficos en una ventana emergente al ejecutar el script,
# puedes añadir plt.show() al final.
plt.show()