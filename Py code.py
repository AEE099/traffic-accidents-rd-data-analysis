#Importar librerias para analisis y graficas de accidentes RD target mes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_vehiculo_mes = pd.read_csv("Estadisticas de Accidentes Segun Tipo de Vehiculo HTNAL 2017 - Jun 2025.csv", encoding='latin1')
df_muerte_mes=pd.read_csv("Muertes accidentes sexo mes.csv" , encoding="latin1")
df_lesion_hora_mes=pd.read_csv("personas-lesionadas-accidente-rango-hora-mes.csv", encoding="latin1")
df_lesion_mes_sexo=pd.read_csv("personas lesionadas sexo mes.csv",encoding="latin1")
df_lesion_hora_mes
# EDA
print ("info : \n", df_lesion_hora_mes.info())
print ("head : \n", df_lesion_hora_mes.head())
print ("tail: \n", df_lesion_hora_mes.tail())
print ("Describe: \n", df_lesion_hora_mes.describe())
print ("Duplicados: \n",df_lesion_hora_mes.duplicated().sum())
df_lesion_hora_mes.drop_duplicates(inplace=True)
print ("nulos: \n", df_lesion_hora_mes.isnull().sum())
df_lesion_hora_mes.replace(np.nan, 0, inplace=True)
df_lesion_hora_mes.dropna(inplace=True)
print(df_lesion_hora_mes)
df_lesion_hora_mes.hist()

# EDA
print ("info : \n", df_lesion_mes_sexo.info())
print ("head : \n", df_lesion_mes_sexo.head())
print ("tail : \n",df_lesion_mes_sexo.tail())
print ("describe : \n",df_lesion_mes_sexo.describe())
print ("duplicated : \n", df_lesion_mes_sexo.duplicated().sum())
df_lesion_mes_sexo.drop_duplicates(inplace=True)
print ("nulos : \n",df_lesion_mes_sexo.isnull().sum())
df_lesion_mes_sexo.dropna(inplace=True)
print(df_lesion_mes_sexo)
#Convertir data
df_lesion_mes_sexo['Cantidad_lesionados'] = pd.to_numeric(df_lesion_mes_sexo['Cantidad_lesionados'], errors='coerce')
# data despues de convertir
print("Data types after conversion:")
print(df_lesion_mes_sexo.dtypes)

# EDA
print ("dataset : \n",df_muerte_mes)
print ("info : \n",df_muerte_mes.info())
print ("head : \n",df_muerte_mes.head())
print ("tail : \n",df_muerte_mes.tail())
print ("describe : \n",df_muerte_mes.describe())
print ("duplicados : \n",df_muerte_mes.duplicated().sum())
df_muerte_mes.drop_duplicates(inplace=True)
print ("nulos : \n",df_muerte_mes.isnull().sum())
df_muerte_mes.dropna(inplace=True)
print(df_muerte_mes)

df_muerte_mes.hist()

# EDA
print ("dataset : \n", df_vehiculo_mes)
print ("info : \n",df_vehiculo_mes.info())
print ("head : \n",df_vehiculo_mes.head())
print ("tail : \n",df_vehiculo_mes.tail())
print ("describe : \n",df_vehiculo_mes.describe())
print ("duplicados : \n",df_vehiculo_mes.duplicated().sum())
df_vehiculo_mes.drop_duplicates(inplace=True)
print ("nulos : \n",df_vehiculo_mes.isnull().sum())
df_vehiculo_mes.dropna(inplace=True)
print(df_vehiculo_mes)

df_vehiculo_mes.hist(figsize=(10,10))

#Analisis cantidad muerte por ano.

# Identificamos anos unicos
all_years = sorted(df_muerte_mes['A¤o'].unique())

# Creamos mapeo para anos intercaldos para el HUE
year_mapping = {}
for i, year in enumerate(all_years):
    if i % 2 == 0:  # Asigna anos reales para ano
        year_mapping[year] = str(year)
    else:  # Agrupa otros anos dentro de una variable
        year_mapping[year] = 'Otros Años'

# Create a temporary DataFrame for plotting to avoid modifying the original
df_muerte_mes_plot = df_muerte_mes.copy()
df_muerte_mes_plot['A¤o_Intercalado'] = df_muerte_mes_plot['A¤o'].map(year_mapping)

plt.figure(figsize=(18,10))
# Use the new 'A¤o_Intercalado' for hue with a line plot
sns.lineplot(data=df_muerte_mes_plot, x="Mes", y="Cantidad_muertes", hue="A¤o_Intercalado", palette='muted', marker='o')
plt.title("Muertes por accidentes, evolución mensual (años intercalados)")
plt.xticks(rotation=45)
plt.xlabel("Mes")
plt.ylabel("Cantidad de Muertes")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Año/Grupo de Año")
plt.tight_layout()
plt.show()

#Barras por sexo
plt.figure(figsize=(12,6))
sns.barplot(data=df_muerte_mes, x="Mes", y="Cantidad_muertes", hue="Sexo", palette="deep")
plt.title("Frecuencia de Muertes por accidente- Distribucion por sexo")
plt.xticks(rotation=45)
plt.xlabel("Mes")
plt.ylabel("Cantidad de Muertes")
plt.show()

#Heat map ano vs mes

pivot= df_muerte_mes.pivot_table(index="Mes", columns="A¤o", values="Cantidad_muertes", aggfunc="sum")
plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap="Reds", annot=True,fmt="d")
plt.title("Heatmap muertes  por ano vs mes")
plt.show()

#Estacionalidades
plt.figure(figsize=(14,10))
muertes_mensuales=df_muerte_mes.groupby("Mes")["Cantidad_muertes"].sum()
sns.barplot(x=muertes_mensuales.index, y=muertes_mensuales.values,palette="muted" )
plt.title("Suma de Muertes por mes")
plt.xlabel("Mes")
plt.ylabel("Cantidad de Muertes")
plt.show()

#Meses criticos vs meses "seguros"
plt.figure(figsize=(14,10))
ranking=df_muerte_mes.groupby("Mes")["Cantidad_muertes"].sum().sort_values(ascending=False)
sns.barplot(x=ranking.index, y=ranking.values, palette="muted")
plt.title("Meses criticos vs meses 'seguros'")
plt.xlabel("Mes")
plt.ylabel("Cantidad de Muertes")
plt.show()

#Variacion interanual dentro de cada mes

# Filter data for the years 2013 to 2023
df_muerte_mes_filtered = df_muerte_mes[(df_muerte_mes['A¤o'] >= 2013) & (df_muerte_mes['A¤o'] <= 2023)]

pivot = df_muerte_mes_filtered.pivot_table(index="A¤o", columns="Mes", values="Cantidad_muertes", aggfunc="sum")

# Sort months to ensure correct plotting order
month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
pivot = pivot[month_order]

plt.figure(figsize=(16,10))
pivot.T.plot(kind="line", figsize=(20,10) , marker='o' )
plt.title("Variación Interanual de Muertes por Mes (2013-2023)")
plt.xlabel("Mes")
plt.ylabel("Cantidad de Muertes")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7, color='gray') # Changed 'Spectral' to a valid color 'gray'
plt.legend(title="Año", bbox_to_anchor=(1.07, 1), loc='upper left')
plt.tight_layout()
plt.show()

df_lesion_mes_sexo

# Estacionalidades menusales de lesionados
lesionados_mensuales= df_lesion_mes_sexo.groupby("Mes")["Cantidad_lesionados"].mean()
plt.figure(figsize=(12,6))
sns.barplot(x=lesionados_mensuales.index, y=lesionados_mensuales.values,palette="Reds" )
plt.title("Pomedio de personas Lesionados por mes en accidentes de transito")
plt.ylabel("Promedio de personas lesionadas")
plt.xticks(rotation=45)
plt.show()

#Compracion de lesionados por sexo
plt.figure(figsize=(12,6))
sns.barplot(data=df_lesion_mes_sexo, x="Mes", y="Cantidad_lesionados", hue="Sexo", palette="muted")
plt.title("Cantidad de personas lesionadas por accidente- Distribucion por sexo")
plt.xticks(rotation=45)
plt.xlabel("Mes")
plt.ylabel("Cantidad de personas lesionadas")
plt.show()

#Heatmap mes vs ano
pivot=df_lesion_mes_sexo.pivot_table(index="Mes", columns="Ano", values="Cantidad_lesionados", aggfunc="sum")
sns.heatmap(pivot, cmap="Reds", annot=True, fmt=".0f")
plt.title("Heatmap de lesionados por mes y ano")
plt.show()

#Boxplot mensuales
plt.figure(figsize=(12,6))
sns.boxplot(data=df_lesion_mes_sexo, x="Mes", y="Cantidad_lesionados", palette="muted")
plt.title("Distribucion de lesionados por mes (deteccion de valores anormales)")
plt.xticks(rotation=45)
plt.xlabel("Mes")
plt.ylabel("Cantidad de lesionados")
plt.show()

#Variacion interanual dentro de cada mes

# Clean the 'Mes' column by stripping whitespace to avoid mismatches
df_lesion_mes_sexo['Mes'] = df_lesion_mes_sexo['Mes'].str.strip()

# Aggregate 'Cantidad_lesionados' by 'Ano' and 'Mes' to get total injured per month/year
df_lesion_aggregated = df_lesion_mes_sexo.groupby(['Ano', 'Mes'])['Cantidad_lesionados'].sum().reset_index()

# Ensure 'Mes' is categorical and ordered for correct plotting
month_order = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
df_lesion_aggregated['Mes'] = pd.Categorical(df_lesion_aggregated['Mes'], categories=month_order, ordered=True)

plt.figure(figsize=(16, 10))
sns.scatterplot(data=df_lesion_aggregated, x="Cantidad_lesionados", y="Mes", hue="Ano", palette='deep', s=100) # s for marker size
plt.title("Evolucion interanual de lesionados por mes (Scatter Plot)")
plt.xlabel("Cantidad de lesionados")
plt.ylabel("Mes")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title="Año", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
