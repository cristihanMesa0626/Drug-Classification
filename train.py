import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import os

# ============================================
# 1. CARGAR LOS DATOS
# ============================================
# Usamos un dataset público de clasificación de drogas

df = pd.read_csv("drug200.csv")


print("✅ Datos cargados correctamente")
print(f"   Forma del dataset: {df.shape}")
print(f"   Columnas: {list(df.columns)}")

# ============================================
# 2. PREPARAR LOS DATOS
# ============================================
# Separamos las características (X) de lo que queremos predecir (y)
X = df.drop("Drug", axis=1)  # Todo menos la columna Drug
y = df["Drug"]               # Solo la columna Drug (lo que predecimos)

# Los modelos ML solo entienden números, no texto
# LabelEncoder convierte texto a números
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()

X = X.copy()
X["Sex"] = le_sex.fit_transform(X["Sex"])           # M/F → 0/1
X["BP"] = le_bp.fit_transform(X["BP"])              # LOW/NORMAL/HIGH → 0/1/2
X["Cholesterol"] = le_chol.fit_transform(X["Cholesterol"])  # NORMAL/HIGH → 0/1

print("\n✅ Datos preparados")

# ============================================
# 3. DIVIDIR EN ENTRENAMIENTO Y PRUEBA
# ============================================
# 80% para entrenar, 20% para evaluar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Datos de entrenamiento: {X_train.shape[0]} filas")
print(f"   Datos de prueba: {X_test.shape[0]} filas")

# ============================================
# 4. ENTRENAR EL MODELO
# ============================================
# RandomForest es como un "comité de expertos" que vota para decidir
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n✅ Modelo entrenado correctamente")

# ============================================
# 5. GUARDAR EL MODELO Y LOS ENCODERS
# ============================================
os.makedirs("Model", exist_ok=True)

joblib.dump(model, "Model/drug_model.pkl")
joblib.dump(le_sex, "Model/le_sex.pkl")
joblib.dump(le_bp, "Model/le_bp.pkl")
joblib.dump(le_chol, "Model/le_chol.pkl")

# Guardamos los datos de prueba para usarlos en evaluate.py
X_test.to_csv("Model/X_test.csv", index=False)
y_test.to_csv("Model/y_test.csv", index=False)

print("✅ Modelo guardado en /Model")
print("\n🎉 Entrenamiento completado exitosamente!")