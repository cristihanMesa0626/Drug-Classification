import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import os

# ============================================
# 1. CARGAR EL MODELO Y LOS DATOS DE PRUEBA
# ============================================
print("📦 Cargando modelo y datos...")

model = joblib.load("Model/drug_model.pkl")
X_test = pd.read_csv("Model/X_test.csv")
y_test = pd.read_csv("Model/y_test.csv").squeeze()  # squeeze convierte DataFrame a Serie

# ============================================
# 2. HACER PREDICCIONES
# ============================================
y_pred = model.predict(X_test)
print("✅ Predicciones realizadas")

# ============================================
# 3. CALCULAR MÉTRICAS
# ============================================
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n📊 Accuracy del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n📋 Reporte completo:")
print(report)

# ============================================
# 4. GUARDAR MÉTRICAS EN ARCHIVO DE TEXTO
# ============================================
os.makedirs("Results", exist_ok=True)

with open("Results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print("✅ Métricas guardadas en Results/metrics.txt")

# ============================================
# 5. GENERAR Y GUARDAR LA MATRIZ DE CONFUSIÓN
# ============================================
# La matriz de confusión muestra cuántas veces el modelo
# acertó y en qué se equivocó para cada tipo de droga

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,          # Muestra los números dentro
    fmt="d",             # Formato entero
    cmap="Blues",        # Color azul
    xticklabels=model.classes_,
    yticklabels=model.classes_
)
plt.title("Matriz de Confusión - Drug Classification", fontsize=14)
plt.ylabel("Valor Real", fontsize=12)
plt.xlabel("Valor Predicho", fontsize=12)
plt.tight_layout()
plt.savefig("Results/model_results.png", dpi=100)
plt.close()

print("✅ Gráfica guardada en Results/model_results.png")
print("\n🎉 Evaluación completada exitosamente!")