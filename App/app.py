import gradio as gr
import joblib
import numpy as np

# ============================================
# 1. CARGAR EL MODELO Y LOS ENCODERS
# ============================================
model = joblib.load("Model/drug_model.pkl")
le_sex = joblib.load("Model/le_sex.pkl")
le_bp = joblib.load("Model/le_bp.pkl")
le_chol = joblib.load("Model/le_chol.pkl")

# ============================================
# 2. FUNCIÓN DE PREDICCIÓN
# ============================================
def predict_drug(age, sex, bp, cholesterol, na_to_k):
    """
    Recibe los datos del paciente y retorna la droga recomendada
    """
    # Convertir texto a números usando los encoders
    sex_encoded = le_sex.transform([sex])[0]
    bp_encoded = le_bp.transform([bp])[0]
    chol_encoded = le_chol.transform([cholesterol])[0]

    # Crear el arreglo de entrada para el modelo
    input_data = np.array([[age, sex_encoded, bp_encoded, chol_encoded, na_to_k]])

    # Hacer la predicción
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max() * 100

    return f"💊 Droga recomendada: **{prediction}**\n\n📊 Confianza: {probability:.2f}%"

# ============================================
# 3. CREAR LA INTERFAZ
# ============================================
demo = gr.Interface(
    fn=predict_drug,
    inputs=[
        gr.Slider(minimum=15, maximum=74, step=1, label="🎂 Edad del paciente", value=30),
        gr.Radio(choices=["M", "F"], label="⚧ Sexo", value="M"),
        gr.Radio(choices=["LOW", "NORMAL", "HIGH"], label="🩸 Presión arterial", value="NORMAL"),
        gr.Radio(choices=["NORMAL", "HIGH"], label="🧪 Colesterol", value="NORMAL"),
        gr.Slider(minimum=6.0, maximum=38.0, step=0.1, label="⚗️ Ratio Sodio/Potasio", value=15.0),
    ],
    outputs=gr.Textbox(label="🔬 Resultado de la predicción"),
    title="💊 Drug Classification - Predictor de Medicamentos",
    description="Ingresa los datos clínicos del paciente para predecir qué medicamento necesita.",
    examples=[
        [23, "F", "HIGH", "HIGH", 25.3],
        [47, "M", "LOW", "HIGH", 14.0],
        [35, "F", "NORMAL", "NORMAL", 10.5],
    ]
)

# ============================================
# 4. LANZAR LA APP
# ============================================
if __name__ == "__main__":
    demo.launch()