import numpy as np
import streamlit as st

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_values = np.exp(x - np.max(x))

    return exp_values / np.sum(exp_values)

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Laboratorio de Red Neuronal",
    layout="wide"
)

st.title("🧠 Laboratorio de Red Neuronal")

st.write(
    "Dibuja un número del 0 al 9. "
    "Por ahora solamente estudiaremos cómo se convierte "
    "el dibujo en información que una red neuronal pueda procesar."
)


# ============================================================
# LIENZO
# ============================================================

st.subheader("1. Entrada: dibuja un número")

canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    return_image_data=True,
    key="digit_canvas"
)

# ============================================================
# PROCESAMIENTO
# ============================================================

if canvas_result.image_data is not None:

    # Imagen producida por el canvas
    image = Image.fromarray(
        canvas_result.image_data.astype("uint8")
    )

    # Convertir a escala de grises
    gray_image = image.convert("L")

    # Reducir de 280x280 a 28x28
    image_28 = gray_image.resize(
        (28, 28),
        Image.Resampling.LANCZOS
    )

    # Convertir imagen a matriz NumPy
    pixels = np.array(
        image_28,
        dtype=np.float32
    )

    # Normalizar:
    # 0   → negro
    # 255 → blanco
    #
    # pasa a:
    #
    # 0.0 → negro
    # 1.0 → blanco

    pixels_normalized = pixels / 255.0

    # Convertir matriz 28x28
    # a vector de 784 valores

    input_vector = pixels_normalized.flatten()

    # ========================================================
    # PRIMERA NEURONA ARTIFICIAL
    # ========================================================

    # Generador aleatorio con semilla fija.
    # Así los pesos siempre serán iguales al reiniciar.
    rng = np.random.default_rng(42)

    # La neurona necesita un peso por cada entrada.
    weights = rng.normal(
        loc=0.0,
        scale=0.1,
        size=784
    )

    # Bias inicial
    bias = 0.0

    # Suma ponderada:
    #
    # x1*w1 + x2*w2 + ... + x784*w784 + bias

    weighted_sum = np.dot(
        input_vector,
        weights
    ) + bias

    # Función de activación
    activation = sigmoid(weighted_sum)
    # ========================================================
    # CAPA DE SALIDA: 10 NEURONAS
    # ========================================================

    rng_output = np.random.default_rng(123)

    # 784 entradas × 10 neuronas
    output_weights = rng_output.normal(
        loc=0.0,
        scale=0.1,
        size=(784, 10)
    )

    # Un bias por cada neurona
    output_biases = np.zeros(10)

    # Suma ponderada de las 10 neuronas
    logits = np.dot(
        input_vector,
        output_weights
    ) + output_biases

    # Convertimos los resultados a probabilidades
    probabilities = softmax(logits)

    # Número con mayor probabilidad
    prediction = np.argmax(probabilities)
    # ========================================================
    # VISUALIZACIÓN
    # ========================================================

    st.subheader("2. Lo que verá la red neuronal")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Imagen original")

        st.image(
            gray_image,
            width=280
        )

    with col2:

        st.write("### Imagen 28 × 28")

        enlarged = image_28.resize(
            (280, 280),
            Image.Resampling.NEAREST
        )

        st.image(
            enlarged,
            width=280
        )


    # ========================================================
    # INFORMACIÓN NUMÉRICA
    # ========================================================

    st.subheader("3. Información convertida a números")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Ancho",
        "28 píxeles"
    )

    col2.metric(
        "Alto",
        "28 píxeles"
    )

    col3.metric(
        "Entradas",
        len(input_vector)
    )


    st.write("### Primeros 40 valores de entrada")

    st.code(
        np.array2string(
            input_vector[:40],
            precision=2
        )
    )


    st.write("### Matriz 28 × 28")

    st.dataframe(
        pixels_normalized
    )

    st.subheader("4. Nuestra primera neurona")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Suma ponderada (z)",
        f"{weighted_sum:.4f}"
    )

    col2.metric(
        "Bias",
        f"{bias:.2f}"
    )

    col3.metric(
        "Activación",
        f"{activation:.4f}"
    )

    st.write("### Algunos pesos de la neurona")

    weight_data = {
        "Entrada": [
            f"x{i}"
            for i in range(20)
        ],
        "Valor píxel": input_vector[:20],
        "Peso": weights[:20],
        "x · w": input_vector[:20] * weights[:20]
    }

    st.dataframe(
        weight_data
    )

    st.subheader("5. Primera capa de clasificación")

    st.write(
        "Cada barra representa la activación de una "
        "de las 10 neuronas de salida."
    )

    probability_data = {
        "Número": list(range(10)),
        "Probabilidad": probabilities
    }

    st.bar_chart(
        probability_data,
        x="Número",
        y="Probabilidad"
    )

    st.metric(
        "Predicción actual",
        int(prediction)
    )