import numpy as np
import streamlit as st

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# FUNCIONES DE ACTIVACIÓN
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    # Restamos el máximo para evitar problemas numéricos
    exp_values = np.exp(x - np.max(x))

    return exp_values / np.sum(exp_values)


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
    "La red procesará los píxeles e intentará clasificarlo. "
    "Después podrás indicarle cuál era el número correcto para enseñarle."
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
# PROCESAMIENTO DE IMAGEN
# ============================================================

if canvas_result.image_data is not None:

    # --------------------------------------------------------
    # Imagen original del canvas
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NORMALIZACIÓN
    # --------------------------------------------------------
    #
    # Original:
    # 0   = negro
    # 255 = blanco
    #
    # Normalizado:
    # 0.0 = negro
    # 1.0 = blanco
    # --------------------------------------------------------

    pixels_normalized = pixels / 255.0

    # Convertir matriz 28x28
    # en vector de 784 valores
    input_vector = pixels_normalized.flatten()


    # ========================================================
    # PRIMERA NEURONA ARTIFICIAL
    # ========================================================
    #
    # Esta neurona es solamente educativa.
    # NO aprende.
    #
    # Sirve para estudiar:
    #
    # entradas
    #    ↓
    # pesos
    #    ↓
    # suma ponderada
    #    ↓
    # bias
    #    ↓
    # sigmoid
    #
    # ========================================================

    rng = np.random.default_rng(42)

    weights = rng.normal(
        loc=0.0,
        scale=0.1,
        size=784
    )

    bias = 0.0

    weighted_sum = np.dot(
        input_vector,
        weights
    ) + bias

    activation = sigmoid(
        weighted_sum
    )


    # ========================================================
    # RED ENTRENABLE
    # ========================================================
    #
    # Arquitectura actual:
    #
    # 784 entradas
    #      ↓
    # 10 neuronas de salida
    #      ↓
    # Softmax
    #
    # ========================================================

    if "output_weights" not in st.session_state:

        rng_output = np.random.default_rng(123)

        # 784 entradas × 10 neuronas
        st.session_state.output_weights = rng_output.normal(
            loc=0.0,
            scale=0.1,
            size=(784, 10)
        )

        # Un bias por neurona
        st.session_state.output_biases = np.zeros(10)

        # Historial de entrenamiento
        st.session_state.training_count = 0
        st.session_state.correct_history = []
        st.session_state.loss_history = []


    # Obtener pesos actuales
    output_weights = st.session_state.output_weights
    output_biases = st.session_state.output_biases


    # ========================================================
    # FORWARD PASS
    # ========================================================
    #
    # Entrada
    #    ↓
    # suma ponderada
    #    ↓
    # logits
    #    ↓
    # softmax
    #    ↓
    # probabilidades
    #
    # ========================================================

    logits = np.dot(
        input_vector,
        output_weights
    ) + output_biases

    probabilities = softmax(
        logits
    )

    prediction = int(
        np.argmax(probabilities)
    )


    # ========================================================
    # VISUALIZACIÓN DE LA ENTRADA
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


    # ========================================================
    # PRIMERA NEURONA EDUCATIVA
    # ========================================================

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
        "Activación Sigmoid",
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


    # ========================================================
    # CLASIFICACIÓN
    # ========================================================

    st.subheader("5. Primera capa de clasificación")

    st.write(
        "Cada barra representa la probabilidad asignada "
        "a cada número por las 10 neuronas de salida."
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


    col1, col2 = st.columns(2)

    col1.metric(
        "Predicción actual",
        prediction
    )

    col2.metric(
        "Confianza",
        f"{probabilities[prediction] * 100:.2f} %"
    )


    # ========================================================
    # ENSEÑAR A LA RED
    # ========================================================

    st.subheader("6. Enseñar a la red")

    correct_number = st.selectbox(
        "¿Qué número dibujaste?",
        options=list(range(10))
    )

    learning_rate = st.slider(
        "Learning rate",
        min_value=0.001,
        max_value=0.5,
        value=0.05,
        step=0.001
    )


    if st.button("🧠 Enseñar a la red"):

        # ----------------------------------------------------
        # TARGET / ONE-HOT ENCODING
        # ----------------------------------------------------

        target = np.zeros(10)

        target[correct_number] = 1


        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------
        #
        # Cross Entropy Loss
        #
        # Cuanto menor sea, mejor.
        #
        # ----------------------------------------------------

        loss = -np.log(
            probabilities[correct_number] + 1e-12
        )


        # ----------------------------------------------------
        # GRADIENTE DE LOS LOGITS
        # ----------------------------------------------------

        gradient_logits = (
            probabilities - target
        )


        # ----------------------------------------------------
        # GRADIENTE DE LOS PESOS
        # ----------------------------------------------------

        gradient_weights = np.outer(
            input_vector,
            gradient_logits
        )


        # ----------------------------------------------------
        # GRADIENTE DE LOS BIAS
        # ----------------------------------------------------

        gradient_biases = gradient_logits


        # ----------------------------------------------------
        # GRADIENT DESCENT
        # ----------------------------------------------------

        st.session_state.output_weights -= (
            learning_rate
            * gradient_weights
        )

        st.session_state.output_biases -= (
            learning_rate
            * gradient_biases
        )


        # ----------------------------------------------------
        # HISTORIAL
        # ----------------------------------------------------

        st.session_state.training_count += 1

        was_correct = int(
            prediction == correct_number
        )

        st.session_state.correct_history.append(
            was_correct
        )

        st.session_state.loss_history.append(
            float(loss)
        )


        st.success(
            f"Ejemplo aprendido. "
            f"Predicción: {prediction} | "
            f"Correcto: {correct_number} | "
            f"Loss: {loss:.4f}"
        )


    # ========================================================
    # ESTADO DEL ENTRENAMIENTO
    # ========================================================

    st.subheader("7. Estado del aprendizaje")

    total_training = (
        st.session_state.training_count
    )

    if total_training > 0:

        accuracy = (
            np.mean(
                st.session_state.correct_history
            )
            * 100
        )

        last_loss = (
            st.session_state.loss_history[-1]
        )

    else:

        accuracy = 0.0
        last_loss = 0.0


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Ejemplos enseñados",
        total_training
    )

    col2.metric(
        "Accuracy acumulado",
        f"{accuracy:.2f} %"
    )

    col3.metric(
        "Último Loss",
        f"{last_loss:.4f}"
    )