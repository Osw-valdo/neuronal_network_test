import numpy as np
import streamlit as st

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