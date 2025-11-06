
import io
import time
import pandas as pd
import streamlit as st
from transformers import pipeline

# ---------- Configuración general ----------
st.set_page_config(
    page_title="Análisis de Sentimientos (ES) — BETO",
    page_icon="🧠",
    layout="wide"
)

# ---------- Estilos (CSS ligero) ----------
st.markdown(
    """
    <style>
    .app-header h1 { font-size: 2rem; margin-bottom: 0.2rem; }
    .subtle { color: #6b7280; }
    .result-card {
        padding: 1rem 1.25rem; border-radius: 14px; border: 1px solid #e5e7eb;
        background: linear-gradient(180deg, #ffffff 0%, #fafafa 100%);
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .pill {
        display:inline-block; padding:.25rem .6rem; border-radius:999px; font-size:.85rem;
        border:1px solid #e5e7eb; background:#fff;
    }
    .footer-note { font-size: 0.9rem; color: #6b7280; }
    .stDownloadButton > button { width: 100%; }
    .gray { color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Carga del pipeline ----------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    return pipeline("sentiment-analysis", model="finiteautomata/beto-sentiment-analysis")

clf = load_pipeline()

# ---------- Mapeo de etiquetas ----------
MAP_ES = {"POS": "positivo", "NEG": "negativo", "NEU": "neutral"}

def format_sentiment(res):
    raw = res.get("label", "")
    score = float(res.get("score", 0.0))
    label_es = MAP_ES.get(raw, raw.lower())
    return label_es, score

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.caption("Modelo: `finiteautomata/beto-sentiment-analysis`")
    st.markdown("---")
    st.markdown("**Modo de uso**")
    modo = st.radio("Selecciona:", ["Texto único", "CSV por lote"], index=0)
    st.markdown("---")
    st.markdown("**Instrucciones**")
    st.write("• Escribe un texto en español o sube un CSV con la columna **texto**.")
    st.write("• Descarga resultados con etiquetas en español y puntajes de confianza.")
    st.write("• Ideal para demos, benchmarking y prototipos.")

# ---------- Encabezado ----------
st.markdown('<div class="app-header">', unsafe_allow_html=True)
st.markdown("## 🧠 Análisis de Sentimientos — Español (BETO)")
st.markdown('<p class="subtle">Interfaz profesional y lista para portafolio</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Contenido ----------
if modo == "Texto único":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        ejemplo = "Estoy satisfecho con el servicio, gracias por la atención."
        texto = st.text_area("Escribe un texto en español", value=ejemplo, height=140, placeholder="Escribe aquí...")
        ejecutar = st.button("Analizar sentimiento", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### ℹ️ Sugerencias")
        st.write("- Usa oraciones completas para mejor contexto.")
        st.write("- El modelo devuelve **positivo, negativo o neutral**.")
        st.write("- Se muestra la **confianza** como porcentaje.")

    st.markdown("")
    if ejecutar:
        with st.spinner("Analizando..."):
            res = clf(texto)[0]
            label_es, score = format_sentiment(res)
            conf_pct = f"{score*100:.2f}%"
            time.sleep(0.15)

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("##### Resultado")
            st.markdown(f"**El análisis de sentimientos es:** :blue[**{label_es}**]")
            st.write(f"Confianza: **{conf_pct}**")
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown("##### Detalle")
            st.caption("Etiqueta original del modelo (en inglés):")
            st.write(f"`{res.get('label','')}`")
            st.caption("Puntaje bruto:")
            st.write(res.get("score", 0.0))

        st.markdown("")
        st.markdown('<p class="footer-note">*Nota:* El modelo puede cometer errores en textos muy cortos o irónicos.</p>', unsafe_allow_html=True)

else:
    st.markdown("#### 📦 Predicción por lote (CSV)")
    st.caption("Sube un CSV con **una columna** llamada `texto`.")
    file = st.file_uploader("Selecciona tu archivo CSV", type=["csv"], accept_multiple_files=False)

    plantilla = pd.DataFrame({"texto": ["Me encanta este producto", "El servicio fue terrible", "Está bien, nada especial"]})
    st.download_button(
        "Descargar plantilla CSV",
        data=plantilla.to_csv(index=False).encode("utf-8-sig"),
        file_name="plantilla_texto.csv",
        mime="text/csv"
    )

    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()

        if "texto" not in df.columns:
            st.error("El CSV debe contener una columna llamada **texto**.")
            st.stop()

        st.info(f"Filas detectadas: **{len(df)}**")
        start = st.button("Analizar CSV", type="primary", use_container_width=True)

        if start:
            progreso = st.progress(0)
            resultados = []
            textos = df["texto"].astype(str).tolist()
            total = len(textos)
            for i, t in enumerate(textos, start=1):
                res = clf(t)[0]
                label_es, score = format_sentiment(res)
                resultados.append({"texto": t, "sentimiento": label_es, "confianza": round(score, 6)})
                if i % 5 == 0 or i == total:
                    progreso.progress(min(i/total, 1.0))

            out = pd.DataFrame(resultados)
            st.success("¡Predicciones listas!")
            st.dataframe(out, use_container_width=True, hide_index=True)

            # Descarga
            csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Descargar resultados (CSV)",
                data=csv_bytes,
                file_name="predicciones_sentimiento.csv",
                mime="text/csv"
            )

# ---------- Pie ----------
st.markdown("---")
st.caption("Hecho con ❤️ usando Streamlit y Hugging Face Transformers · Modelo: finiteautomata/beto-sentiment-analysis")
