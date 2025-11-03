# app_streamlit_hr_levels_tree.py
# Streamlit: Compañía / Departamento / Individual / Árbol del modelo.
# Ejecuta: streamlit run app_streamlit_hr_levels_tree.py
# Requisitos: streamlit pandas scikit-learn plotly matplotlib

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.io as pio
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
from sklearn import tree as sktree
import matplotlib.pyplot as plt
import io

# =========================
# CONFIGURACIÓN
# =========================
st.set_page_config(page_title="HR — Niveles y Árbol", layout="wide")

DATA_PATH = "HRDataset_v14.csv"
TARGET = "PerfScoreID"
FEATURES = [
    "EmpSatisfaction",
    "EngagementSurvey",
    "Salary",
    "SpecialProjectsCount",
    "Absences",
    "DaysLateLast30",
]
LEVEL_MIN, LEVEL_MAX = 1, 5
COMPANY_LEVELS_TO_SHOW = [1, 2, 3, 4]

pio.templates.default = "plotly_white"
DEFAULT_SEQ = px.colors.qualitative.Set2


def style_fig(fig):
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    return fig


# =========================
# ESTILO
# =========================
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 15px; }
      h1 { font-size: 1.6rem !important; margin-bottom: .2rem; }
      h2 { font-size: 1.3rem !important; margin-bottom: .2rem; }
      h3 { font-size: 1.1rem !important; margin-bottom: .2rem; }
      .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(DATA_PATH)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas en el CSV: {missing}")
    st.stop()

if "Department" not in df.columns:
    st.error("El CSV no tiene la columna 'Department'.")
    st.stop()

# =========================
# MODELO
# =========================
X = df[FEATURES].copy()
y = df[TARGET].astype(float) if TARGET in df.columns else np.full(len(df), 3.0)
tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, random_state=42)
tree.fit(X, y)


def to_level(pred: np.ndarray) -> np.ndarray:
    arr = np.rint(pred).astype(int)
    return np.clip(arr, LEVEL_MIN, LEVEL_MAX)


df["_pred_base"] = tree.predict(df[FEATURES])
df["_level_base"] = to_level(df["_pred_base"])

id_cols = [c for c in ["EmpID", "Employee_Name"] if c in df.columns]
id_col = id_cols[0] if id_cols else "_row_id"
if "_row_id" not in df.columns:
    df["_row_id"] = df.index

# =========================
# PALANCAS
# =========================
st.sidebar.header("Palancas (±100%)")
delta_salary_pct = st.sidebar.slider("Salary (%)", -100, 100, 0, 1)
delta_projects_pct = st.sidebar.slider("SpecialProjectsCount (%)", -100, 100, 0, 1)
delta_absences_pct = st.sidebar.slider("Absences (%)", -100, 100, 0, 1)


def apply_levers(subset: pd.DataFrame) -> pd.DataFrame:
    out = subset.copy()
    out["Salary"] = out["Salary"] * (1 + delta_salary_pct / 100.0)
    out["SpecialProjectsCount"] = out["SpecialProjectsCount"] * (1 + delta_projects_pct / 100.0)
    out["Absences"] = out["Absences"] * (1 + delta_absences_pct / 100.0)
    out = out.clip(lower=0)
    return out


# =========================
# INTERFAZ
# =========================
st.title("📊 HR — Niveles y Árbol de Decisión")
st.caption("Modelo predictivo de niveles basado en HRDataset_v14.csv (solo predicciones).")

# Tabs principales
tab_company, tab_dept, tab_individual, tab_tree = st.tabs([
    "🏢 Compañía",
    "🏬 Departamento",
    "👤 Individual",
    "🌳 Árbol del modelo"
])

# =========================
# COMPAÑÍA
# =========================
with tab_company:
    st.header("Distribución por nivel")
    mask = df["_level_base"].isin(COMPANY_LEVELS_TO_SHOW)
    dist = df.loc[mask, "_level_base"].value_counts().reindex(COMPANY_LEVELS_TO_SHOW, fill_value=0)
    dist_tbl = pd.DataFrame({"Nivel": dist.index, "Personas": dist.values})
    dist_tbl["%"] = (dist_tbl["Personas"] / dist_tbl["Personas"].sum() * 100).round(1)

    fig = px.bar(dist_tbl, x="Nivel", y="Personas", text="Personas", title="Distribución por nivel (predicho)")
    fig.update_traces(textposition="outside")
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dist_tbl, hide_index=True, use_container_width=True)

# =========================
# DEPARTAMENTO
# =========================
with tab_dept:
    st.header("Comparar departamentos")
    dept_opts = sorted(df["Department"].dropna().unique())
    dept_sel = st.selectbox("Departamento", dept_opts)

    df_dept = df[df["Department"] == dept_sel]
    dist = df_dept["_level_base"].value_counts().sort_index()
    tbl = pd.DataFrame({"Nivel": dist.index, "Personas": dist.values})
    tbl["%"] = (tbl["Personas"] / tbl["Personas"].sum() * 100).round(1)

    fig = px.bar(tbl, x="Nivel", y="Personas", text="Personas", title=f"Distribución — {dept_sel}")
    fig.update_traces(textposition="outside")
    style_fig(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(tbl, use_container_width=True, hide_index=True)

# =========================
# INDIVIDUAL
# =========================
with tab_individual:
    st.header("Predicción individual")
    emp_opts = df[id_col].astype(str).tolist()
    emp_sel = st.selectbox("Empleado", emp_opts)

    row = df[df[id_col].astype(str) == emp_sel].iloc[0:1].copy()
    st.write("**Datos actuales:**")
    st.dataframe(row[FEATURES], use_container_width=True)

    pred = float(tree.predict(row[FEATURES])[0])
    st.metric("PerfScoreID predicho", f"{pred:.3f}")
    st.metric("Nivel predicho", f"{to_level([pred])[0]}")

# =========================
# ÁRBOL DEL MODELO
# =========================
with tab_tree:
    st.header("🌳 Árbol de Decisión del Modelo")
    st.caption("Visualización del árbol entrenado (profundidad ≤ 5).")

    # Estructura textual
    st.subheader("Estructura textual del árbol")
    tree_text = sktree.export_text(tree, feature_names=FEATURES, show_weights=True)
    st.text(tree_text)

    # Visualización gráfica
    st.subheader("Visualización gráfica")
    fig, ax = plt.subplots(figsize=(18, 8))
    sktree.plot_tree(tree, feature_names=FEATURES, filled=True, rounded=True, fontsize=8)
    st.pyplot(fig)

    # Descargar PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    st.download_button(
        "⬇️ Descargar árbol como imagen (PNG)",
        data=buf.getvalue(),
        file_name="arbol_modelo.png",
        mime="image/png",
        use_container_width=True
    )

    # Exportar DOT
    st.subheader("Exportar a formato DOT (Graphviz)")
    dot_data = sktree.export_graphviz(
        tree,
        out_file=None,
        feature_names=FEATURES,
        filled=True,
        rounded=True,
        special_characters=True
    )
    st.download_button(
        "⬇️ Descargar archivo .dot",
        data=dot_data.encode("utf-8"),
        file_name="arbol_modelo.dot",
        mime="text/plain",
        use_container_width=True
    )

# =========================
# PIE DE PÁGINA
# =========================
st.caption(
    "• Este tablero usa predicciones del modelo DecisionTreeRegressor (profundidad 5). "
    "• La pestaña 'Árbol del modelo' permite inspeccionar y exportar su estructura."
)
