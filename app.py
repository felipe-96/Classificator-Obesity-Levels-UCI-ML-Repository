import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from scipy.stats import mstats
import os

# ══════════════════════════════════════════
# PALETA USS + CONFIG
# ══════════════════════════════════════════
USS_RED      = "#1E3A5F"
USS_NAVY      = "#0F2747"
USS_GRAY      = "#0B0F14"
USS_DARK      = "#05070A"
USS_WHITE     = "#FFFFFF"
USS_ACCENT    = "#2C5D8A"
USS_LIGHT     = "#E8F1FA"

# Paleta de 7 clases (degradado de verde a rojo oscuro)
CLASE_COLORS = [
    "#27AE60",   # Insufficient_Weight  — verde
    "#2ECC71",   # Normal_Weight        — verde claro
    "#F39C12",   # Overweight_Level_I   — naranja
    "#E67E22",   # Overweight_Level_II  — naranja oscuro
    "#E74C3C",   # Obesity_Type_I       — rojo
    "#C0392B",   # Obesity_Type_II      — rojo oscuro
    "#7B241C",   # Obesity_Type_III     — granate
]

ORDEN_CLASES = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I",  "Overweight_Level_II",
    "Obesity_Type_I",      "Obesity_Type_II",  "Obesity_Type_III"
]
LABELS_ES = [
    "Peso Insuficiente", "Peso Normal",
    "Sobrepeso I", "Sobrepeso II",
    "Obesidad I",  "Obesidad II",  "Obesidad III"
]
LABEL_MAP  = dict(zip(ORDEN_CLASES, LABELS_ES))
COLOR_MAP  = dict(zip(ORDEN_CLASES, CLASE_COLORS))

st.set_page_config(
    page_title="Clasificador de Obesidad · USS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS USS
st.markdown(f"""
<style>
  /* Fondo general */
  .stApp {{ background-color: {USS_GRAY}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, {USS_NAVY} 0%, #0F1C30 100%);
  }}
  [data-testid="stSidebar"] * {{ color: {USS_WHITE} !important; }}
  [data-testid="stSidebar"] .stRadio label {{ color: {USS_WHITE} !important; }}

  /* Títulos */
  h1 {{ color: {USS_NAVY}; border-bottom: 4px solid {USS_RED}; padding-bottom: 8px; }}
  h2 {{ color: {USS_NAVY}; }}
  h3 {{ color: {USS_RED}; }}

  /* Métricas */
  [data-testid="stMetric"] {{
      background: {USS_WHITE};
      border-radius: 10px;
      padding: 16px;
      border-left: 5px solid {USS_RED};
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  [data-testid="stMetricLabel"] {{ color: {USS_NAVY} !important; font-weight: 700; }}
  [data-testid="stMetricValue"] {{ color: {USS_RED} !important; font-size: 2rem !important; }}

  /* Botón principal */
  .stButton > button, .stFormSubmitButton > button {{
      background: {USS_RED} !important;
      color: white !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 700 !important;
      font-size: 1rem !important;
      padding: 0.6rem 1.5rem !important;
      transition: background 0.2s;
  }}
  .stButton > button:hover, .stFormSubmitButton > button:hover {{
      background: {USS_ACCENT} !important;
  }}

  /* Tarjetas info */
  .uss-card {{
      background: {USS_WHITE};
      border-radius: 12px;
      padding: 20px 24px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.07);
      margin-bottom: 16px;
  }}
  .uss-result {{
      border-left: 6px solid {USS_RED};
  }}

  /* Divider */
  hr {{ border-color: {USS_RED}44; }}

  /* Selectbox / slider */
  [data-baseweb="select"] {{ border-color: {USS_NAVY} !important; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
# CARGA Y ENTRENAMIENTO
# ══════════════════════════════════════════
@st.cache_data(show_spinner="⏳ Entrenando modelo Random Forest…")
def cargar_y_entrenar():
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=544)
        X_raw = dataset.data.features
        y_raw = dataset.data.targets.squeeze()
        fuente = "UCI ML Repository (id=544)"
    except Exception:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        df  = pd.read_csv(os.path.join(base_dir, "obesity_data.csv"))
        y_raw = df["NObeyesdad"]
        X_raw = df.drop(columns=["NObeyesdad"])
        fuente = "Dataset local (réplica UCI 544)"

    # Limpieza
    df_c = X_raw.copy(); y_c = y_raw.copy()
    mask = ~df_c.duplicated(keep="first")
    df_c = df_c[mask].reset_index(drop=True)
    y_c  = y_c[mask].reset_index(drop=True)

    for col in ["NCP","CH2O","FAF","TUE","FCVC"]:
        if col in df_c.columns:
            df_c[col] = df_c[col].round().astype(float)

    for col in df_c.select_dtypes(include="number").columns:
        datos = df_c[col].dropna()
        Q1, Q3 = datos.quantile(0.25), datos.quantile(0.75)
        if ((datos < Q1-1.5*(Q3-Q1)) | (datos > Q3+1.5*(Q3-Q1))).sum() > 20:
            df_c[col] = mstats.winsorize(df_c[col], limits=[0.01,0.01])

    # Feature Engineering
    df_fe = df_c.copy()
    favc_b = (df_c["FAVC"]=="yes").astype(float)
    fam_b  = (df_c["family_history_with_overweight"]=="yes").astype(float)
    df_fe["IMC"]               = df_c["Weight"] / df_c["Height"]**2
    df_fe["Riesgo_Calorico"]   = favc_b * (4 - df_c["FAF"].clip(0,3))
    df_fe["Actividad_Neta"]    = df_c["FAF"] - df_c["TUE"]
    df_fe["Carga_Hidratacion"] = df_c["CH2O"] / (df_c["NCP"]+1)
    df_fe["Herencia_Riesgo"]   = fam_b * favc_b

    # Encoding
    df_f = df_fe.copy()
    cols_cat = df_f.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cols_cat:
        le = LabelEncoder()
        df_f[col] = le.fit_transform(df_f[col].astype(str))
        encoders[col] = le

    le_y  = LabelEncoder()
    y_enc = le_y.fit_transform(y_c)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_f)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)

    y_pred  = rf.predict(X_te)
    y_proba = rf.predict_proba(X_te)
    acc     = accuracy_score(y_te, y_pred)
    auc     = roc_auc_score(
        label_binarize(y_te, classes=list(range(7))),
        y_proba, multi_class="ovr", average="macro")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(rf, X_scaled, y_enc, cv=skf, scoring="accuracy")
    cm  = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred,
                                   target_names=le_y.classes_, output_dict=True)
    pca2    = PCA(n_components=2)
    X_pca2  = pca2.fit_transform(X_scaled)
    exp_var = PCA().fit(X_scaled).explained_variance_ratio_

    return dict(
        rf=rf, scaler=scaler, encoders=encoders, le_y=le_y,
        y_enc=y_enc, y_clean=y_c, X_scaled=X_scaled,
        X_pca2=X_pca2, exp_var=exp_var,
        acc=acc, auc=auc, cv=cv, cm=cm, report=report,
        feat_names=list(df_f.columns),
        importances=rf.feature_importances_,
        df_clean=df_c, df_fe=df_fe,
        X_train=X_tr, y_train=y_tr, y_test=y_te, y_pred=y_pred,
        fuente=fuente,
    )


def predecir(datos_usuario, R):
    df_u = pd.DataFrame([datos_usuario])
    for col, le in R["encoders"].items():
        if col in df_u.columns:
            val = str(df_u[col].values[0])
            if val not in le.classes_: val = le.classes_[0]
            df_u[col] = le.transform([val])
    h = float(df_u["Height"].values[0]); w = float(df_u["Weight"].values[0])
    faf = float(df_u["FAF"].values[0]);  tue = float(df_u["TUE"].values[0])
    ch2o= float(df_u["CH2O"].values[0]); ncp = float(df_u["NCP"].values[0])
    favc= float(df_u["FAVC"].values[0]); fam = float(df_u["family_history_with_overweight"].values[0])
    df_u["IMC"]               = w / h**2
    df_u["Riesgo_Calorico"]   = favc * (4 - min(faf,3))
    df_u["Actividad_Neta"]    = faf - tue
    df_u["Carga_Hidratacion"] = ch2o / (ncp+1)
    df_u["Herencia_Riesgo"]   = fam * favc
    df_u = df_u[R["feat_names"]]
    X_u  = R["scaler"].transform(df_u.values.astype(float))
    pred_enc   = R["rf"].predict(X_u)[0]
    pred_proba = R["rf"].predict_proba(X_u)[0]
    clase      = R["le_y"].inverse_transform([pred_enc])[0]
    return clase, pred_proba, R["le_y"].classes_


# ══════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <div style="font-size:42px;">🏥</div>
        <div style="font-size:20px; font-weight:800; letter-spacing:1px;">USS</div>
        <div style="font-size:11px; opacity:0.75; margin-top:2px;">
            Universidad San Sebastián
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    pagina = st.radio(
        "Navegación",
        ["📈  Análisis Exploratorio",
         "📊  Resultados del Modelo",
         "🔮  Predecir Nuevo Dato"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px; opacity:0.8; line-height:1.8;">
    <b>Modelo:</b> Random Forest 100 árboles<br>
    <b>Split:</b> 75 / 25 estratificado<br>
    <b>CV:</b> StratifiedKFold 5-fold<br>
    <b>Dataset:</b> UCI ML Repo id=544<br>
    <b>Clases:</b> 7 niveles OMS<br>
    <b>Curso:</b> Taller de Aplicaciones<br>
    <b>Prof.:</b> Dr. Mauricio Sepúlveda<br>
    <b>Alumnos:</b> F. Carrasco · C. Tello
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════
# CARGA
# ══════════════════════════════════════════
R = cargar_y_entrenar()

NUEVAS = {"IMC","Riesgo_Calorico","Actividad_Neta","Carga_Hidratacion","Herencia_Riesgo"}

# helper Plotly layout
def uss_layout(fig, title="", height=None):
    kwargs = dict(
        title=dict(text=title, font=dict(color=USS_NAVY, size=15, family="Arial Black")),
        paper_bgcolor="white", plot_bgcolor="#FAFAFA",
        font=dict(family="Arial", color=USS_DARK),
        margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    if height: kwargs["height"] = height
    fig.update_layout(**kwargs)
    fig.update_xaxes(gridcolor="#EEEEEE", zerolinecolor="#CCCCCC")
    fig.update_yaxes(gridcolor="#EEEEEE", zerolinecolor="#CCCCCC")
    return fig


# ══════════════════════════════════════════════════════════
# PÁGINA 1 — ANÁLISIS EXPLORATORIO
# ══════════════════════════════════════════════════════════
if pagina == "📈  Análisis Exploratorio":
    st.title("📈 Análisis Exploratorio de Datos")
    st.caption(f"Dataset: Obesity Levels · {R['fuente']} · {len(R['y_clean'])} muestras · 7 clases OMS")

    # ── Conteos de clase ──
    conteos = R["y_clean"].value_counts().reindex(ORDEN_CLASES)
    total   = conteos.sum()

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=[LABEL_MAP[c] for c in ORDEN_CLASES],
            y=conteos.values,
            marker_color=CLASE_COLORS,
            text=[f"{v}<br>({v/total*100:.1f}%)" for v in conteos.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Frecuencia: %{y}<extra></extra>",
        ))
        fig.add_hline(y=total/7, line_dash="dot", line_color=USS_RED,
                      annotation_text="Distribución ideal", annotation_position="top right")
        uss_layout(fig, "Distribución de Clases", height=380)
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(go.Pie(
            labels=LABELS_ES, values=conteos.values,
            marker_colors=CLASE_COLORS,
            hole=0.38,
            textinfo="percent",
            hovertemplate="<b>%{label}</b><br>%{value} muestras (%{percent})<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{total}</b><br>muestras",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color=USS_NAVY))
        uss_layout(fig, "Proporción de Clases", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Variables numéricas por clase (violin) ──
    st.subheader("Distribución de Variables Numéricas por Clase")
    vars_num = ["Age","Height","Weight","FAF"]
    nombres_num = ["Edad (años)","Altura (m)","Peso (kg)","Actividad Física (días/sem)"]

    var_sel = st.selectbox("Selecciona variable:", vars_num,
                           format_func=lambda x: nombres_num[vars_num.index(x)])

    df_plot = R["df_clean"][["Age","Height","Weight","FAF"]].copy()
    df_plot["Clase"] = [LABEL_MAP.get(c, c) for c in R["y_clean"].values]
    df_plot["Color"] = [COLOR_MAP.get(c, "#888") for c in R["y_clean"].values]

    fig = go.Figure()
    for clase_key, clase_es, color in zip(ORDEN_CLASES, LABELS_ES, CLASE_COLORS):
        mask = R["y_clean"].values == clase_key
        fig.add_trace(go.Violin(
            x=[clase_es]*mask.sum(),
            y=df_plot.loc[mask, var_sel],
            name=clase_es,
            fillcolor=color,
            line_color=USS_NAVY,
            opacity=0.82,
            box_visible=True,
            meanline_visible=True,
            points="outliers",
            hovertemplate=f"<b>{clase_es}</b><br>{var_sel}: %{{y:.2f}}<extra></extra>",
        ))
    uss_layout(fig, f"Distribución de {nombres_num[vars_num.index(var_sel)]} por Nivel de Obesidad", height=420)
    fig.update_layout(showlegend=False, violingap=0.05, violinmode="group")
    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── IMC por clase (box) ──
    st.subheader("IMC por Nivel de Obesidad — Validación Umbrales OMS")
    imc_vals = R["df_fe"]["IMC"].values

    fig = go.Figure()
    for clase_key, clase_es, color in zip(ORDEN_CLASES, LABELS_ES, CLASE_COLORS):
        mask = R["y_clean"].values == clase_key
        fig.add_trace(go.Box(
            y=imc_vals[mask], name=clase_es,
            marker_color=color, line_color=USS_NAVY,
            boxmean=True,
            hovertemplate=f"<b>{clase_es}</b><br>IMC: %{{y:.1f}}<extra></extra>",
        ))
    # líneas OMS
    for y_val, label, color in [
        (18.5, "OMS: Bajo peso (<18.5)",   "#27AE60"),
        (25.0, "OMS: Sobrepeso (≥25)",      "#F39C12"),
        (30.0, "OMS: Obesidad (≥30)",       USS_RED),
    ]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1.8,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_color=color)
    uss_layout(fig, "IMC por Nivel de Obesidad (Feature Engineering)", height=420)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Scatter Age vs Weight coloreado ──
    st.subheader("Dispersión: Edad vs Peso por Clase")
    df_sc = R["df_clean"][["Age","Weight","Height"]].copy()
    df_sc["Clase"]  = [LABEL_MAP.get(c,c)  for c in R["y_clean"].values]
    df_sc["Color"]  = [COLOR_MAP.get(c,"#888") for c in R["y_clean"].values]
    df_sc["IMC"]    = R["df_fe"]["IMC"].values

    fig = go.Figure()
    for clase_key, clase_es, color in zip(ORDEN_CLASES, LABELS_ES, CLASE_COLORS):
        mask = R["y_clean"].values == clase_key
        sub  = df_sc[mask]
        fig.add_trace(go.Scatter(
            x=sub["Age"], y=sub["Weight"],
            mode="markers", name=clase_es,
            marker=dict(color=color, size=5, opacity=0.65,
                        line=dict(width=0.3, color="white")),
            hovertemplate=(
                f"<b>{clase_es}</b><br>"
                "Edad: %{x:.0f} años<br>"
                "Peso: %{y:.1f} kg<extra></extra>"
            ),
        ))
    uss_layout(fig, "Dispersión Edad vs Peso coloreado por Nivel de Obesidad", height=450)
    fig.update_xaxes(title="Edad (años)")
    fig.update_yaxes(title="Peso (kg)")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── PCA 2D ──
    st.subheader("Proyección PCA 2D — Separación de Clases en el Espacio Reducido")
    c1, c2 = st.columns(2)

    for col_obj, titulo, colorear in [(c1,"Sin etiquetas",False),(c2,"Con clases reales",True)]:
        with col_obj:
            fig = go.Figure()
            if not colorear:
                fig.add_trace(go.Scatter(
                    x=R["X_pca2"][:,0], y=R["X_pca2"][:,1],
                    mode="markers",
                    marker=dict(color=USS_NAVY, size=4, opacity=0.3),
                    hoverinfo="skip",
                ))
            else:
                for clase_key, clase_es, color in zip(ORDEN_CLASES, LABELS_ES, CLASE_COLORS):
                    mask = R["y_clean"].values == clase_key
                    fig.add_trace(go.Scatter(
                        x=R["X_pca2"][mask,0], y=R["X_pca2"][mask,1],
                        mode="markers", name=clase_es,
                        marker=dict(color=color, size=4, opacity=0.65),
                        hovertemplate=f"<b>{clase_es}</b><extra></extra>",
                    ))
            uss_layout(fig, titulo, height=380)
            fig.update_xaxes(title=f"PC1 ({R['exp_var'][0]*100:.1f}%)")
            fig.update_yaxes(title=f"PC2 ({R['exp_var'][1]*100:.1f}%)")
            if not colorear: fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════
# PÁGINA 2 — RESULTADOS DEL MODELO
# ══════════════════════════════════════════════════════════
elif pagina == "📊  Resultados del Modelo":
    st.title("📊 Resultados del Clasificador — Random Forest")
    st.caption(f"Fuente: {R['fuente']} · 100 árboles · split 75/25 estratificado · CV 5-fold")

    # ── Métricas ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎯 Accuracy (Test)",   f"{R['acc']:.1%}")
    m2.metric("🔁 CV 5-fold",         f"{R['cv'].mean():.1%}", f"±{R['cv'].std():.3f}")
    m3.metric("📐 AUC-ROC macro",     f"{R['auc']:.4f}")
    m4.metric("🗂 Muestras train",    f"{len(R['X_train'])}")

    st.divider()

    # ── Validación cruzada ──
    st.subheader("Validación Cruzada Stratified 5-Fold")
    folds = [f"Fold {i+1}" for i in range(5)]
    colores_cv = [USS_RED if v == R["cv"].max() else USS_NAVY for v in R["cv"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=folds, y=R["cv"],
        marker_color=colores_cv,
        text=[f"{v:.4f}" for v in R["cv"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=R["cv"].mean(), line_dash="dash", line_color=USS_RED, line_width=2,
                  annotation_text=f"Promedio: {R['cv'].mean():.4f}",
                  annotation_position="top right", annotation_font_color=USS_RED)
    # banda ±std
    fig.add_hrect(y0=R["cv"].mean()-R["cv"].std(),
                  y1=R["cv"].mean()+R["cv"].std(),
                  fillcolor=USS_RED, opacity=0.07, line_width=0,
                  annotation_text=f"±{R['cv'].std():.3f} std",
                  annotation_position="top left")
    uss_layout(fig, "Accuracy por Fold — Estabilidad del modelo", height=380)
    fig.update_yaxes(range=[max(0, R["cv"].min()-0.05), 1.02], title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Matriz de confusión (heatmap interactivo) ──
    st.subheader("Matriz de Confusión (7 clases)")
    etq = [c.replace("_","<br>") for c in R["le_y"].classes_]

    fig = go.Figure(go.Heatmap(
        z=R["cm"],
        x=etq, y=etq,
        colorscale=[[0,"#FFFFFF"],[0.001,"#FAE8EB"],[1,USS_RED]],
        text=R["cm"],
        texttemplate="%{text}",
        textfont=dict(size=13, color=USS_DARK),
        hovertemplate="Real: %{y}<br>Predicho: %{x}<br>N: %{z}<extra></extra>",
        showscale=True,
        colorbar=dict(title="N"),
    ))
    uss_layout(fig, f"Matriz de Confusión · Accuracy = {R['acc']:.1%} · AUC = {R['auc']:.4f}", height=480)
    fig.update_xaxes(title="Predicho", side="bottom")
    fig.update_yaxes(title="Real", autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Precision / Recall / F1 ──
    st.subheader("Precision, Recall y F1-Score por Clase")
    clases_ = list(R["le_y"].classes_)
    metricas_names = ["precision","recall","f1-score"]
    met_colors = [USS_NAVY, USS_RED, "#8E44AD"]

    fig = go.Figure()
    for met, col in zip(metricas_names, met_colors):
        vals = [R["report"][c][met] for c in clases_]
        fig.add_trace(go.Bar(
            name=met.capitalize(),
            x=[LABEL_MAP.get(c,c) for c in clases_],
            y=vals,
            marker_color=col,
            opacity=0.88,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
            hovertemplate=f"<b>%{{x}}</b><br>{met}: %{{y:.3f}}<extra></extra>",
        ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="#CCCCCC")
    uss_layout(fig, "Métricas por Clase de Obesidad", height=420)
    fig.update_layout(barmode="group", yaxis_range=[0,1.15])
    fig.update_xaxes(tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── Feature Importance ──
    st.subheader("Importancia de Variables — Gini (Random Forest)")
    feat_names  = R["feat_names"]
    importances = R["importances"]
    orden_imp   = np.argsort(importances)[::-1]

    nombres_ord = [feat_names[i] for i in orden_imp]
    imp_ord     = importances[orden_imp]
    colores_imp = [USS_RED if n in NUEVAS else USS_NAVY for n in nombres_ord]

    fig = go.Figure(go.Bar(
        x=imp_ord[::-1],
        y=nombres_ord[::-1],
        orientation="h",
        marker_color=colores_imp[::-1],
        text=[f"{v:.3f}" for v in imp_ord[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importancia: %{x:.4f}<extra></extra>",
    ))
    # leyenda manual
    fig.add_trace(go.Bar(x=[None], y=[None], name="Feature engineered",
                         marker_color=USS_RED, showlegend=True))
    fig.add_trace(go.Bar(x=[None], y=[None], name="Feature original",
                         marker_color=USS_NAVY, showlegend=True))
    uss_layout(fig, "Feature Importance Gini — Rojo = variables de Feature Engineering", height=520)
    fig.update_layout(showlegend=True, legend=dict(x=0.72, y=0.02))
    fig.update_xaxes(title="Importancia Gini")
    st.plotly_chart(fig, use_container_width=True)

    # Top 5 tarjetas
    st.subheader("🏆 Top 5 Variables")
    cols_top = st.columns(5)
    for i, (col_obj, idx) in enumerate(zip(cols_top, orden_imp[:5])):
        nombre = feat_names[idx]; imp = importances[idx]
        emoji  = "🥇🥈🥉4️⃣5️⃣"[i]
        tag    = " 🆕" if nombre in NUEVAS else ""
        col_obj.markdown(f"""
        <div style="background:white;border-radius:10px;padding:14px;
                    border-top:4px solid {USS_RED if nombre in NUEVAS else USS_NAVY};
                    text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
            <div style="font-size:22px;">{emoji}</div>
            <div style="font-weight:700;font-size:13px;color:{USS_NAVY};margin:6px 0 4px;">
                {nombre}{tag}</div>
            <div style="font-size:20px;font-weight:800;color:{USS_RED};">{imp:.3f}</div>
            <div style="font-size:10px;color:#888;">Gini</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Reporte detallado
    with st.expander("📋 Ver reporte completo de clasificación"):
        df_rep = pd.DataFrame(R["report"]).T
        df_rep = df_rep.drop(columns=["support"], errors="ignore").round(4)
        st.dataframe(df_rep.style.background_gradient(cmap="Reds", subset=["f1-score"]),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════
# PÁGINA 3 — PREDICCIÓN
# ══════════════════════════════════════════════════════════
elif pagina == "🔮  Predecir Nuevo Dato":
    st.title("🔮 Predecir Nivel de Obesidad")
    st.markdown("Ingresa los datos de una persona y el modelo **Random Forest** clasificará su nivel de obesidad según criterios de la **OMS**.")

    st.markdown(f"""
    <div class="uss-card" style="border-left:5px solid {USS_RED};">
        💡 Completa todos los campos del formulario y presiona <b>Predecir</b>.
        El modelo calculará automáticamente el IMC y retornará la clase con su probabilidad.
    </div>
    """, unsafe_allow_html=True)

    with st.form("form_pred"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"#### 🧍 Datos físicos")
            Gender = st.selectbox("Género", ["Male","Female"],
                                  format_func=lambda x:"Masculino" if x=="Male" else "Femenino")
            Age    = st.number_input("Edad (años)", 14, 70, 25, step=1)
            Height = st.number_input("Altura (m)", 1.45, 2.00, 1.70, step=0.01, format="%.2f")
            Weight = st.number_input("Peso (kg)", 39.0, 173.0, 70.0, step=0.5, format="%.1f")

        with c2:
            st.markdown(f"#### 🥗 Hábitos alimentarios")
            fam_hist = st.selectbox("Historial familiar de sobrepeso", ["yes","no"],
                                    format_func=lambda x:"Sí" if x=="yes" else "No")
            FAVC = st.selectbox("¿Come frecuentemente comida hipercalórica?", ["yes","no"],
                                format_func=lambda x:"Sí" if x=="yes" else "No")
            FCVC = st.select_slider("Frecuencia de consumo de verduras",
                                    options=[1.0,2.0,3.0],
                                    format_func=lambda x:{1.0:"Nunca",2.0:"A veces",3.0:"Siempre"}[x],
                                    value=2.0)
            NCP  = st.select_slider("N° de comidas principales al día",
                                    options=[1.0,2.0,3.0,4.0], value=3.0)
            CAEC = st.selectbox("Come entre comidas", ["no","Sometimes","Frequently","Always"],
                                format_func=lambda x:{"no":"No","Sometimes":"A veces",
                                                       "Frequently":"Frecuentemente","Always":"Siempre"}[x])
            CH2O = st.select_slider("Consumo de agua diario",
                                    options=[1.0,2.0,3.0],
                                    format_func=lambda x:{1.0:"< 1 litro",2.0:"1-2 litros",3.0:"> 2 litros"}[x],
                                    value=2.0)

        with c3:
            st.markdown(f"#### 🏃 Estilo de vida")
            SMOKE  = st.selectbox("¿Fuma?", ["no","yes"],
                                  format_func=lambda x:"No" if x=="no" else "Sí")
            SCC    = st.selectbox("¿Monitorea calorías?", ["no","yes"],
                                  format_func=lambda x:"No" if x=="no" else "Sí")
            FAF    = st.select_slider("Actividad física semanal",
                                      options=[0.0,1.0,2.0,3.0],
                                      format_func=lambda x:{0.0:"Ninguna",1.0:"1-2 días",
                                                             2.0:"2-4 días",3.0:"4-5 días"}[x],
                                      value=1.0)
            TUE    = st.select_slider("Uso de tecnología al día",
                                      options=[0.0,1.0,2.0],
                                      format_func=lambda x:{0.0:"0-2h",1.0:"3-5h",2.0:">5h"}[x])
            CALC   = st.selectbox("Consumo de alcohol", ["no","Sometimes","Frequently","Always"],
                                  format_func=lambda x:{"no":"No","Sometimes":"A veces",
                                                         "Frequently":"Frecuentemente","Always":"Siempre"}[x])
            MTRANS = st.selectbox("Medio de transporte", ["Public_Transportation","Walking",
                                                           "Automobile","Motorbike","Bike"],
                                  format_func=lambda x:{"Public_Transportation":"Transporte público",
                                                         "Walking":"A pie","Automobile":"Automóvil",
                                                         "Motorbike":"Motocicleta","Bike":"Bicicleta"}[x])

        submitted = st.form_submit_button("🔍 Predecir nivel de obesidad",
                                          use_container_width=True, type="primary")

    if submitted:
        datos = dict(Gender=Gender, Age=float(Age), Height=float(Height), Weight=float(Weight),
                     family_history_with_overweight=fam_hist, FAVC=FAVC,
                     FCVC=float(FCVC), NCP=float(NCP), CAEC=CAEC,
                     SMOKE=SMOKE, CH2O=float(CH2O), SCC=SCC,
                     FAF=float(FAF), TUE=float(TUE), CALC=CALC, MTRANS=MTRANS)
        try:
            clase, proba, clases_le = predecir(datos, R)
            imc     = float(Weight) / float(Height)**2
            color_c = "#0F2747"
            nom_es  = LABEL_MAP.get(clase, clase)
            conf    = proba[list(clases_le).index(clase)]

            st.divider()
            st.subheader("🎯 Resultado de la Predicción")

            res1, res2 = st.columns([1, 2])
            with res1:
                st.markdown(f"""
                <div style="background:white;border-radius:14px;padding:28px 20px;
                            border-top:6px solid {color_c};
                            box-shadow:0 4px 18px rgba(0,0,0,0.10);text-align:center;">
                    <div style="font-size:42px;">⚕️</div>
                    <div style="font-size:22px;font-weight:800;color:{color_c};margin:10px 0 4px;">
                        {nom_es}</div>
                    <div style="font-size:12px;color:#888;">Clasificación OMS · Random Forest</div>
                    <hr style="border-color:{color_c}44;margin:14px 0;">
                    <div style="font-size:34px;font-weight:900;color:{"#0F2747"};">
                        {imc:.1f}</div>
                    <div style="font-size:12px;color:#888;margin-top:2px;">IMC (kg/m²)</div>
                    <hr style="border-color:{color_c}44;margin:14px 0;">
                    <div style="font-size:28px;font-weight:800;color:{color_c};">
                        {conf:.1%}</div>
                    <div style="font-size:12px;color:"#2C5D8A";">Confianza del modelo</div>
                </div>
                """, unsafe_allow_html=True)

            with res2:
                # Gauge confianza
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=conf*100,
                    number=dict(suffix="%", font=dict(size=36, color="#0F2747")),
                    gauge=dict(
                        axis=dict(range=[0,100], tickfont=dict(size=11)),
                        bar=dict(color=color_c),
                        bgcolor="black",
                        steps=[
                            dict(range=[0,50],  color="#F0F0F0"),
                            dict(range=[50,75], color="#D3D3D3"),
                            dict(range=[75,100],color="#A9A9A9"),
                        ],
                        threshold=dict(line=dict(color=USS_NAVY,width=3), value=conf*100)
                    ),
                    title=dict(text="Confianza del modelo", font=dict(size=14, color=USS_NAVY))
                ))
                uss_layout(fig_g, height=250)
                fig_g.update_layout(margin=dict(t=40,b=10,l=30,r=30))
                st.plotly_chart(fig_g, use_container_width=True)

                # Barras de probabilidad por clase
                etq_es = [LABEL_MAP.get(c,c) for c in clases_le]
                cols_b = [color_c if c==clase else "#DDDDDD" for c in clases_le]
                fig_p  = go.Figure(go.Bar(
                    x=proba, y=etq_es, orientation="h",
                    marker_color=cols_b,
                    text=[f"{v:.1%}" for v in proba],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.2%}<extra></extra>",
                ))
                uss_layout(fig_p, "Distribución de probabilidades por clase", height=300)
                fig_p.update_xaxes(range=[0,1.15], title="Probabilidad")
                fig_p.update_layout(showlegend=False)
                # --- CONFIGURACIÓN DEL FONDO AZUL OSCURO Y CONTRASTE ---
                fig_p.update_layout(
                    plot_bgcolor="#1E293B",   # Azul oscuro (Slate 800) para el área del gráfico
                    paper_bgcolor="#0F172A",  # Azul más oscuro (Slate 900) para el fondo total
                    font=dict(color="#F8FAFC"), # Texto general en blanco hueso para que resalte
                    title_font=dict(color="#F8FAFC") # Título en blanco
                )

                # Ajustar las líneas de los ejes para que combinen con el fondo oscuro
                fig_p.update_xaxes(
                    showgrid=True, 
                    gridcolor="#334155",  # Líneas de cuadrícula sutiles
                    linecolor="#475569",  # Línea del eje
                    title_font=dict(color="#94A3B8"),
                    tickfont=dict(color="#94A3B8")
                )
                fig_p.update_yaxes(
                    linecolor="#475569", 
                    tickfont=dict(color="#F8FAFC") # Etiquetas de las clases bien legibles
                )
                st.plotly_chart(fig_p, use_container_width=True)

            st.divider()

            # Tabla resumen + referencia IMC
            ti1, ti2 = st.columns(2)
            with ti1:
                st.markdown("#### 📌 Datos ingresados")
                st.dataframe(pd.DataFrame({
                    "Variable": ["Edad","Altura","Peso","IMC calculado",
                                 "Actividad física","Agua diaria","Hist. familiar sobrepeso"],
                    "Valor": [f"{Age} años", f"{float(Height):.2f} m",
                              f"{float(Weight):.1f} kg", f"{imc:.1f} kg/m²",
                              {0.0:"Ninguna",1.0:"1-2 días",2.0:"2-4 días",3.0:"4-5 días"}[FAF],
                              {1.0:"< 1L",2.0:"1-2L",3.0:"> 2L"}[CH2O],
                              "Sí" if fam_hist=="yes" else "No"]
                }), use_container_width=True, hide_index=True)

            with ti2:
                st.markdown("#### 🏥 Referencia IMC (OMS)")
                flechas = ["◀ Tu IMC" if imc<18.5 else "",
                           "◀ Tu IMC" if 18.5<=imc<25 else "",
                           "◀ Tu IMC" if 25<=imc<30 else "",
                           "◀ Tu IMC" if imc>=30 else ""]
                st.dataframe(pd.DataFrame({
                    "Rango IMC":   ["< 18.5","18.5 – 24.9","25.0 – 29.9","≥ 30"],
                    "Categoría":   ["Bajo peso","Normal","Sobrepeso","Obesidad"],
                    "":            flechas
                }), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.exception(e)
