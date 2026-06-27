import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from scipy.stats import mstats
from matplotlib.patches import Patch
import os

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Obesidad",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PALETTE = ['#2B4C7E','#E05C2A','#27A96F','#8E44AD','#C0392B','#F39C12','#1ABC9C']
ORDEN_CLASES = ['Insufficient_Weight','Normal_Weight','Overweight_Level_I',
                'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']
LABELS_ES = ['Peso Insuficiente','Peso Normal','Sobrepeso I','Sobrepeso II',
             'Obesidad Tipo I','Obesidad Tipo II','Obesidad Tipo III']
LABEL_MAP = dict(zip(ORDEN_CLASES, LABELS_ES))
COLORES_NIVEL = {
    'Insufficient_Weight': '#1ABC9C', 'Normal_Weight': '#27A96F',
    'Overweight_Level_I': '#F39C12',  'Overweight_Level_II': '#E05C2A',
    'Obesity_Type_I': '#C0392B',      'Obesity_Type_II': '#8E44AD',
    'Obesity_Type_III': '#2C3E50',
}

# ─────────────────────────────────────────
# CARGA Y ENTRENAMIENTO (cacheado)
# ─────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset UCI y entrenando modelo...")
def cargar_y_entrenar():
    # ── 1. Carga del dataset ──
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=544)
        X_raw = dataset.data.features
        y_raw = dataset.data.targets.squeeze()
        fuente = "UCI ML Repository (id=544)"
    except Exception:
        # Fallback: dataset incluido localmente
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path  = os.path.join(base_dir, "obesity_data.csv")
        df = pd.read_csv(csv_path)
        y_raw = df['NObeyesdad']
        X_raw = df.drop(columns=['NObeyesdad'])
        fuente = "Dataset local (réplica sintética UCI 544)"

    # ── 2. Limpieza ──
    df_clean = X_raw.copy()
    y_clean  = y_raw.copy()
    mask_dup = ~df_clean.duplicated(keep='first')
    df_clean = df_clean[mask_dup].reset_index(drop=True)
    y_clean  = y_clean[mask_dup].reset_index(drop=True)

    for col in ['NCP','CH2O','FAF','TUE','FCVC']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].round().astype(float)

    cols_num = df_clean.select_dtypes(include='number').columns.tolist()
    for col in cols_num:
        datos = df_clean[col].dropna()
        Q1, Q3 = datos.quantile(0.25), datos.quantile(0.75)
        IQR = Q3 - Q1
        n_out = ((datos < Q1 - 1.5*IQR) | (datos > Q3 + 1.5*IQR)).sum()
        if n_out > 20:
            df_clean[col] = mstats.winsorize(df_clean[col], limits=[0.01, 0.01])

    # ── 3. Feature Engineering ──
    df_fe = df_clean.copy()
    df_fe['IMC'] = df_clean['Weight'] / (df_clean['Height'] ** 2)
    favc_bin = (df_clean['FAVC'] == 'yes').astype(float)
    fam_bin  = (df_clean['family_history_with_overweight'] == 'yes').astype(float)
    df_fe['Riesgo_Calorico']   = favc_bin * (4 - df_clean['FAF'].clip(0, 3))
    df_fe['Actividad_Neta']    = df_clean['FAF'] - df_clean['TUE']
    df_fe['Carga_Hidratacion'] = df_clean['CH2O'] / (df_clean['NCP'] + 1)
    df_fe['Herencia_Riesgo']   = fam_bin * favc_bin

    # ── 4. Encoding y escalado ──
    df_final = df_fe.copy()
    cols_cat = df_final.select_dtypes(include='object').columns.tolist()
    encoders = {}
    for col in cols_cat:
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col].astype(str))
        encoders[col] = le

    le_y  = LabelEncoder()
    y_enc = le_y.fit_transform(y_clean)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_final)

    # ── 5. Entrenamiento ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(
        label_binarize(y_test, classes=list(range(7))),
        y_proba, multi_class='ovr', average='macro'
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(rf, X_scaled, y_enc, cv=skf, scoring='accuracy')
    cm  = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=le_y.classes_, output_dict=True)

    # ── 6. PCA para visualización ──
    pca2    = PCA(n_components=2)
    X_pca2  = pca2.fit_transform(X_scaled)
    exp_var = PCA().fit(X_scaled).explained_variance_ratio_

    return {
        'rf': rf, 'scaler': scaler, 'encoders': encoders,
        'le_y': le_y, 'y_enc': y_enc, 'y_clean': y_clean,
        'X_scaled': X_scaled, 'X_pca2': X_pca2, 'exp_var': exp_var,
        'acc': acc, 'auc': auc, 'cv': cv, 'cm': cm, 'report': report,
        'feat_names': list(df_final.columns),
        'importances': rf.feature_importances_,
        'df_clean': df_clean, 'df_fe': df_fe,
        'X_train': X_train, 'y_train': y_train,
        'y_test': y_test, 'y_pred': y_pred,
        'fuente': fuente,
    }


def predecir(datos_usuario, R):
    """Predicción de un nuevo dato."""
    rf = R['rf']; scaler = R['scaler']; encoders = R['encoders']
    le_y = R['le_y']; feat_names = R['feat_names']

    df_u = pd.DataFrame([datos_usuario])
    for col, le in encoders.items():
        if col in df_u.columns:
            val = str(df_u[col].values[0])
            if val not in le.classes_:
                val = le.classes_[0]
            df_u[col] = le.transform([val])

    # Feature engineering
    h = float(df_u['Height'].values[0])
    w = float(df_u['Weight'].values[0])
    faf_v  = float(df_u['FAF'].values[0])
    tue_v  = float(df_u['TUE'].values[0])
    ch2o_v = float(df_u['CH2O'].values[0])
    ncp_v  = float(df_u['NCP'].values[0])
    favc_v = float(df_u['FAVC'].values[0])
    fam_v  = float(df_u['family_history_with_overweight'].values[0])

    df_u['IMC']              = w / (h ** 2)
    df_u['Riesgo_Calorico']  = favc_v * (4 - min(faf_v, 3))
    df_u['Actividad_Neta']   = faf_v - tue_v
    df_u['Carga_Hidratacion']= ch2o_v / (ncp_v + 1)
    df_u['Herencia_Riesgo']  = fam_v * favc_v

    df_u = df_u[feat_names]
    X_u  = scaler.transform(df_u.values.astype(float))

    pred_enc   = R['rf'].predict(X_u)[0]
    pred_proba = R['rf'].predict_proba(X_u)[0]
    clase      = le_y.inverse_transform([pred_enc])[0]
    return clase, pred_proba, le_y.classes_


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚕️ Clasificador de Obesidad")
    st.markdown("**Solemne 1 — Taller de Aplicaciones**")
    st.markdown("*Dr. Mauricio Sepúlveda*")
    st.markdown("*Felipe Carrasco, Carlos Tello*")
    st.divider()
    pagina = st.radio(
        "Sección:",
        ["📊 Resultados del Modelo",
         "🔮 Predecir Nuevo Dato",
         "📈 Análisis Exploratorio"]
    )
    st.divider()
    st.markdown("""
    **Modelo:** Random Forest  
    **100 árboles · split 75/25**  
    **Validación:** StratifiedKFold 5-fold  
    **Dataset:** UCI ML Repo 544  
    **Clases:** 7 niveles OMS  
    """)

# ─────────────────────────────────────────
# CARGA DATOS
# ─────────────────────────────────────────
with st.spinner("Entrenando modelo Random Forest..."):
    R = cargar_y_entrenar()

# ═══════════════════════════════════════════════════════════
# PÁGINA 1 — RESULTADOS
# ═══════════════════════════════════════════════════════════
if pagina == "📊 Resultados del Modelo":
    st.title("📊 Resultados del Clasificador — Random Forest")
    st.caption(f"Fuente de datos: {R['fuente']} · Dataset: Obesity Levels")

    # Métricas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy (Test 25%)", f"{R['acc']:.1%}")
    c2.metric("CV 5-fold media", f"{R['cv'].mean():.1%}", f"±{R['cv'].std():.3f}")
    c3.metric("AUC-ROC macro", f"{R['auc']:.4f}")
    c4.metric("Muestras entrenamiento", f"{len(R['X_train'])}")

    st.divider()

    # Matriz de confusión + CV
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Matriz de Confusión")
        fig, ax = plt.subplots(figsize=(7, 5))
        etq = [c.replace('_','\n') for c in R['le_y'].classes_]
        sns.heatmap(R['cm'], annot=True, fmt='d', cmap='Greens', ax=ax,
                    xticklabels=etq, yticklabels=etq,
                    linewidths=1, linecolor='white', cbar=False,
                    annot_kws={'size': 9, 'fontweight': 'bold'})
        ax.set_title(f"Accuracy = {R['acc']:.1%}  |  AUC = {R['auc']:.4f}",
                     fontweight='bold', fontsize=10)
        ax.set_ylabel('Real'); ax.set_xlabel('Predicho')
        ax.tick_params(axis='x', rotation=30, labelsize=7)
        ax.tick_params(axis='y', rotation=0,  labelsize=7)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Validación Cruzada Stratified 5-Fold")
        fig, ax = plt.subplots(figsize=(7, 5))
        folds  = [f'Fold {i+1}' for i in range(5)]
        cols_  = [PALETTE[2] if v == max(R['cv']) else PALETTE[0] for v in R['cv']]
        bars   = ax.bar(folds, R['cv'], color=cols_, edgecolor='white', width=0.5, alpha=0.9)
        for bar, val in zip(bars, R['cv']):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.axhline(R['cv'].mean(), color=PALETTE[1], linestyle='--', linewidth=2,
                   label=f'Promedio: {R["cv"].mean():.4f}')
        ax.fill_between(range(-1,6), R['cv'].mean()-R['cv'].std(),
                        R['cv'].mean()+R['cv'].std(),
                        alpha=0.1, color=PALETTE[1], label=f'±std: {R["cv"].std():.4f}')
        ax.set_ylabel('Accuracy'); ax.set_ylim(max(0, R['cv'].min()-0.05), 1.02)
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.4)
        ax.set_title('Estabilidad entre folds', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()

    # Precision / Recall / F1
    st.subheader("Precision, Recall y F1-Score por Clase")
    clases_  = list(R['le_y'].classes_)
    metricas = ['precision','recall','f1-score']
    x_pos    = np.arange(len(clases_)); w_ = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (met, col) in enumerate(zip(metricas, PALETTE[:3])):
        vals = [R['report'][c][met] for c in clases_]
        bars = ax.bar(x_pos+i*w_, vals, w_, label=met.capitalize(),
                      color=col, alpha=0.88, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x_pos+w_)
    ax.set_xticklabels([c.replace('_','\n') for c in clases_], fontsize=8)
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.4)
    ax.set_title('Métricas por clase de obesidad', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # Feature Importance
    st.subheader("Importancia de Variables (Gini)")
    feat_names  = R['feat_names']
    importances = R['importances']
    orden_imp   = np.argsort(importances)[::-1]
    nuevas = {'IMC','Riesgo_Calorico','Actividad_Neta','Carga_Hidratacion','Herencia_Riesgo'}

    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        fig, ax = plt.subplots(figsize=(9, 6))
        colores_imp = ['#E05C2A' if feat_names[i] in nuevas else PALETTE[0]
                       for i in orden_imp]
        ax.barh([feat_names[i] for i in orden_imp][::-1],
                importances[orden_imp][::-1],
                color=colores_imp[::-1], edgecolor='white', alpha=0.88)
        for i, v in enumerate(importances[orden_imp][::-1]):
            ax.text(v+0.002, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
        ax.set_xlabel('Importancia Gini')
        ax.set_title('Importancia de features\n(naranja = Feature Engineering)', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.legend(handles=[
            Patch(color='#E05C2A', label='Feature engineered'),
            Patch(color=PALETTE[0],  label='Feature original')
        ], fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col_f2:
        st.markdown("#### 🏆 Top 5 variables")
        for i, idx in enumerate(orden_imp[:5]):
            nombre = feat_names[idx]; imp = importances[idx]
            emoji  = "🏆" if i == 0 else f"{i+1}."
            tag    = " 🆕" if nombre in nuevas else ""
            st.markdown(f"**{emoji} {nombre}**{tag}")
            st.progress(float(imp))
            st.caption(f"Importancia Gini: {imp:.4f}")

    st.divider()

    # PCA 2D
    st.subheader("Proyección PCA 2D — Distribución de Clases")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, usar_color in zip(axes, [False, True]):
        if not usar_color:
            ax.scatter(R['X_pca2'][:,0], R['X_pca2'][:,1],
                       c='#2B4C7E', alpha=0.3, s=10, edgecolors='none')
            ax.set_title('Sin etiquetas (vista clustering)', fontweight='bold')
        else:
            for i, clase in enumerate(ORDEN_CLASES):
                mask = R['y_clean'].values == clase
                ax.scatter(R['X_pca2'][mask,0], R['X_pca2'][mask,1],
                           c=PALETTE[i], alpha=0.6, s=10,
                           label=LABEL_MAP[clase], edgecolors='none')
            ax.legend(fontsize=7, loc='upper right')
            ax.set_title('Con clases reales (7 niveles)', fontweight='bold')
        ax.set_xlabel(f'PC1 ({R["exp_var"][0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({R["exp_var"][1]*100:.1f}%)')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Reporte detallado
    with st.expander("📋 Reporte completo de clasificación"):
        df_rep = pd.DataFrame(R['report']).T.drop(columns=['support'], errors='ignore')
        st.dataframe(df_rep.round(4), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PÁGINA 2 — PREDICCIÓN
# ═══════════════════════════════════════════════════════════
elif pagina == "🔮 Predecir Nuevo Dato":
    st.title("🔮 Predecir Nivel de Obesidad")
    st.markdown("Ingresa los datos de una persona. El modelo **Random Forest** clasificará su nivel de obesidad según los criterios de la OMS.")
    st.info("💡 Completa todos los campos y presiona **Predecir** al final del formulario.")

    with st.form("form_pred"):
        st.subheader("📋 Datos personales y hábitos")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🧍 Datos físicos**")
            Gender = st.selectbox("Género", ["Male", "Female"],
                                   format_func=lambda x: "Masculino" if x=="Male" else "Femenino")
            Age    = st.number_input("Edad (años)", 14, 70, 25, step=1)
            Height = st.number_input("Altura (m)", 1.45, 2.00, 1.70, step=0.01, format="%.2f")
            Weight = st.number_input("Peso (kg)", 39.0, 173.0, 70.0, step=0.5, format="%.1f")

        with col2:
            st.markdown("**🥗 Hábitos alimentarios**")
            fam_hist = st.selectbox("Historial familiar de sobrepeso",
                                    ["yes","no"], format_func=lambda x: "Sí" if x=="yes" else "No")
            FAVC = st.selectbox("¿Come frecuentemente comida hipercalórica?",
                                ["yes","no"], format_func=lambda x: "Sí" if x=="yes" else "No")
            FCVC = st.select_slider("Frecuencia de consumo de verduras",
                                    options=[1.0, 2.0, 3.0],
                                    format_func=lambda x: {1.0:"Nunca",2.0:"A veces",3.0:"Siempre"}[x],
                                    value=2.0)
            NCP  = st.select_slider("Número de comidas principales al día",
                                    options=[1.0, 2.0, 3.0, 4.0], value=3.0)
            CAEC = st.selectbox("Come entre comidas",
                                ["no","Sometimes","Frequently","Always"],
                                format_func=lambda x: {"no":"No","Sometimes":"A veces",
                                                        "Frequently":"Frecuentemente","Always":"Siempre"}[x])
            CH2O = st.select_slider("Consumo de agua diario",
                                    options=[1.0, 2.0, 3.0],
                                    format_func=lambda x: {1.0:"< 1 litro",2.0:"1-2 litros",3.0:"> 2 litros"}[x],
                                    value=2.0)

        with col3:
            st.markdown("**🏃 Estilo de vida**")
            SMOKE  = st.selectbox("¿Fuma?", ["no","yes"],
                                  format_func=lambda x: "No" if x=="no" else "Sí")
            SCC    = st.selectbox("¿Monitorea calorías?", ["no","yes"],
                                  format_func=lambda x: "No" if x=="no" else "Sí")
            FAF    = st.select_slider("Actividad física semanal",
                                      options=[0.0, 1.0, 2.0, 3.0],
                                      format_func=lambda x: {0.0:"Ninguna",1.0:"1-2 días",
                                                              2.0:"2-4 días",3.0:"4-5 días"}[x],
                                      value=1.0)
            TUE    = st.select_slider("Uso de tecnología al día",
                                      options=[0.0, 1.0, 2.0],
                                      format_func=lambda x: {0.0:"0-2h",1.0:"3-5h",2.0:">5h"}[x])
            CALC   = st.selectbox("Consumo de alcohol",
                                  ["no","Sometimes","Frequently","Always"],
                                  format_func=lambda x: {"no":"No","Sometimes":"A veces",
                                                          "Frequently":"Frecuentemente","Always":"Siempre"}[x])
            MTRANS = st.selectbox("Medio de transporte habitual",
                                  ["Public_Transportation","Walking","Automobile","Motorbike","Bike"],
                                  format_func=lambda x: {"Public_Transportation":"Transporte público",
                                                          "Walking":"A pie","Automobile":"Automóvil",
                                                          "Motorbike":"Motocicleta","Bike":"Bicicleta"}[x])

        submitted = st.form_submit_button("🔍 Predecir nivel de obesidad",
                                          use_container_width=True, type="primary")

    if submitted:
        datos = {
            'Gender': Gender, 'Age': float(Age), 'Height': float(Height), 'Weight': float(Weight),
            'family_history_with_overweight': fam_hist, 'FAVC': FAVC,
            'FCVC': float(FCVC), 'NCP': float(NCP), 'CAEC': CAEC,
            'SMOKE': SMOKE, 'CH2O': float(CH2O), 'SCC': SCC,
            'FAF': float(FAF), 'TUE': float(TUE), 'CALC': CALC, 'MTRANS': MTRANS,
        }
        try:
            clase, proba, clases_le = predecir(datos, R)
            imc   = float(Weight) / (float(Height) ** 2)
            color = COLORES_NIVEL.get(clase, '#888')
            nombre_es = LABEL_MAP.get(clase, clase)
            idx_pred  = list(clases_le).index(clase)
            confianza = proba[idx_pred]

            st.divider()
            st.subheader("🎯 Resultado de la Predicción")

            cr1, cr2 = st.columns([1, 2])
            with cr1:
                st.markdown(f"""
                <div style="background:{color}22;border-left:6px solid {color};
                            padding:20px;border-radius:10px;text-align:center;">
                    <h2 style="color:{color};margin:0;">{nombre_es}</h2>
                    <p style="color:#666;margin:6px 0 0 0;font-size:13px;">Clasificación OMS · Random Forest</p>
                    <hr style="border-color:{color}44;margin:12px 0;">
                    <p style="font-size:26px;font-weight:bold;color:{color};margin:0;">IMC: {imc:.1f}</p>
                    <p style="color:#888;font-size:12px;margin:2px 0 0 0;">kg/m²</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Confianza del modelo", f"{confianza:.1%}")
                st.progress(float(confianza))

            with cr2:
                fig, ax = plt.subplots(figsize=(8, 4))
                etq_es = [LABEL_MAP.get(c, c) for c in clases_le]
                cols_b = [color if c == clase else '#DDDDDD' for c in clases_le]
                bars   = ax.barh(etq_es, proba, color=cols_b,
                                 edgecolor='white', alpha=0.9, height=0.6)
                for bar, val in zip(bars, proba):
                    if val > 0.01:
                        ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
                                f'{val:.1%}', va='center', fontsize=9, fontweight='bold')
                ax.set_xlabel('Probabilidad')
                ax.set_title('Distribución de probabilidades por clase', fontweight='bold')
                ax.set_xlim(0, 1.15)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig); plt.close()

            st.divider()
            ci1, ci2 = st.columns(2)
            with ci1:
                st.markdown("#### 📌 Datos ingresados")
                st.dataframe(pd.DataFrame({
                    'Variable': ['Edad','Altura','Peso','IMC calculado',
                                 'Actividad física','Agua diaria','Hist. familiar'],
                    'Valor': [f'{Age} años', f'{float(Height):.2f} m', f'{float(Weight):.1f} kg',
                              f'{imc:.1f} kg/m²',
                              {0.0:"Ninguna",1.0:"1-2 días",2.0:"2-4 días",3.0:"4-5 días"}[FAF],
                              {1.0:"<1L",2.0:"1-2L",3.0:">2L"}[CH2O],
                              "Sí" if fam_hist=="yes" else "No"]
                }), use_container_width=True, hide_index=True)

            with ci2:
                st.markdown("#### 🏥 Referencia IMC (OMS)")
                st.dataframe(pd.DataFrame({
                    'Rango IMC': ['< 18.5','18.5 – 24.9','25.0 – 29.9','≥ 30'],
                    'Categoría': ['Bajo peso','Normal','Sobrepeso','Obesidad'],
                    '': ['◀ Tu IMC' if imc < 18.5 else '',
                         '◀ Tu IMC' if 18.5 <= imc < 25 else '',
                         '◀ Tu IMC' if 25 <= imc < 30 else '',
                         '◀ Tu IMC' if imc >= 30 else '']
                }), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.exception(e)


# ═══════════════════════════════════════════════════════════
# PÁGINA 3 — EDA
# ═══════════════════════════════════════════════════════════
elif pagina == "📈 Análisis Exploratorio":
    st.title("📈 Análisis Exploratorio de Datos")
    st.caption("Dataset: Obesity Levels — UCI ML Repository (id=544) · 2.087 muestras · 7 clases")

    # Distribución clases
    st.subheader("Distribución de Clases")
    conteos = R['y_clean'].value_counts().reindex(ORDEN_CLASES)
    total   = len(R['y_clean'])
    pcts    = (conteos / total * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bars = axes[0].bar(range(len(ORDEN_CLASES)), conteos.values,
                       color=PALETTE[:7], edgecolor='white', width=0.7)
    for bar, val, pct in zip(bars, conteos.values, pcts.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, val+5,
                     f'{val}\n({pct}%)', ha='center', va='bottom', fontsize=8, fontweight='bold')
    axes[0].set_xticks(range(len(ORDEN_CLASES)))
    axes[0].set_xticklabels([LABEL_MAP[c] for c in ORDEN_CLASES],
                             fontsize=8, rotation=30, ha='right')
    axes[0].set_title('Frecuencia absoluta por clase', fontweight='bold')
    axes[0].set_ylabel('Frecuencia')
    axes[0].axhline(total/7, color='red', linestyle='--', alpha=0.5, label='Distribución perfecta')
    axes[0].legend(fontsize=8); axes[0].grid(axis='y', alpha=0.4)

    axes[1].pie(conteos.values, labels=[LABEL_MAP[c] for c in ORDEN_CLASES],
                colors=PALETTE[:7], autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 8})
    axes[1].set_title('Proporción de clases', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # Variables numéricas
    st.subheader("Distribución de Variables Numéricas por Clase")
    vars_plot  = ['Age','Height','Weight','FAF']
    nombres_pl = ['Edad (años)','Altura (m)','Peso (kg)','Actividad física']

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    df_num = R['df_clean'][['Age','Height','Weight','FAF']].copy()
    df_num['clase'] = R['y_clean'].values
    for ax, var, nom in zip(axes, vars_plot, nombres_pl):
        for i, clase in enumerate(ORDEN_CLASES):
            datos = df_num[df_num['clase']==clase][var]
            ax.hist(datos, bins=15, alpha=0.5, color=PALETTE[i],
                    label=LABEL_MAP[clase], density=True)
        ax.set_title(nom, fontweight='bold', fontsize=10)
        ax.set_xlabel('Valor'); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=5)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # IMC por clase
    st.subheader("IMC por Nivel de Obesidad (validación criterios OMS)")
    fig, ax = plt.subplots(figsize=(11, 4))
    imc_vals = R['df_fe']['IMC'].values
    data_box = [imc_vals[R['y_clean'].values == c] for c in ORDEN_CLASES]
    bp = ax.boxplot(data_box, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(color); patch.set_alpha(0.75)
    for y_val, label, col in [(18.5,'OMS: <18.5 bajo peso','#27A96F'),
                                (25.0,'OMS: 25 sobrepeso','#F59E0B'),
                                (30.0,'OMS: 30 obesidad','#E05C2A')]:
        ax.axhline(y_val, color=col, linestyle='--', linewidth=1.5, alpha=0.8, label=label)
    ax.set_xticklabels([LABEL_MAP[c] for c in ORDEN_CLASES],
                       rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('IMC (kg/m²)')
    ax.set_title('IMC por nivel de obesidad (feature engineered)', fontweight='bold')
    ax.legend(fontsize=9, loc='upper left'); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.divider()

    # Resumen
    st.subheader("📋 Resumen del Proceso")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        | Métrica | Valor |
        |---------|-------|
        | Muestras originales | 2.111 |
        | Duplicados eliminados | 24 |
        | **Muestras finales** | **2.087** |
        | Features originales | 16 |
        | Features con FE | 21 |
        | Nulos | 0 |
        | Clases | 7 |
        """)
    with c2:
        st.markdown("""
        | Transformación | Detalle |
        |----------------|---------|
        | Duplicados | Eliminados (keep='first') |
        | Decimales SMOTE | Redondeados |
        | Outliers | Winsorizing p1-p99 |
        | Encoding | LabelEncoder |
        | Escalado | StandardScaler |
        | Feature Eng. | IMC + 4 variables clínicas |
        | Clasificador | Random Forest 100 árboles |
        | Validación | StratifiedKFold 5-fold |
        """)
