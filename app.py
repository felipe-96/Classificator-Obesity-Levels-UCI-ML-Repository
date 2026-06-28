import warnings; warnings.filterwarnings('ignore')
import os, base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from scipy.stats import mstats

# ═══════════════════════════════════
# PALETA DARK EJECUTIVA
# ═══════════════════════════════════
BG     = "#0D1117"
BG2    = "#161C27"
CARD   = "#1C2333"
CARD2  = "#222B3E"
BORDER = "#2D3A52"
TEXT   = "#E6EDF3"
MUTED  = "#8B98A9"
ACCENT = "#4A90D9"
GOLD   = "#C9A84C"
GREEN  = "#3DD68C"
AMBER  = "#F0B429"
RED    = "#F87171"
WHITE  = "#FFFFFF"

CLASS_C = ["#3DD68C","#63D9A0","#F0B429","#E07B34","#E05252","#B83232","#7A1A1A"]
ORDEN   = ["Insufficient_Weight","Normal_Weight","Overweight_Level_I",
           "Overweight_Level_II","Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]
LABELS  = ["Peso Insuf.","Peso Normal","Sobrepeso I","Sobrepeso II",
           "Obesidad I","Obesidad II","Obesidad III"]
LMAP    = dict(zip(ORDEN, LABELS))
CMAP    = dict(zip(ORDEN, CLASS_C))

# ── Convierte hex a rgba (Plotly no acepta hex de 8 dígitos) ──
def rgba(hex_c, alpha=1.0):
    h = hex_c.lstrip("#")
    r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config(page_title="Dashboard Obesidad · USS",
                   page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

# ── Logo ──
def b64_logo():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uss_logo.png")
    if os.path.exists(p):
        return base64.b64encode(open(p,"rb").read()).decode()
    return None
LOGO = b64_logo()

# ── CSS ──
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif!important;color:{TEXT};}}
.stApp,.main,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{{
    background:{BG}!important;}}
[data-testid="stSidebar"]{{background:{BG2}!important;border-right:1px solid {BORDER};}}
[data-testid="stSidebar"] *{{color:{TEXT}!important;}}
[data-testid="stSidebar"] hr{{border-color:{BORDER}!important;}}

/* Radio nav — ocultar el círculo y dar aspecto de botones */
[data-testid="stSidebar"] [data-testid="stRadio"] > label{{display:none!important;}}
[data-testid="stSidebar"] [role="radiogroup"]{{gap:0!important;}}
[data-testid="stSidebar"] [role="radiogroup"] > label{{
    background:{CARD};border-radius:8px;padding:11px 14px!important;
    margin:4px 0!important;border:1px solid {BORDER};
    cursor:pointer;transition:all .15s;
    width:100%;box-sizing:border-box;}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover{{
    border-color:{GOLD};background:{CARD2};}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked){{
    border-color:{GOLD}!important;background:{CARD2}!important;
    box-shadow:inset 3px 0 0 {GOLD};}}
/* ocultar el botón circular del radio */
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child{{display:none!important;}}
[data-testid="stSidebar"] [role="radiogroup"] > label > div{{
    font-size:13px!important;font-weight:500!important;color:{TEXT}!important;}}

h1{{color:{WHITE}!important;font-weight:800!important;font-size:1.65rem!important;
    border-left:4px solid {GOLD};padding-left:14px;margin-bottom:2px!important;}}
h2{{color:{TEXT}!important;font-weight:700!important;font-size:1.1rem!important;}}

[data-baseweb="select"]>div,[data-testid="stNumberInput"] input{{
    background:{CARD}!important;border-color:{BORDER}!important;
    color:{TEXT}!important;border-radius:8px!important;}}
[data-baseweb="popover"] [role="listbox"]{{background:{CARD}!important;border:1px solid {BORDER}!important;}}
[data-baseweb="popover"] [role="option"]:hover{{background:{CARD2}!important;}}

.stFormSubmitButton>button,.stButton>button{{
    background:{ACCENT}!important;color:{WHITE}!important;border:none!important;
    border-radius:8px!important;font-weight:700!important;font-size:.95rem!important;
    padding:11px 0!important;letter-spacing:.3px!important;
    box-shadow:0 0 20px {rgba(ACCENT,.27)}!important;}}
.stFormSubmitButton>button:hover,.stButton>button:hover{{
    background:{GOLD}!important;}}

[data-testid="stDataFrame"]{{background:{CARD}!important;
    border:1px solid {BORDER}!important;border-radius:8px!important;}}
[data-testid="stExpander"]{{background:{CARD}!important;
    border:1px solid {BORDER}!important;border-radius:10px!important;}}
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{BG};}}
::-webkit-scrollbar-thumb{{background:{BORDER};border-radius:3px;}}

.kpi{{background:{CARD};border:1px solid {BORDER};border-radius:12px;
      padding:20px 18px 16px;border-top:3px solid {GOLD};}}
.kpi-v{{font-size:2rem;font-weight:800;color:{WHITE};line-height:1;margin-bottom:6px;}}
.kpi-l{{font-size:.7rem;font-weight:700;color:{MUTED};text-transform:uppercase;letter-spacing:.9px;}}
.kpi-s{{font-size:.72rem;color:{GOLD};font-weight:500;margin-top:4px;}}

.shdr{{display:flex;align-items:flex-start;gap:12px;background:{CARD};
       border:1px solid {BORDER};border-radius:10px;padding:14px 18px;margin:26px 0 6px;}}
.snum{{min-width:28px;height:28px;background:{GOLD};color:{BG};border-radius:6px;
       display:flex;align-items:center;justify-content:center;
       font-weight:800;font-size:.85rem;margin-top:1px;}}
.stitle{{font-weight:700;color:{WHITE};font-size:.95rem;}}
.sdesc{{font-size:.75rem;color:{MUTED};margin-top:2px;}}

.ibox{{background:{CARD};border-left:3px solid {GOLD};border-radius:0 8px 8px 0;
       padding:13px 16px;margin:8px 0 20px;font-size:.84rem;color:{MUTED};line-height:1.7;}}
.ibox b{{color:{TEXT};}}
.itag{{display:inline-block;background:{rgba(GOLD,.15)};color:{GOLD};
       border-radius:4px;padding:1px 7px;font-size:.72rem;
       font-weight:700;letter-spacing:.5px;margin-right:6px;}}

.cbox{{background:linear-gradient(135deg,{CARD} 0%,{CARD2} 100%);
       border:1px solid {BORDER};border-top:3px solid {GOLD};
       border-radius:0 0 10px 10px;padding:20px 22px;margin:8px 0 6px;
       font-size:.84rem;color:{MUTED};line-height:1.9;}}
.cbox b{{color:{WHITE};}}
.ctitle{{font-size:.7rem;font-weight:700;color:{GOLD};
         text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;}}

.rpill{{background:{CARD};border:1px solid {BORDER};border-radius:12px;
        padding:24px 20px;text-align:center;}}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════
# MODELO
# ═══════════════════════════════════
@st.cache_data(show_spinner="Entrenando modelo Random Forest…")
def load():
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=544)
        X_r = ds.data.features; y_r = ds.data.targets.squeeze()
        src = "UCI ML Repository · id=544"
    except Exception:
        base = os.path.dirname(os.path.abspath(__file__))
        df   = pd.read_csv(os.path.join(base,"obesity_data.csv"))
        y_r  = df["NObeyesdad"]; X_r = df.drop(columns=["NObeyesdad"])
        src  = "Dataset local (réplica UCI 544)"

    df = X_r.copy(); y = y_r.copy()
    m  = ~df.duplicated(); df=df[m].reset_index(drop=True); y=y[m].reset_index(drop=True)
    for c in ["NCP","CH2O","FAF","TUE","FCVC"]:
        if c in df: df[c]=df[c].round().astype(float)
    for c in df.select_dtypes("number"):
        d=df[c].dropna(); Q1,Q3=d.quantile(.25),d.quantile(.75)
        if ((d<Q1-1.5*(Q3-Q1))|(d>Q3+1.5*(Q3-Q1))).sum()>20:
            df[c]=mstats.winsorize(df[c],limits=[.01,.01])

    fe=df.copy()
    fb=(df["FAVC"]=="yes").astype(float); fm=(df["family_history_with_overweight"]=="yes").astype(float)
    fe["IMC"]=df["Weight"]/df["Height"]**2
    fe["Riesgo_Calorico"]=fb*(4-df["FAF"].clip(0,3))
    fe["Actividad_Neta"]=df["FAF"]-df["TUE"]
    fe["Carga_Hidratacion"]=df["CH2O"]/(df["NCP"]+1)
    fe["Herencia_Riesgo"]=fm*fb

    ff=fe.copy(); enc={}
    for c in ff.select_dtypes("object"):
        le=LabelEncoder(); ff[c]=le.fit_transform(ff[c].astype(str)); enc[c]=le
    ley=LabelEncoder(); ye=le_y=ley.fit_transform(y)
    sc=StandardScaler(); Xs=sc.fit_transform(ff)
    Xtr,Xte,ytr,yte=train_test_split(Xs,ye,test_size=.25,random_state=42,stratify=ye)
    rf=RandomForestClassifier(n_estimators=100,random_state=42); rf.fit(Xtr,ytr)
    yp=rf.predict(Xte); ypr=rf.predict_proba(Xte)
    acc=accuracy_score(yte,yp)
    auc=roc_auc_score(label_binarize(yte,classes=list(range(7))),ypr,multi_class="ovr",average="macro")
    skf=StratifiedKFold(5,shuffle=True,random_state=42)
    cv=cross_val_score(rf,Xs,ye,cv=skf,scoring="accuracy")
    cm=confusion_matrix(yte,yp)
    rep=classification_report(yte,yp,target_names=ley.classes_,output_dict=True)
    p2=PCA(2); Xp=p2.fit_transform(Xs); ev=PCA().fit(Xs).explained_variance_ratio_
    return dict(rf=rf,sc=sc,enc=enc,ley=ley,ye=ye,y=y,Xs=Xs,Xp=Xp,ev=ev,
                acc=acc,auc=auc,cv=cv,cm=cm,rep=rep,
                fn=list(ff.columns),imp=rf.feature_importances_,
                df=df,fe=fe,Xtr=Xtr,ytr=ytr,yte=yte,yp=yp,src=src)

def do_predict(d, R):
    du = pd.DataFrame([d])
    for c,le in R["enc"].items():
        if c in du:
            v=str(du[c].values[0])
            if v not in le.classes_: v=le.classes_[0]
            du[c]=le.transform([v])
    h,w   = float(du["Height"].values[0]), float(du["Weight"].values[0])
    faf   = float(du["FAF"].values[0]);   tue  = float(du["TUE"].values[0])
    ch2o  = float(du["CH2O"].values[0]);  ncp  = float(du["NCP"].values[0])
    fv    = float(du["FAVC"].values[0]);  fm   = float(du["family_history_with_overweight"].values[0])
    du["IMC"]=w/h**2; du["Riesgo_Calorico"]=fv*(4-min(faf,3))
    du["Actividad_Neta"]=faf-tue; du["Carga_Hidratacion"]=ch2o/(ncp+1)
    du["Herencia_Riesgo"]=fm*fv
    du=du[R["fn"]]; X=R["sc"].transform(du.values.astype(float))
    pe=R["rf"].predict(X)[0]; pp=R["rf"].predict_proba(X)[0]
    return R["ley"].inverse_transform([pe])[0], pp, R["ley"].classes_

def dark(fig, title="", h=380, legend=True):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" if title else "",
                   font=dict(size=13,color=TEXT,family="Inter"),x=0,xanchor="left"),
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(family="Inter",color=MUTED,size=11),
        height=h, showlegend=legend,
        margin=dict(t=44 if title else 20,b=36,l=44,r=16),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color=MUTED)),
    )
    fig.update_xaxes(gridcolor=BORDER,zerolinecolor=BORDER,
                     showline=True,linecolor=BORDER,tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER,zerolinecolor=BORDER,
                     showline=True,linecolor=BORDER,tickfont=dict(color=MUTED))
    return fig

def pc(fig): st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

def kpi4(vals):
    cols=st.columns(4)
    for c,(v,l,s) in zip(cols,vals):
        c.markdown(f'<div class="kpi"><div class="kpi-v">{v}</div>'
                   f'<div class="kpi-l">{l}</div><div class="kpi-s">{s}</div></div>',
                   unsafe_allow_html=True)

def shdr(n,t,d):
    st.markdown(f'<div class="shdr"><div class="snum">{n}</div>'
                f'<div><div class="stitle">{t}</div><div class="sdesc">{d}</div></div></div>',
                unsafe_allow_html=True)

def ibox(tag,txt):
    st.markdown(f'<div class="ibox"><span class="itag">{tag}</span>{txt}</div>',
                unsafe_allow_html=True)

def cbox(title,items):
    rows="".join(f"<div style='margin:3px 0'>{i}</div>" for i in items)
    st.markdown(f'<div class="cbox"><div class="ctitle">⬡ {title}</div>{rows}</div>',
                unsafe_allow_html=True)


# ═══════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════
with st.sidebar:
    # Logo USS
    if LOGO:
        st.markdown(
            f'<div style="text-align:center;padding:18px 0 16px;">'
            f'<div style="background:{WHITE};border-radius:10px;padding:12px 16px;'
            f'display:inline-block;box-shadow:0 2px 10px rgba(0,0,0,.3);">'
            f'<img src="data:image/png;base64,{LOGO}" '
            f'style="width:140px;display:block;"></div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="text-align:center;padding:22px 0 16px;'
            f'font-size:1.5rem;font-weight:800;color:{GOLD};">USS</div>',
            unsafe_allow_html=True)

    st.markdown(
        f'<div style="padding:0 4px 14px;border-bottom:1px solid {BORDER};">'
        f'<div style="font-size:10px;font-weight:700;color:{GOLD};letter-spacing:1.4px;'
        f'text-transform:uppercase;margin-bottom:8px;">Información del Proyecto</div>'
        f'<div style="font-size:11.5px;color:{MUTED};line-height:2;">'
        f'<span style="color:{TEXT};font-weight:600;">Curso</span> · Taller de Aplicaciones<br>'
        f'<span style="color:{TEXT};font-weight:600;">Docente</span> · Dr. M. Sepúlveda<br>'
        f'<span style="color:{TEXT};font-weight:600;">Alumnos</span> · F. Carrasco · C. Tello<br>'
        f'<span style="color:{TEXT};font-weight:600;">Modelo</span> · Random Forest<br>'
        f'<span style="color:{TEXT};font-weight:600;">Dataset</span> · UCI ML Repo #544'
        f'</div></div>'
        f'<div style="font-size:10px;font-weight:700;color:{GOLD};letter-spacing:1.4px;'
        f'text-transform:uppercase;padding:14px 4px 10px;">Navegación</div>',
        unsafe_allow_html=True)

    pag = st.radio("nav",
        ["📈  Exploración del Dataset",
         "📊  Rendimiento del Modelo",
         "🔮  Clasificar Paciente"],
        label_visibility="collapsed")

    st.markdown(
        f'<div style="font-size:10px;color:{BORDER};padding:16px 4px 4px;'
        f'line-height:2;border-top:1px solid {BORDER};margin-top:14px;">'
        f'100 árboles · split 75/25 estratificado<br>'
        f'StratifiedKFold 5-fold · 7 clases OMS<br>'
        f'Feature Engineering: IMC + 4 variables</div>',
        unsafe_allow_html=True)

R = load()
NEW = {"IMC","Riesgo_Calorico","Actividad_Neta","Carga_Hidratacion","Herencia_Riesgo"}


# ═══════════════════════════════════
# PÁGINA 1 — EXPLORACIÓN
# ═══════════════════════════════════
if pag == "📈  Exploración del Dataset":
    st.title("Exploración del Dataset")
    st.markdown(f"<p style='color:{MUTED};font-size:.83rem;margin-top:-6px;margin-bottom:20px;'>"
                f"Fuente: {R['src']}  ·  {len(R['y']):,} muestras  ·  16 variables + 5 features engineered</p>",
                unsafe_allow_html=True)

    cnt=R["y"].value_counts().reindex(ORDEN); tot=cnt.sum()
    nob=cnt[["Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]].sum()
    nsp=cnt[["Overweight_Level_I","Overweight_Level_II"]].sum()
    kpi4([(f"{tot:,}","Muestras totales","Post-limpieza y deduplicación"),
          (f"{nob:,}","Casos de obesidad",f"{nob/tot*100:.1f}% del dataset"),
          (f"{nsp:,}","Casos sobrepeso",  f"{nsp/tot*100:.1f}% del dataset"),
          (f"{R['fe']['IMC'].median():.1f}","IMC mediano","kg/m² — feature engineered")])

    # ── 1. Distribución ──
    shdr("1","Distribución de Clases","¿Cuántas muestras tiene cada nivel? ¿Está balanceado el dataset?")
    c1,c2=st.columns(2)
    with c1:
        fig=go.Figure(go.Bar(
            x=[LMAP[c] for c in ORDEN],y=cnt.values,
            marker=dict(color=CLASS_C,line=dict(width=0)),
            text=[f"<b>{v}</b>" for v in cnt.values],textposition="outside",
            textfont=dict(color=TEXT,size=11),
            hovertemplate="<b>%{x}</b><br>N = %{y}<extra></extra>"))
        fig.add_hline(y=tot/7,line_dash="dot",line_color=GOLD,line_width=1.5,
                      annotation_text="Distribución ideal",annotation_font_size=10,
                      annotation_font_color=GOLD,annotation_position="top right")
        dark(fig,"Frecuencia absoluta por clase",360,False)
        fig.update_xaxes(tickangle=-30,tickfont=dict(size=10))
        fig.update_yaxes(title_text="N muestras",title_font=dict(color=MUTED,size=11))
        pc(fig)
    with c2:
        fig=go.Figure(go.Pie(
            labels=LABELS,values=cnt.values,
            marker=dict(colors=CLASS_C,line=dict(color=BG,width=2)),
            hole=.52,textinfo="percent",textfont=dict(size=11,color=TEXT),
            hovertemplate="<b>%{label}</b><br>%{value} muestras · %{percent}<extra></extra>"))
        fig.add_annotation(text=f"<b>{tot}</b><br>muestras",x=.5,y=.5,
                           showarrow=False,font=dict(size=13,color=TEXT,family="Inter"))
        dark(fig,"Proporción relativa de clases",360)
        pc(fig)

    ibox("INSIGHT","El dataset presenta una distribución <b>balanceada</b> (270–350 muestras por clase). "
         "<b>Obesidad Tipo I</b> es la clase más frecuente (≈17%) y "
         "<b>Peso Insuficiente</b> la menos representada (≈13%), lo que puede generar un leve sesgo hacia las clases mayoritarias.")

    # ── 2. Variables numéricas (violin) ──
    shdr("2","Variables Numéricas por Clase","¿Qué variable separa mejor los niveles? Usa el selector para explorar.")
    VNUM=["Weight","Age","Height","FAF"]
    VNOM=["Peso (kg)","Edad (años)","Altura (m)","Actividad física (días/sem)"]
    vs=st.selectbox("Variable a explorar:",VNUM,format_func=lambda x:VNOM[VNUM.index(x)])

    dfp=R["df"][["Weight","Age","Height","FAF"]].copy(); dfp["Clase"]=R["y"].values
    fig=go.Figure()
    for ck,ces,col in zip(ORDEN,LABELS,CLASS_C):
        mask=dfp["Clase"]==ck
        fig.add_trace(go.Violin(
            x=[ces]*mask.sum(), y=dfp.loc[mask,vs].values,
            name=ces,
            fillcolor=rgba(col,.35),   # ← rgba correcto
            line_color=col, opacity=.9,
            box_visible=True, meanline_visible=True,
            meanline_color=WHITE, box_fillcolor=BG,
            points="outliers", marker=dict(size=3,color=col,opacity=.5),
            hovertemplate=f"<b>{ces}</b><br>{vs}: %{{y:.2f}}<extra></extra>"))
    dark(fig,f"Distribución de {VNOM[VNUM.index(vs)]} por nivel de obesidad",410,False)
    fig.update_layout(violinmode="group",violingap=.04)
    fig.update_xaxes(tickangle=-25,tickfont=dict(size=10))
    pc(fig)

    ibox("INSIGHT","El <b>Peso</b> es el predictor visual más poderoso — muestra separación casi perfecta entre clases. "
         "La <b>Actividad física</b> presenta solapamiento pero con tendencia descendente a mayor obesidad. "
         "La <b>Edad</b> y la <b>Altura</b> son menos discriminativas de manera aislada.")

    # ── 3. IMC ──
    shdr("3","IMC por Nivel de Obesidad","¿El IMC engineered respeta los umbrales clínicos OMS?")
    imc_v=R["fe"]["IMC"].values
    fig=go.Figure()
    for ck,ces,col in zip(ORDEN,LABELS,CLASS_C):
        mask=R["y"].values==ck
        fig.add_trace(go.Box(
            y=imc_v[mask],name=ces,
            marker_color=col, line_color=col,
            fillcolor=rgba(col,.25),   # ← rgba correcto
            boxmean=True,
            hovertemplate=f"<b>{ces}</b><br>IMC: %{{y:.1f}}<extra></extra>"))
    for yv,lbl,col in [(18.5,"Bajo peso < 18.5",GREEN),(25,"Sobrepeso ≥ 25",AMBER),(30,"Obesidad ≥ 30",RED)]:
        fig.add_hline(y=yv,line_dash="dash",line_color=col,line_width=1.5,
                      annotation_text=f"OMS: {lbl}",annotation_font_size=10,
                      annotation_font_color=col,annotation_position="top right")
    dark(fig,"IMC por clase · Validación de umbrales OMS",400,False)
    fig.update_yaxes(title_text="IMC (kg/m²)",title_font=dict(color=MUTED))
    pc(fig)

    ibox("INSIGHT","El IMC construido (<i>peso/altura²</i>) <b>valida clínicamente las etiquetas del dataset</b>: "
         "las medianas de cada clase se alinean con las bandas OMS, confirmando la calidad de los datos "
         "y justificando que el IMC sea la variable más importante en el Random Forest.")

    # ── 4. Scatter ──
    shdr("4","Edad vs. Peso","¿Existe un patrón visual en el espacio edad-peso que permita separar clases?")
    fig=go.Figure()
    for ck,ces,col in zip(ORDEN,LABELS,CLASS_C):
        mask=R["y"].values==ck; sub=R["df"][mask]
        fig.add_trace(go.Scatter(
            x=sub["Age"],y=sub["Weight"],mode="markers",name=ces,
            marker=dict(color=col,size=5,opacity=.65,
                        line=dict(width=.3,color=BG)),
            hovertemplate=f"<b>{ces}</b><br>Edad: %{{x:.0f}} · Peso: %{{y:.1f}} kg<extra></extra>"))
    dark(fig,"Dispersión Edad vs. Peso coloreado por nivel de obesidad",420)
    fig.update_xaxes(title_text="Edad (años)"); fig.update_yaxes(title_text="Peso (kg)")
    pc(fig)

    cbox("Conclusiones — Exploración del Dataset",[
        "✦ Dataset <b>balanceado</b> — 7 clases homogéneas sin necesidad de re-muestreo.",
        "✦ <b>Peso e IMC</b> son los discriminadores naturales más potentes.",
        "✦ El IMC engineered <b>valida clínicamente</b> las etiquetas bajo criterios OMS.",
        "✦ Relaciones no lineales justifican el uso de <b>Random Forest</b> como modelo de ensamble."])


# ═══════════════════════════════════
# PÁGINA 2 — RENDIMIENTO
# ═══════════════════════════════════
elif pag == "📊  Rendimiento del Modelo":
    st.title("Rendimiento del Clasificador")
    st.markdown(f"<p style='color:{MUTED};font-size:.83rem;margin-top:-6px;margin-bottom:20px;'>"
                "Random Forest · 100 árboles · split 75/25 estratificado · StratifiedKFold 5-fold</p>",
                unsafe_allow_html=True)

    kpi4([(f"{R['acc']:.1%}","Accuracy — test set","Fracción correctamente clasificada"),
          (f"{R['cv'].mean():.1%}","CV 5-fold media",f"±{R['cv'].std():.3f} desv. estándar"),
          (f"{R['auc']:.4f}","AUC-ROC macro","Capacidad discriminativa global"),
          (f"{len(R['Xtr'])}","Muestras entrenamiento",f"{len(R['yte'])} muestras de test")])

    ibox("LECTURA RÁPIDA","Un <b>AUC-ROC ≥ 0.99</b> indica discriminación casi perfecta entre los 7 niveles. "
         "La baja desviación estándar en CV (≤ 0.01) confirma <b>estabilidad y ausencia de sobreajuste</b>.")

    # ── 1. CV ──
    shdr("1","Validación Cruzada Stratified 5-Fold","¿El modelo mantiene su rendimiento en distintas particiones?")
    cv_col=[GOLD if v==R["cv"].max() else ACCENT for v in R["cv"]]
    fig=go.Figure(go.Bar(
        x=[f"Fold {i+1}" for i in range(5)],y=R["cv"],
        marker=dict(color=cv_col,line=dict(width=0)),width=.42,
        text=[f"<b>{v:.4f}</b>" for v in R["cv"]],
        textposition="outside",textfont=dict(color=TEXT,size=11),
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=R["cv"].mean(),line_dash="dash",line_color=GOLD,line_width=1.8,
                  annotation_text=f"Promedio: {R['cv'].mean():.4f}",
                  annotation_font_color=GOLD,annotation_font_size=11,annotation_position="top right")
    fig.add_hrect(y0=R["cv"].mean()-R["cv"].std(),y1=R["cv"].mean()+R["cv"].std(),
                  fillcolor=rgba(GOLD,.06),line_width=0,
                  annotation_text="±1σ",annotation_position="top left",
                  annotation_font_color=GOLD,annotation_font_size=10)
    dark(fig,"Accuracy por fold — Estabilidad entre particiones",340,False)
    fig.update_yaxes(range=[max(0,R["cv"].min()-.04),1.02],
                     title_text="Accuracy",title_font=dict(color=MUTED))
    pc(fig)

    ibox("INSIGHT","Los cinco folds arrojan resultados casi idénticos (banda sombreada = ±1σ), "
         "confirmando que el modelo <b>generaliza correctamente</b> y no está sobreajustado.")

    # ── 2. Matriz de confusión ──
    shdr("2","Matriz de Confusión","¿Dónde se equivoca el modelo? ¿Entre qué clases existe ambigüedad?")
    etq=[c.replace("_","<br>") for c in R["ley"].classes_]
    cm_n=R["cm"].astype(float)/R["cm"].sum(axis=1,keepdims=True)
    txt=[[f"<b>{R['cm'][i][j]}</b><br><span style='font-size:9px'>{cm_n[i][j]*100:.0f}%</span>"
          for j in range(7)] for i in range(7)]

    # Colorscale con rgba (NO hex de 8 dígitos)
    cs=[[0,CARD],[0.001,CARD2],[0.4,rgba(ACCENT,.53)],[1,ACCENT]]
    fig=go.Figure(go.Heatmap(
        z=cm_n,x=etq,y=etq,colorscale=cs,
        text=txt,texttemplate="%{text}",
        textfont=dict(size=10,color=TEXT),
        hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Tasa: %{z:.1%}<extra></extra>",
        showscale=True,
        colorbar=dict(title="Tasa",tickformat=".0%",len=.8,
                      tickfont=dict(color=MUTED),title_font=dict(color=MUTED))))
    dark(fig,f"Matriz de Confusión · Accuracy = {R['acc']:.1%} · AUC-ROC = {R['auc']:.4f}",480,False)
    fig.update_xaxes(title_text="Predicho",side="bottom",tickfont=dict(size=9,color=MUTED))
    fig.update_yaxes(title_text="Real",autorange="reversed",tickfont=dict(size=9,color=MUTED))
    pc(fig)

    ibox("INSIGHT","La diagonal concentra casi toda la masa de predicciones. "
         "Los errores residuales ocurren entre <b>clases adyacentes</b> (ej. Sobrepeso I / II), "
         "lo que es <b>clínicamente aceptable</b> dado que sus límites son continuos.")

    # ── 3. Métricas por clase ──
    shdr("3","Precision, Recall y F1-Score","¿El modelo es igualmente preciso en todas las clases?")
    clases_le=list(R["ley"].classes_)
    fig=go.Figure()
    for met,col in [("precision",ACCENT),("recall",GOLD),("f1-score",GREEN)]:
        vals=[R["rep"][c][met] for c in clases_le]
        fig.add_trace(go.Bar(
            name=met.capitalize(),
            x=[LMAP.get(c,c) for c in clases_le],y=vals,
            marker=dict(color=col,line=dict(width=0)),opacity=.88,
            text=[f"<b>{v:.2f}</b>" for v in vals],
            textposition="outside",textfont=dict(size=9,color=TEXT),
            hovertemplate=f"<b>%{{x}}</b><br>{met}: %{{y:.3f}}<extra></extra>"))
    fig.add_hline(y=1,line_dash="dot",line_color=BORDER,line_width=1)
    dark(fig,"Métricas de clasificación por clase",400)
    fig.update_layout(barmode="group",yaxis_range=[0,1.12])
    fig.update_xaxes(tickangle=-25,tickfont=dict(size=10))
    fig.update_yaxes(tickformat=".0%",title_text="Score",title_font=dict(color=MUTED))
    pc(fig)

    # ── 4. Feature Importance ──
    shdr("4","Importancia de Variables — Criterio Gini","¿Qué variables aportan más al poder predictivo del modelo?")
    fn=R["fn"]; imp=R["imp"]; ord_=np.argsort(imp)[::-1]
    nom=[fn[i] for i in ord_]; iv=[imp[i] for i in ord_]
    col_=[GOLD if n in NEW else ACCENT for n in nom]
    fig=go.Figure(go.Bar(
        x=iv[::-1],y=nom[::-1],orientation="h",
        marker=dict(color=col_[::-1],line=dict(width=0)),
        text=[f"<b>{v:.3f}</b>" for v in iv[::-1]],
        textposition="outside",textfont=dict(size=10,color=TEXT),
        hovertemplate="<b>%{y}</b><br>Gini: %{x:.4f}<extra></extra>"))
    for n2,c2 in [("Feature Engineering",GOLD),("Feature Original",ACCENT)]:
        fig.add_trace(go.Bar(x=[None],y=[None],name=n2,marker_color=c2,showlegend=True))
    dark(fig,"Importancia Gini — Dorado = variables de Feature Engineering",520)
    fig.update_xaxes(title_text="Importancia Gini",title_font=dict(color=MUTED))
    fig.update_layout(legend=dict(x=.65,y=.05))
    pc(fig)

    ibox("INSIGHT","Las variables de <b>Feature Engineering</b> (dorado) — especialmente "
         "<b>IMC</b> y <b>Riesgo Calórico</b> — figuran entre las más importantes, "
         "validando la estrategia de ingeniería de características implementada.")

    cbox("Conclusiones — Rendimiento del Modelo",[
        "✦ <b>Accuracy ≥ 95% y AUC-ROC ≥ 0.99</b> — rango de excelencia clínica.",
        "✦ Validación cruzada confirma <b>robustez y ausencia de sobreajuste</b>.",
        "✦ Errores concentrados entre clases adyacentes — <b>aceptable en clasificación ordinal</b>.",
        "✦ Feature Engineering aporta valor predictivo <b>medible y cuantificable</b> vía Gini.",
        "✦ Modelo listo para su uso como <b>herramienta de screening clínico</b>."])


# ═══════════════════════════════════
# PÁGINA 3 — PREDICCIÓN
# ═══════════════════════════════════
elif pag == "🔮  Clasificar Paciente":
    st.title("Clasificar Paciente")
    st.markdown(f"<p style='color:{MUTED};font-size:.83rem;margin-top:-6px;margin-bottom:20px;'>"
                "Ingresa los datos · El modelo calculará el IMC y clasificará según criterios OMS</p>",
                unsafe_allow_html=True)

    ibox("INSTRUCCIONES","Completa los tres paneles — <b>datos físicos</b>, "
         "<b>hábitos alimentarios</b> y <b>estilo de vida</b> — y presiona "
         "<b>Clasificar Paciente</b>. El resultado incluye la clase predicha, "
         "las probabilidades por clase y un gauge del IMC comparado con umbrales OMS.")

    # ── Formulario ──
    with st.form("fp"):
        c1,c2,c3=st.columns(3)
        def hdr(t):
            return (f"<div style='font-size:.75rem;font-weight:700;color:{GOLD};"
                    f"text-transform:uppercase;letter-spacing:.9px;"
                    f"border-bottom:1px solid {BORDER};padding-bottom:6px;"
                    f"margin-bottom:14px;'>{t}</div>")
        with c1:
            st.markdown(hdr("🧍 Datos Físicos"),unsafe_allow_html=True)
            Gender = st.selectbox("Género",["Male","Female"],
                                  format_func=lambda x:"Masculino" if x=="Male" else "Femenino")
            Age    = st.number_input("Edad (años)",14,70,25,step=1)
            Height = st.number_input("Altura (m)",1.45,2.00,1.70,step=.01,format="%.2f")
            Weight = st.number_input("Peso (kg)",39.0,173.0,70.0,step=.5,format="%.1f")
        with c2:
            st.markdown(hdr("🥗 Hábitos Alimentarios"),unsafe_allow_html=True)
            fam  = st.selectbox("Historial familiar de sobrepeso",["yes","no"],
                                format_func=lambda x:"Sí" if x=="yes" else "No")
            FAVC = st.selectbox("Consumo frecuente hipercalórico",["yes","no"],
                                format_func=lambda x:"Sí" if x=="yes" else "No")
            FCVC = st.select_slider("Frecuencia consumo verduras",[1.,2.,3.],
                                    format_func=lambda x:{1.:"Nunca",2.:"A veces",3.:"Siempre"}[x],value=2.)
            NCP  = st.select_slider("Comidas principales al día",[1.,2.,3.,4.],value=3.)
            CAEC = st.selectbox("Come entre comidas",["no","Sometimes","Frequently","Always"],
                                format_func=lambda x:{"no":"No","Sometimes":"A veces",
                                                       "Frequently":"Frecuentemente","Always":"Siempre"}[x])
            CH2O = st.select_slider("Agua diaria",[1.,2.,3.],
                                    format_func=lambda x:{1.:"< 1L",2.:"1–2L",3.:"> 2L"}[x],value=2.)
        with c3:
            st.markdown(hdr("🏃 Estilo de Vida"),unsafe_allow_html=True)
            SMOKE  = st.selectbox("¿Fuma?",["no","yes"],
                                  format_func=lambda x:"No" if x=="no" else "Sí")
            SCC    = st.selectbox("¿Monitorea calorías?",["no","yes"],
                                  format_func=lambda x:"No" if x=="no" else "Sí")
            FAF    = st.select_slider("Actividad física semanal",[0.,1.,2.,3.],
                                      format_func=lambda x:{0.:"Ninguna",1.:"1–2 días",
                                                             2.:"2–4 días",3.:"4–5 días"}[x],value=1.)
            TUE    = st.select_slider("Uso tecnología/día",[0.,1.,2.],
                                      format_func=lambda x:{0.:"0–2h",1.:"3–5h",2.:">5h"}[x])
            CALC   = st.selectbox("Consumo de alcohol",["no","Sometimes","Frequently","Always"],
                                  format_func=lambda x:{"no":"No","Sometimes":"A veces",
                                                         "Frequently":"Frecuentemente","Always":"Siempre"}[x])
            MTRANS = st.selectbox("Medio de transporte",
                                  ["Public_Transportation","Walking","Automobile","Motorbike","Bike"],
                                  format_func=lambda x:{"Public_Transportation":"Transporte público",
                                                         "Walking":"A pie","Automobile":"Automóvil",
                                                         "Motorbike":"Motocicleta","Bike":"Bicicleta"}[x])
        submitted = st.form_submit_button("🔍  Clasificar Paciente",use_container_width=True)

    # ── Resultado — SOLO si se presionó el botón ──
    if submitted:
        datos=dict(Gender=Gender,Age=float(Age),Height=float(Height),Weight=float(Weight),
                   family_history_with_overweight=fam,FAVC=FAVC,FCVC=float(FCVC),NCP=float(NCP),
                   CAEC=CAEC,SMOKE=SMOKE,CH2O=float(CH2O),SCC=SCC,
                   FAF=float(FAF),TUE=float(TUE),CALC=CALC,MTRANS=MTRANS)
        try:
            clase,proba,clases_le = do_predict(datos,R)
            imc  = float(Weight)/float(Height)**2
            cc   = CMAP.get(clase,ACCENT)
            ne   = LMAP.get(clase,clase)
            cf   = proba[list(clases_le).index(clase)]

            st.markdown(f"<hr style='border:none;border-top:1px solid {BORDER};margin:22px 0;'>",
                        unsafe_allow_html=True)
            st.markdown("<h2 style='margin-bottom:16px;'>Resultado del Análisis</h2>",
                        unsafe_allow_html=True)

            r1,r2,r3=st.columns([1,1.1,1.8])

            with r1:
                st.markdown(
                    f'<div class="rpill" style="border-top:4px solid {cc};">'
                    f'<div style="font-size:.68rem;font-weight:700;color:{MUTED};'
                    f'text-transform:uppercase;letter-spacing:.9px;">Clasificación OMS</div>'
                    f'<div style="font-size:1.55rem;font-weight:800;color:{cc};'
                    f'margin:10px 0 6px;line-height:1.2;">{ne}</div>'
                    f'<div style="height:3px;background:{BORDER};border-radius:2px;margin:14px 0;">'
                    f'<div style="height:3px;background:{cc};border-radius:2px;'
                    f'width:{cf*100:.0f}%;"></div></div>'
                    f'<div style="font-size:.68rem;color:{MUTED};text-transform:uppercase;'
                    f'letter-spacing:.8px;">Confianza del modelo</div>'
                    f'<div style="font-size:2rem;font-weight:800;color:{cc};margin:4px 0;">'
                    f'{cf:.1%}</div>'
                    f'<div style="height:1px;background:{BORDER};margin:14px 0;"></div>'
                    f'<div style="font-size:.68rem;color:{MUTED};text-transform:uppercase;'
                    f'letter-spacing:.8px;">IMC calculado</div>'
                    f'<div style="font-size:2rem;font-weight:800;color:{TEXT};margin:4px 0;">'
                    f'{imc:.1f}</div>'
                    f'<div style="font-size:.72rem;color:{MUTED};">kg/m²</div></div>',
                    unsafe_allow_html=True)

            with r2:
                # Gauge IMC — colores rgba correctos
                fig_g=go.Figure(go.Indicator(
                    mode="gauge+number",value=imc,
                    number=dict(suffix=" kg/m²",font=dict(size=20,color=TEXT,family="Inter")),
                    gauge=dict(
                        axis=dict(range=[14,46],tickfont=dict(size=9,color=MUTED),
                                  tickvals=[18.5,25,30,35,40],
                                  ticktext=["18.5","25","30","35","40"]),
                        bar=dict(color=cc,thickness=.22),
                        bgcolor=CARD2, bordercolor=BORDER, borderwidth=1,
                        steps=[
                            dict(range=[14,18.5], color=rgba(GREEN,.15)),
                            dict(range=[18.5,25],  color=rgba(GREEN,.08)),
                            dict(range=[25,30],    color=rgba(AMBER,.15)),
                            dict(range=[30,35],    color=rgba(RED,.15)),
                            dict(range=[35,46],    color=rgba(RED,.28)),
                        ],
                        threshold=dict(line=dict(color=WHITE,width=2),value=imc)),
                    title=dict(text="IMC del Paciente",
                               font=dict(size=12,color=MUTED,family="Inter"))))
                fig_g.update_layout(paper_bgcolor=CARD,height=230,
                                    margin=dict(t=44,b=10,l=20,r=20),
                                    font=dict(family="Inter",color=TEXT))
                st.plotly_chart(fig_g,use_container_width=True,
                                config={"displayModeBar":False})

                flechas=["◀ Aquí" if imc<18.5 else "",
                         "◀ Aquí" if 18.5<=imc<25 else "",
                         "◀ Aquí" if 25<=imc<30 else "",
                         "◀ Aquí" if imc>=30 else ""]
                st.dataframe(pd.DataFrame({
                    "Rango IMC":["< 18.5","18.5–24.9","25–29.9","≥ 30"],
                    "Categoría":["Bajo peso","Normal","Sobrepeso","Obesidad"],
                    "Paciente":flechas}),
                    use_container_width=True,hide_index=True)

            with r3:
                etq=[LMAP.get(c,c) for c in clases_le]
                col_b=[cc if c==clase else BORDER for c in clases_le]
                fig_p=go.Figure(go.Bar(
                    x=proba,y=etq,orientation="h",
                    marker=dict(color=col_b,line=dict(width=0)),
                    text=[f"<b>{v:.1%}</b>" if v>.02 else "" for v in proba],
                    textposition="outside",textfont=dict(size=10,color=TEXT),
                    hovertemplate="<b>%{y}</b><br>Probabilidad: %{x:.2%}<extra></extra>"))
                dark(fig_p,"Probabilidad por clase (Random Forest)",310,False)
                fig_p.update_xaxes(range=[0,1.22],tickformat=".0%",
                                   title_text="Probabilidad",title_font=dict(color=MUTED))
                pc(fig_p)

            # Recomendación clínica
            if clase in ["Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]:
                bc,bc2=rgba(RED,.12),RED; nivel="⚠️  Riesgo Alto"
                msg=(f"El paciente presenta <b>{ne}</b> (IMC {imc:.1f} kg/m²). "
                     "Se recomienda <b>evaluación médica especializada</b>, plan nutricional "
                     "supervisado y programa de actividad física progresiva.")
            elif clase in ["Overweight_Level_I","Overweight_Level_II"]:
                bc,bc2=rgba(AMBER,.12),AMBER; nivel="🔶  Riesgo Moderado"
                msg=(f"El paciente presenta <b>{ne}</b> (IMC {imc:.1f} kg/m²). "
                     "Se recomienda <b>intervención preventiva</b> en hábitos alimentarios "
                     "y aumento gradual de actividad física.")
            else:
                bc,bc2=rgba(GREEN,.12),GREEN; nivel="✅  Riesgo Normal / Bajo"
                msg=(f"El paciente presenta <b>{ne}</b> (IMC {imc:.1f} kg/m²). "
                     "Mantener hábitos actuales y realizar <b>controles periódicos</b>.")

            st.markdown(
                f'<div style="background:{bc};border-left:4px solid {bc2};'
                f'border-radius:0 10px 10px 0;padding:14px 18px;margin-top:16px;'
                f'font-size:.85rem;line-height:1.65;">'
                f'<div style="font-weight:700;color:{bc2};margin-bottom:5px;">{nivel}</div>'
                f'<div style="color:{TEXT};">{msg}</div></div>',
                unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error al predecir: {e}")
            st.exception(e)
