# Clasificador de Obesidad — Streamlit App
**Solemne 1 · Taller de Aplicaciones · Dr. Mauricio Sepúlveda**  
*Felipe Carrasco, Carlos Tello*

## Archivos
- `app.py` — Aplicación Streamlit principal
- `obesity_data.csv` — Dataset de respaldo (se usa automáticamente si UCI no está disponible)
- `requirements.txt` — Dependencias Python

## Despliegue en Streamlit Community Cloud (gratuito)

1. Subir estos archivos a un repositorio de GitHub (público o privado)
2. Ir a https://share.streamlit.io → "New app"
3. Conectar el repositorio → seleccionar `app.py` → Deploy

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secciones de la app

### 📊 Resultados del Modelo
- Métricas: Accuracy, AUC-ROC, Validación cruzada 5-fold
- Matriz de confusión (7 clases)
- Precision / Recall / F1 por clase
- Feature Importance (Gini) con features engineered destacados
- Proyección PCA 2D

### 🔮 Predecir Nuevo Dato
- Formulario interactivo con todos los campos del dataset
- Predicción con probabilidades por clase
- Cálculo automático del IMC
- Comparación con umbrales OMS

### 📈 Análisis Exploratorio
- Distribución de clases
- Histogramas por clase
- Boxplot IMC vs niveles OMS

## Modelo
- **Random Forest** (100 árboles, random_state=42)
- **Split**: 75% train / 25% test, estratificado por clase
- **Validación**: StratifiedKFold 5-fold
- **Feature Engineering**: IMC + 4 variables clínicas derivadas
- **Dataset**: UCI ML Repo id=544 (con fallback a CSV local)
