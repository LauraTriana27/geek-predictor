# 🔮 ¿Qué tipo de geek eres?

Experiencia interactiva de Machine Learning desarrollada como propuesta para acercar la Ciencia de Datos al público de SOFA Colombia.

## Objetivo

El visitante responde 10 preguntas sobre gustos y preferencias. Un modelo de clasificación aprende patrones a partir de datos sintéticos y predice uno de seis perfiles: Gamer, Lore Master, Creador, Estratega, Tech Geek o Explorador.

> **Importante:** este prototipo utiliza datos sintéticos. Los resultados no representan una población real de visitantes de SOFA ni una población científica de geeks.

## Stack

Python · Streamlit · pandas · NumPy · scikit-learn · matplotlib · seaborn · joblib · Pillow

## Modelos

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors

Se calculan Accuracy, Precision, Recall y F1-score. El mejor modelo se selecciona automáticamente por F1-score, usando Accuracy como métrica complementaria. También se genera una matriz de confusión.

## Estructura

```text
geek-predictor/
├── app/
│   └── main.py
├── data/
│   └── geek_dataset.csv        # se genera automáticamente
├── models/                     # artefactos locales opcionales
├── notebooks/
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   ├── train_models.py
│   └── predict.py
├── assets/
├── requirements.txt
├── README.md
└── .gitignore
```

## Ejecutar localmente

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/train_models.py
streamlit run app/main.py
```

La aplicación también puede generar automáticamente el dataset de 3.000 registros y entrenar los modelos si todavía no existen los artefactos locales.

## Publicar en Streamlit Community Cloud

1. Abre Streamlit Community Cloud.
2. Crea una nueva aplicación desde GitHub.
3. Selecciona `LauraTriana27/geek-predictor`.
4. Rama: `main`.
5. Archivo principal: `app/main.py`.
6. Pulsa Deploy.

No se requieren secretos ni servicios externos.

## Datos sintéticos

`src/generate_data.py` crea 3.000 registros con semilla fija (42). Las seis clases tienen firmas de preferencias diferentes y cada respuesta incorpora variabilidad y ruido. Esto evita que una sola pregunta determine artificialmente el perfil.

## Flujo

```text
INICIO
  ↓
¿QUÉ TIPO DE GEEK ERES?
  ↓
10 PREGUNTAS
  ↓
ANALIZANDO...
  ↓
TU RESULTADO
  ↓
¿LA IA ACERTÓ?
  ↓
COMPARTIR
```

## Evolución en SOFA

**Fase 1 — Prototipo:** datos sintéticos → modelo inicial.

**Fase 2 — SOFA:** respuestas reales y voluntarias → nuevos datos.

**Fase 3 — Después de SOFA:** evaluación → mejora del modelo.

## Limitaciones

- Los datos son sintéticos.
- La distribución de perfiles fue diseñada para una demostración equilibrada.
- Las probabilidades del modelo no equivalen a una certeza psicológica.
- El feedback se mantiene únicamente en la sesión del prototipo; una versión productiva requeriría almacenamiento anónimo y consentimiento.
