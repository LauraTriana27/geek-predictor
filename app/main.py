from pathlib import Path
import io,sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from generate_data import generate_dataset,PROFILES
from train_models import train_and_evaluate

st.set_page_config(page_title="¿Qué tipo de geek eres?",page_icon="🔮",layout="wide")

QUESTIONS=[
("Tienes una tarde libre. ¿Qué te gustaría hacer?",["Jugar videojuegos","Ver anime o una serie","Dibujar, crear o hacer manualidades","Resolver puzzles o jugar estrategia","Experimentar con tecnología","Buscar algo completamente nuevo"]),
("¿Qué te engancha más de una historia?",["La acción y los desafíos","El universo y su lore","Los personajes y la estética","Las estrategias y decisiones","La tecnología y las ideas","Explorar un mundo desconocido"]),
("En un videojuego, ¿qué disfrutas más?",["Competir y ganar","Seguir la historia","Personalizar personajes o escenarios","Crear estrategias","Experimentar con mecánicas","Explorar el mapa"]),
("¿Qué actividad te parece más divertida?",["Competir contra otras personas","Descubrir teorías sobre una historia","Crear algo desde cero","Resolver un desafío","Aprender cómo funciona algo","Probar una experiencia nueva"]),
("Si encuentras un nuevo universo ficticio, ¿qué haces primero?",["Busco cómo jugarlo","Investigo todo el lore","Me fijo en el diseño de personajes","Intento entender sus reglas","Analizo cómo fue construido","Exploro sin investigar demasiado"]),
("¿Qué tipo de reto te atrae más?",["Ganar una partida difícil","Resolver un misterio","Crear algo impresionante","Resolver un problema complejo","Construir o programar algo","Descubrir algo que nadie te ha mostrado"]),
("¿Qué objeto llevarías a una convención geek?",["Consola portátil","Libro o manga","Material para dibujar","Juego de mesa","Gadget tecnológico","Cámara para registrar la experiencia"]),
("Cuando algo no funciona, ¿qué haces primero?",["Lo intento otra vez","Busco información sobre ello","Intento modificarlo","Analizo qué estrategia utilizar","Investigo cómo funciona","Pruebo una alternativa diferente"]),
("¿Qué contenido podrías consumir durante horas?",["Gameplays o esports","Anime, películas o series","Arte, cosplay o creación","Estrategias, puzzles o teorías","Tecnología, IA o programación","Contenido sobre temas nuevos"]),
("¿Qué frase te representa más?",["Una partida más.","Necesito conocer todo el lore.","Yo podría diseñarlo mejor.","Tiene que existir una estrategia.","Quiero saber cómo funciona.","¿Qué habrá después?"])
]
INFO={
"Gamer":("🎮","PLAYER 01","Te mueven los retos, la competencia y esa sensación de que una partida más puede convertirse en una aventura épica.",["Competitivo","Persistente","Orientado al reto"]),
"Lore Master":("🧙","LORE ARCHIVIST","No te basta con vivir una historia: quieres entender su universo, personajes, teorías y cada detalle escondido.",["Curioso","Narrativo","Detallista"]),
"Creador":("🎨","CREATIVE MODE","Tu lugar favorito es donde una idea puede convertirse en algo que puedas diseñar, construir, editar o personalizar.",["Creativo","Expresivo","Visual"]),
"Estratega":("🧩","TACTICAL MIND","Antes de lanzarte, analizas. Te encantan los puzzles, las decisiones difíciles y encontrar la jugada que nadie vio.",["Analítico","Planificador","Resolutivo"]),
"Tech Geek":("🤖","SYSTEM EXPLORER","Si algo funciona, quieres saber por qué. Tecnología, IA, programación y gadgets despiertan tu curiosidad.",["Curioso","Lógico","Explorador tech"]),
"Explorador":("🌌","DISCOVERY MODE","Tu superpoder es la curiosidad. Te emociona probar cosas nuevas, descubrir comunidades y entrar en mundos que todavía no conoces.",["Curioso","Aventurero","Flexible"])
}
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
.stApp{background:radial-gradient(circle at 80% 10%,#233b78 0,transparent 28%),#071225;color:#f7f5ef;font-family:Inter,sans-serif}
.block-container{max-width:1150px;padding-top:2rem}.hero{text-align:center;min-height:65vh;display:flex;flex-direction:column;justify-content:center;align-items:center}
.hero h1{font-family:'Space Grotesk';font-size:clamp(3.2rem,9vw,7rem);line-height:.92;letter-spacing:-.06em;background:linear-gradient(100deg,#fff,#70e7ff,#b994ff);-webkit-background-clip:text;color:transparent}
.hero p,.muted{color:#aebbd0}.card{border:1px solid #ffffff18;background:#ffffff09;border-radius:26px;padding:1.5rem}.tag{color:#62e5ff;letter-spacing:.15em;font-size:.8rem;font-weight:800}
.big{font-family:'Space Grotesk';font-size:4rem}.score{font-size:2rem;font-weight:800}.trait{display:inline-block;background:#ffffff10;border-radius:999px;padding:.45rem .7rem;margin:.2rem}.flow{display:flex;gap:.6rem;justify-content:center;align-items:center;flex-wrap:wrap}.node{background:#102442;padding:.8rem 1rem;border-radius:14px}
</style>""",unsafe_allow_html=True)

@st.cache_data
def data_and_models():
    data_path=ROOT/"data"/"geek_dataset.csv"; data_path.parent.mkdir(exist_ok=True)
    if data_path.exists(): df=pd.read_csv(data_path)
    else:
        df=generate_dataset(3000,42); df.to_csv(data_path,index=False)
    metrics,best,model,cm,_=train_and_evaluate(df)
    return df,metrics,best,model,cm

def card_image(profile,pct):
    emoji,_,desc,_=INFO[profile]; img=Image.new("RGB",(1200,630),(7,18,37)); d=ImageDraw.Draw(img)
    try: big=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",72); med=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",34)
    except: big=med=ImageFont.load_default()
    d.text((70,70),"¿QUÉ TIPO DE GEEK ERES?",fill=(98,229,255),font=med); d.text((70,145),f"{emoji}  {profile.upper()}",fill="white",font=big); d.text((75,250),f"{pct}% de probabilidad",fill=(169,225,255),font=med); d.text((75,330),desc[:70],fill=(185,198,217),font=med)
    out=io.BytesIO(); img.save(out,"PNG"); out.seek(0); return out

def home():
    st.markdown('<div class="hero"><div class="tag">SOFA · DATA SCIENCE EXPERIENCE</div><h1>¿QUÉ TIPO<br>DE GEEK ERES?</h1><p>La IA quiere descubrirlo.<br>Responde 10 preguntas y descubre qué perfil geek predice nuestro modelo.</p></div>',unsafe_allow_html=True)
    if st.button("🔮 DESCUBRIR MI TIPO",use_container_width=True,type="primary"):
        st.session_state.started=True; st.rerun()

def quiz():
    i=st.session_state.get("q",0); answers=st.session_state.setdefault("answers",{})
    st.markdown(f"### Pregunta {i+1} de 10")
    st.progress((i+1)/10)
    q,opts=QUESTIONS[i]
    st.markdown(f'<div class="card"><h2>{q}</h2></div>',unsafe_allow_html=True)
    choice=st.radio("Selecciona una opción",opts,index=None,key=f"q_{i}",label_visibility="collapsed")
    if st.button("Siguiente →",use_container_width=True,type="primary",disabled=choice is None):
        answers[i]=opts.index(choice)
        if i==9:
            with st.spinner("🔮 ANALIZANDO TUS RESPUESTAS…"): 
                st.session_state.result_ready=True
        else: st.session_state.q=i+1
        st.rerun()

def result(df,metrics,best,model,cm):
    values=[st.session_state.answers[i] for i in range(10)]
    x=pd.DataFrame([values],columns=[f"q{i}" for i in range(1,11)])
    profile=model.predict(x)[0]; probs=model.predict_proba(x)[0]; prob=dict(zip(model.classes_,probs))
    pct=round(max(prob.values())*100)
    emoji,tag,desc,traits=INFO[profile]
    st.markdown(f'<div class="card"><div class="big">{emoji}</div><div class="tag">{tag}</div><div class="big">{profile.upper()}</div><div class="score">{pct}% de probabilidad</div><p>{desc}</p>'+''.join(f'<span class="trait">{t}</span>' for t in traits)+'</div>',unsafe_allow_html=True)
    st.markdown("### Distribución de probabilidades")
    p=pd.DataFrame({"Perfil":list(prob),"Probabilidad":[v*100 for v in prob.values()]}).sort_values("Probabilidad",ascending=False)
    st.bar_chart(p.set_index("Perfil"))
    st.markdown("### ¿LA IA ACERTÓ?")
    a,b=st.columns(2)
    if a.button("🟢 Sí, totalmente",use_container_width=True): st.session_state.feedback="Sí, totalmente"
    if b.button("🔴 Nada que ver",use_container_width=True): st.session_state.feedback="Nada que ver"
    if st.session_state.get("feedback"): st.success("¡Gracias! Tu respuesta queda registrada de forma anónima en esta sesión.")
    st.download_button("📸 Compartir mi resultado",card_image(profile,pct),file_name="mi-geek.png",mime="image/png",use_container_width=True)
    st.markdown("### ¿QUÉ HAY DETRÁS?")
    st.markdown('<div class="flow"><div class="node">Tus respuestas</div>→<div class="node">Datos</div>→<div class="node">Machine Learning</div>→<div class="node">Predicción</div>→<div class="node">Tu perfil</div></div>',unsafe_allow_html=True)
    st.info("Tus respuestas se convierten en datos. El modelo analiza patrones aprendidos durante su entrenamiento y utiliza esos patrones para predecir el perfil que más se parece a tus respuestas.")
    st.warning("Nota del prototipo: el modelo fue entrenado con datos sintéticos. Los resultados no representan una población real de visitantes de SOFA.")
    if st.button("🔁 Hacer el cuestionario de nuevo",use_container_width=True):
        for k in ["started","result_ready","q","answers","feedback"]: st.session_state.pop(k,None)
        st.rerun()

def technical(df,metrics,best,cm):
    st.markdown("## 🧠 ¿Cómo se creó?")
    st.write("3.000 registros sintéticos → 80/20 train-test → 4 modelos → métricas → mejor F1-score → predicción.")
    display=metrics.copy()
    for c in ["Accuracy","Precision","Recall","F1"]: display[c]=display[c].map(lambda x:f"{x:.3f}")
    st.dataframe(display,use_container_width=True,hide_index=True)
    st.success(f"🏆 MODELO SELECCIONADO: {best}")
    st.markdown("### Matriz de confusión")
    st.dataframe(pd.DataFrame(cm,index=sorted(PROFILES),columns=sorted(PROFILES)),use_container_width=True)
    st.caption("La matriz muestra dónde el modelo acierta y en qué perfiles tiende a confundirse.")
    st.markdown("### Transparencia y evolución")
    st.warning("Los datos son sintéticos y no representan científicamente a los geeks de Colombia ni a visitantes de SOFA.")
    st.markdown("**FASE 1:** Datos sintéticos → modelo inicial  
**FASE 2:** SOFA → respuestas reales y voluntarias  
**FASE 3:** Evaluación → mejora del modelo")

def presentation():
    st.markdown("## 🎤 Presentación del proyecto")
    st.markdown("""**Problema.** Queremos acercar Ciencia de Datos a una audiencia relacionada con tecnología, gaming, anime y cultura geek.

**Idea.** Creamos “¿Qué tipo de geek eres?”. El visitante responde diez preguntas sobre preferencias y un modelo predice uno de seis perfiles.

**Datos.** Como todavía no tenemos datos reales, usamos 3.000 registros sintéticos con patrones, variabilidad y ruido.

**Modelos.** Entrenamos Logistic Regression, Decision Tree, Random Forest y KNN. Los comparamos con accuracy, precision, recall y F1-score, además de una matriz de confusión.

**Resultado.** Seleccionamos automáticamente el modelo con mejor F1-score y lo utilizamos para nuevas predicciones.

**Futuro.** En SOFA podríamos recopilar respuestas reales y voluntarias para evaluar y mejorar el modelo.

> La idea es demostrar que la Ciencia de Datos no solamente está detrás de números y empresas: también puede utilizarse para entender patrones en las cosas que nos apasionan.""")

df,metrics,best,model,cm=data_and_models()
if "result_ready" in st.session_state:
    result(df,metrics,best,model,cm)
elif st.session_state.get("started"):
    quiz()
else:
    home()
st.divider()
t1,t2=st.tabs(["🧠 ¿Cómo se creó?","🎤 Presentación del proyecto"])
with t1: technical(df,metrics,best,cm)
with t2: presentation()
st.caption("SOFA · Prototipo académico de Ciencia de Datos · Sin datos personales")
