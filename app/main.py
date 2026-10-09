from pathlib import Path
import io,sys,re
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image,ImageDraw,ImageFont
import requests

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
CHARACTER_SOURCES={
"Gamer":"https://fd.sao-game.jp/character/detail.php?chara=kirito",
"Estratega":"https://spy-family.net/tvseries/",
"Tech Geek":"https://www.ytv.co.jp/heroaca/character/hatsume/",
"Lore Master":"https://bleach-anime.com/en/character/?chara=69",
"Creador":"https://fullmetalalchemistusa.com/character/",
"Explorador":"https://delicious-in-dungeon.com/"
}
DIRECT_CHARACTER_IMAGES={
"Explorador":"https://delicious-in-dungeon.com/assets/character/1c.png"
}

@st.cache_data(ttl=86400,show_spinner=False)
def get_character_image(profile):
    try:
        url=DIRECT_CHARACTER_IMAGES.get(profile)
        if not url:
            page=requests.get(CHARACTER_SOURCES[profile],timeout=8,headers={"User-Agent":"Mozilla/5.0"}).text
            name=INFO[profile][2].split(" · ",1)[0]
            patterns=[
                rf'<img[^>]+src=["\']([^"\']+)["\'][^>]*alt=["\'][^"\']*{re.escape(name)}[^"\']*["\']',
                rf'<img[^>]+alt=["\'][^"\']*{re.escape(name)}[^"\']*["\'][^>]*src=["\']([^"\']+)["\']'
            ]
            match=None
            for pattern in patterns:
                match=re.search(pattern,page,re.I)
                if match: break
            if not match:
                # Find an image URL close to the character name in the page source.
                idx=page.lower().find(name.lower())
                if idx>=0:
                    window=page[max(0,idx-4000):idx+4000]
                    match=re.search(r'<img[^>]+src=["\']([^"\']+)["\']',window,re.I)
            if not match: return None
            url=requests.compat.urljoin(CHARACTER_SOURCES[profile],match.group(1))
        r=requests.get(url,timeout=10,headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        return r.content
    except Exception:
        return None

def square_crop(image,size=520):
    image=image.convert("RGB")
    side=min(image.size)
    left=(image.width-side)//2; top=(image.height-side)//2
    image=image.crop((left,top,left+side,top+side))
    return image.resize((size,size),Image.Resampling.LANCZOS)

INFO={
"Gamer":("🎮","PLAYER 01","Kirito · Sword Art Online","Te mueven los retos, la competencia y la emoción de superar una partida difícil.",["Competitivo","Persistente","Orientado al reto"]),
"Lore Master":("🧙","LORE ARCHIVIST","Sōsuke Aizen · Bleach","No te basta con conocer una historia: quieres entender su universo, sus personajes y cada detalle escondido.",["Curioso","Narrativo","Detallista"]),
"Creador":("🎨","CREATIVE MODE","Father · Fullmetal Alchemist","Transformas ideas en posibilidades y te atraen la creación, la experimentación y construir algo desde cero.",["Creativo","Experimental","Visionario"]),
"Estratega":("🧩","TACTICAL MIND","Loid Forger · Spy × Family","Antes de actuar analizas. Te gustan los planes, las decisiones difíciles y encontrar la jugada que nadie vio.",["Analítico","Planificador","Resolutivo"]),
"Tech Geek":("🤖","SYSTEM EXPLORER","Mei Hatsume · My Hero Academia","Si algo funciona, quieres saber por qué. Tecnología, inventos, IA y gadgets despiertan tu curiosidad.",["Inventivo","Lógico","Explorador tech"]),
"Explorador":("🌌","DISCOVERY MODE","Laios Touden · Delicious in Dungeon","Tu superpoder es la curiosidad. Te emociona probar cosas nuevas, descubrir mundos y aprender mientras exploras.",["Curioso","Aventurero","Flexible"])
}
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Martian+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--cream:#FFFDF8;--navy:#18263A;--blue:#5397C8;--sky:#80B5D7;--orange:#F19F39;--gold:#F4BD62;--muted:#53657A}
.stApp{background:linear-gradient(135deg,var(--cream) 0%,#F7FAFC 55%,#FFF6E7 100%);color:var(--navy);font-family:'Space Grotesk',sans-serif}
.block-container{max-width:1120px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3,.hero h1,.big,.tag{font-family:'Martian Mono',monospace}
h1,h2,h3{color:var(--navy)!important}
p,li,span{color:var(--navy)}
.hero{text-align:center;min-height:62vh;display:flex;flex-direction:column;justify-content:center;align-items:center}
.hero h1{font-size:clamp(2.8rem,8vw,6.5rem);line-height:.98;letter-spacing:-.06em;color:var(--navy)!important}
.hero p{color:var(--muted)!important;font-size:1.1rem;line-height:1.55}
.tag{color:var(--blue)!important;letter-spacing:.12em;font-size:.75rem;font-weight:700}
.card{border:2px solid #DCE8EF;background:rgba(255,255,255,.92);border-radius:26px;padding:1.6rem;box-shadow:0 10px 30px rgba(24,38,58,.08)}
.big{font-size:clamp(2.4rem,6vw,4.5rem);color:var(--navy)!important}
.score{font-family:'Martian Mono';font-size:1.8rem;font-weight:700;color:#D87810!important}
.trait{display:inline-block;background:#EAF4FA!important;color:var(--navy)!important;border:1px solid #BFDCEC;border-radius:999px;padding:.45rem .75rem;margin:.2rem;font-weight:600}
.flow{display:flex;gap:.6rem;justify-content:center;align-items:center;flex-wrap:wrap}
.node{background:var(--navy)!important;color:white!important;padding:.8rem 1rem;border-radius:14px;font-family:'Martian Mono';font-size:.82rem}
div[data-testid="stRadio"] label{background:white!important;border:2px solid #BFDCEC!important;border-radius:16px!important;padding:13px 16px!important;margin:8px 0!important;transition:.15s!important}
div[data-testid="stRadio"] label:hover{background:#FFF4DE!important;border-color:var(--orange)!important}
div[data-testid="stRadio"] label p{color:var(--navy)!important;font-family:'Space Grotesk',sans-serif!important;font-weight:600!important;font-size:1rem!important}
div[data-testid="stRadio"] label:has(input:checked){background:#EAF4FA!important;border-color:var(--blue)!important;box-shadow:0 0 0 2px rgba(83,151,200,.12)!important}
div.stButton>button,div.stDownloadButton>button{background:var(--orange)!important;color:var(--navy)!important;border:2px solid #D87810!important;border-radius:14px!important;font-family:'Martian Mono',monospace!important;font-weight:700!important;min-height:48px!important}
div.stButton>button:hover,div.stDownloadButton>button:hover{background:var(--gold)!important;color:var(--navy)!important}
div.stButton>button p,div.stDownloadButton>button p{color:var(--navy)!important;font-weight:700!important}
div[data-testid="stAlert"] p{color:var(--navy)!important}
[data-testid="stDataFrame"]{border:2px solid #DCE8EF;border-radius:16px}
.stProgress>div>div>div>div{background:var(--blue)!important}
hr{border-color:#DCE8EF!important}
@media(max-width:640px){.block-container{padding-left:1rem;padding-right:1rem}.hero{min-height:55vh}.hero h1{font-size:clamp(2.3rem,12vw,4rem)}.card{padding:1.1rem}}
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
    emoji,tag,character,desc,traits=INFO[profile]
    img=Image.new("RGB",(1600,1000),(255,253,248)); d=ImageDraw.Draw(img)
    try:
        title=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",42)
        big=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",68)
        med=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",30)
        small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",22)
    except: title=big=med=small=ImageFont.load_default()
    d.rectangle((0,0,1600,18),fill=(83,151,200))
    d.rectangle((18,18,1582,982),outline=(241,159,57),width=4)
    # Image panel
    raw=get_character_image(profile)
    if raw:
        try:
            import PIL.Image
            char_img=square_crop(Image.open(io.BytesIO(raw)),520)
            img.paste(char_img,(80,220))
            d.rectangle((80,220,600,740),outline=(128,181,215),width=6)
        except Exception:
            raw=None
    if not raw:
        d.rounded_rectangle((80,220,600,740),radius=30,fill=(234,244,250),outline=(128,181,215),width=5)
        d.text((285,420),emoji,fill=(24,38,58),font=big)
    d.text((80,70),"¿QUÉ TIPO DE GEEK ERES?",fill=(24,38,58),font=title)
    d.text((680,190),profile.upper(),fill=(83,151,200),font=big)
    d.text((680,285),f"{pct}% de probabilidad",fill=(216,120,16),font=title)
    parts=character.split(" · ",1); name=parts[0]; universe=parts[1] if len(parts)>1 else ""
    d.text((680,385),name,fill=(24,38,58),font=big)
    d.text((680,465),universe,fill=(83,101,122),font=med)
    d.text((680,540),tag,fill=(83,151,200),font=title)
    words=desc.split(); lines=[]; line=""
    for w in words:
        if len(line)+len(w)+1>48: lines.append(line); line=w
        else: line=(line+" "+w).strip()
    if line: lines.append(line)
    y=610
    for ln in lines[:3]: d.text((680,y),ln,fill=(24,38,58),font=med); y+=42
    d.text((680,770),"  •  ".join(traits),fill=(83,101,122),font=small)
    d.text((80,900),"SOFA · Prototipo académico de Ciencia de Datos",fill=(83,151,200),font=small)
    out=io.BytesIO(); img.save(out,"PNG",optimize=True); out.seek(0); return out


def character_card(profile):
    emoji,tag,character,desc,traits=INFO[profile]
    parts=character.split(" · ",1); name=parts[0]; universe=parts[1] if len(parts)>1 else ""
    traits_html="".join(f'<span class="trait">{t}</span>' for t in traits)
    raw=get_character_image(profile)
    image_html=""
    if raw:
        import base64
        image_html=f'<img src="data:image/jpeg;base64,{base64.b64encode(raw).decode()}" style="width:240px;height:240px;object-fit:cover;border-radius:22px;border:4px solid #80B5D7;box-shadow:0 8px 24px rgba(24,38,58,.14)">'
    else:
        image_html=f'<div style="width:240px;height:240px;border-radius:22px;background:#EAF4FA;border:4px solid #80B5D7;display:flex;align-items:center;justify-content:center;font-size:6rem">{emoji}</div>'
    return f'''<div class="card" style="margin-top:1.2rem;background:linear-gradient(135deg,#FFFDF8 0%,#EAF4FA 100%);border-color:#80B5D7">
<div class="tag">FICHA DEL PERSONAJE</div>
<div style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;margin-top:.8rem">
{image_html}
<div style="flex:1;min-width:260px"><div class="big" style="font-size:2rem">{name}</div><div style="color:#53657A;font-weight:600">{universe}</div><div class="tag" style="margin-top:.4rem">{tag}</div>
<p style="font-size:1.05rem;line-height:1.55;margin-top:1rem">{desc}</p><div>{traits_html}</div></div>
</div></div>'''


def home():
    st.markdown('<div class="hero"><div class="tag">SOFA · DATA SCIENCE EXPERIENCE</div><h1>¿QUÉ TIPO<br>DE GEEK ERES?</h1><p>La IA quiere descubrirlo.<br>Responde 10 preguntas y descubre qué perfil geek predice nuestro modelo.</p></div>',unsafe_allow_html=True)
    if st.button("🔮 DESCUBRIR MI TIPO",use_container_width=True,type="primary"):
        st.session_state.started=True
        st.rerun()


def quiz():
    i=st.session_state.get("q",0)
    answers=st.session_state.setdefault("answers",{})
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
        else:
            st.session_state.q=i+1
        st.rerun()


def result(df,metrics,best,model,cm):
    values=[st.session_state.answers[i] for i in range(10)]
    x=pd.DataFrame([values],columns=[f"q{i}" for i in range(1,11)])
    profile=model.predict(x)[0]; probs=model.predict_proba(x)[0]; prob=dict(zip(model.classes_,probs))
    pct=round(max(prob.values())*100)
    emoji,tag,character,desc,traits=INFO[profile]
    st.markdown(f'<div class="card"><div class="big">{emoji}</div><div class="tag">{tag}</div><div class="big">{profile.upper()}</div><div class="score">{pct}% de probabilidad</div><p>{desc}</p>'+''.join(f'<span class="trait">{t}</span>' for t in traits)+'</div>',unsafe_allow_html=True)
    st.markdown(character_card(profile),unsafe_allow_html=True)
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
    st.markdown("""**FASE 1:** Datos sintéticos → modelo inicial

**FASE 2:** SOFA → respuestas reales y voluntarias

**FASE 3:** Evaluación → mejora del modelo""")

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
