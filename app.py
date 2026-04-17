import streamlit as st
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Projet de groupe",
    page_icon="🎟️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
.main > div { padding-top: 1rem; }

.narrative-box {
    background: rgba(74,144,226,0.08);
    border-left: 4px solid #4A90E2;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 16px;
    line-height: 1.8;
    margin: 1rem 0;
    font-style: italic;
}

.result-card {
    border-radius: 16px;
    padding: 20px 16px;
    text-align: center;
    color: white;
    margin-bottom: 8px;
}

.card-accept {
    background: linear-gradient(135deg, #2ecc71, #27ae60);
}

.card-refuse {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
}
</style>
""", unsafe_allow_html=True)

# =========================
# CHARGEMENT MODELES
# =========================
@st.cache_resource
def load_models():
    with open("C:/Users/artur/Desktop/Forage de données/Streamlite/modeles.pkl", "rb") as f:
        return pickle.load(f)

data = load_models()

dt_model = data["dt"]
rf_model = data["rf"]
mlp_model = data["mlp"]
COLUMNS = data["columns"]
SELECTED_FEATURES = data.get("selected_features", COLUMNS)

DEFAULT_METRICS = {
    "Arbre de decision": {"Accuracy": 0.6893, "F1": 0.7311},
    "Random Forest": {"Accuracy": 0.7317, "F1": 0.7706},
    "MLP": {"Accuracy": 0.7092, "F1": 0.7503},
}
METRICS = data.get("metrics", DEFAULT_METRICS)

# =========================
# CONSTANTES
# =========================
TEMP_F_TO_C = {30: "-1 C", 55: "13 C", 80: "27 C"}

ORDRE_EDU = {
    "Some High School": 0,
    "High School Graduate": 1,
    "Some college - no degree": 2,
    "Associates degree": 3,
    "Bachelors degree": 4,
    "Graduate degree (Masters or Doctorate)": 5,
}

EDU_FR = {
    "Some High School": "Secondaire partiel",
    "High School Graduate": "Diplome du secondaire",
    "Some college - no degree": "Collegial sans diplome",
    "Associates degree": "DEC / Diplome associe",
    "Bachelors degree": "Baccalaureat",
    "Graduate degree (Masters or Doctorate)": "Maitrise ou Doctorat",
}

AGE_LBL = {
    0: "<21", 1: "21-25", 2: "26-30", 3: "31-35",
    4: "36-40", 5: "41-45", 6: "46-50", 7: ">50"
}

INCOME_LBL = {
    0: "<12k$", 1: "12-25k$", 2: "25-37k$", 3: "37-50k$",
    4: "50-62k$", 5: "62-75k$", 6: "75-87k$", 7: "87-100k$", 8: ">100k$"
}

FREQ_LBL = {
    0: "Jamais",
    1: "Rarement",
    2: "Parfois",
    3: "Souvent",
    4: "Tres souvent"
}

# =========================
# SESSION STATE
# =========================
defaults = {
    "show_results": False,

    # expérience utilisateur
    "weather": "Sunny",
    "destination": "No Urgent Place",

    # variables utiles au modèle
    "time": "2PM",
    "temperature": 80,
    "passanger": "Alone",
    "coupon": "Coffee House",
    "expiration": "1d",
    "toCoupon_GEQ15min": 0,
    "gender": "Male",
    "age": 2,
    "education": "Bachelors degree",
    "income": 3,
    "Bar": 1,
    "CoffeeHouse": 2,
    "CarryAway": 2,
    "RestaurantLessThan20": 3,
    "Restaurant20To50": 1,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# =========================
# FONCTIONS
# =========================
def randomize():
    import random

    # expérience utilisateur
    st.session_state.weather = random.choice(["Sunny", "Rainy", "Snowy"])
    st.session_state.destination = random.choice(["Home", "No Urgent Place", "Work"])

    # variables utiles au modèle
    st.session_state.time = random.choice(["7AM", "10AM", "2PM", "6PM", "10PM"])
    st.session_state.temperature = random.choice([30, 55, 80])
    st.session_state.passanger = random.choice(["Alone", "Friend(s)", "Kid(s)", "Partner"])
    st.session_state.coupon = random.choice(["Bar", "Carry out & Take away", "Coffee House", "Restaurant(20-50)", "Restaurant(<20)"])
    st.session_state.expiration = random.choice(["2h", "1d"])
    st.session_state.toCoupon_GEQ15min = random.choice([0, 1])
    st.session_state.gender = random.choice(["Male", "Female"])
    st.session_state.age = random.randint(0, 7)
    st.session_state.education = random.choice(list(ORDRE_EDU.keys()))
    st.session_state.income = random.randint(0, 8)
    st.session_state.Bar = random.randint(0, 4)
    st.session_state.CoffeeHouse = random.randint(0, 4)
    st.session_state.CarryAway = random.randint(0, 4)
    st.session_state.RestaurantLessThan20 = random.randint(0, 4)
    st.session_state.Restaurant20To50 = random.randint(0, 4)

    st.session_state.show_results = True

def make_svg():
    w, h = 680, 370
    time = st.session_state.time
    weather = st.session_state.weather
    is_night = time == "10PM"
    is_evening = time == "6PM"
    is_morning = time == "7AM"

    if is_night:
        sky1, sky2 = "#0a0a2e", "#1a1a4e"
        gnd, road, lc = "#111", "#222", "#ffff00"
    elif is_evening:
        sky1, sky2 = "#ff6b35", "#c94b4b"
        gnd, road, lc = "#2d5a27", "#3a3a3a", "#fff"
    elif is_morning:
        sky1, sky2 = "#ffeaa7", "#fdcb6e"
        gnd, road, lc = "#27ae60", "#555", "#fff"
    else:
        gnd, road, lc = "#27ae60", "#555", "#fff"
        if weather == "Sunny":
            sky1, sky2 = "#87CEEB", "#5ba3d9"
        elif weather == "Rainy":
            sky1, sky2 = "#6b7a8d", "#4a5568"
        else:
            sky1, sky2 = "#c8d6e5", "#a4b0be"

    hy = 195

    s = f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
    s += f'<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{sky1}"/><stop offset="100%" stop-color="{sky2}"/></linearGradient></defs>'
    s += f'<rect width="{w}" height="{h}" fill="url(#sg)"/>'

    if weather == "Sunny" and not is_night and not is_evening:
        s += '<circle cx="590" cy="65" r="42" fill="#FFD700" opacity="0.9"/>'
        s += '<circle cx="590" cy="65" r="54" fill="#FFD700" opacity="0.15"/>'
        s += '<ellipse cx="140" cy="75" rx="65" ry="26" fill="white" opacity="0.75"/>'
        s += '<ellipse cx="190" cy="60" rx="50" ry="22" fill="white" opacity="0.8"/>'
    elif weather == "Rainy":
        s += '<ellipse cx="200" cy="55" rx="110" ry="40" fill="#4a5568" opacity="0.9"/>'
        s += '<ellipse cx="420" cy="48" rx="130" ry="44" fill="#4a5568" opacity="0.85"/>'
        s += '<ellipse cx="580" cy="60" rx="95" ry="36" fill="#4a5568" opacity="0.9"/>'
        for rx2, ry2 in [(110,105),(155,100),(200,108),(330,103),(370,110),(440,100),(520,107),(560,102)]:
            s += f'<line x1="{rx2}" y1="{ry2}" x2="{rx2-8}" y2="{ry2+30}" stroke="#74b9ff" stroke-width="1.5" opacity="0.65"/>'
    elif weather == "Snowy":
        s += '<ellipse cx="180" cy="50" rx="115" ry="40" fill="#dfe6e9" opacity="0.88"/>'
        s += '<ellipse cx="420" cy="44" rx="130" ry="44" fill="#dfe6e9" opacity="0.85"/>'
        s += '<ellipse cx="590" cy="56" rx="95" ry="36" fill="#dfe6e9" opacity="0.88"/>'
        for fx, fy, fs in [(90,118,18),(195,132,14),(315,112,20),(450,125,16),(555,118,18),(150,158,12),(375,142,14),(260,102,16),(490,148,12)]:
            s += f'<text x="{fx}" y="{fy}" font-size="{fs}" fill="white" opacity="0.95" font-family="sans-serif">\u2744</text>'

    if is_night:
        for sx, sy, sr in [(50,28,1.5),(130,48,1),(220,22,1.5),(310,38,1),(420,18,2),(500,42,1),(575,28,1.5),(645,52,1)]:
            s += f'<circle cx="{sx}" cy="{sy}" r="{sr}" fill="white" opacity="0.9"/>'
        s += '<circle cx="590" cy="62" r="28" fill="#f0e68c" opacity="0.9"/>'
        s += f'<circle cx="606" cy="55" r="22" fill="{sky1}"/>'

    s += f'<rect x="0" y="{hy}" width="{w}" height="{h-hy}" fill="{gnd}"/>'
    s += f'<polygon points="255,{hy} 425,{hy} {w},{h} 0,{h}" fill="{road}"/>'

    for dy, dw in [(5,12),(30,20),(75,35),(130,55)]:
        cx_road = (255 + 425) // 2
        s += f'<line x1="{cx_road-dw//2}" y1="{hy+dy}" x2="{cx_road+dw//2}" y2="{hy+dy}" stroke="{lc}" stroke-width="3" opacity="0.7"/>'

    dest = st.session_state.destination
    if dest == "Home":
        s += f'<rect x="320" y="{hy-52}" width="62" height="52" rx="4" fill="#e17055" opacity="0.9"/>'
        s += f'<polygon points="308,{hy-52} 351,{hy-90} 394,{hy-52}" fill="#d63031" opacity="0.9"/>'
        s += f'<rect x="338" y="{hy-28}" width="18" height="28" rx="2" fill="#2d3436" opacity="0.8"/>'
        s += f'<rect x="322" y="{hy-48}" width="13" height="11" rx="2" fill="#74b9ff" opacity="0.7"/>'
        s += f'<rect x="367" y="{hy-48}" width="13" height="11" rx="2" fill="#74b9ff" opacity="0.7"/>'
    elif dest == "Work":
        s += f'<rect x="290" y="{hy-82}" width="33" height="82" rx="2" fill="#636e72" opacity="0.9"/>'
        s += f'<rect x="327" y="{hy-62}" width="28" height="62" rx="2" fill="#74b9ff" opacity="0.8"/>'
        s += f'<rect x="359" y="{hy-98}" width="38" height="98" rx="2" fill="#4a5568" opacity="0.9"/>'
    else:
        s += f'<circle cx="340" cy="{hy-12}" r="8" fill="#fdcb6e" opacity="0.9"/>'
        s += f'<circle cx="340" cy="{hy-12}" r="16" fill="#fdcb6e" opacity="0.25"/>'

    coupon = st.session_state.coupon
    icons = {
        "Bar": "🍺",
        "Carry out & Take away": "🥡",
        "Coffee House": "☕",
        "Restaurant(20-50)": "🍽️",
        "Restaurant(<20)": "🍔"
    }
    colors = {
        "Bar": "#e17055",
        "Carry out & Take away": "#00b894",
        "Coffee House": "#6c5ce7",
        "Restaurant(20-50)": "#d63031",
        "Restaurant(<20)": "#e67e22"
    }

    icon = icons.get(coupon, "🎟")
    pc = colors.get(coupon, "#4A90E2")
    exp = "2h" if st.session_state.expiration == "2h" else "1 jour"

    # Carte coupon propre
    s += f'<rect x="510" y="110" width="150" height="130" rx="14" fill="{pc}" opacity="0.95"/>'

    # Bande du haut
    s += f'<rect x="510" y="110" width="150" height="38" rx="14" fill="rgba(0,0,0,0.2)"/>'
    s += f'<text x="585" y="135" text-anchor="middle" font-size="13" fill="white" font-weight="bold" font-family="sans-serif">COUPON</text>'

    # Séparation
    s += f'<rect x="510" y="148" width="150" height="10" fill="rgba(0,0,0,0.2)"/>'

    # Icône
    s += f'<text x="585" y="185" text-anchor="middle" font-size="32" font-family="sans-serif">{icon}</text>'

    # Expiration
    s += f'<text x="585" y="210" text-anchor="middle" font-size="12" fill="white" opacity="0.95" font-family="sans-serif">Valable {exp}</text>'

    # Promo
    s += f'<text x="585" y="232" text-anchor="middle" font-size="11" fill="white" opacity="0.8" font-family="sans-serif">-20% immediat</text>'
    

    cx, cy = 240, 265
    s += f'<rect x="{cx}" y="{cy}" width="204" height="78" rx="8" fill="#2d3436"/>'
    s += f'<rect x="{cx+14}" y="{cy-42}" width="176" height="52" rx="10" fill="#636e72"/>'
    s += f'<rect x="{cx+24}" y="{cy-37}" width="156" height="40" rx="6" fill="#74b9ff" opacity="0.55"/>'
    s += f'<rect x="{cx+4}" y="{cy+8}" width="24" height="15" rx="3" fill="#e74c3c" opacity="0.9"/>'
    s += f'<rect x="{cx+176}" y="{cy+8}" width="24" height="15" rx="3" fill="#e74c3c" opacity="0.9"/>'
    s += f'<rect x="{cx+77}" y="{cy+53}" width="50" height="18" rx="3" fill="white"/>'
    s += f'<text x="{cx+102}" y="{cy+66}" text-anchor="middle" font-size="9" fill="#2d3436" font-weight="bold" font-family="sans-serif">QC 2026</text>'

    for wx in [cx+38, cx+162]:
        s += f'<circle cx="{wx}" cy="{cy+80}" r="22" fill="#2d3436"/>'
        s += f'<circle cx="{wx}" cy="{cy+80}" r="13" fill="#636e72"/>'

    p = st.session_state.passanger
    if p == "Alone":
        s += f'<circle cx="{cx+102}" cy="{cy-22}" r="14" fill="#fdcb6e" opacity="0.9"/>'
        s += f'<rect x="{cx+88}" y="{cy-9}" width="28" height="17" rx="4" fill="#0984e3" opacity="0.8"/>'

    elif p == "Friend(s)":
        for px2, py2, pc2, cc in [
            (cx+72, cy-22, "#fdcb6e", "#0984e3"),
            (cx+118, cy-22, "#fd79a8", "#6c5ce7"),
            (cx+162, cy-20, "#55efc4", "#00b894")
        ]:
            s += f'<circle cx="{px2}" cy="{py2}" r="12" fill="{pc2}" opacity="0.9"/>'
            s += f'<rect x="{px2-12}" y="{py2+11}" width="24" height="14" rx="4" fill="{cc}" opacity="0.8"/>'

    elif p == "Kid(s)":
        s += f'<circle cx="{cx+85}" cy="{cy-22}" r="13" fill="#fdcb6e" opacity="0.9"/>'
        s += f'<rect x="{cx+72}" y="{cy-10}" width="26" height="15" rx="4" fill="#0984e3" opacity="0.8"/>'
        s += f'<circle cx="{cx+140}" cy="{cy-16}" r="9" fill="#fd79a8" opacity="0.9"/>'
        s += f'<rect x="{cx+131}" y="{cy-8}" width="18" height="13" rx="3" fill="#e17055" opacity="0.8"/>'

    elif p == "Partner":
        for px2, pc2, cc in [
            (cx+82, "#fdcb6e", "#0984e3"),
            (cx+132, "#fd79a8", "#e84393")
        ]:
            s += f'<circle cx="{px2}" cy="{cy-22}" r="13" fill="{pc2}" opacity="0.9"/>'
            s += f'<rect x="{px2-13}" y="{cy-10}" width="26" height="15" rx="4" fill="{cc}" opacity="0.8"/>'

    temp_c = TEMP_F_TO_C.get(st.session_state.temperature, "?")
    time_disp = {"7AM":"07:00","10AM":"10:00","2PM":"14:00","6PM":"18:00","10PM":"22:00"}
    s += f'<rect x="12" y="12" width="108" height="34" rx="17" fill="rgba(0,0,0,0.38)"/>'
    s += f'<text x="66" y="34" text-anchor="middle" font-size="14" fill="white" font-weight="bold" font-family="sans-serif">{time_disp.get(time,time)}</text>'
    s += f'<rect x="128" y="12" width="82" height="34" rx="17" fill="rgba(0,0,0,0.38)"/>'
    s += f'<text x="169" y="34" text-anchor="middle" font-size="14" fill="white" font-family="sans-serif">{temp_c}</text>'
    s += '</svg>'
    return s

def narrative():
    tf = {"7AM":"7h du matin","10AM":"10h du matin","2PM":"14h","6PM":"18h","10PM":"22h"}
    wf = {"Sunny":"ensoleille","Rainy":"pluvieux","Snowy":"neigeux"}
    pf = {"Alone":"tu conduis seul-e","Friend(s)":"tu conduis avec des amis","Kid(s)":"tu as tes enfants avec toi","Partner":"tu conduis avec ton partenaire"}
    df = {"Home":"vers chez toi","No Urgent Place":"sans destination precise","Work":"vers le travail"}
    cf = {"Bar":"un bar","Carry out & Take away":"un restaurant a emporter","Coffee House":"un coffee house","Restaurant(20-50)":"un restaurant","Restaurant(<20)":"un fast-food"}
    ef = {"2h":"2 heures","1d":"1 jour"}

    s = st.session_state
    temp_c = TEMP_F_TO_C.get(s.temperature, "?")

    return (
        f"Il est **{tf.get(s.time, s.time)}**, temps **{wf.get(s.weather, s.weather)}** "
        f"a **{temp_c}**. **{pf.get(s.passanger, s.passanger)}** "
        f"**{df.get(s.destination, s.destination)}**. "
        f"Ton GPS te propose un coupon pour **{cf.get(s.coupon, s.coupon)}** "
        f"qui expire dans **{ef.get(s.expiration, s.expiration)}**. "
        f"*Est-ce que tu l'acceptes ?*"
    )

def encode():
    s = st.session_state
    f = {col: 0 for col in COLUMNS}

    # numériques / ordinales
    if "CoffeeHouse" in f:
        f["CoffeeHouse"] = s.CoffeeHouse
    if "income" in f:
        f["income"] = s.income
    if "age" in f:
        f["age"] = s.age
    if "Bar" in f:
        f["Bar"] = s.Bar
    if "education" in f:
        f["education"] = ORDRE_EDU.get(s.education, 4)
    if "CarryAway" in f:
        f["CarryAway"] = s.CarryAway
    if "RestaurantLessThan20" in f:
        f["RestaurantLessThan20"] = s.RestaurantLessThan20
    if "Restaurant20To50" in f:
        f["Restaurant20To50"] = s.Restaurant20To50
    if "temperature" in f:
        f["temperature"] = s.temperature
    if "toCoupon_GEQ15min" in f:
        f["toCoupon_GEQ15min"] = s.toCoupon_GEQ15min

    # binaires one-hot
    if s.expiration == "2h" and "expiration_2h" in f:
        f["expiration_2h"] = 1
    if s.coupon == "Carry out & Take away" and "coupon_Carry out & Take away" in f:
        f["coupon_Carry out & Take away"] = 1
    if s.coupon == "Restaurant(<20)" and "coupon_Restaurant(<20)" in f:
        f["coupon_Restaurant(<20)"] = 1
    if s.coupon == "Coffee House" and "coupon_Coffee House" in f:
        f["coupon_Coffee House"] = 1
    if s.gender == "Male" and "gender_Male" in f:
        f["gender_Male"] = 1
    if s.time == "6PM" and "time_6PM" in f:
        f["time_6PM"] = 1
    if s.passanger == "Friend(s)" and "passanger_Friend(s)" in f:
        f["passanger_Friend(s)"] = 1

    return pd.DataFrame([f])[COLUMNS]

# =========================
# EN-TETE
# =========================
col_logo, col_title = st.columns([1, 1.5])

with col_logo:
    st.image("logo.png", width=250)

with col_title:
    st.title("Forage de données")

# =========================
# LAYOUT PRINCIPAL
# =========================
col_title2, col_rand = st.columns([4, 1])

with col_title2:
    st.markdown(
        "<h2 style='color:#e74c3c;'>Est-ce qu'il va accepter ?</h2>",
        unsafe_allow_html=True
    )

with col_rand:
    st.write("")
    if st.button("🎲 Aléatoire !", use_container_width=True):
        randomize()
        st.rerun()

col_scene, col_form = st.columns([3, 2])

with col_scene:
    st.markdown(make_svg(), unsafe_allow_html=True)
    st.markdown(f'<div class="narrative-box">{narrative()}</div>', unsafe_allow_html=True)
    
with col_form:
    tab1, tab2, tab3, tab4 = st.tabs(["Situation", "Coupon", "Profil", "Habitudes"])

    with tab1:
        st.selectbox(
            "Heure",
            ["7AM", "10AM", "2PM", "6PM", "10PM"],
            key="time",
            format_func=lambda x: {"7AM":"07:00 matin","10AM":"10:00 matin","2PM":"14:00","6PM":"18:00","10PM":"22:00 nuit"}[x]
        )
        st.selectbox(
            "Météo",
            ["Sunny", "Rainy", "Snowy"],
            key="weather",
            format_func=lambda x: {"Sunny":"Ensoleille","Rainy":"Pluvieux","Snowy":"Neigeux"}[x]
        )
        st.selectbox(
            "Température",
            [30, 55, 80],
            key="temperature",
            format_func=lambda x: TEMP_F_TO_C[x]
        )
        st.selectbox(
            "Passager(s)",
            ["Alone", "Friend(s)", "Kid(s)", "Partner"],
            key="passanger",
            format_func=lambda x: {"Alone":"Seul-e","Friend(s)":"Avec des amis","Kid(s)":"Avec les enfants","Partner":"Avec partenaire"}[x]
        )
        st.selectbox(
            "Destination",
            ["Home", "No Urgent Place", "Work"],
            key="destination",
            format_func=lambda x: {"Home":"Maison","No Urgent Place":"Sans destination","Work":"Travail"}[x]
        )

    with tab2:
        st.selectbox(
            "Type de coupon",
            ["Bar", "Carry out & Take away", "Coffee House", "Restaurant(20-50)", "Restaurant(<20)"],
            key="coupon",
            format_func=lambda x: {
                "Bar":"Bar",
                "Carry out & Take away":"A emporter",
                "Coffee House":"Coffee House",
                "Restaurant(20-50)":"Restaurant 20-50$",
                "Restaurant(<20)":"Restaurant moins de 20$"
            }[x]
        )
        st.selectbox(
            "Expiration",
            ["2h", "1d"],
            key="expiration",
            format_func=lambda x: "2 heures" if x == "2h" else "1 jour"
        )
        st.selectbox(
            "Venue à plus de 15 min ?",
            [0, 1],
            key="toCoupon_GEQ15min",
            format_func=lambda x: "Oui" if x == 1 else "Non"
        )

    with tab3:
        st.selectbox(
            "Genre",
            ["Male", "Female"],
            key="gender",
            format_func=lambda x: "Homme" if x == "Male" else "Femme"
        )
        st.selectbox(
            "Age",
            list(AGE_LBL.keys()),
            key="age",
            format_func=lambda x: AGE_LBL[x]
        )
        st.selectbox(
            "Education",
            list(ORDRE_EDU.keys()),
            key="education",
            format_func=lambda x: EDU_FR[x]
        )
        st.selectbox(
            "Revenu annuel",
            list(INCOME_LBL.keys()),
            key="income",
            format_func=lambda x: INCOME_LBL[x]
        )

    with tab4:
        for field, label in [
            ("Bar", "Bar"),
            ("CoffeeHouse", "Coffee House"),
            ("CarryAway", "A emporter"),
            ("RestaurantLessThan20", "Restaurant moins de 20$"),
            ("Restaurant20To50", "Restaurant 20-50$")
        ]:
            st.select_slider(
                label,
                options=[0, 1, 2, 3, 4],
                key=field,
                format_func=lambda x: FREQ_LBL[x]
            )

    st.write("")
    if st.button("🔮 Predire !", type="primary", use_container_width=True):
        st.session_state.show_results = True
        st.rerun()

# =========================
# INFOS MODELE
# =========================
st.caption(f"Modèle mis à jour : {len(SELECTED_FEATURES)} variables retenues par sélection Random Forest")
st.caption("Variables utilisées : " + ", ".join(SELECTED_FEATURES))

# =========================
# RESULTATS
# =========================
if st.session_state.get("show_results", False):
    st.divider()
    st.markdown("## Résultat")

    X = encode()

    probas = {
        "Arbre de decision": dt_model.predict_proba(X)[0][1],
        "Random Forest": rf_model.predict_proba(X)[0][1],
        "MLP": mlp_model.predict_proba(X)[0][1],
    }

    cols = st.columns(3)

    for col, (nom, prob) in zip(cols, probas.items()):
        accepted = prob >= 0.5
        verdict = "ACCEPTE" if accepted else "REFUSE"
        css = "card-accept" if accepted else "card-refuse"
        m = METRICS.get(nom, {"Accuracy": 0.0, "F1": 0.0})

        col.markdown(f"""
        <div class="result-card {css}">
            <div style="font-size:12px;opacity:0.85;margin-bottom:6px;">{nom}</div>
            <div style="font-size:28px;font-weight:bold;margin-bottom:6px;">{verdict}</div>
            <div style="font-size:22px;margin-bottom:10px;">{prob*100:.1f}%</div>
            <div style="background:rgba(255,255,255,0.25);border-radius:8px;height:8px;margin:6px 0;">
                <div style="width:{prob*100:.0f}%;height:8px;border-radius:8px;background:rgba(255,255,255,0.7);"></div>
            </div>
            <div style="font-size:11px;opacity:0.8;margin-top:8px;">
                Accuracy {m['Accuracy']:.4f} · F1 {m['F1']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)