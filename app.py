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

# Style CSS pour l'interface
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

# Chargement des modeles entrainés
@st.cache_resource
def charger_modeles():
    with open("modeles.pkl", "rb") as f:
        return pickle.load(f)

donnees_modeles = charger_modeles()

modele_arbre = donnees_modeles["dt"]
modele_rf = donnees_modeles["rf"]
modele_mlp = donnees_modeles["mlp"]
colonnes_features = donnees_modeles["columns"]
features_selectionnees = donnees_modeles.get("selected_features", colonnes_features)

# Métriques obtenues avec nos modèles
metriques_defaut = {
    "Arbre de decision": {"Accuracy": 0.6941, "F1": 0.7351},
    "Random Forest": {"Accuracy": 0.7267 , "F1": 0.7654},
    "MLP": {"Accuracy": 0.7067 , "F1": 0.7485},
}
metriques_modeles = donnees_modeles.get("metrics", metriques_defaut)

# Convertion température Fahrenheit vers Celsius
TEMP_F_VERS_C = {30: "-1 C", 55: "13 C", 80: "27 C"}

# Ordre pour l'education (variable ordinale)
ORDRE_EDUCATION = {
    "Some High School": 0,
    "High School Graduate": 1,
    "Some college - no degree": 2,
    "Associates degree": 3,
    "Bachelors degree": 4,
    "Graduate degree (Masters or Doctorate)": 5,
}

# Traduction en francais pour l'interface
EDUCATION_FR = {
    "Some High School": "Secondaire partiel",
    "High School Graduate": "Diplome du secondaire",
    "Some college - no degree": "Collegial sans diplome",
    "Associates degree": "DEC / Diplome associe",
    "Bachelors degree": "Baccalaureat",
    "Graduate degree (Masters or Doctorate)": "Maitrise ou Doctorat",
}

LABEL_AGE = {
    0: "<21", 1: "21-25", 2: "26-30", 3: "31-35",
    4: "36-40", 5: "41-45", 6: "46-50", 7: ">50"
}

LABEL_REVENU = {
    0: "<12k$", 1: "12-25k$", 2: "25-37k$", 3: "37-50k$",
    4: "50-62k$", 5: "62-75k$", 6: "75-87k$", 7: "87-100k$", 8: ">100k$"
}

LABEL_FREQUENCE = {
    0: "Jamais",
    1: "Rarement",
    2: "Parfois",
    3: "Souvent",
    4: "Tres souvent"
}

# Valeurs par defaut pour l'interface
valeurs_defaut = {
    "afficher_resultats": False,
    "meteo": "Sunny",
    "destination": "No Urgent Place",
    "heure": "2PM",
    "temperature": 80,
    "passager": "Alone",
    "type_coupon": "Coffee House",
    "expiration": "1d",
    "distance_15min": 0,
    "genre": "Male",
    "age": 2,
    "education": "Bachelors degree",
    "revenu": 3,
    "freq_bar": 1,
    "freq_cafe": 2,
    "freq_emporter": 2,
    "freq_resto_20": 3,
    "freq_resto_50": 1,
}

for cle, valeur in valeurs_defaut.items():
    if cle not in st.session_state:
        st.session_state[cle] = valeur

# Fonction pour generer des valeurs aléatoires
def generer_aleatoire():
    import random
    
    st.session_state.meteo = random.choice(["Sunny", "Rainy", "Snowy"])
    st.session_state.destination = random.choice(["Home", "No Urgent Place", "Work"])
    st.session_state.heure = random.choice(["7AM", "10AM", "2PM", "6PM", "10PM"])
    st.session_state.temperature = random.choice([30, 55, 80])
    st.session_state.passager = random.choice(["Alone", "Friend(s)", "Kid(s)", "Partner"])
    st.session_state.type_coupon = random.choice(["Bar", "Carry out & Take away", "Coffee House", "Restaurant(20-50)", "Restaurant(<20)"])
    st.session_state.expiration = random.choice(["2h", "1d"])
    st.session_state.distance_15min = random.choice([0, 1])
    st.session_state.genre = random.choice(["Male", "Female"])
    st.session_state.age = random.randint(0, 7)
    st.session_state.education = random.choice(list(ORDRE_EDUCATION.keys()))
    st.session_state.revenu = random.randint(0, 8)
    st.session_state.freq_bar = random.randint(0, 4)
    st.session_state.freq_cafe = random.randint(0, 4)
    st.session_state.freq_emporter = random.randint(0, 4)
    st.session_state.freq_resto_20 = random.randint(0, 4)
    st.session_state.freq_resto_50 = random.randint(0, 4)
    
    st.session_state.afficher_resultats = True

# Fonction pour créer le SVG de la scene
def creer_svg_scene():
    largeur, hauteur = 680, 370
    heure = st.session_state.heure
    meteo = st.session_state.meteo
    est_nuit = heure == "10PM"
    est_soir = heure == "6PM"
    est_matin = heure == "7AM"

    # Couleurs selon l'heure
    if est_nuit:
        ciel1, ciel2 = "#0a0a2e", "#1a1a4e"
        sol, route, ligne = "#111", "#222", "#ffff00"
    elif est_soir:
        ciel1, ciel2 = "#ff6b35", "#c94b4b"
        sol, route, ligne = "#2d5a27", "#3a3a3a", "#fff"
    elif est_matin:
        ciel1, ciel2 = "#ffeaa7", "#fdcb6e"
        sol, route, ligne = "#27ae60", "#555", "#fff"
    else:
        sol, route, ligne = "#27ae60", "#555", "#fff"
        if meteo == "Sunny":
            ciel1, ciel2 = "#87CEEB", "#5ba3d9"
        elif meteo == "Rainy":
            ciel1, ciel2 = "#6b7a8d", "#4a5568"
        else:
            ciel1, ciel2 = "#c8d6e5", "#a4b0be"

    horizon_y = 195

    svg = f'<svg width="{largeur}" height="{hauteur}" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{ciel1}"/><stop offset="100%" stop-color="{ciel2}"/></linearGradient></defs>'
    svg += f'<rect width="{largeur}" height="{hauteur}" fill="url(#sg)"/>'

    # Elements meteo
    if meteo == "Sunny" and not est_nuit and not est_soir:
        svg += '<circle cx="590" cy="65" r="42" fill="#FFD700" opacity="0.9"/>'
        svg += '<circle cx="590" cy="65" r="54" fill="#FFD700" opacity="0.15"/>'
        svg += '<ellipse cx="140" cy="75" rx="65" ry="26" fill="white" opacity="0.75"/>'
        svg += '<ellipse cx="190" cy="60" rx="50" ry="22" fill="white" opacity="0.8"/>'
    elif meteo == "Rainy":
        svg += '<ellipse cx="200" cy="55" rx="110" ry="40" fill="#4a5568" opacity="0.9"/>'
        svg += '<ellipse cx="420" cy="48" rx="130" ry="44" fill="#4a5568" opacity="0.85"/>'
        svg += '<ellipse cx="580" cy="60" rx="95" ry="36" fill="#4a5568" opacity="0.9"/>'
        for rx, ry in [(110,105),(155,100),(200,108),(330,103),(370,110),(440,100),(520,107),(560,102)]:
            svg += f'<line x1="{rx}" y1="{ry}" x2="{rx-8}" y2="{ry+30}" stroke="#74b9ff" stroke-width="1.5" opacity="0.65"/>'
    elif meteo == "Snowy":
        svg += '<ellipse cx="180" cy="50" rx="115" ry="40" fill="#dfe6e9" opacity="0.88"/>'
        svg += '<ellipse cx="420" cy="44" rx="130" ry="44" fill="#dfe6e9" opacity="0.85"/>'
        svg += '<ellipse cx="590" cy="56" rx="95" ry="36" fill="#dfe6e9" opacity="0.88"/>'
        for fx, fy, fs in [(90,118,18),(195,132,14),(315,112,20),(450,125,16),(555,118,18),(150,158,12),(375,142,14),(260,102,16),(490,148,12)]:
            svg += f'<text x="{fx}" y="{fy}" font-size="{fs}" fill="white" opacity="0.95" font-family="sans-serif">\u2744</text>'

    if est_nuit:
        # Etoiles
        for sx, sy, sr in [(50,28,1.5),(130,48,1),(220,22,1.5),(310,38,1),(420,18,2),(500,42,1),(575,28,1.5),(645,52,1)]:
            svg += f'<circle cx="{sx}" cy="{sy}" r="{sr}" fill="white" opacity="0.9"/>'
        svg += '<circle cx="590" cy="62" r="28" fill="#f0e68c" opacity="0.9"/>'
        svg += f'<circle cx="606" cy="55" r="22" fill="{ciel1}"/>'

    # Route et sol
    svg += f'<rect x="0" y="{horizon_y}" width="{largeur}" height="{hauteur-horizon_y}" fill="{sol}"/>'
    svg += f'<polygon points="255,{horizon_y} 425,{horizon_y} {largeur},{hauteur} 0,{hauteur}" fill="{route}"/>'

    # Lignes de la route
    for dy, dlarg in [(5,12),(30,20),(75,35),(130,55)]:
        centre_route = (255 + 425) // 2
        svg += f'<line x1="{centre_route-dlarg//2}" y1="{horizon_y+dy}" x2="{centre_route+dlarg//2}" y2="{horizon_y+dy}" stroke="{ligne}" stroke-width="3" opacity="0.7"/>'

    # Batiment selon destination
    dest = st.session_state.destination
    if dest == "Home":
        svg += f'<rect x="320" y="{horizon_y-52}" width="62" height="52" rx="4" fill="#e17055" opacity="0.9"/>'
        svg += f'<polygon points="308,{horizon_y-52} 351,{horizon_y-90} 394,{horizon_y-52}" fill="#d63031" opacity="0.9"/>'
        svg += f'<rect x="338" y="{horizon_y-28}" width="18" height="28" rx="2" fill="#2d3436" opacity="0.8"/>'
        svg += f'<rect x="322" y="{horizon_y-48}" width="13" height="11" rx="2" fill="#74b9ff" opacity="0.7"/>'
        svg += f'<rect x="367" y="{horizon_y-48}" width="13" height="11" rx="2" fill="#74b9ff" opacity="0.7"/>'
    elif dest == "Work":
        svg += f'<rect x="290" y="{horizon_y-82}" width="33" height="82" rx="2" fill="#636e72" opacity="0.9"/>'
        svg += f'<rect x="327" y="{horizon_y-62}" width="28" height="62" rx="2" fill="#74b9ff" opacity="0.8"/>'
        svg += f'<rect x="359" y="{horizon_y-98}" width="38" height="98" rx="2" fill="#4a5568" opacity="0.9"/>'
    else:
        svg += f'<circle cx="340" cy="{horizon_y-12}" r="8" fill="#fdcb6e" opacity="0.9"/>'
        svg += f'<circle cx="340" cy="{horizon_y-12}" r="16" fill="#fdcb6e" opacity="0.25"/>'

    # Carte du coupon
    coupon = st.session_state.type_coupon
    icones_coupon = {
        "Bar": "🍺",
        "Carry out & Take away": "🥡",
        "Coffee House": "☕",
        "Restaurant(20-50)": "🍽️",
        "Restaurant(<20)": "🍔"
    }
    couleurs_coupon = {
        "Bar": "#e17055",
        "Carry out & Take away": "#00b894",
        "Coffee House": "#6c5ce7",
        "Restaurant(20-50)": "#d63031",
        "Restaurant(<20)": "#e67e22"
    }

    icone = icones_coupon.get(coupon, "🎟")
    couleur_c = couleurs_coupon.get(coupon, "#4A90E2")
    txt_exp = "2h" if st.session_state.expiration == "2h" else "1 jour"

    svg += f'<rect x="510" y="110" width="150" height="130" rx="14" fill="{couleur_c}" opacity="0.95"/>'
    svg += f'<rect x="510" y="110" width="150" height="38" rx="14" fill="rgba(0,0,0,0.2)"/>'
    svg += f'<text x="585" y="135" text-anchor="middle" font-size="13" fill="white" font-weight="bold" font-family="sans-serif">COUPON</text>'
    svg += f'<rect x="510" y="148" width="150" height="10" fill="rgba(0,0,0,0.2)"/>'
    svg += f'<text x="585" y="185" text-anchor="middle" font-size="32" font-family="sans-serif">{icone}</text>'
    svg += f'<text x="585" y="210" text-anchor="middle" font-size="12" fill="white" opacity="0.95" font-family="sans-serif">Valable {txt_exp}</text>'
    svg += f'<text x="585" y="232" text-anchor="middle" font-size="11" fill="white" opacity="0.8" font-family="sans-serif">-20% immediat</text>'
    
    # La voiture
    cx, cy = 240, 265
    svg += f'<rect x="{cx}" y="{cy}" width="204" height="78" rx="8" fill="#2d3436"/>'
    svg += f'<rect x="{cx+14}" y="{cy-42}" width="176" height="52" rx="10" fill="#636e72"/>'
    svg += f'<rect x="{cx+24}" y="{cy-37}" width="156" height="40" rx="6" fill="#74b9ff" opacity="0.55"/>'
    svg += f'<rect x="{cx+4}" y="{cy+8}" width="24" height="15" rx="3" fill="#e74c3c" opacity="0.9"/>'
    svg += f'<rect x="{cx+176}" y="{cy+8}" width="24" height="15" rx="3" fill="#e74c3c" opacity="0.9"/>'
    svg += f'<rect x="{cx+77}" y="{cy+53}" width="50" height="18" rx="3" fill="white"/>'
    svg += f'<text x="{cx+102}" y="{cy+66}" text-anchor="middle" font-size="9" fill="#2d3436" font-weight="bold" font-family="sans-serif">QC 2026</text>'

    for wx in [cx+38, cx+162]:
        svg += f'<circle cx="{wx}" cy="{cy+80}" r="22" fill="#2d3436"/>'
        svg += f'<circle cx="{wx}" cy="{cy+80}" r="13" fill="#636e72"/>'

    # Passagers
    p = st.session_state.passager
    if p == "Alone":
        svg += f'<circle cx="{cx+102}" cy="{cy-22}" r="14" fill="#fdcb6e" opacity="0.9"/>'
        svg += f'<rect x="{cx+88}" y="{cy-9}" width="28" height="17" rx="4" fill="#0984e3" opacity="0.8"/>'
    elif p == "Friend(s)":
        for px2, py2, pc2, cc in [(cx+72, cy-22, "#fdcb6e", "#0984e3"), (cx+118, cy-22, "#fd79a8", "#6c5ce7"), (cx+162, cy-20, "#55efc4", "#00b894")]:
            svg += f'<circle cx="{px2}" cy="{py2}" r="12" fill="{pc2}" opacity="0.9"/>'
            svg += f'<rect x="{px2-12}" y="{py2+11}" width="24" height="14" rx="4" fill="{cc}" opacity="0.8"/>'
    elif p == "Kid(s)":
        svg += f'<circle cx="{cx+85}" cy="{cy-22}" r="13" fill="#fdcb6e" opacity="0.9"/>'
        svg += f'<rect x="{cx+72}" y="{cy-10}" width="26" height="15" rx="4" fill="#0984e3" opacity="0.8"/>'
        svg += f'<circle cx="{cx+140}" cy="{cy-16}" r="9" fill="#fd79a8" opacity="0.9"/>'
        svg += f'<rect x="{cx+131}" y="{cy-8}" width="18" height="13" rx="3" fill="#e17055" opacity="0.8"/>'
    elif p == "Partner":
        for px2, pc2, cc in [(cx+82, "#fdcb6e", "#0984e3"), (cx+132, "#fd79a8", "#e84393")]:
            svg += f'<circle cx="{px2}" cy="{cy-22}" r="13" fill="{pc2}" opacity="0.9"/>'
            svg += f'<rect x="{px2-13}" y="{cy-10}" width="26" height="15" rx="4" fill="{cc}" opacity="0.8"/>'

    # Affichage heure et temp
    temp_c = TEMP_F_VERS_C.get(st.session_state.temperature, "?")
    affichage_heure = {"7AM":"07:00","10AM":"10:00","2PM":"14:00","6PM":"18:00","10PM":"22:00"}
    svg += f'<rect x="12" y="12" width="108" height="34" rx="17" fill="rgba(0,0,0,0.38)"/>'
    svg += f'<text x="66" y="34" text-anchor="middle" font-size="14" fill="white" font-weight="bold" font-family="sans-serif">{affichage_heure.get(heure,heure)}</text>'
    svg += f'<rect x="128" y="12" width="82" height="34" rx="17" fill="rgba(0,0,0,0.38)"/>'
    svg += f'<text x="169" y="34" text-anchor="middle" font-size="14" fill="white" font-family="sans-serif">{temp_c}</text>'
    svg += '</svg>'
    return svg

# Texte narratif de la situation
def generer_narratif():
    trad_heure = {"7AM":"7h du matin","10AM":"10h du matin","2PM":"14h","6PM":"18h","10PM":"22h"}
    trad_meteo = {"Sunny":"ensoleille","Rainy":"pluvieux","Snowy":"neigeux"}
    trad_passager = {"Alone":"tu conduis seul-e","Friend(s)":"tu conduis avec des amis","Kid(s)":"tu as tes enfants avec toi","Partner":"tu conduis avec ton partenaire"}
    trad_dest = {"Home":"vers chez toi","No Urgent Place":"sans destination precise","Work":"vers le travail"}
    trad_coupon = {"Bar":"un bar","Carry out & Take away":"un restaurant a emporter","Coffee House":"un coffee house","Restaurant(20-50)":"un restaurant","Restaurant(<20)":"un fast-food"}
    trad_exp = {"2h":"2 heures","1d":"1 jour"}

    s = st.session_state
    temp_c = TEMP_F_VERS_C.get(s.temperature, "?")

    return (
        f"Il est **{trad_heure.get(s.heure, s.heure)}**, temps **{trad_meteo.get(s.meteo, s.meteo)}** "
        f"a **{temp_c}**. **{trad_passager.get(s.passager, s.passager)}** "
        f"**{trad_dest.get(s.destination, s.destination)}**. "
        f"Ton GPS te propose un coupon pour **{trad_coupon.get(s.type_coupon, s.type_coupon)}** "
        f"qui expire dans **{trad_exp.get(s.expiration, s.expiration)}**. "
        f"*Est-ce que tu l'acceptes ?*"
    )

# Encoder les donnees pour les modeles
def encoder_donnees():
    s = st.session_state
    features = {col: 0 for col in colonnes_features}

    # Variables numeriques et ordinales
    if "CoffeeHouse" in features:
        features["CoffeeHouse"] = s.freq_cafe
    if "income" in features:
        features["income"] = s.revenu
    if "age" in features:
        features["age"] = s.age
    if "Bar" in features:
        features["Bar"] = s.freq_bar
    if "education" in features:
        features["education"] = ORDRE_EDUCATION.get(s.education, 4)
    if "CarryAway" in features:
        features["CarryAway"] = s.freq_emporter
    if "RestaurantLessThan20" in features:
        features["RestaurantLessThan20"] = s.freq_resto_20
    if "Restaurant20To50" in features:
        features["Restaurant20To50"] = s.freq_resto_50
    if "temperature" in features:
        features["temperature"] = s.temperature
    if "toCoupon_GEQ15min" in features:
        features["toCoupon_GEQ15min"] = s.distance_15min

    # Variables one-hot
    if s.expiration == "2h" and "expiration_2h" in features:
        features["expiration_2h"] = 1
    if s.type_coupon == "Carry out & Take away" and "coupon_Carry out & Take away" in features:
        features["coupon_Carry out & Take away"] = 1
    if s.type_coupon == "Restaurant(<20)" and "coupon_Restaurant(<20)" in features:
        features["coupon_Restaurant(<20)"] = 1
    if s.type_coupon == "Coffee House" and "coupon_Coffee House" in features:
        features["coupon_Coffee House"] = 1
    if s.genre == "Male" and "gender_Male" in features:
        features["gender_Male"] = 1
    if s.heure == "6PM" and "time_6PM" in features:
        features["time_6PM"] = 1
    if s.passager == "Friend(s)" and "passanger_Friend(s)" in features:
        features["passanger_Friend(s)"] = 1

    return pd.DataFrame([features])[colonnes_features]

# En-tete
col_logo, col_titre = st.columns([1, 1.5])

with col_logo:
    st.image("logo.png", width=250)

with col_titre:
    st.title("Forage de données")

# Layout principal
col_titre2, col_aleatoire = st.columns([4, 1])

with col_titre2:
    st.markdown(
        "<h2 style='color:#e74c3c;'>Est-ce qu'il va accepter ?</h2>",
        unsafe_allow_html=True
    )

with col_aleatoire:
    st.write("")
    if st.button("Aléatoire !", use_container_width=True):
        generer_aleatoire()
        st.rerun()

col_scene, col_formulaire = st.columns([3, 2])

with col_scene:
    st.markdown(creer_svg_scene(), unsafe_allow_html=True)
    st.markdown(f'<div class="narrative-box">{generer_narratif()}</div>', unsafe_allow_html=True)
    
with col_formulaire:
    tab1, tab2, tab3, tab4 = st.tabs(["Situation", "Coupon", "Profil", "Habitudes"])

    with tab1:
        st.selectbox(
            "Heure",
            ["7AM", "10AM", "2PM", "6PM", "10PM"],
            key="heure",
            format_func=lambda x: {"7AM":"07:00 matin","10AM":"10:00 matin","2PM":"14:00","6PM":"18:00","10PM":"22:00 nuit"}[x]
        )
        st.selectbox(
            "Météo",
            ["Sunny", "Rainy", "Snowy"],
            key="meteo",
            format_func=lambda x: {"Sunny":"Ensoleille","Rainy":"Pluvieux","Snowy":"Neigeux"}[x]
        )
        st.selectbox(
            "Température",
            [30, 55, 80],
            key="temperature",
            format_func=lambda x: TEMP_F_VERS_C[x]
        )
        st.selectbox(
            "Passager(s)",
            ["Alone", "Friend(s)", "Kid(s)", "Partner"],
            key="passager",
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
            key="type_coupon",
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
            key="distance_15min",
            format_func=lambda x: "Oui" if x == 1 else "Non"
        )

    with tab3:
        st.selectbox(
            "Genre",
            ["Male", "Female"],
            key="genre",
            format_func=lambda x: "Homme" if x == "Male" else "Femme"
        )
        st.selectbox(
            "Age",
            list(LABEL_AGE.keys()),
            key="age",
            format_func=lambda x: LABEL_AGE[x]
        )
        st.selectbox(
            "Education",
            list(ORDRE_EDUCATION.keys()),
            key="education",
            format_func=lambda x: EDUCATION_FR[x]
        )
        st.selectbox(
            "Revenu annuel",
            list(LABEL_REVENU.keys()),
            key="revenu",
            format_func=lambda x: LABEL_REVENU[x]
        )

    with tab4:
        for champ, libelle in [
            ("freq_bar", "Bar"),
            ("freq_cafe", "Coffee House"),
            ("freq_emporter", "A emporter"),
            ("freq_resto_20", "Restaurant moins de 20$"),
            ("freq_resto_50", "Restaurant 20-50$")
        ]:
            st.select_slider(
                libelle,
                options=[0, 1, 2, 3, 4],
                key=champ,
                format_func=lambda x: LABEL_FREQUENCE[x]
            )

    st.write("")
    if st.button("Prédire !", type="primary", use_container_width=True):
        st.session_state.afficher_resultats = True
        st.rerun()

# Informations sur le modele
st.caption(f"Modèle mis à jour : {len(features_selectionnees)} variables retenues par sélection Random Forest")
st.caption("Variables utilisées : " + ", ".join(features_selectionnees))

# Resultats de prediction
if st.session_state.get("afficher_resultats", False):
    st.divider()
    st.markdown("## Résultat")

    donnees_X = encoder_donnees()

    # Predictions des 3 modeles
    probabilites = {
        "Arbre de decision": modele_arbre.predict_proba(donnees_X)[0][1],
        "Random Forest": modele_rf.predict_proba(donnees_X)[0][1],
        "MLP": modele_mlp.predict_proba(donnees_X)[0][1],
    }

    colonnes = st.columns(3)

    for col, (nom_modele, proba) in zip(colonnes, probabilites.items()):
        est_accepte = proba >= 0.5
        verdict = "ACCEPTE" if est_accepte else "REFUSE"
        style_css = "card-accept" if est_accepte else "card-refuse"
        metriques = metriques_modeles.get(nom_modele, {"Accuracy": 0.0, "F1": 0.0})

        col.markdown(f"""
        <div class="result-card {style_css}">
            <div style="font-size:12px;opacity:0.85;margin-bottom:6px;">{nom_modele}</div>
            <div style="font-size:28px;font-weight:bold;margin-bottom:6px;">{verdict}</div>
            <div style="font-size:22px;margin-bottom:10px;">{proba*100:.1f}%</div>
            <div style="background:rgba(255,255,255,0.25);border-radius:8px;height:8px;margin:6px 0;">
                <div style="width:{proba*100:.0f}%;height:8px;border-radius:8px;background:rgba(255,255,255,0.7);"></div>
            </div>
            <div style="font-size:11px;opacity:0.8;margin-top:8px;">
                Accuracy {metriques['Accuracy']:.4f} · F1 {metriques['F1']:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)