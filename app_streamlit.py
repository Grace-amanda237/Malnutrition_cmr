
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# ============== CONFIG ==============
st.set_page_config(
    page_title="Malnutrition lié au poids",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== STYLE ==============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1F4E79;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1F4E79;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    .metric-value { font-size: 2.5rem; font-weight: bold; }
    .metric-label { font-size: 1rem; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ============== DONNEES ==============
@st.cache_data
def load_data():
    """Charge et nettoie les données. Mis en cache pour éviter de recharger."""
    df = pd.read_csv("df_femme.csv")
    df['IMC'] = df['imc_100'] / 100.0
    df['poids_kg'] = df['poids_brut'] / 10.0
    df['taille_cm'] = df['taille_brut'] / 10.0
    df = df[df['enceinte'].fillna(0) != 1]
    df = df.dropna(subset=['IMC'])
    df = df[(df['poids_kg'].between(30, 200)) & (df['taille_cm'].between(120, 200)) & df['IMC'].between(12, 60)]

    def classer(x):
        if x < 18.5: return 'Maigreur'
        elif x < 25:  return 'Normal'
        elif x < 30:  return 'Surpoids'
        else:         return 'Obésité'
    df['IMC_label'] = df['IMC'].apply(classer)
    df['IMC_cat'] = df['IMC_label'].map({'Maigreur':1, 'Normal':2, 'Surpoids':3, 'Obésité':4})

    df['milieu_lbl'] = df['milieu'].map({1:'Urbain', 2:'Rural'})
    df['education_lbl'] = df['education'].map({0:'Aucune', 1:'Primaire', 2:'Secondaire', 3:'Supérieur'})
    df['quintile_lbl'] = df['quintile'].map({1:'Très pauvre', 2:'Pauvre', 3:'Moyen', 4:'Riche', 5:'Très riche'})
    df['age_cat'] = pd.cut(df['age_continu'], bins=[14,24,34,44,49], labels=['15-24','25-34','35-44','45-49'])
    return df

@st.cache_resource
def train_model(df):
    """Entraîne le Random Forest et le met en cache."""
    features = ['age_continu','milieu','region','education','quintile','travail',
                'matrimonial','parite','radio','tv','journal','religion']
    cat_cols = ['milieu','region','education','quintile','travail','matrimonial',
                'radio','tv','journal','religion']
    num_cols = ['age_continu','parite']

    X = df[features]
    y = df['IMC_cat'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    prep = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols),
    ])
    X_tr_p = prep.fit_transform(X_tr)
    X_te_p = prep.transform(X_te)
    if hasattr(X_tr_p, 'toarray'):
        X_tr_p = X_tr_p.toarray(); X_te_p = X_te_p.toarray()

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_tr_b, y_tr_b = sm.fit_resample(X_tr_p, y_tr)

    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42,
                                class_weight='balanced', n_jobs=-1)
    rf.fit(X_tr_b, y_tr_b)
    pred = rf.predict(X_te_p)

    return rf, prep, features, cat_cols, num_cols, X_te_p, y_te, pred

# ============== HEADER ==============
st.markdown('<div class="main-header">🩺 Malnutrition féminine — Analyse EDS</div>',
            unsafe_allow_html=True)

df = load_data()
rf, prep, features, cat_cols, num_cols, X_te_p, y_te, pred = train_model(df)

# ============== SIDEBAR NAVIGATION ==============
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Choisir une page :", [
    " Accueil",
    " Exploration des données",
    " Analyse bivariée",
    " Modèle prédictif",
    " Performance du modèle"
])

st.sidebar.markdown("---")
st.sidebar.info(f"**Échantillon** : {len(df):,} femmes\n\n**Variables** : 12 explicatives\n\n**Modèle** : Random Forest")

# ============== PAGE ACCUEIL ==============
if page == " Accueil":
    st.markdown("## Vue d'ensemble")

    c1, c2, c3, c4 = st.columns(4)
    distrib = df['IMC_label'].value_counts()
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Femmes analysées</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#e74c3c 0%,#c0392b 100%)">
            <div class="metric-value">{distrib.get('Maigreur',0)/len(df)*100:.1f}%</div>
            <div class="metric-label">Maigreur</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        with_excess = (distrib.get('Surpoids',0) + distrib.get('Obésité',0)) / len(df) * 100
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#f39c12 0%,#d35400 100%)">
            <div class="metric-value">{with_excess:.1f}%</div>
            <div class="metric-label">Surpoids + Obésité</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#27ae60 0%,#16a085 100%)">
            <div class="metric-value">{df['IMC'].mean():.1f}</div>
            <div class="metric-label">IMC moyen</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### Distribution de l'IMC")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df['IMC'], bins=50, color='#3498db', edgecolor='white')
        ax.axvline(18.5, color='red', linestyle='--', label='Maigreur')
        ax.axvline(25, color='orange', linestyle='--', label='Surpoids')
        ax.axvline(30, color='darkred', linestyle='--', label='Obésité')
        ax.set_xlabel('IMC (kg/m²)'); ax.set_ylabel('Fréquence')
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.markdown("### Catégories OMS")
        order = ['Maigreur','Normal','Surpoids','Obésité']
        colors_cat = ['#e74c3c','#2ecc71','#f39c12','#8e44ad']
        cnt = df['IMC_label'].value_counts().reindex(order)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pie(cnt.values, labels=order, colors=colors_cat, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

    st.markdown("### 📌 À propos du projet")
    st.info("""
    Cette application interactive permet d'explorer les déterminants de la **malnutrition**
    chez les femmes adultes à partir des données EDS. Elle s'appuie sur :
    - Une analyse descriptive et bivariée complète
    - Un modèle prédictif (**Random Forest**) entraîné avec rééquilibrage **SMOTE**
    - Des visualisations interactives pour explorer les associations
    """)

# ============== EXPLORATION ==============
elif page == " Exploration des données":
    st.markdown("##  Exploration des données")

    var_choice = st.selectbox("Variable à explorer :",
        ['milieu_lbl', 'education_lbl', 'quintile_lbl', 'age_cat'])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### Distribution de {var_choice}")
        fig, ax = plt.subplots(figsize=(8, 5))
        cnt = df[var_choice].value_counts()
        ax.bar(cnt.index.astype(str), cnt.values, color='#3498db')
        ax.set_ylabel('Effectif')
        plt.xticks(rotation=30)
        st.pyplot(fig)
    with col2:
        st.markdown(f"### IMC par {var_choice}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x=var_choice, y='IMC', ax=ax, palette='Set2')
        ax.axhline(25, color='red', ls='--', alpha=0.5)
        plt.xticks(rotation=30)
        st.pyplot(fig)

    st.markdown("### Tableau croisé")
    ct = pd.crosstab(df[var_choice], df['IMC_label'], normalize='index')*100
    order = ['Maigreur','Normal','Surpoids','Obésité']
    ct = ct[order].round(1)
    st.dataframe(ct.style.background_gradient(cmap='YlOrRd'), use_container_width=True)

# ============== BIVARIE ==============
elif page == " Analyse bivariée":
    st.markdown("##  Tests d'association")

    qual_vars = ['milieu_lbl','education_lbl','quintile_lbl','age_cat']
    results = []
    for v in qual_vars:
        ct = pd.crosstab(df[v], df['IMC_label'])
        chi2, p, dof, _ = chi2_contingency(ct)
        n = ct.sum().sum()
        phi2 = chi2/n
        r, k = ct.shape
        phi2c = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rc = r - ((r-1)**2)/(n-1)
        kc = k - ((k-1)**2)/(n-1)
        v_c = np.sqrt(phi2c / min((kc-1), (rc-1)))
        results.append({'Variable': v, 'Chi²': round(chi2,1),
                        'p-value': f"{p:.2e}", 'V Cramer': round(v_c,3)})
    rdf = pd.DataFrame(results).sort_values('V Cramer', ascending=False)
    st.dataframe(rdf, use_container_width=True)

    st.markdown("### Stacked bar chart interactif")
    var = st.selectbox("Variable :", qual_vars)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = ['Maigreur','Normal','Surpoids','Obésité']
    colors_cat = ['#e74c3c','#2ecc71','#f39c12','#8e44ad']
    pct = pd.crosstab(df[var], df['IMC_label'], normalize='index')[order] * 100
    pct.plot(kind='bar', stacked=True, ax=ax, color=colors_cat)
    ax.set_ylabel('%')
    ax.legend(title='IMC', bbox_to_anchor=(1.02,1), loc='upper left')
    plt.xticks(rotation=30)
    st.pyplot(fig)

# ============== PREDICTION ==============
elif page == " Modèle prédictif":
    st.markdown("##  Prédiction individuelle")
    st.info("Remplissez le profil pour obtenir la prédiction de catégorie d'IMC.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Âge", 15, 49, 30)
        parite = st.slider("Nombre d'enfants", 0, 15, 2)
        milieu = st.selectbox("Milieu", [1,2], format_func=lambda x: 'Urbain' if x==1 else 'Rural')
    with col2:
        region = st.selectbox("Région", list(range(1, 13)))
        education = st.selectbox("Éducation", [0,1,2,3],
            format_func=lambda x: ['Aucune','Primaire','Secondaire','Supérieur'][x])
        quintile = st.selectbox("Quintile", [1,2,3,4,5],
            format_func=lambda x: ['Très pauvre','Pauvre','Moyen','Riche','Très riche'][x-1])
    with col3:
        travail = st.selectbox("Travail (0=sans)", [0,1,2,3,4,5,6,7,8,9])
        matrimonial = st.selectbox("Matrimonial", [0,1,2,3,4,5],
            format_func=lambda x: ['Jamais','Marié','Partenaire','Veuf','Divorcé','Séparé'][x])
        religion = st.selectbox("Religion", [1,2,3,4,5,7,96])

    col4, col5, col6 = st.columns(3)
    with col4:
        radio = st.selectbox("Radio", [0,1,2], format_func=lambda x:['Aucune','<1x/sem','≥1x/sem'][x])
    with col5:
        tv = st.selectbox("TV", [0,1,2], format_func=lambda x:['Aucune','<1x/sem','≥1x/sem'][x])
    with col6:
        journal = st.selectbox("Journal", [0,1,2], format_func=lambda x:['Aucun','<1x/sem','≥1x/sem'][x])

    if st.button("🔮 Prédire", type="primary"):
        X_new = pd.DataFrame([{
            'age_continu': age, 'milieu': milieu, 'region': region,
            'education': education, 'quintile': quintile, 'travail': travail,
            'matrimonial': matrimonial, 'parite': parite, 'radio': radio,
            'tv': tv, 'journal': journal, 'religion': religion
        }])
        X_new_p = prep.transform(X_new)
        if hasattr(X_new_p, 'toarray'): X_new_p = X_new_p.toarray()
        pred_class = rf.predict(X_new_p)[0]
        proba = rf.predict_proba(X_new_p)[0]
        labels = ['Maigreur','Normal','Surpoids','Obésité']
        colors = ['#e74c3c','#2ecc71','#f39c12','#8e44ad']

        col_a, col_b = st.columns([1, 2])
        with col_a:
            color = colors[pred_class - 1]
            st.markdown(f"""<div style="background:{color};color:white;padding:2rem;
                border-radius:12px;text-align:center">
                <div style="font-size:1.2rem">Catégorie prédite</div>
                <div style="font-size:3rem;font-weight:bold">{labels[pred_class-1]}</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(labels, proba*100, color=colors)
            for i, v in enumerate(proba):
                ax.text(v*100+1, i, f'{v*100:.1f}%', va='center')
            ax.set_xlabel('Probabilité (%)')
            ax.set_xlim(0, 100)
            st.pyplot(fig)

# ============== PERFORMANCE ==============
elif page == " Performance du modèle":
    st.markdown("##  Performance du Random Forest")

    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average='macro')

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("F1 macro", f"{f1:.3f}")

    st.markdown("### Matrice de confusion")
    cm = confusion_matrix(y_te, pred, labels=[1,2,3,4])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Maigreur','Normal','Surpoids','Obésité'],
                yticklabels=['Maigreur','Normal','Surpoids','Obésité'], ax=ax)
    ax.set_xlabel('Prédit'); ax.set_ylabel('Réel')
    st.pyplot(fig)

    st.markdown("### Rapport de classification")
    rep = classification_report(y_te, pred,
        target_names=['Maigreur','Normal','Surpoids','Obésité'], output_dict=True)
    st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

    st.markdown("### Importance des variables")
    feature_names = num_cols + list(prep.named_transformers_['cat'].get_feature_names_out(cat_cols))
    imp = pd.DataFrame({'variable': feature_names,
                        'importance': rf.feature_importances_}).sort_values('importance', ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(imp.iloc[::-1]['variable'], imp.iloc[::-1]['importance'], color='#2ecc71')
    ax.set_xlabel('Importance')
    st.pyplot(fig)

st.markdown("---")
