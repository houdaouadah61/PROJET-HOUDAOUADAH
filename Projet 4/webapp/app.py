
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.datasets import load_breast_cancer


def charger_objets():


    base_dir = Path(__file__).resolve().parents[1]
    chemin_modele = base_dir / "models" / "modele.joblib"

    objets = joblib.load(chemin_modele)

    modele = objets["modele"]
    scaler = objets["scaler"]
    features_names = objets["features_names"]

    return modele, scaler, features_names


def charger_plages(features_names):


    data = load_breast_cancer()
    X_reference = pd.DataFrame(data.data, columns=data.feature_names)

    feature_mins = X_reference[features_names].min()
    feature_maxs = X_reference[features_names].max()
    feature_medians = X_reference[features_names].median()

    return feature_mins, feature_maxs, feature_medians


def valider_entrees(valeurs, features_names, feature_mins, feature_maxs):


    erreurs = []
    avertissements = []
    valeurs_converties = []

    for feature in features_names:
        valeur = valeurs.get(feature)

        if valeur is None or str(valeur).strip() == "":
            erreurs.append(f"Le champ '{feature}' est vide.")
            continue

        try:
            valeur_float = float(valeur)
        except ValueError:
            erreurs.append(f"Le champ '{feature}' doit contenir un nombre.")
            continue

        if valeur_float < 0:
            erreurs.append(f"Le champ '{feature}' ne peut pas être négatif.")

        if valeur_float < feature_mins[feature] or valeur_float > feature_maxs[feature]:
            avertissements.append(
                f"Attention : '{feature}' = {valeur_float} est hors de la plage observée "
                f"({feature_mins[feature]:.2f} à {feature_maxs[feature]:.2f})."
            )

        valeurs_converties.append(valeur_float)

    return erreurs, avertissements, valeurs_converties


def lancer_webapp():


    st.title("WebApp de prédiction - Cancer du sein")

    st.write(
        "Cette application utilise le modèle sauvegardé pendant la phase 5. "
        "L'utilisateur saisit les mesures, puis l'application renvoie une prédiction."
    )

    modele, scaler, features_names = charger_objets()
    feature_mins, feature_maxs, feature_medians = charger_plages(features_names)

    st.subheader("Saisie des features")

    st.info(
        "Les champs sont préremplis avec les valeurs médianes du dataset. "
        "Vous pouvez les modifier pour tester le modèle."
    )

    valeurs = {}

    with st.expander("Afficher les champs de saisie", expanded=True):
        for feature in features_names:
            valeurs[feature] = st.text_input(
                label=feature,
                value=str(round(feature_medians[feature], 4))
            )

    if st.button("Prédire"):
        erreurs, avertissements, valeurs_converties = valider_entrees(
            valeurs,
            features_names,
            feature_mins,
            feature_maxs
        )

        if erreurs:
            st.error("La prédiction est impossible car certaines valeurs sont incorrectes.")
            for erreur in erreurs:
                st.write("-", erreur)
            st.stop()

        if avertissements:
            st.warning("Certaines valeurs sont hors de la plage observée pendant l'entraînement.")
            for avertissement in avertissements:
                st.write("-", avertissement)

        features_df = pd.DataFrame(
            [valeurs_converties],
            columns=features_names
        )

        features_scaled = scaler.transform(features_df)

        prediction = int(modele.predict(features_scaled)[0])
        proba = float(modele.predict_proba(features_scaled)[0][prediction])

        labels_fr = {
            0: "tumeur maligne",
            1: "tumeur bénigne"
        }

        label = labels_fr[prediction]

        st.subheader("Résultat")

        st.success(f"Prédiction : {label}")
        st.metric("Probabilité", f"{proba * 100:.2f} %")

        st.progress(min(max(proba, 0), 1))


lancer_webapp()
