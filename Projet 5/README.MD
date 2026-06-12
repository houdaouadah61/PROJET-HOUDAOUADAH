# Projet 5 — Maintenance prédictive industrielle

## 1. Présentation du projet

Ce projet est le projet final du module de Machine Learning.
L’objectif est de reprendre les principales étapes vues pendant les projets précédents : analyse de données, preprocessing, entraînement de plusieurs modèles, comparaison des performances, choix d’un modèle champion et déploiement dans une WebApp.

Le sujet choisi est la **maintenance prédictive industrielle**.

L’idée est de prédire si une machine risque de tomber en panne à partir de données capteurs comme la température, la vitesse de rotation, le couple et l’usure de l’outil.

---

## 2. Problématique

**Comment prédire si une machine industrielle risque de tomber en panne à partir de données capteurs, afin d’aider une équipe de maintenance à anticiper les pannes ?**

Dans un contexte industriel, anticiper une panne peut permettre de limiter :

* les arrêts de production ;
* les coûts de réparation ;
* les pertes de temps ;
* les interventions de maintenance non prévues.

Le modèle ne remplace pas un diagnostic technique réel, mais il peut servir d’outil d’aide à la décision.

---

## 3. Dataset utilisé

Le dataset utilisé est :

```text
predictive_maintenance.csv
```

Il contient :

```text
10 000 lignes
10 colonnes
```

Colonnes présentes dans le dataset :

* `UDI`
* `Product ID`
* `Type`
* `Air temperature [K]`
* `Process temperature [K]`
* `Rotational speed [rpm]`
* `Torque [Nm]`
* `Tool wear [min]`
* `Target`
* `Failure Type`

La variable cible est :

```text
Target
```

Elle prend deux valeurs :

* `0` : pas de panne ;
* `1` : panne machine.

Le problème est donc une **classification binaire supervisée**.

---

## 4. Analyse de la cible

La répartition de la cible est la suivante :

| Classe | Signification | Nombre d'observations | Pourcentage |
| ------ | ------------- | --------------------: | ----------: |
| 0      | Pas de panne  |                 9 661 |     96,61 % |
| 1      | Panne         |                   339 |      3,39 % |

Le dataset est donc fortement déséquilibré.

Cela signifie que l’accuracy seule peut être trompeuse.
Par exemple, un modèle qui prédit toujours “pas de panne” peut avoir une accuracy élevée, mais il ne détecte aucune panne.

Pour cette raison, les métriques les plus importantes dans ce projet sont :

* le recall ;
* la precision ;
* le F1-score ;
* la ROC-AUC ;
* la matrice de confusion.

---

## 5. Colonnes supprimées

Avant l’entraînement, certaines colonnes ont été retirées :

| Colonne        | Raison                                                                                  |
| -------------- | --------------------------------------------------------------------------------------- |
| `UDI`          | Identifiant technique, pas utile pour généraliser                                       |
| `Product ID`   | Identifiant produit, peu utile pour la prédiction                                       |
| `Failure Type` | Risque de fuite de données car cette colonne donne déjà une information liée à la panne |

La colonne `Failure Type` est importante à supprimer.
Si on la garde, le modèle peut apprendre une information trop directement liée à la panne, ce qui rendrait l’évaluation moins honnête.

---

## 6. Feature engineering

J’ai ajouté trois nouvelles variables pour enrichir les données :

| Nouvelle variable  | Explication                                                            |
| ------------------ | ---------------------------------------------------------------------- |
| `temperature_diff` | Différence entre la température du process et la température de l’air  |
| `power_estimate`   | Estimation simple de l’effort mécanique : couple × vitesse de rotation |
| `wear_level`       | Niveau d’usure de l’outil : faible, moyenne ou forte                   |

Ces variables permettent d’ajouter une logique métier simple au modèle.

---

## 7. Préprocessing

Le preprocessing est réalisé avec un `ColumnTransformer` et un `Pipeline` scikit-learn.

Variables numériques :

* remplacement des valeurs manquantes par la médiane ;
* standardisation avec `StandardScaler`.

Variables catégorielles :

* remplacement des valeurs manquantes par la modalité la plus fréquente ;
* encodage avec `OneHotEncoder`.

Les variables numériques utilisées sont :

* `Air temperature [K]`
* `Process temperature [K]`
* `Rotational speed [rpm]`
* `Torque [Nm]`
* `Tool wear [min]`
* `temperature_diff`
* `power_estimate`

Les variables catégorielles utilisées sont :

* `Type`
* `wear_level`

L’intérêt du pipeline est de garder exactement le même preprocessing pendant l’entraînement et pendant l’utilisation du modèle dans la WebApp.

---

## 8. Baseline

Avant de comparer les vrais modèles, une baseline a été créée avec `DummyClassifier`.

Cette baseline prédit toujours la classe majoritaire, donc “pas de panne”.

Résultat obtenu :

| Modèle                   | Accuracy | Precision | Recall | F1-score |
| ------------------------ | -------: | --------: | -----: | -------: |
| Baseline DummyClassifier |    0.966 |     0.000 |  0.000 |    0.000 |

Même si l’accuracy est élevée, la baseline ne détecte aucune panne.
Cela confirme que l’accuracy seule n’est pas adaptée à ce problème.

---

## 9. Algorithmes comparés

J’ai comparé quatre modèles de Machine Learning classique :

| Modèle              | Rôle dans le projet                                  |
| ------------------- | ---------------------------------------------------- |
| Logistic Regression | Modèle simple et interprétable                       |
| Random Forest       | Modèle robuste sur données tabulaires                |
| Gradient Boosting   | Modèle souvent performant sur les données tabulaires |
| SVM                 | Modèle de comparaison avec une approche différente   |

Tous les modèles ont été comparés avec :

* le même dataset ;
* le même preprocessing ;
* le même train/test split ;
* les mêmes métriques.

Cela permet une comparaison plus juste entre les modèles.

---

## 10. Leaderboard

Les résultats obtenus dans le notebook sont les suivants :

| Modèle              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Gradient Boosting   |   0.9910 |    0.9464 | 0.7794 |   0.8548 |  0.9759 |
| Random Forest       |   0.9875 |    0.9388 | 0.6765 |   0.7863 |  0.9652 |
| SVM                 |   0.9235 |    0.2947 | 0.8971 |   0.4436 |  0.9677 |
| Logistic Regression |   0.8655 |    0.1889 | 0.8971 |   0.3120 |  0.9318 |

La métrique principale utilisée pour choisir le champion est le **F1-score**.

Le F1-score est adapté ici car il prend en compte à la fois :

* la precision : éviter trop de fausses alertes ;
* le recall : détecter le maximum de vraies pannes.

---

## 11. Modèle champion

Le modèle champion retenu est :

```text
Gradient Boosting
```

Il obtient le meilleur F1-score du leaderboard :

```text
F1-score : 0.8548
```

Ses résultats principaux sont :

| Métrique  |  Score |
| --------- | -----: |
| Accuracy  | 0.9910 |
| Precision | 0.9464 |
| Recall    | 0.7794 |
| F1-score  | 0.8548 |
| ROC-AUC   | 0.9759 |

Ce modèle représente le meilleur compromis entre la détection des pannes et la limitation des fausses alertes.

Dans ce projet, le recall reste important, car rater une panne peut entraîner un arrêt de production non anticipé.
Cependant, le Gradient Boosting garde aussi une très bonne precision, ce qui évite de générer trop d’alertes inutiles.

---

## 12. Évaluation du champion

Le rapport de classification du champion montre :

| Classe           | Precision | Recall | F1-score | Support |
| ---------------- | --------: | -----: | -------: | ------: |
| 0 — Pas de panne |      0.99 |   1.00 |     1.00 |    1932 |
| 1 — Panne        |      0.95 |   0.78 |     0.85 |      68 |

La matrice de confusion obtenue est :

|                     | Prédit : pas de panne | Prédit : panne |
| ------------------- | --------------------: | -------------: |
| Réel : pas de panne |                  1929 |              3 |
| Réel : panne        |                    15 |             53 |

Le modèle détecte donc 53 pannes sur 68 dans le jeu de test.
Il rate encore 15 pannes, ce qui reste une limite importante à prendre en compte.

---

## 13. Validation croisée

Une validation croisée à 5 folds a été réalisée avec le F1-score.

Résultats obtenus :

```text
[0.6531, 0.6000, 0.3109, 0.7937, 0.7156]
```

F1-score moyen :

```text
0.6146
```

Écart-type :

```text
0.1650
```

La validation croisée montre que le modèle est performant, mais que ses résultats restent sensibles au découpage des données.

Cela peut s’expliquer par le fort déséquilibre de la classe panne.
Le score reste tout de même supérieur à la baseline, dont le F1-score est égal à 0.

---

## 14. Courbes d’évaluation

Deux courbes ont été ajoutées dans le notebook :

* la courbe ROC ;
* la courbe Precision-Recall.

La courbe ROC permet de voir si le modèle sépare correctement les machines en panne et sans panne.

La courbe Precision-Recall est particulièrement utile ici, car la classe panne est rare.
Elle permet de mieux visualiser le compromis entre les fausses alertes et les pannes détectées.

---

## 15. Sauvegarde du modèle

Le modèle champion a été sauvegardé avec `joblib`.

Fichier généré :

```text
champion.joblib
```

Le pipeline complet est sauvegardé, pas uniquement le modèle.
C’est important car la WebApp doit appliquer exactement le même preprocessing que celui utilisé pendant l’entraînement.

---

## 16. WebApp Streamlit

Une WebApp a été créée avec Streamlit.

Fichier principal :

```text
app.py
```

L’utilisateur peut saisir :

* le type de produit ;
* la température de l’air ;
* la température du process ;
* la vitesse de rotation ;
* le couple ;
* l’usure de l’outil.

L’application affiche ensuite :

* la probabilité de panne ;
* une décision lisible : risque faible, moyen ou élevé.

La WebApp permet donc de tester le modèle directement sans relancer tout le notebook.

---

## 17. Structure du projet

Structure attendue du dossier :

```text
Projet 5/
│
├── dernierprojet.ipynb
├── app.py
├── champion.joblib
├── leaderboard.csv
├── predictive_maintenance.csv
├── requirements.txt
└── README.md
```

---

## 19. Installation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## 20. Lancer la WebApp

Depuis le dossier du projet :

```bash
python -m streamlit run app.py
```

Puis ouvrir dans le navigateur :

```text
http://localhost:8501
```

---




