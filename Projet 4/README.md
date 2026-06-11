# Projet 4 - Évaluation rigoureuse et déploiement d’un modèle de classification

## Présentation du projet

Ce projet a pour objectif de réaliser une démarche complète de machine learning, depuis la préparation des données jusqu’au déploiement d’un modèle de prédiction.

Le but n’est pas seulement d’entraîner un modèle, mais de savoir l’évaluer correctement, comparer plusieurs modèles, choisir une métrique adaptée au problème, puis rendre le meilleur modèle utilisable à travers une API et une WebApp.

Le projet sera réalisé étape par étape. Chaque phase sera travaillée dans un notebook séparé 

---

## Objectifs du projet

Les objectifs principaux sont les suivants :

* choisir ou utiliser un dataset de classification binaire ;
* séparer proprement les données en train, validation et test ;
* vérifier que la séparation conserve bien la répartition des classes ;
* évaluer la stabilité d’un modèle avec le bootstrap ;
* utiliser la validation croisée pour obtenir une évaluation plus fiable ;
* comparer plusieurs modèles avec des métriques adaptées ;
* prendre en compte le coût métier des erreurs ;
* sauvegarder le meilleur modèle entraîné ;
* créer une API minimale avec Flask ;
* créer une WebApp simple avec Streamlit ou Gradio ;
* justifier clairement le choix du modèle final.

---

## Dataset utilisé

Le projet portera sur un dataset de classification binaire qui est donné par le prof  

## Organisation du dépôt

Chaque phase du projet sera réalisée dans un notebook différent. Cela permettra de suivre clairement l’avancement du travail et de faire un commit GitHub à chaque étape terminée.

```text
Projet 4/
│
├── README.md
│
├── Phase0_mise_en_route.ipynb
├── Phase1_split_train_validation_test.ipynb
├── Phase2_bootstrap_bagging.ipynb
├── Phase3_validation_croisee.ipynb
├── Phase4_metriques_metier.ipynb
├── Phase5_serialisation_api.ipynb
├── Phase6_webapp_prediction.ipynb
├── Phase7_arbitrage_final.ipynb

## Phase 0 - Mise en route

La phase 0 sert à préparer l’environnement de travail.

Dans cette phase, je vais :

* créer l’organisation du dossier projet ;
* installer les bibliothèques nécessaires ;
* vérifier que les imports fonctionnent correctement ;
* créer le fichier README ;
* préparer le suivi du projet sur GitHub ;
* faire un premier commit propre.

Les principales bibliothèques utilisées seront :

* `scikit-learn` pour les modèles classiques, les métriques et la validation croisée ;
* `tensorflow` pour le réseau de neurones ;
* `joblib` pour sauvegarder le modèle ;
* `flask` pour créer une API ;
* `streamlit` ou `gradio` pour créer une WebApp de prédiction.

---

## Phase 1 - Séparation train / validation / test

Dans cette phase, je vais séparer les données en trois parties :

* un jeu d’entraînement ;
* un jeu de validation ;
* un jeu de test.

Le jeu d’entraînement sert à entraîner les modèles.

Le jeu de validation sert à comparer les modèles et à faire des choix pendant le projet.

Le jeu de test ne doit être utilisé qu’à la fin, pour obtenir une estimation honnête des performances du modèle final.

Je vais créer une fonction `split_train_val_test` qui renvoie :

* `X_train`
* `X_val`
* `X_test`
* `y_train`
* `y_val`
* `y_test`

La séparation devra utiliser la stratification afin de conserver la même proportion de classes dans les trois jeux de données.

Des vérifications seront faites sur :

* un cas normal ;
* un cas limite ;
* un cas avec des classes déséquilibrées.

---

## Phase 2 - Bootstrap et bagging

Dans cette phase, je vais utiliser le bootstrap pour mesurer la stabilité d’un modèle.

Le bootstrap consiste à tirer plusieurs échantillons avec remise à partir des données.

Pour chaque itération :

* un échantillon est tiré avec remise ;
* le modèle est entraîné sur cet échantillon ;
* les données non tirées sont utilisées comme données out-of-bag ;
* un score est calculé.

L’objectif est d’obtenir plusieurs scores afin d’observer :

* le score moyen ;
* l’écart-type ;
* la stabilité du modèle.

Cette phase permettra de comprendre qu’un seul score peut être fragile, car il dépend du découpage des données.

---

## Phase 3 - Validation croisée k-fold

Dans cette phase, je vais utiliser la validation croisée k-fold.

La validation croisée consiste à découper les données en plusieurs parties appelées folds.

Le modèle est entraîné plusieurs fois :

* à chaque tour, un fold est utilisé pour tester ;
* les autres folds sont utilisés pour entraîner ;
* chaque fold sert une fois de test.

Je vais afficher :

* les scores obtenus à chaque fold ;
* la moyenne des scores ;
* l’écart-type des scores.

L’objectif est d’avoir une évaluation plus fiable qu’avec un seul découpage train/test.

---

## Phase 4 - Choix de la bonne métrique métier

Dans cette phase, je vais montrer que l’accuracy ne suffit pas toujours pour évaluer un modèle.

Sur un dataset déséquilibré, un modèle peut avoir une accuracy élevée tout en étant mauvais. Par exemple, si une classe est très majoritaire, un modèle peut prédire uniquement cette classe et obtenir un bon score d’accuracy, sans réellement résoudre le problème.

Je vais donc utiliser plusieurs métriques :

* matrice de confusion ;
* précision ;
* recall ;
* score F1 ;
* coût métier.

Le coût métier permettra de comparer les modèles selon les erreurs qu’ils commettent.

Par exemple :

* un faux négatif peut être très grave dans un problème médical ou de fraude ;
* un faux positif peut être gênant dans un filtre anti-spam.

L’objectif est de choisir le modèle le plus adapté au problème, et pas seulement celui qui a la meilleure accuracy.

---

## Phase 5 - Sérialisation du modèle et API Flask

Dans cette phase, je vais sauvegarder le meilleur modèle.

La sauvegarde du modèle permet de le réutiliser plus tard sans devoir le réentraîner à chaque exécution.

Le modèle sera sauvegardé avec les éléments nécessaires à la prédiction, par exemple :

* le modèle entraîné ;
* le scaler ;
* les transformations appliquées aux données.

Ensuite, je vais créer une API minimale avec Flask.

L’API devra :

* recevoir des données au format JSON ;
* vérifier que les données envoyées sont correctes ;
* appliquer les mêmes transformations que pendant l’entraînement ;
* renvoyer une prédiction ;
* renvoyer une probabilité si le modèle le permet.

L’API devra aussi gérer les erreurs simples, par exemple :

* absence de la clé `features` ;
* mauvais nombre de valeurs ;
* valeurs non numériques ;
* tableau vide.

---

## Phase 6 - WebApp de prédiction

Dans cette phase, je vais créer une interface utilisateur simple.

Cette WebApp permettra d’utiliser le modèle sans écrire de code Python.

L’utilisateur pourra saisir des valeurs dans des champs, cliquer sur un bouton, puis obtenir une prédiction lisible.

La WebApp sera réalisée avec Streamlit ou Gradio.

Elle devra :

* charger le modèle sauvegardé ;
* afficher des champs de saisie ;
* vérifier les valeurs entrées ;
* afficher la prédiction ;
* afficher la probabilité associée si disponible ;
* signaler les valeurs incohérentes ou trop éloignées des données d’entraînement.

L’objectif est de rendre le modèle utilisable par un utilisateur non technique.

---

## Phase 7 - Arbitrage final

Dans cette dernière phase, je vais comparer les modèles obtenus.

Je vais construire un tableau récapitulatif avec plusieurs critères :

* accuracy ;
* recall ;
* coût métier ;
* temps d’entraînement ;
* latence de prédiction.

Les modèles comparés pourront être par exemple :

* Random Forest ;
* Gradient Boosting ;
* réseau de neurones Keras.

Le choix du modèle champion devra être justifié avec des chiffres.

Le meilleur modèle ne sera pas forcément celui qui a la meilleure accuracy. Le choix final devra prendre en compte :

* la performance globale ;
* la stabilité ;
* le coût métier ;
* le temps d’entraînement ;
* la rapidité de prédiction ;
* la simplicité de déploiement.


## Conclusion

Ce projet permet de mettre en pratique une démarche complète de machine learning.

