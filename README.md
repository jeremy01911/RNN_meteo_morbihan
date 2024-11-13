***Projet de prediction de température avec un LSTM et un Transformer***

l'objectif de ce projet est de prédire la 21eme mesure de température d'une série de 20 avec un LSTM et un Transformer et de comparer leurs résultats.

**Dataset**

Le dataset est issu de mesures météo france réalisées dans une ville du Morbihan en Bretagne.

**preprocessing :** on met en forme le dataset, récupère les features utiles, interpole et construit les séries de mesures.

**utilitaires :** fonctions de support pour dénormaliser les données, tracer les courbes de température ou mesurer la différence moyenne entre les prédictions et les températures réelles

**LSTM :** modèle LSTM utilisé et boucle d'entrainement.

**Transformer :** modèle transformer utilisé et boucle d'entrainement.
