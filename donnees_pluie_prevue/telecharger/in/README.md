# `telecharger/in` — entrée de l'ETL

Déposer ici les fichiers à nettoyer. L'entrée peut être n'importe quoi :
le premier DAG ne retient que les CSV.

Il les remet dans l'ordre des 30 colonnes attendues par le modèle et écrit
`../../out/features_modele.csv`.

Les GeoTIFF, HDF et textes laissés à côté sont listés dans
`out/rapport_etl.txt` et ne vont pas au modèle.

Le deuxième DAG ne lit que ce CSV, entraîne le modèle, puis l'API sert
le JSON (`/api/afrique`).
