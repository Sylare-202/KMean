<h1> K-means algorithme </h1>

<h2> Librairies </h2>
Ce projet a entierement ete realiser en pyhon avec l'aide des librairie suivante :

```python
import random
import numpy
import matplotlib
import pylab
import csv
import sys
import time
import argparse
```

Utilisation de "*random*" afin de generer nos cluster avant de les calibrer.</p>
Utilisation de "*numpy*" et "*pylab*" pour la partie calcule avec des matrices et des fonction mathematique. </p>
Utilisation de "*csv*" pour pouvoir extraire nos données. </p>
Utilisation de "*Time*" a des fin de benchmarking. </p>
Utilisation de "*Argparse*" pour crée un programe responsive et adaptable. </p>

___

<h2> Spécification </h2>

Ce programme a été réalisé afin de pouvoir gerer des données en 2D et en 3D en fonction des arguments passé en parametre. </p>

Lancer "*python3 Kmeans.py -h*" afin d'avoir toute les info lié au programme. </p>

<h2> Lancement du programme </h2>

Pour pouvoir lancer le programme plusieur commande sont disponible :</p>
1. "**python3 Kmeans.py -l**" </p>
   Lancer le programme avec la light database
2. "**python3 Kmeans.py -f -i 36 -c 6**" </p>
   Lancer le programme avec la havy database. On peux preciser le nbre de cluster avec le -c et le nombre d'iterations avec le -i </p>
   Il est perferable avec n cluster d'avoir n^2 iteration pour avoir le plus de precision possible
3. "**python3 Kmeans.py -d3 -i 16 -c 4**"
   Lancer le programme avec la database en 3D et un nombre diteration et de cluster fix. </p>

Vous pouvez rajouter l'argument "-w" afin de pouvoir generais une image de vos donner ainsi qu'un csv pour pouvoir recuperer vos données a nimport quel moment. </p>
Vous pouvez rajouter l'argument "-r" suivie de "-d <Data_path>" afin de pouvoir charger un fichier csv existant et l'afficher. </p>
Vous pouvez rajouter l'argument "-d <Data_path>" afin de charger n'importe quel csv voulu afin d'en calculer les culuster (ne pas preciser "-f" ou "-i" et le nombre de cluster est par defaut a 4 et le nbre d'iteration a 4^2). </p>
Vous pouvez rajouter l'argument "-ch" afin de cree un outline des clusters grace a un algorithme de converx hull. </p>

<h2> Commande exemple </h2>

1. 3D : python3 Kmeans.py -d3 -i 16 -c 4
2. 2D havy db : python3 Kmeans.py -f -i 36 -c 6 -ch
3. 2D light db : python3 Kmeans.py -l -ch

<h4> Pour d'autre info concernant les differentes commandes executable vous pouvais utiliser la commande "-h" afin d'avoir plus d'info sur le fonctionnement du programme. Vous pouvez aussi m'addreser un email si vous en estimez le besoin. </h4>


<h6> Email : bonnefon.julien@isen.yncrea.fr </h6>

