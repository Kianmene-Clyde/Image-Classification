# Projet Annuel 2024
> Projet annuel de 3ème année IA&amp;BD ESGI

------

### Elèves :

Tom DORDET
Clyde KIANMENE TAKUEFOU
Houda SLIMANI



### GitHub :

Lien `GitHub` du projet : https://github.com/TomDordet/ProjetAnnuel2024



### Explication :

Notre projet porte sur la classification de panneau de signalisation. Pour faciliter la mise en place du projet, nous en avons sélectionné 3 types pour commencer : les `panneaux stop`, les `panneaux sens interdit`, et les `panneaux de limitation de vitesse 50`.



### Langages utilisés :

Nous avons décidé d'utiliser le `C++` afin de construire notre algorithme et nos librairies. Nous utiliserons `Python` afin d'appeler les librairies et de procéder à l'entraînement et à l'évaluation de nos modèles de classification de panneaux. Enfin, nous utiliserons du `PHP` avec des langages de balisages `HTML/CSS` pour l'interface du site qui permettra d'interagir avec nos modèles.



### Dataset :

Nous avons commencé à récupérer des images pour constituer notre dataset, les méthodes classiques sont utilisés. Nous récupérons des images par recherche Google. Pour gagner en temps, nous avons programmé un **wrapper** permettant de chercher et de télécharger des milliers d'images rapidement et de façon automatique. Autre méthode, **recueillir à la main des centaines d'images que l'on trouve sur d'autre datasets déjà mis en place** (nous avons bien pris conscience du fait que nous devons récupérer seulement une maigre partie des datasets pour éviter les biais d'un dataset que nous ne maîtrisons pas et qui nuirait à l'apprentissage de nos modèles.). Enfin, la dernière méthode pour récupérer de la donnée et personnaliser réellement notre dataset est de **prendre nous-même en photo les panneaux que l'on souhaite voir apparaître dans notre dataset** en se déplaçant nous-même devant les panneaux avec un appareil photo.

------

**Quelques liens** pour récupérer des données intéressantes :

[IconFinder](https://www.iconfinder.com/search?q=traffic+sign)

[Commons](https://commons.wikimedia.org/w/index.php?search=traffic+sign&title=Special:MediaSearch&go=Go&type=image)

[ShutterStock](https://www.shutterstock.com/fr/search/traffic-signs?consentChanged=true&ds_ag=FF%3DShutterstock-Shutterstock-Exact_AU%3DProspecting&ds_agid=58700002001420666&ds_cid=71700000017549998&ds_eid=700000001507159&gclid=CjwKCAjw_YShBhAiEiwAMomsEEFywfOumbyjPjgoDtEjHx7vhDnURi7KaS8_JbpaI-kdVlbb2u9SbRoCxgIQAvD_BwE&gclsrc=aw.ds&kw=shutterstock&utm_campaign=CO%3DFR_LG%3DFR_BU%3DIMG_AD%3DBRAND_TS%3Dlggeneric_RG%3DEUAF_AB%3DACQ_CH%3DSEM_OG%3DCONV_PB%3DGoogle&utm_medium=cpc&utm_source=GOOGLE)

[Flickr](https://www.flickr.com/search/?text=traffic+signs)

------

Faire fonctionner le `wrapper.py` :

Pour cela, il faut créer un nouveau projet dans la console `Google Cloud Platform`, chercher `Google Custom Search API` dans `Bibliothèque` et activer l'API. De là, on va dans `Identifiants` et on génère une `Clé d'API`
Deuxièmement, créer un moteur de recherche avec [Custom Search Engine](https://programmablesearchengine.google.com/about/). On suit les étapes classiques de création du moteur de recherche en cochant `Recherche d'images` et `Rechercher sur l'ensemble du Web` ainsi que `SafeSearch`. Enfin, cliquer sur `Personnaliser` pour accéder aux paramètres et détails du moteur de recherche et récupérer l'id du moteur.

Maintenant équipé de notre clé d'API et de notre ID de moteur de recherche, on remplace `API_KEY` (ligne 13) et `CSE_ID` (ligne 14).

```python
    # Remplacer 'API_KEY' par votre clé d'API
    # Remplacer 'CSE_ID' par l'ID de votre moteur de recherche personnalisé
    api_key = 'API_KEY'
    cse_id = 'CSE_ID'
```

Enfin, installer le module requis pour lancer le programme :

```powershell
pip install google-api-python-client
```

------

