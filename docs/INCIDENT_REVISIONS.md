# Fusion à un instant

La version candidate 0.1.3 ajoute `update_incident_state` : état précédent explicite, versions de
preuves, instant de validité, coupure de connaissance et initialisation spatiale de l'incident/épisode.
La fonction ne possède ni base métier, ni ordonnanceur, ni décision de publication.

Une preuve d'un autre incident ou épisode est refusée. Une preuve tardive ne peut pas entrer dans une
reconstruction causale passée. Les corrections et retraits imposent un rejeu depuis un état non affecté ;
le backend actuel choisit un rejeu conservateur depuis l'initialisation, sans réutiliser une preuve retirée.

Le périmètre est une couche de l'incident dans son contexte spatial. Le terrain générique UWD reste
indépendant. Les trous, composantes séparées, champs actif/parcouru/observable et grilles de provenance
gardent leur sémantique. Le profil 3.3.0 conserve son statut de calibration ; une validation humaine ne
constitue pas une calibration scientifique.

`fuse_daily_fire_state` reste un adaptateur compatible. Les bornes quotidiennes utilisent le calendrier
Europe/Paris, y compris les journées de 23 ou 25 heures. Les tests comparent les grilles et la provenance
de l'adaptateur, l'évolution intrajournalière, la décroissance de l'activité, les preuves tardives et le
refus des preuves hors contexte. Le protocole quotidien historique n'est pas retiré.

`python tools/ci.py verify` construit et teste le wheel installé isolément. Les dépendances candidates
sont copiées dans `vendor/` avec leur licence et vérifiées par SHA-256 ; aucun dépôt voisin n'est requis.
La qualification reste CPU, sans modèle déployé ni rendu terrain natif.
