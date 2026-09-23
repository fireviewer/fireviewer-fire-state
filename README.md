# fireviewer-fire-state

## Repères documentaires — 19 septembre 2026

- **Rôle :** Calcul déterministe Part.4, profils de fusion, calibration et évaluation.
- **Statut :** Actif — package v0.1.2. La baseline Part.4 documentée reste non calibrée pour publication autonome.
- **Entrées :** Contexte spatial validé, observations géoréférencées admissibles, état précédent et provenance.
- **Sorties :** États `affected`, `active`, `observable`, incertitude et produits probabilistes/provenance retournés au backend.
- **Limites :** Ne crée pas ses propres transactions métier. Interpolation seule ne doit pas créer de nouvelle surface brûlée; un calcul n’est pas une observation.

[Fiche du dépôt](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/repositories/fireviewer-fire-state.md) · [Architecture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ARCHITECTURE.md) · [Statuts et vocabulaire](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/STATUTS_ET_VOCABULAIRE.md).

Cette revue documentaire ne renouvelle aucun test ni aucune réception. Les procédures, versions et preuves techniques ci-dessous conservent leur périmètre et leur date.

> **Source active FV · public.** Calcul Part.4, profils, calibration et évaluation. Voir [où travailler, quoi commiter et comment reprendre](ORGANISATION.md).

Deterministic Part.4 fusion and calibration, with injected storage ports.

Python package: `fireviewer_fire_state`. Version: `0.1.2`.

## Installation

Install the versioned release wheels (including versioned FireViewer dependencies) from the release bundle. No sibling source checkout is required.

```sh
python -m pip install --find-links /path/to/release/wheels fireviewer-fire-state==0.1.2
python -m pytest tests -q
```

Optional model/provider environments are separate extras and retain their existing upstream constraints. Model weights, credentials, datasets and local evidence are external inputs.

## Canonical repository and rights

Canonical source: [`fireviewer/fireviewer-fire-state`](https://github.com/fireviewer/fireviewer-fire-state). Technical stewardship: FIRE-VIEWER. Repository access: public.

Historical authorship, AGPL-3.0-or-later notices and third-party rights are retained. Technical stewardship and repository placement are not a signed assignment of intellectual-property rights. Any pre-association assets remain subject to their documented licences or agreements.

This repository is the maintained implementation location for the responsibility stated above. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. Older `firewarning_worker` or backend imports remain compatibility adapters where required; they are not alternative locations for new component logic.

## Delivery and qualification

Versioned packages are distributed through the versioned release bundles. Current container locks, reconstruction inputs and dated acceptance records are maintained in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker).

Package installation, CPU/schema tests, service deployment and real-data acceptance are separate checks. CPU/schema tests do not qualify GPU, visual or scientific performance. This documentation update does not publish a package, rebuild an image or change production configuration.

Extraction correspondence and hashes remain in the historical migration dossier. They record the restructuring, not the current deployment state.

## Sources et commandes propres au composant

API de calcul : `fireviewer_fire_state.fire_state_fusion.fuse_daily_fire_state`, profils dans `part4_fusion_profiles`. Un contexte spatial validé reste obligatoire. Les résultats et grilles sont retournés au backend ; la librairie ne crée aucune transaction ou révision métier.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels versionnés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les commandes de reprise et leurs prérequis sont décrits dans [ORGANISATION.md](ORGANISATION.md). Les reçus du dossier de migration restent des preuves historiques, pas une nouvelle qualification.

## Ouverture du code source — 19 septembre 2026

Ce dépôt fait partie du premier lot de huit composants FIRE-VIEWER ouvert au public sur décision du mainteneur. Le code original reste sous **AGPL-3.0-or-later** et la documentation originale sous **CC BY 4.0**, avec les notices et droits tiers existants.

Cette ouverture porte sur le code, son historique et les artefacts de développement déjà associés au dépôt. Les services déployés, comptes, données, corpus, modèles, secrets et autorisations des ressources externes gardent leur propre périmètre. Les sources des sites, du backend, des applications Android et de l’infrastructure restent privées. La visibilité publique ne constitue ni une nouvelle recette fonctionnelle ni un acte de cession des droits.

[Inventaire et périmètre d’ouverture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/OPEN_SOURCE.md).

## Migration par révisions d’incident

La version candidate 0.1.3 introduit les révisions temporelles documentées dans [le guide de migration](docs/INCIDENT_REVISIONS.md). Les contrôles CPU ne qualifient pas les modèles, la production ou le rendu Unreal. La compatibilité quotidienne reste maintenue.
