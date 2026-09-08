# fireviewer-fire-state

Deterministic Part.4 fusion and calibration, with injected storage ports.

Python package: `fireviewer_fire_state`. Version: `0.1.0`.

## Installation

Install the versioned release wheels (including private FireViewer dependencies) from the release bundle. No sibling source checkout is required.

```sh
python -m pip install --find-links /path/to/release/wheels fireviewer-fire-state==0.1.0
python -m pytest tests -q
```

Optional model/provider environments are separate extras and retain their existing upstream constraints. Model weights, credentials, datasets and local evidence are external inputs.

## Ownership and compatibility

Target account: `fireviewer`. Target stewardship: Association FIRE-VIEWER. Historical authorship and AGPL-3.0-or-later notices are retained. This technical extraction is not a signed assignment of rights.

Source correspondence and hashes are recorded in the migration dossier. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. The former `firewarning_worker` or backend module paths are compatibility adapters in their original repository.

## Delivery boundary

Docker was deferred during the initial source delivery. The resumed private container phase, pinned images and acceptance limits are documented in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker). Production deployment remains separate. CPU/schema tests do not qualify GPU, visual or scientific performance.

## Sources et commandes propres au composant

API de calcul : `fireviewer_fire_state.fire_state_fusion.fuse_daily_fire_state`, profils dans `part4_fusion_profiles`. Un contexte spatial validé reste obligatoire. Les résultats et grilles sont retournés au backend ; la librairie ne crée aucune transaction ou révision métier.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels privés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les tests de composant et leurs dépendances de test sont recensés dans le dossier unique de migration.
