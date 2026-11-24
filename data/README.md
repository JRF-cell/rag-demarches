# Data

## Source

- Dataset: "Fiches pratiques et ressources de Service-Public.gouv.fr Particuliers", published on data.gouv.fr:
  https://www.data.gouv.fr/datasets/fiches-pratiques-et-ressources-de-service-public-gouv-fr-particuliers
- Author: Service-Public.gouv.fr / Direction de l'information légale et administrative (DILA)
- License: Licence Ouverte / Open Licence v2.0 (Etalab)
- The pre-vectorised version of these sheets, also on data.gouv.fr, is not used.

## Snapshot in use

| Field | Value |
|---|---|
| Download URL | https://lecomarquage.service-public.gouv.fr/vdd/3.5/part/zip/vosdroits-latest.zip |
| Local file | `data/raw/20261124/vosdroits-3.5-20261124.zip` |
| Server `Last-Modified` | Tue, 24 Nov 2026 14:26:09 GMT |
| Downloaded (UTC) | 2026-11-24T15:50:00 |
| SHA-256 | `70c890bee948ab2dd268f3fec9843e70af0b621905f2ebf1fd57d4aa261fa1bd` |
| Feed / schema version | 3.5 (URL path, and `xsi:noNamespaceSchemaLocation="../Schemas/3.5/Publication.xsd"` in every sheet) |
| Schema archive | https://echanges.dila.gouv.fr/OPENDATA/SERVICE-PUBLIC_DTD/schema_3.5.zip |

Version 3.3 of the feed was withdrawn on 4 May 2026; 3.5 is the current version.

## Reproduce

```bash
make data      # downloads into data/raw/<YYYYMMDD>/ and writes manifest.json
make explore   # writes results/corpus_stats_raw.json
```

`data/raw/` and `data/processed/` are not versioned. `manifest.json` records URL, HTTP metadata, size and SHA-256 of each downloaded file.
