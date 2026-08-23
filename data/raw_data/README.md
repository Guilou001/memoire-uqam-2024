# data/raw_data (non versionné)

Fichiers attendus par le pipeline (noms exacts) :

| Fichier | Contenu | Source | Obtention |
|---|---|---|---|
| `canadian_stocks_2000-01-01_to_2024-06-01.csv` | clôtures ajustées quotidiennes, 49 titres TSX | Yahoo Finance via yfinance | `make data` |
| `us_stocks_2000-01-01_to_2024-06-01.csv` | clôtures ajustées quotidiennes, 50 titres S&P 500 | Yahoo Finance via yfinance | `make data` |
| `TSX60_2000-01-01_to_2024-06-01.csv`, `SP500_...`, `NASDAQ_...` | indices ^GSPTSE, ^GSPC, ^IXIC (clôture) | Yahoo Finance | `make data` |
| `macro_data.csv` | LCDMA, panel mensuel équilibré (1981M01 → 2024M04, 410 variables) | Stevanovic (UQAM), https://www.stevanovic.uqam.ca/DS_LCMD.html | manuel (`balanced_can_md.csv` du zip, renommé) |
| `Fred-MD.csv` | FRED-MD, millésime juin 2024, séparateur « ; » | McCracken et Ng (St. Louis Fed) | manuel |

Le mémoire a utilisé le millésime de juin 2024 pour les prix Yahoo ; les prix ajustés étant révisés avec le
temps, un téléchargement ultérieur ne les reproduit pas à l'identique (voir README, section Reproduire).
Les fichiers de prix ne sont pas redistribués (conditions d'utilisation de Yahoo Finance) ; LCDMA et FRED-MD
se citent (Fortin-Gagnon, Leroux, Stevanovic et Surprenant, 2022 ; McCracken et Ng, 2016).
