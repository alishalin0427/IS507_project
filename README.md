# How much do body part and play pattern add to a location-based xG model?

IS507 Midterm Project · Mengling Chu & IChen Lin

**Research question:** How much do body part and play pattern improve a location-based expected-goals (xG) model, and are its probabilities trustworthy?

- 📄 Report: [`IS507_Midterm_Report.pdf`](IS507_Midterm_Report.pdf)
- 📊 Slides: [`IS507_Midterm_Slides.pdf`](IS507_Midterm_Slides.pdf)
- 📝 Data decisions: [`data_log.md`](data_log.md)

## Project in one paragraph

Expected goals (xG) is the probability that a shot becomes a goal. We build xG models with logistic regression on 37,488 non-penalty shots from four complete 2015/16 leagues (Premier League, La Liga, Serie A, Ligue 1) in StatsBomb Open Data. We start from a location-only baseline (distance + angle) and plan to add body part and play pattern one group at a time, judging each step by log loss and calibration on a validation set. Pre-match Bet365 odds from football-data.co.uk will be used to check whether the probabilities are equally reliable for strong and weak teams.

## Midterm scope

| Stage | Status |
|---|---|
| Data download, cleaning and profiling | ✅ Done |
| Exploratory analysis | ✅ Done |
| Baseline 0 (average goal rate) and Baseline 1 (distance + angle) | ✅ Done, evaluated on validation |
| Model 2 (+ body part), Model 3 (+ play pattern) | Planned for the final |
| Calibration check by team strength (odds) | Planned for the final |
| Test-set evaluation | Sealed until the final |

## Key numbers

| | |
|---|---|
| Matches | 1,517 (4 leagues; Ligue 1 has 377 of 380 in the source) |
| Raw events | 5,321,459 |
| Shots / non-penalty shots | 37,888 / 37,488 |
| Non-penalty goals | 3,569 (9.5%) |
| Split by match (train / validation / test) | 70% / 15% / 15% → 26,115 / 5,739 / 5,634 shots |
| Validation log loss | Baseline 0: 0.3108 · Baseline 1: 0.2725 |
| Baseline 1 coefficients | intercept −1.570 · distance −0.076 · angle +1.188 |

## Repository structure

```
.
├── README.md
├── IS507_Midterm_Report.pdf         # report (≤ 5 pages)
├── IS507_Midterm_Slides.pdf         # presentation slides
├── data_log.md                      # every data decision, with evidence
│
├── download_leagues.py              # 1. download StatsBomb events, build the shot table
├── download_odds.py                 # 2. download football-data.co.uk odds files
├── odd.py                           # 3. join odds to StatsBomb matches via the team-name map
├── check_bundesliga.py              # 4a. why the Bundesliga has only 34 matches
├── find_missing_matches.py          # 4b. which 3 Ligue 1 matches are missing
├── data_profile.py                  # 5. size, columns and types of every dataset
├── eda.py                           # 6. exploratory analysis and figures
├── baseline.py                      # 7. train/validation/test split, Baselines 0 and 1, error analysis
├── show_shot.py                     # (optional) one raw shot JSON vs. one row of the shot table
│
├── outputs/                         # terminal output of each script
│   ├── download_output.txt
│   ├── download_odds_output.txt
│   ├── profile_output.txt
│   ├── eda_output_5leagues.txt     # first EDA, before the Bundesliga was excluded (D3)
│   ├── eda_output_4leagues.txt     # EDA used in the report
│   └── baseline_output.txt
├── docs/
│   └── data_profile.md              # data profile written by data_profile.py
├── figures/                         # charts used in the report and slides
└── data/
    ├── team_name_map.csv            # football-data name → StatsBomb name (45 teams)
    ├── processed/                   # shot tables (.parquet), match lists (.csv), split.csv
    └── raw/                         # not in the repo (large); recreated by steps 1–2
```

## How to reproduce

**Requirements:** Python 3.10+ and the packages below. The raw event files are saved with pandas, so use the **same numpy major version** to save and to read them (we use numpy < 2).

```bash
pip install "numpy<2" pandas pyarrow matplotlib scikit-learn statsbombpy requests
```

**Run the scripts in order from the project folder** (each one reads files written by the step before):

```bash
python download_leagues.py   > outputs/download_output.txt 2>&1
python download_odds.py      > outputs/download_odds_output.txt 2>&1
python odd.py
python check_bundesliga.py
python find_missing_matches.py
python data_profile.py       > outputs/profile_output.txt 2>&1
python eda.py pl laliga ligue1 seriea > outputs/eda_output_4leagues.txt 2>&1
# (optional) python eda.py > outputs/eda_output_5leagues.txt 2>&1   # all five leagues, incl. the incomplete Bundesliga
python baseline.py           > outputs/baseline_output.txt 2>&1
```

Steps 1–2 need an internet connection and take a while (about 1,500 match files). If `data/processed/` is already present, you can start from step 5.

## Data sources

- **StatsBomb Open Data** — https://github.com/statsbomb/open-data (event data, 2015/16). Data provided by StatsBomb under their open-data terms.
- **football-data.co.uk** — https://www.football-data.co.uk/ (match results and pre-match odds, 2015/16).

## Team

Mengling Chu · IChen Lin
