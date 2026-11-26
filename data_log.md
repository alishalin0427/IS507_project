# Data log

Every data-processing and modelling decision, with the evidence behind it. Newest entries at the bottom of each section.
Counts are for the four complete leagues (Premier League, La Liga, Serie A, Ligue 1) unless stated otherwise. The Bundesliga covers only Bayer Leverkusen's 34 matches (entry D3) and is excluded from the analysis.

## Data acquisition

| # | Date | Who | Issue | Evidence | Decision | Impact | Status |
|---|---|---|---|---|---|---|---|
| A1 | 10/6 | Mengling, IChen | StatsBomb raw data is nested JSON across linked files (competitions, matches, events), not a ready-made table | One events file per match; shot locations and freeze frames are nested lists | Download each match with `statsbombpy` (Mengling: 3 leagues, IChen: 2 leagues), keep shot events, flatten `location` into `x`, `y`, store `shot_freeze_frame` as a JSON string | Built our own shot table (`download_leagues.py`) | Done |
| A2 | 10/6 | Mengling | Raw event files could not be saved reliably as parquet | Nested columns (e.g. freeze frames) break parquet conversion | Cache raw events as `.pkl`; save only the flattened shot table as parquet | Raw cache is local only (`data/raw/` is git-ignored) | Done |
| A3 | 10/6 | Mengling | ClubElo API not reachable from our machines | Requests to clubelo.com failed | Use pre-match Bet365 odds from football-data.co.uk as the team-strength measure instead | Team strength = bookmaker-implied win probability | Done |
| A4 | 10/11 | Mengling | Raw `.pkl` files could not be read by our Python environment | Files were saved with numpy 2; the environment used for analysis has numpy 1 (`ModuleNotFoundError: numpy._core.numeric`) | Re-downloaded all five leagues with numpy 1, so every script uses the same environment | Shot counts unchanged (37,888 shots; 37,488 non-penalty) | Done |
| A5 | 10/11 | Mengling | Odds files lost after cleaning up disk space | `data/raw/odds/` missing | Wrote `download_odds.py` to re-download the five 2015/16 files (E0, SP1, F1, I1, D1) | 1,520 matches for the 4 leagues, 380 per league | Done |

## Coverage: sample vs population

| # | Date | Who | Issue | Evidence | Decision | Impact | Status |
|---|---|---|---|---|---|---|---|
| D1 | 10/7 | Mengling | Ligue 1 has 377 matches, not 380 | StatsBomb's own match list has 377 rows; all 377 downloaded | Keep; document as a source gap | ~75 shots (0.8% of Ligue 1) | Done |
| D2 | 10/7 | Mengling | Which 3 Ligue 1 matches are missing | Outer merge with football-data on (home team, away team): 377 matched, 3 only in football-data, 0 only in StatsBomb, 0 date mismatches | Missing: Bastia 1–2 Gazélec Ajaccio (2015-11-22), Saint-Étienne 0–2 PSG (2016-01-31), Troyes 2–4 Bordeaux (2016-04-30). No shared team, month or circumstance | Treated as a random coverage gap (`find_missing_matches.py`) | Done |
| D3 | 10/7 | IChen | Bundesliga has 34 matches, not 306 | Season selection correct (competition 9, season 27; dates 2015-08-15 to 2016-05-14; 18 clubs). Leverkusen plays in all 34 matches, every other club exactly twice. StatsBomb's public match file `matches/9/27.json` itself holds only these 34. vs football-data: 34 matched, 272 missing | Not fixable from the source: StatsBomb publishes only Leverkusen's 2015/16 matches. Exclude from the analysis; keep the raw download to document the finding (`check_bundesliga.py`) | Analysis uses 4 complete leagues: 1,517 matches, 3,569 non-penalty goals | Done |

## Data quality: duplicates, missing values, anomalies

| # | Date | Who | Issue | Evidence | Decision | Impact | Status |
|---|---|---|---|---|---|---|---|
| Q1 | 10/7 | Mengling | Duplicate records | 0 duplicate shot IDs, 0 duplicate match IDs | No action | None | Done |
| Q2 | 10/7 | Mengling | Penalties behave differently from other shots | 400 penalties, 75.0% scored vs 9.5% for non-penalty shots | Exclude from modelling | 400 shots removed from the modelling sample | Done |
| Q3 | 10/7 | Mengling | `under_pressure` mostly blank | 74.1% blank | Treat blank as False: StatsBomb records these flags only when true. Not a feature in current models | No rows lost | Done |
| Q4 | 10/7 | Mengling | `shot_first_time` mostly blank | 68.7% blank | Treat blank as False (same convention). Not a feature in current models | No rows lost | Done |
| Q5 | 10/7 | Mengling | No assisting pass (`shot_key_pass_id` blank) | 28.6% blank | Structural (dribbles, rebounds, loose balls): blank = "unassisted". Not a feature in current models | No rows lost | Done |
| Q6 | 10/7 | Mengling | Shots without a freeze frame | 354 shots (0.9%); all are penalties | No action: penalties are excluded, so every modelling shot has a freeze frame | Freeze frame available for future defensive features | Done |
| Q7 | 10/7 | Mengling | Coordinates outside the pitch (x 0–120, y 0–80) | 1 shot in the 4 leagues, direct from the corner flag, x = 120.2 | Keep; clip x to 120 and y to 80 | 1 shot adjusted | Done |
| Q8 | 10/7 | Mengling | Shots from own half | 44 shots with x < 60; genuine long-range attempts, no sign of flipped direction | Keep | None | Done |
| Q9 | 10/7 | Mengling | Shots direct from a corner | 7 shots | Keep; grouped with corners through `play_pattern` | Too few for a separate class | Done |

## Joining football-data.co.uk

| # | Date | Who | Issue | Evidence | Decision | Impact | Status |
|---|---|---|---|---|---|---|---|
| J1 | 10/7 | Mengling | Inconsistent date formats | Some files dd/mm/yyyy, others dd/mm/yy; parsing failed at row 99 | `pd.to_datetime(..., format="mixed", dayfirst=True)`; check all dates fall in Aug 2015–May 2016 | All dates parsed | Done |
| J2 | 10/7 | Mengling (found), IChen (map) | Team names differ between sources | 45 of 98 teams do not match exactly, one-to-one. Types: abbreviations (Man United), dropped prefixes/suffixes, missing accents, different spellings | Manual mapping table `data/team_name_map.csv` (fd_name, sb_name, league, note) | IChen completed the map for all 45 teams; Ligue 1 verified with 0 unmatched | Done |
| J3 | 10/7 | Mengling | Blank trailing row in football-data CSV | One all-NaN row after merging | Drop rows with missing `HomeTeam` | 1 empty row per affected file | Done |
| J4 | 10/7 | Mengling | Post-match columns in football-data files | Files include final score and match statistics | Use only pre-match odds (B365H/D/A) as features; scores used only to identify matches | Avoids leakage | Done |

## Modelling decisions

| # | Date | Who | Decision | Evidence / reason | Status |
|---|---|---|---|---|---|
| M1 | 10/7 | Mengling | Split train / validation / test **by match** (70 / 15 / 15, random seed 507) | Shots are clustered: 25 per match; top 10% of shooters take 36% of shots. Splitting by shot would put the same match in two sets (leakage) | Done |
| M2 | 10/7 | Mengling | Evaluate with **log loss** and a **calibration curve**, not accuracy | Only 9.5% of shots are goals, so predicting "no goal" every time is 90.5% accurate | Done |
| M3 | 10/7 | Mengling | Baseline 0 = training goal rate (9.8%); Baseline 1 = logistic regression on distance + angle | Goal rate falls steeply with distance (EDA). Validation log loss: 0.3108 → 0.2725. Baseline 1 is reasonably calibrated but slightly over-predicts on average (mean prediction 9.9% vs 9.4% actual), partly because the validation goal rate is lower than the training rate. Numbers updated after M6 | Done |
| M4 | 10/7 | Mengling | Next features: body part (Model 2), play pattern (Model 3) | Error analysis of Baseline 1 (training set): headers predicted 864 goals vs 454 actual (z = −16.0); corners over-predicted by 276 (z = −12.5); counter-attacks under-predicted by 84 (z = +9.0) | Planned for the final |
| M5 | 10/12 | Team | Drop Model 4 (+ log distance) from the plan | Not part of the research question; keep the plan simple. Revisit if error analysis still shows bias by distance band | Done |
| M6 | 11/17 | Mengling | Sort match IDs before shuffling for the split | Same seed gave a different split after the re-download (A4), because `unique()` follows file order. Sorting makes the split independent of file order. New split: 1,061 / 227 / 229 matches; 26,115 / 5,739 / 5,634 shots. Baseline numbers updated in M3, M4 | Done |
| M7 | 11/26 | Mengling | Final model = Model 3 (distance, angle, body part, play pattern) | Lowest validation log loss: B1 0.2725 → M2 0.2636 → M3 0.2606 (StatsBomb xG 0.2433; M3 closes 41% of the gap). Body part adds about 3× as much as play pattern. On validation, header bias 93 → 7 goals, corner bias 57 → 3, counter bias 25 → 8 | Done |
| M8 | 11/26 | Mengling | Judge error analysis of Model 3 on validation, not training | On the training set, a dummy forces its group's predicted total to equal actual goals, so z ≈ 0 there by construction | Done |
| M9 | 11/26 | Mengling | Calibration by team strength: Model 3 over-predicts underdogs | Validation, favourite = higher implied win probability than the opponent (rule set by IChen before seeing results). Favourites: 354 goals vs 356 predicted (z = −0.1). Underdogs: 183 vs 219 (z = −2.7), same as Baseline 1, so body part and play pattern do not explain it. Gap concentrated in high-probability chances | Done |

## Feature rules (leakage)

| # | Date | Who | Rule | Why |
|---|---|---|---|---|
| F1 | 10/7 | Team | Never use `shot_outcome`, `shot_end_location` or post-match data as features | They are the answer, or known only after the shot |
| F2 | 10/7 | Team | `shot_statsbomb_xg` is a benchmark only, never a feature | It is another model's prediction of the target |
| F3 | 10/7 | Team | External features must be known before kick-off | Pre-match odds only |

## Open items

- [x] **J2** IChen: finish `team_name_map.csv` for all leagues
- [x] **D3** IChen: check why the Bundesliga is missing matches
- [x] Mengling: build features and Baselines 0 and 1 on the four complete leagues
- [x] **Split reproducibility.** After re-downloading (A4), the Baseline 1 coefficients changed slightly (intercept −1.495 → −1.527) with the same seed, because the split shuffles match IDs in file order. Fix: sort match IDs before shuffling.
- [ ] Final: fit Models 2 and 3 and compare with Baseline 1 on validation
- [ ] Final: join odds and check calibration for favourites vs underdogs
- [ ] Final: evaluate the chosen model once on the test set
- [x] **Distance non-linearity.** Model 3: 12–18 yd z = +2.5 on training but +0.6 on validation; mild, mentioned as a limitation. Log distance not added.
