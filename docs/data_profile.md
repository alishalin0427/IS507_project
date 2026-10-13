# Data profile

## A. StatsBomb raw events (one row = one on-ball action)

- Match files: 1,517
- Total events (rows): 5,321,459
- Events per match: mean 3,508, min 2,614, max 4,632
- Distinct columns across all files: 120
- Shots: 37,888 (0.7% of all events)

| Event type | Share of events |
| --- | --- |
| Pass | 27.8% |
| Ball Receipt* | 26.0% |
| Carry | 21.0% |
| Pressure | 9.2% |
| Ball Recovery | 3.1% |
| Duel | 2.4% |
| Clearance | 1.4% |
| Block | 1.1% |
| Dribble | 1.0% |
| Goal Keeper | 0.9% |

## B. StatsBomb match list (one row = one match)

- Rows: 1,517; columns: 55

## C. Shot table we built (one row = one shot)

- Rows: 37,888 shots (37,488 after removing 400 penalties); columns: 20

| Column | Type | Example | Missing | Role in project |
| --- | --- | --- | --- | --- |
| `match_id` | numeric | 3754173 | 0.0% | Used to split train/valid/test |
| `id` | text / category | 4805e39a-634f-44f2-aaef-42fe1e0fec29 | 0.0% |  |
| `team` | text / category | AFC Bournemouth | 0.0% | Identifier |
| `player` | text / category | Joshua King | 0.0% | Identifier |
| `period` | numeric | 1 | 0.0% |  |
| `minute` | numeric | 4 | 0.0% |  |
| `second` | numeric | 18 | 0.0% |  |
| `shot_outcome` | text / category | Saved | 0.0% | Y (goal or not) |
| `shot_type` | text / category | Open Play | 0.0% | Used to remove penalties |
| `shot_body_part` | text / category | Right Foot | 0.0% | X (Model 2) |
| `shot_technique` | text / category | Normal | 0.0% | Not used yet |
| `play_pattern` | text / category | From Counter | 0.0% | X (Model 3) |
| `shot_first_time` | boolean | True | 68.7% | Not used yet; blank = False |
| `under_pressure` | boolean | True | 74.1% | Not used yet; blank = False |
| `shot_statsbomb_xg` | numeric | 0.03997139 | 0.0% | Reference only, never a feature |
| `shot_key_pass_id` | text / category | ec40b213-1cb6-4cf0-9921-8077d230ee70 | 28.6% | Not used yet; blank = no assist |
| `shot_freeze_frame` | text / category | [{"location": [99.0, 34.7], "player": {"... | 0.9% | Not used yet: player positions (nested list stored as JSON text) |
| `x` | numeric | 100.0 | 0.0% | X (location) |
| `y` | numeric | 27.9 | 0.0% | X (location) |
| `league` | text / category | Premier League | 0.0% | Used for error analysis |

## D. football-data.co.uk odds files (one row = one match)

- Rows: 1,520 matches; columns: 65
- Rows per file: {'E0': 380, 'F1': 380, 'I1': 380, 'SP1': 380}
- Columns we use: Date, HomeTeam, AwayTeam, B365H, B365D, B365A (pre-match only)
