# 2025 Formula 1 telemetry analysis: pace, energy and aerodynamics

Data: OpenF1 (all 24 rounds; 26,265 laps; 22,514 representative laps; qualifying position/speed traces for 23 circuits).
Wet rounds (1, 12, 13) and neutralised laps are excluded from pace and physics (20,832 clean dry laps).

## 1. Driver and team pace (cluster-robust regression)
Model: lap time ~ driver fixed effects + compound + tyre age x compound + fuel (lap number), clustered by driver stint, per round, then pooled across 21 dry rounds.
- Fuel effect: -0.054 s per lap (95% CI -0.062 to -0.045), about 3.3 s over a race.
- Tyre degradation: SOFT 0.055, MEDIUM 0.046, HARD 0.034 s per lap of age.
- Fastest on pace: PIA -1.11%, NOR -1.09%, VER -0.99% vs field mean. Team order: McLaren, Ferrari, Mercedes, Red Bull, Williams, Racing Bulls, Haas, Aston Martin, Sauber, Alpine.
- Teammate head-to-heads (paired t-test, Wilcoxon, Cohen's d, Holm-corrected): 10 pairs, 4 with p<0.05, 3 survive correction: VER vs TSU (+1.14 s, d=2.2), LEC vs HAM (+0.29 s, d=1.2), ALO vs STR (0.29 s, d=-0.74, p_holm=0.047). NOR vs PIA is statistically indistinguishable (-0.02 s, p=0.69).

## 2. Braking energy vs the hybrid recovery limit
Kinetic energy shed under braking per lap = sum of 1/2 m (v1^2 - v2^2) over brake zones (m = 800 kg + fuel, assumed).
- Median 10.7 MJ per lap (8.7 Dutch to 14.8 Azerbaijan), about 5.3x the 2 MJ per lap MGU-K recovery cap.
- Because 2 MJ at 120 kW takes only 16.7 s and median brake time is 16.8 s, recovery is power-limited on 10 circuits and cap-limited on 11.
- Street circuits shed more (11.8 vs 10.3 MJ) but not significantly (Welch p=0.18, d=0.90, n=6 vs 15). Braking energy does not correlate with top speed (r=0.15, p=0.51).
- Most braking energy must therefore go to the friction brakes and aerodynamic drag.

## 3. Full-throttle acceleration and power
- Acceleration: 1.27 g at 150 km/h, 0.81 g at 200, 0.39 at 250, 0.07 at 300; falls below 0.01 g near 335 km/h (drag = thrust).
- Power into kinetic energy (m a v): 387 kW at 200 km/h, 229 at 250, 49 at 300; peak 455 kW near 155 km/h (traction-limited below that).
- MGU-K torque at its 120 kW limit: 89-110 Nm over the 10,400-12,900 rpm seen at full throttle (T = P/omega). This is inferred; no electrical channel exists in public telemetry.

## 4. DRS and drag
At equal speed and engine power, extra force = m x (a_open - a_closed), so dCdA = 2 dF / (rho v^2), with rho from weather (p, T).
- DRS open adds 1.62 m/s^2 on average, a force of 1.38 kN (CI 1.24-1.52), equal to a drag-area reduction of 0.37 m^2 (95% CI 0.33-0.42) over 21 rounds.
- The effect shrinks at fast circuits (r=-0.66 with top speed, p=0.001; Las Vegas 0.18, Austria 0.51). Slipstream and DRS-zone placement confound this, so treat it as an effective value.

## 5. Cornering grip and downforce (qualifying traces)
Lateral acceleration a = v^2 x curvature from smoothed X,Y; 97th-percentile envelope vs speed pooled over 20 dry rounds, fitted as mu + (mu rho CLA / 2 m g) v^2.
- Envelope rises from 2.9 g at 90 km/h to 5.6 g at 270 km/h (R^2=0.90). Peak sampled lateral acceleration reaches about 6.7 g.
- Effective mechanical grip 3.25 g (bootstrap CI 3.03-3.49) and effective CLA 1.88 m^2 (CI 1.52-2.15); aero load is about 66% of weight at 250 km/h. These are lower-bound, effective values: the 97th percentile includes no guarantee that every corner is at the limit, and GPS-derived curvature is noisy.

## Assumptions and limits
- Start fuel 100 kg (assumed), qualifying mass 810 kg. Electrical quantities are inferred, not measured.
- Speed is sampled at about 3.7 Hz, so acceleration is noisy; bins are averaged over thousands of samples.
- Fuel mass estimated from acceleration change is far too noisy to use (IQR 55-128 kg); not reported as a result.
- A coast-down drag fit was tried and dropped: coasting samples include lift-and-coast, harvesting and brake-flag lag.
- Miami has about 33 of 57 laps; Azerbaijan has no qualifying traces; Spain lacks Stroll.

---

# Part 2: predictive models for tyre strategy and battery use

Run with `python src/f1_strategy.py f1_2025 analysis_out` (after `f1_analysis.py`). Everything below is validated leave-one-round-out: each circuit is predicted by a model that never saw it.

## A. Tyre strategy

**Lap-time model.** A gradient-boosted regressor (monotone in tyre age and fuel burn) predicts a lap's delta to the driver's own median from compound, tyre age, race progress and circuit descriptors (track and air temperature, base lap time, mean/top speed, braking energy). 20,858 dry representative laps.
- Held-out RMSE 0.82 s vs 1.08 s for predicting zero (R^2 0.42). A plain linear age/fuel model gets 0.81 s (R^2 0.45): **boosting does not beat the linear model on RMSE**. It tracks per-circuit age trends slightly better (correlation 0.62 vs 0.54, mean error 0.023 vs 0.025 s per lap, 46 round-compound pairs). Circuit differences in tyre wear are only weakly predictable from 21 circuits.
- The "trend with tyre age" includes fuel burn-off because age and lap number rise together, so many observed slopes are negative.

**Pit loss.** From 478 in/out-lap pairs: median 22.5 s, ranging 18.5 s (Dutch) to 28.7 s (Emilia Romagna).

**Strategy optimiser.** For every dry round it simulates all legal 1-, 2- and 3-stop plans (at least two compounds, stints within the observed tyre-age envelope: soft 28, medium 35, hard 44 laps) and picks the fastest simulated race time.
- Backtest vs what the race winner actually did (21 dry rounds): the winner's plan ranked in the top 10% of all simulated plans in 19 of 21 rounds (median rank: top 0.7%) and was a median 9.3 s (mean 14.2 s) slower than the model's pick in simulation; within 5 s in 6 rounds, within 10 s in 11.
- Stop count: the model picked a one-stop 20 times out of 21 (two-stop at Monaco); winners one-stopped 12 times. It matched the winner's stop count in 13 of 21, **barely better than always guessing one stop (12 of 21)**. Winners' extra stops came from safety cars, traffic and undercuts that the simulation ignores (Canada, with five stops, is the 65 s outlier).
- The model's picks lean on the longest allowed hard stint, so the stint-length caps matter as much as the learned lap-time trend. Treat the output as a baseline plan, not a race-day call.

**Pit-window classifier.** Predicts whether a driver pits within the next 3 laps from compound, tyre age, race progress, laps left, stint number, recent pace loss within the stint, safety-car flag and track temperature (22,506 laps, 8.2% positive).
- Leave-one-round-out AUC **0.83** (median per-round 0.88), vs 0.73 using tyre age and compound alone. Brier score 0.062 vs 0.075 for always predicting the base rate. Pace loss and safety-car flags carry real signal.

## B. Battery harvest and deploy plan (physics-based optimiser, not machine learning)
There is no state-of-charge channel in public telemetry, so this is an optimisation grounded in measured acceleration, not a trained model, and it cannot be validated against real battery data.

- Marginal benefit of extra energy at each full-throttle sample = extra speed (dE / m v) that decays along the straight according to the measured full-throttle acceleration curve, converted to time saved to the end of that straight. Energy is allocated greedily to the highest benefit per joule, within the 120 kW limit and an energy budget, on each circuit's fastest qualifying lap (20 dry circuits).
- Estimated lap time saved by deployment: **3.1 s per lap for a 4 MJ budget (2.0 s for 2 MJ)**, best at Miami (4.0 s) and worst at Monaco (2.1 s). Planning where to spend it beats spending the same energy from the start of the lap by about 0.6 s (4 MJ) or 0.8 s (2 MJ), and beats spreading it evenly by 0.7 s (4 MJ) or 0.8 s (2 MJ).
- Harvest: capturable braking energy under the 120 kW limit averages 1.5 MJ per lap (0.97 to 2.31 MJ), so the 2 MJ cap is reached on only 3 of 20 circuits (Singapore, USA, Las Vegas). Harvest is power-limited, so the plan is to harvest in every braking zone (median 7 zones; 6.8 needed on average), ranked by capturable energy, with the largest single zone giving 14-31% of the total. Per-circuit zones are in `battery_plans.json`.
- Limits: no electrical measurements; ignores braking-point shift after deployment, tyre/temperature limits and the MGU-H; absolute seconds depend on the assumed mass (810 kg) and the smoothed acceleration curve, so use the ranking and relative gains more than the absolute values.

---

# Part 3: racing lines, brake and tyre thermals, qualifying vs race pace

Scripts: `f1_lines.py`, `f1_thermal.py`, `f1_qrpace.py` (run after `f1_analysis.py`). Qualifying-trace work covers 20 dry circuits (wet rounds and Azerbaijan, which has no traces, are excluded).

## A. Racing lines and the g-g diagram
Each driver's fastest qualifying lap is resampled onto a common distance grid (one point per 20 m). Corners are found on the fastest lap as stretches with curvature radius below 400 m; every lap is then split into corner and straight time.
- Corners are 49% of the distance but 59% of lap time.
- Time against the field mean in the same round, averaged over 20 rounds (negative = faster; figure 14):
  - McLaren gains most in corners (-0.70 s per lap) and also on the straights (-0.23 s).
  - Mercedes gains in corners (-0.40 s) but is average on the straights (-0.01 s).
  - Williams is the clearest low-drag car: faster on the straights (-0.13 s, CI -0.20 to -0.06) and slower in corners (+0.17 s, CI 0.06 to 0.26).
  - Aston Martin is slow in both (corners +0.42 s, straights +0.30 s), and Kick Sauber loses mainly in corners (+0.45 s).
  - Across teams, corner and straight time are correlated (r = 0.71): the fast cars are fast everywhere.
- Apex speed at the same corners (141 corners, 2,796 driver-corner samples), against the field mean: Piastri +3.9 km/h and Norris +3.7 km/h are fastest through the middle of corners, Stroll -3.3 km/h slowest. The spread between driver means is 7.2 km/h; the corner-to-corner scatter is 5.2 km/h. In teammate pairs 8 of 14 differ at p < 0.05, but only Verstappen vs Tsunoda (+2.9 km/h) survives Holm correction.
- g-g envelope (98th percentile of total acceleration, by direction and speed class): about 6.5 g of cornering above 220 km/h versus 3.5 g below 120 km/h, and about 5.1 g of braking at high speed. This is the aero-grip effect seen directly.
- Friction-circle utilisation (share of the envelope used above 1.5 g) is 0.64 to 0.67 for every driver, with overlapping confidence intervals: it does **not** separate drivers.
- Limits: positions are sampled every 20 m, so curvature is noisy. The per-corner minimum-radius ("how wide the line is") is in `lines_corner_driver.csv` but too noisy to interpret; apex speed is the metric to trust.

## B. Brake disc and tyre thermal models
**Brake disc.** A lumped heat balance for one front disc (mass 1.4 kg, carbon heat capacity rising with temperature, convective cooling that grows with speed, radiation), driven by friction-brake power: deceleration power minus aerodynamic drag minus MGU-K recovery (120 kW up to 2 MJ per lap), 60% to the front axle. Solved to lap-periodic steady state on each circuit's fastest qualifying lap.
- Disc mass, area, drag area and the front-brake share are assumptions, and there is no public brake-temperature channel. The cooling scale is calibrated so the median circuit's mean disc temperature is 600 C, so **absolute temperatures are not predictions; the ranking between circuits is the result**.
- Friction brakes absorb a median 74% of the braking energy (7.9 MJ per lap).
- Hottest circuits: Singapore (peak 924 C), Canada (919), Mexico City (885), Austria (866). Coolest: Japan (585), Qatar (598), Netherlands (653). Typical temperature swing within a lap is about 360 C.
- The ranking is robust: across nine parameter variations (drag area, disc mass, cooling, front share) its Spearman correlation with the baseline is at least 0.98. Peak temperature correlates with friction energy per lap (r = 0.74).

**Tyre load.** A physics-motivated index per circuit (sum of squared total g times distance, plus a heating-rate and a high-g share proxy) was tested against each circuit's observed tyre degradation.
- **Null result.** None of the 15 correlations (3 compounds x 5 features, 12-20 circuits) is significant; the best is r = 0.22 (p = 0.36, Holm 1.0). The load index for medium tyres has r = 0.13, and track temperature r = 0.11.
- Likely reasons: only 20 circuits, degradation slopes that also carry fuel and traffic effects, and a load index built from low-fuel qualifying laps instead of race stints. It does not show that tyre wear is unrelated to cornering load.

## C. Qualifying pace vs race pace
Both are expressed as % against the field mean of the same drivers in each dry round (20 rounds, 376 driver-rounds).
- Team ordering is almost identical in the two sessions (Spearman 0.94, p < 0.001); McLaren leads both.
- Race pace minus qualifying pace per team (positive = slower in the race):
  - Racing Bulls +0.40% (p = 0.009, but 0.09 after Holm correction, so only suggestive), better in the race in just 4 of 20 rounds.
  - Aston Martin -0.50%, but the interval is very wide (-2.1 to +0.4%) because of a few outlier qualifying laps.
  - No team is significant after correction, and the other teams are within ±0.15%.
- Team tyre-age trends (seconds per lap of age, fuel-corrected, within round and compound): Mercedes 0.013 (best), McLaren 0.021, Red Bull 0.024, Ferrari 0.032, Alpine and Kick Sauber 0.039 (worst). They do not explain the race-minus-qualifying gap (Spearman 0.22, p = 0.53, 10 teams).
- Circuit conditions (track temperature, street vs permanent) do not explain how widely the teams' gaps spread across a round (p = 0.15 and p = 0.98).

## Part 4: Pitwall app and qualifying/winner predictor

`app/index.html` is a single-file web app (open it in a browser; data is embedded). Build chain:
`f1_predict.py` -> `build_app_data.py` -> `build_app.py` (template in `app/template.html`).

Tools: tyre strategy (21 dry circuits), battery plan and racing line (20 circuits; wet rounds 1, 12, 13 and Azerbaijan lack usable traces), qualifying and winner prediction (24).

Leave-one-round-out backtest: pole-sitter predicted in 6 of 20 dry rounds; qualifying-order rank correlation 0.72; the circuit adjustment adds nothing measurable; race winner top-1 5 of 21, top-3 19 of 21 (odds temperature-scaled, tau fit on the same backtest, so slightly flattering). Grid = qualifying order (penalties ignored). The battery plan is a physics optimiser and is not validated; the tyre model is a baseline (winner's plan in the top 10% in 19 of 21 races).
