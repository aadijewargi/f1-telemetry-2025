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
