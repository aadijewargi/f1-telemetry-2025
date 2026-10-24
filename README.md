# F1 2025 Telemetry Analysis

Physics and statistics on the full 2025 Formula 1 season using the free [OpenF1](https://openf1.org) API: lap pace, braking energy vs the MGU-K recovery limit, full-throttle power, DRS drag, and cornering grip/downforce.

See [REPORT.md](REPORT.md) for methods and results, `figures/` for charts, `results/` for output tables.

## Run
```
pip install -r requirements.txt
python src/f1_openf1_export.py --year 2025 --out f1_2025     # download data (resumable)
python src/f1_analysis.py f1_2025 analysis_out               # statistics + physics
python src/make_figs.py analysis_out                         # extra figures
```
(`f1_openf1_export.py` imports helpers from `f1_season_export.py`; keep them together. OpenF1 refuses historical data during a live session, so retry later if you get a 401.)

## Assumptions
Start fuel 100 kg and car mass are assumed; electrical quantities are inferred from speed data (no electrical channel is public); speed is sampled at ~3.7 Hz.
