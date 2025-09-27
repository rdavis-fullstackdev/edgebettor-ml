from __future__ import annotations

from pathlib import Path
import glob
import pandas as pd


def main() -> None:
    sched_path = Path('.data/raw/schedules.csv')
    if not sched_path.exists():
        print('0 of 0 games missing spread_home (0.00%) [no schedules.csv]')
        return
    sched = pd.read_csv(sched_path)
    for c in ['season','week','home_team','away_team']:
        if c not in sched.columns:
            sched[c] = pd.NA
    sched = sched[sched['season'] >= 2020].copy()
    sched_total = len(sched.dropna(subset=['home_team','away_team']))
    if sched_total == 0:
        print('0 of 0 games missing spread_home (0.00%) [no games >=2020]')
        return

    # Load all consolidated odds CSVs
    odds_files = glob.glob('.data/odds/*/wk*.csv')
    if not odds_files:
        print(f'{sched_total} of {sched_total} games missing spread_home (100.00%) [no odds files]')
        return
    parts = []
    for p in odds_files:
        try:
            df = pd.read_csv(p)
            parts.append(df)
        except Exception:
            continue
    if not parts:
        print(f'{sched_total} of {sched_total} games missing spread_home (100.00%) [failed to read odds files]')
        return
    odds = pd.concat(parts, axis=0, ignore_index=True)
    for c in ['home_team','away_team','spread_home']:
        if c not in odds.columns:
            odds[c] = pd.NA

    # We also want season/week for reliable join; if not present, infer from file paths later; otherwise join by teams only
    # For now, perform join by home/away teams; schedules filtered to 2020+
    merged = sched.merge(odds[['home_team','away_team','spread_home']], on=['home_team','away_team'], how='left')
    missing = int(merged['spread_home'].isna().sum())
    pct = (missing / sched_total * 100.0) if sched_total else 0.0
    print(f'{missing} of {sched_total} games missing spread_home ({pct:.2f}%) for seasons >= 2020')


if __name__ == '__main__':
    main()


