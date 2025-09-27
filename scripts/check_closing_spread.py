from __future__ import annotations

from pathlib import Path
import pandas as pd


def main() -> None:
    raw = Path('.data/raw')
    sched_path = raw / 'schedules.csv'
    lines_path = raw / 'betting_lines.csv'
    if not sched_path.exists():
        print('0 of 0 games missing closing_spread (0.00%)')
        return

    schedules = pd.read_csv(sched_path)
    keep = ['game_id', 'season', 'week']
    for c in keep:
        if c not in schedules.columns:
            schedules[c] = pd.NA
    sched = schedules[keep].drop_duplicates('game_id')

    if lines_path.exists():
        lines = pd.read_csv(lines_path)
        # Normalize column names commonly from nfl_data_py
        if 'spread_close' in lines.columns:
            lines = lines[['game_id', 'spread_close']].drop_duplicates('game_id')
            lines = lines.rename(columns={'spread_close': 'closing_spread'})
        elif 'closing_spread' in lines.columns:
            lines = lines[['game_id', 'closing_spread']].drop_duplicates('game_id')
        else:
            lines = pd.DataFrame({'game_id': [], 'closing_spread': []})
    else:
        lines = pd.DataFrame({'game_id': [], 'closing_spread': []})

    df = sched.merge(lines, on='game_id', how='left')
    total = len(df)
    missing = int(df['closing_spread'].isna().sum()) if total else 0
    pct = (missing / total * 100.0) if total else 0.0
    print(f"{missing} of {total} games missing closing_spread ({pct:.2f}%)")


if __name__ == '__main__':
    main()


