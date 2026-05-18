#!/usr/bin/env python3
import argparse
import pandas as pd


METRICS = [
    ("NC", "No At-Fault Collisions", "no_at_fault_collisions"),
    ("DAC", "Drivable Area Compliance", "drivable_area_compliance"),
    ("DDC", "Driving Direction Compliance", "driving_direction_compliance"),
    ("TLC", "Traffic Light Compliance", "traffic_light_compliance"),
    ("EP", "Ego Progress", "ego_progress"),
    ("TTC", "Time To Collision", "time_to_collision_within_bound"),
    ("LK", "Lane Keeping", "lane_keeping"),
    ("HC", "History Comfort", "history_comfort"),
    ("EC", "Extended Comfort", "two_frame_extended_comfort"),
]


def get_row(df: pd.DataFrame, token: str) -> pd.Series:
    rows = df[df["token"] == token]
    if len(rows) == 0:
        raise ValueError(f"Cannot find summary row: {token}")
    return rows.iloc[0]


def fmt(x, digits=4):
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def make_markdown_table(table_df: pd.DataFrame) -> str:
    headers = list(table_df.columns)
    rows = table_df.astype(str).values.tolist()

    widths = []
    for i, h in enumerate(headers):
        max_cell = max([len(str(r[i])) for r in rows], default=0)
        widths.append(max(len(h), max_cell))

    def line(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"

    out = [line(headers), sep]
    for r in rows:
        out.append(line(r))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="/data/llh/navsim_workspace/exp/cv_agent/2026.05.18.15.35.07/2026.05.18.15.45.00.csv", help="Path to NAVSIM v2 evaluation csv")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--out", type=str, default=None, help="Optional output csv path")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    # Remove unnamed index column if present
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    stage_one = get_row(df, "extended_pdm_score_stage_one")
    stage_two = get_row(df, "extended_pdm_score_stage_two")
    combined = get_row(df, "extended_pdm_score_combined")

    rows = []

    for short_name, full_name, base_col in METRICS:
        s1_col = f"{base_col}_stage_one"
        s2_col = f"{base_col}_stage_two"

        rows.append({
            "Metric": short_name,
            "Name": full_name,
            "Stage-One": fmt(combined.get(s1_col), args.digits),
            "Stage-Two": fmt(combined.get(s2_col), args.digits),
        })

    rows.append({
        "Metric": "Score",
        "Name": "EPDMS Score",
        "Stage-One": fmt(stage_one["score"], args.digits),
        "Stage-Two": fmt(stage_two["score"], args.digits),
    })

    rows.append({
        "Metric": "Final",
        "Name": "Final Combined Score",
        "Stage-One": "",
        "Stage-Two": fmt(combined["score"], args.digits),
    })

    table_df = pd.DataFrame(rows)

    print(make_markdown_table(table_df))

    if args.out is not None:
        table_df.to_csv(args.out, index=False)
        print(f"\nSaved table to: {args.out}")


if __name__ == "__main__":
    main()