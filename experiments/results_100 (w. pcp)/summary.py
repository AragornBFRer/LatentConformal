"""Summarise experiment statistics by mixture separation (delta)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


METHOD_SPECS = {
	"EM-soft": {
		"coverage": "coverage_soft",
		"length": "length_soft",
		"len_ratio": "len_ratio_soft",
	},
	"Ignore-Z": {
		"coverage": "coverage_ignore",
		"length": "length_ignore",
		"len_ratio": "len_ratio_ignore",
	},
	"PCP": {
		"coverage": "coverage_pcp",
		"length": "length_pcp",
		"len_ratio": "len_ratio_pcp",
	},
}


def _format_value(mean: float, std: float) -> str:
	if pd.isna(mean):
		return "—"
	if pd.isna(std):
		return f"{mean:.3f}"
	return f"{mean:.3f} ± {std:.3f}"


def summarise(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if "delta" not in df.columns:
		raise ValueError("Results CSV does not contain a 'delta' column")

	rows = []
	for delta, group in df.groupby("delta"):
		for label, cols in METHOD_SPECS.items():
			coverage = group[cols["coverage"]].dropna()
			len_ratio_col = cols["len_ratio"]
			if len_ratio_col in group.columns:
				length_ratio = group[len_ratio_col].dropna()
			else:
				length_ratio = (group[cols["length"]] / group["length_oracle"]).dropna()
			length = group[cols["length"]].dropna()

			rows.append(
				{
					"delta": float(delta),
					"method": label,
					"runs": int(len(group)),
					"coverage_mean": float(coverage.mean()) if not coverage.empty else float("nan"),
					"coverage_std": float(coverage.std(ddof=0)) if coverage.size > 0 else float("nan"),
					"len_ratio_mean": float(length_ratio.mean()) if not length_ratio.empty else float("nan"),
					"len_ratio_std": float(length_ratio.std(ddof=0)) if length_ratio.size > 0 else float("nan"),
					"length_mean": float(length.mean()) if not length.empty else float("nan"),
					"length_std": float(length.std(ddof=0)) if length.size > 0 else float("nan"),
				}
			)

	summary_df = pd.DataFrame(rows)
	summary_df.sort_values(["delta", "method"], inplace=True)
	summary_df.reset_index(drop=True, inplace=True)
	return summary_df


def build_markdown(summary_df: pd.DataFrame, source_csv: Path) -> str:
	total_rows = summary_df["runs"].groupby(summary_df["delta"]).first().sum()
	summary_df = summary_df.copy()
	summary_df["delta"] = summary_df["delta"].map(lambda x: f"{x:g}")
	summary_df["coverage"] = summary_df.apply(
		lambda row: _format_value(row["coverage_mean"], row["coverage_std"]), axis=1
	)
	summary_df["len_ratio"] = summary_df.apply(
		lambda row: _format_value(row["len_ratio_mean"], row["len_ratio_std"]), axis=1
	)
	summary_df["length"] = summary_df.apply(
		lambda row: _format_value(row["length_mean"], row["length_std"]), axis=1
	)

	table_df = summary_df[["delta", "method", "runs", "coverage", "len_ratio", "length"]]
	table_df = table_df.rename(
		columns={
			"delta": "Δ",
			"method": "Method",
			"runs": "Rows",
			"coverage": "Coverage",
			"len_ratio": "Length Ratio",
			"length": "Length",
		}
	)

	header = "# Summary by Mixture Separation (Δ)\n\n"
	intro = (
		f"Source: `{source_csv.name}` with {int(total_rows)} rows. "
		"Values are mean ± std across all runs at each Δ.\n\n"
	)
	table_md = table_df.to_markdown(index=False)
	return header + intro + table_md + "\n"


def main() -> None:
	here = Path(__file__).resolve().parent
	csv_path = here / "gmm_em_results.csv"
	out_path = here / "summary.md"

	summary_df = summarise(csv_path)
	markdown = build_markdown(summary_df, csv_path)
	out_path.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
	main()
