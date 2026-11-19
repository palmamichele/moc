from pathlib import Path
import pandas as pd

root_path = Path("experiments")
experiment_tables = []

for folder_path in sorted(root_path.iterdir()):
    if not folder_path.is_dir():
        continue

    # Find all df_transposed*.pkl files in this experiment folder
    table_paths = sorted(folder_path.glob("df_transposed*.pkl"))

    for table_path in table_paths:
        df = pd.read_pickle(table_path)

        # Use the filename as the column name
        column_name = table_path.stem.replace("df_transposed", "").lstrip("_")

        if not column_name:
            column_name = folder_path.name

        df.columns = [f"{folder_path.name}_{column_name}"]

        experiment_tables.append(df)

if not experiment_tables:
    raise FileNotFoundError(
        f"No df_transposed*.pkl files found under {root_path}"
    )

# Append all tables column-wise
df_combined = pd.concat(experiment_tables, axis=1)
df_combined.index.names = ["Metric", "Model type"]

latex_table = df_combined.to_latex(
    escape=False,
    column_format="ll|" + "c" * len(df_combined.columns),
    caption=(
        "Performance across experimental runs "
        f"(mean $\\pm$ standard deviation), "
        f""
    ),
    label="tab:experiment_results",
    multirow=True,
)

output_path = root_path / "experiments_summary.tex"
output_path.write_text(latex_table, encoding="utf-8")

print(f"Combined {len(experiment_tables)} tables.")
print(f"LaTeX table saved to: {output_path}")