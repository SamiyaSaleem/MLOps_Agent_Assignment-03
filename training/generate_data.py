import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

def generate_dataset(rows: int) -> None:
	dataset = load_breast_cancer(as_frame=True)
	
	df = dataset.frame.copy()
	
	if rows < 1 or rows > len(df):
		raise ValueError(f"Rows must be between 1 and {len(df)}" )
		
	df = df.sample(n=rows, random_state=42).sort_index()
	
	output_path = Path("data/train.csv")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	df.to_csv(output_path, index=False)
	
	print("Dataset generated successfully!")
	print(f"Rows: {len(df)}")
	print(f"Columns: {len(df.columns)}")
	print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rows",
        type=int,
        default=300
    )

    args = parser.parse_args()
    generate_dataset(args.rows)
