import json
from pathlib import Path
import pandas as pd


def save_csv(
    output_directory: Path, file_name: str, dataframe: pd.DataFrame
) -> None:
    dataframe.to_csv(output_directory / file_name, index=False)


def save_json(
    output_directory: Path, file_name: str, data: dict
) -> None:
    with open(output_directory / file_name, "w") as json_file:
        json.dump(data, json_file, indent=2)
