"""
This script generates a list of duplicate or manually excluded tiles
in the raw data folder. It then copies the non-excluded tiles to the
processed data folder.
"""

from pathlib import Path
import shutil


# Define script parameters
DATA_PATH = Path("../data/raw")

DUPLICATE_MAP = {
    "Au_01-vol_01": [(174, 408), (851, 885)],
    "Au_01-vol_02": [(1, 110)],
}


def generate_excluded_numbers(extrema_list: list) -> list:
    """Generate list of excluded tile numbers based on the extrema"""
    excluded = [list(range(extrema[0], extrema[1] + 1)) for extrema in extrema_list]
    return [number for sublist in excluded for number in sublist]


def generate_excluded_tiles() -> None:
    """Generate list of excluded tile names based on the DUPLICATE_MAP"""
    excluded_tiles = []
    for dataset, extrema_list in DUPLICATE_MAP.items():
        excluded_numbers = generate_excluded_numbers(extrema_list)
        excluded_tiles.extend(
            [f"{dataset}-z_{str(number).zfill(4)}.tif" for number in excluded_numbers]
        )
    return excluded_tiles


def update_excluded_tiles_file(tile_path: Path) -> None:
    """Update the file containing containing the list of excluded files or create it"""
    excluded_dir = tile_path.parent / "excluded"
    excluded_dir.mkdir(exist_ok=True)
    excluded_file = excluded_dir / "excluded_tiles.txt"

    if not excluded_file.exists():
        excluded_file.touch()
        existing_content = []
    else:
        existing_content = excluded_file.read_text().splitlines()

    if tile_path.name not in existing_content:
        with open(excluded_file, "a") as file:
            file.write(f"{tile_path.name}\n")


def main():
    excluded_tiles = generate_excluded_tiles()
    for tile_path in sorted(DATA_PATH.rglob("*/*.tif")):
        tile_name = tile_path.name
        if (tile_name in excluded_tiles) and ("excluded" not in tile_path.parent.name):
            update_excluded_tiles_file(tile_path)
            shutil.move(tile_path, tile_path.parent / "excluded" / tile_path.name)


if __name__ == "__main__":
    main()
