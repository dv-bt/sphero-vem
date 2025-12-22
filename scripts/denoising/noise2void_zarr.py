from pathlib import Path
from sphero_vem.denoising import DenosingConfig, train_n2v


def main():
    root_path = Path("data/images/Au_01-vol_01.zarr")
    src_path = "images/raw/50-10-10"
    config = DenosingConfig(root_path, src_path)
    train_n2v(config)


if __name__ == "__main__":
    main()
