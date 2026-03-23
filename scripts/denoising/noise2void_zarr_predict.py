from pathlib import Path
from sphero_vem.denoising import denoise_stack


def main():
    root_path = Path("data/images/Au_01-vol_01.zarr")
    src_path = "images/raw/50-10-10"
    denoise_stack(
        root_path,
        src_path,
        model_name="n2v-depth3-patch128-nimages10",
        rescale_mode="per_slice",
    )


if __name__ == "__main__":
    main()
