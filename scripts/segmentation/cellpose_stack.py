"""
Segment a volume stack using cellpose
"""

import tyro
from sphero_vem.segmentation import SegmentationConfig, segment_stack


def main():
    config = tyro.cli(SegmentationConfig)
    segment_stack(config)


if __name__ == "__main__":
    main()
