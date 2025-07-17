#!/bin/bash
#
# This script removes macOS extended metadata files
# from the data directory and all its subdirectories.
# These metadata files are often not compatible with other
# file systems, and are generally safe to remove
#
# Usage:
#   ./cleanup_mac_metadata.sh             # to remove the files
#   ./cleanup_mac_metadata.sh --dry-run   # to perform a dry-run (list files without deleting them)

DRY_RUN=0

# Check if dry-run flag is provided
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=1
    echo "Dry-run mode: The following files would be removed:"
fi

# Define patterns for files to remove
PATTERNS=(
    "*:com.apple.*"
    "*.DS_Store*"
)

# Define path to the data directory
DATA_DIR="../data"

for pattern in "${PATTERNS[@]}"; do
    if [ $DRY_RUN -eq 1 ]; then
        echo "Listing files matching pattern: $pattern"
        find $DATA_DIR -type f -name "$pattern" -print
    else
        echo "Removing files matching pattern: $pattern"
        find $DATA_DIR -type f -name "$pattern" -exec rm -v {} \;
    fi
done

echo "Cleanup complete."
