#!/bin/bash

# Script to efficiently delete all .xml files in a specified folder
# Designed to handle millions of files without memory issues

# Parse command line arguments
TARGET_DIR="${1:-./cla}"

# Display usage if help is requested
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [directory]"
    echo "Deletes all .xml files in the specified directory and its subdirectories"
    echo "Default directory: ./cla"
    echo ""
    echo "Example: $0 ./data"
    exit 0
fi

echo "Starting deletion of .xml files in $TARGET_DIR folder..."
echo "Warning: This will permanently delete all .xml files in $TARGET_DIR and its subdirectories!"
read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Operation cancelled."
  exit 1
fi

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: $TARGET_DIR directory not found!"
    exit 1
fi

echo "Counting .xml files to be deleted..."
# Count files first (optional, comment out if too slow)
# xml_count=$(find "$TARGET_DIR" -name "*.xml" -type f | wc -l)
# echo "Found $xml_count .xml files to delete"

echo "Deleting .xml files..."
start_time=$(date +%s)

# Use find with -delete for maximum efficiency
# -name "*.xml" : matches files ending with .xml
# -type f : only regular files (not directories)
# -delete : delete matched files directly (more efficient than -exec rm)
find "$TARGET_DIR" -name "*.xml" -type f -delete

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Deletion completed in $duration seconds"
echo "All .xml files in $TARGET_DIR have been deleted."
