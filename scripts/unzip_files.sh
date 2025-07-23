#!/bin/bash

# Script to unzip ch2_* files and move them to processed folder
# Preserves directory structure during extraction

# Create processed directory if it doesn't exist
mkdir -p processed

# Counter for tracking progress
count=0
total=$(ls ch2_*.zip 2>/dev/null | wc -l)

echo "Found $total ch2_*.zip files to process"

# Process each ch2_*.zip file
for zipfile in ch2_*.zip; do
    # Check if file exists (in case no ch2_*.zip files are found)
    if [ ! -f "$zipfile" ]; then
        echo "No ch2_*.zip files found"
        exit 1
    fi

    count=$((count + 1))
    echo "[$count/$total] Processing: $zipfile"

    # Unzip the file preserving directory structure
    # -q for quiet mode, -o to overwrite without prompting
    if unzip -q -o "$zipfile"; then
        echo "  ✓ Successfully extracted $zipfile"

        # Move the zip file to processed folder
        if mv "$zipfile" processed/; then
            echo "  ✓ Moved $zipfile to processed/"
        else
            echo "  ✗ Failed to move $zipfile to processed/"
        fi
    else
        echo "  ✗ Failed to extract $zipfile"
    fi

    echo ""
done

echo "Processing complete! All ch2_*.zip files have been extracted and moved to processed/"
