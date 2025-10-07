#!/bin/bash

DATA_DIR="data"

SUBDIRS=("models" "raw_data" "processed_data" "embeddings" "queries" "synthetic_data" "search_results" "charts")

if [ ! -d "$DATA_DIR" ]; then
    echo "Directory '$DATA_DIR' does not exist. Creating..."
    mkdir -p "$DATA_DIR"
else
    echo "Directory '$DATA_DIR' already exists."
fi

for SUBDIR in "${SUBDIRS[@]}"; do
    SUBPATH="$DATA_DIR/$SUBDIR"
    if [ ! -d "$SUBPATH" ]; then
        echo "Creating subdirectory: $SUBPATH"
        mkdir -p "$SUBPATH"
    else
        echo "Subdirectory '$SUBPATH' already exists."
    fi
done

echo "✅ All directories are set up."
