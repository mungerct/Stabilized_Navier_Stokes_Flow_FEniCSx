#!/bin/bash

# Loop through all .png files in the InletImages folder
for img in ./InletImages/*.png; do
    echo "Processing $img..."
    ./InletBatchScript.py 1 "$img" 0.5 0.04
done

