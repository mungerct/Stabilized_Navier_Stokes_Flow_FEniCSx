#!/bin/bash

# Set the constant image path
img="./InletImages/L_mesh_screenshot2_final.png"

# Loop from 2 to 40
for i in $(seq 2 40); do
    echo "Processing input $i with image $img..."
    ./InletBatchScript.py "$i" "$img" 0.5 0.04
done

