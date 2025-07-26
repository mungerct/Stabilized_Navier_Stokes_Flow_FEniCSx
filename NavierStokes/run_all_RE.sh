#!/bin/bash

# Set the constant image path
img="./InletImages/L_mesh_screenshot2_final.png"

# Loop from 41 to 100
for i in $(seq 10 10 70); do
    echo "Processing input $i with image $img..."
    mpirun -n 6 ./InletBatchScript.py "$i" "$img" 0.5 0.04
done

