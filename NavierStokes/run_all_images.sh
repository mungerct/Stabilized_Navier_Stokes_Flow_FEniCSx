#!/bin/bash

# Loop through all .png files in the InletImages folder
for img in ./InletImages/*.png; do
    echo "Processing $img..."
    mpirun -n 6 ./InletBatchScript.py 10 "$img" 0.5 0.04
done

