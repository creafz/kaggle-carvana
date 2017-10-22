#!/bin/bash

for i in $(seq 0 5)
do
    python train.py --experiment-name se_refinenet_1024 --model-name se_refinenet_1024 --fold-num "$i"
    python predict.py --experiment-name se_refinenet_1024 --model-name se_refinenet_1024 --fold-num "$i" --dirname se_refinenet_1024
done

python make_submission.py --dirname se_refinenet_1024 --submission-filename se_refinenet_1024
