#!/bin/sh
batchs=(10000 30000 50000)
lrs=(0.005 0.01 0.02)
for batch in "${batchs[@]}"
do
    for lr in "${lrs[@]}"
    do
        python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b ${batch} -lr ${lr} -rtg --nn_baseline --exp_name hc_b${batch}_r${lr}
    done
done