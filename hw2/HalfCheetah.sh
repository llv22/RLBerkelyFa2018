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

## args: Namespace(batch_size=50000, debug=False, discount=0.9, dont_normalize_advantages=False, env_name='HalfCheetah-v2', ep_len=150.0, exp_name='hc_b50000_r0.01', learning_rate=0.01, n_experiments=3, n_iter=100, n_layers=2, nn_baseline=True, process_in_parallel=0, render=False, reward_to_go=True, seed=1, size=32, tensorboard_debug_address=None, ui_type='curses') [just finished 1 experiment]
# python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b50000_r0.01
# python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b50000_r0.02