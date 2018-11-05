# CS294-112 HW 2: Policy Gradient

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==**2.3.2**

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only file that you need to look at is `train_pg_f18.py`, which you will implement.

See the [HW2 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf) for further instructions.

## 2. Exercise
### 5.2 [Problem 4] CartPole - Discrete Action Space
```bash
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na
```
Result:
OOM on 2*8G GPU environment

```bash
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 2 -dna --exp_name sb_no_rtg_dna
```
Result:
OOM on 2*8G GPU environment

1) on local mac with 2*8G GPU (in sequential mode)
* batch size = 1000
```bash
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna [OK, not normalize advantages, for all rewards]
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna [OK, not normalize advantages, rewards to go]
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na [OK, normalize advantages, rewards to go]
```
Analysis for result
```bash
python plot.py --logdir data/sb_no_rtg_dna_CartPole-v0_05-11-2018_16-07-02/ data/sb_rtg_dna_CartPole-v0_05-11-2018_16-09-30/ data/sb_rtg_na_CartPole-v0_05-11-2018_16-11-30/ --legend sb_no_rtg_dna sb_rtg_dna sb_rtg_na --value AverageReturn StdReturn EpLenMean TimestepsThisBatch
```
1. Average Return Figure:  

<img src="data/00_sb_plots/AverageReturn.png" width="60%"/>

2. Eposide Length Mean:  

<img src="data/00_sb_plots/EpLenMean.png" width="60%"/>

3. Standard Deviation Return:  

<img src="data/00_sb_plots/StdReturn.png" width="60%"/>

4. Time Steps used is this batch:  

<img src="data/00_sb_plots/TimestepsThisBatch.png" width="60%"/>

* batch size = 5000
```bash
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna [OK, not normalize advantages, for all rewards]
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna [OK, not normalize advantages, rewards to go]
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na [OK, normalize advantages, rewards to go]
```
Analysis for result
```bash
python plot.py --logdir data/lb_no_rtg_dna_CartPole-v0_05-11-2018_16-16-35/ data/lb_rtg_dna_CartPole-v0_05-11-2018_16-25-42/ data/lb_rtg_na_CartPole-v0_05-11-2018_16-35-59/ --legend lb_no_rtg_dna lb_rtg_dna lb_rtg_na --value AverageReturn StdReturn EpLenMean TimestepsThisBatch
```
1. Average Return Figure:  

<img src="data/01_lb_plots/AverageReturn.png" width="60%"/>

2. Eposide Length Mean:  

<img src="data/01_lb_plots/EpLenMean.png" width="60%"/>

3. Standard Deviation Return:  

<img src="data/01_lb_plots/StdReturn.png" width="60%"/>

4. Time Steps used is this batch:  

<img src="data/01_lb_plots/TimestepsThisBatch.png" width="60%"/>


### 5.3 [Problem 5] InvertedPendulum - **Continous Action Space**

```bash
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 6000 -lr 0.01 -rtg --exp_name hc_b6000_r0.01 [OK]
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 8000 -lr 0.01 -rtg --exp_name hc_b8000_r0.01 [OK]
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 10000 -lr 0.01 -rtg --exp_name hc_b10000_r0.01
```

Analysis for result
```bash
python plot.py --logdir data/hc_b6000_r0.01_InvertedPendulum-v2_05-11-2018_16-46-42/ data/hc_b8000_r0.01_InvertedPendulum-v2_05-11-2018_16-59-08/ data/hc_b10000_r0.01_InvertedPendulum-v2_05-11-2018_17-20-03/ --legend hc_b6000_r0.01 hc_b8000_r0.01 hc_b10000_r0.01 --value AverageReturn StdReturn EpLenMean TimestepsThisBatch
```

1. Average Return Figure:  

<img src="data/02_iv_plots/AverageReturn.png" width="60%"/>

2. Eposide Length Mean:  

<img src="data/02_iv_plots/EpLenMean.png" width="60%"/>

3. Standard Deviation Return:  

<img src="data/02_iv_plots/StdReturn.png" width="60%"/>

4. Time Steps used is this batch:  

<img src="data/02_iv_plots/TimestepsThisBatch.png" width="60%"/>

### 7 More Complex Tasks

#### 1. [Problem 7] LunarLander - Discrete Action Space

```bash
pip install box2d-py # install to support after swig installed
python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005
```

#### 2. [Problem 8] HalfCheetah - Discrete Action Space