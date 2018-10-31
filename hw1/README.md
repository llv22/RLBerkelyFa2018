# CS294-112 HW 1: Imitation Learning

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**

Once Python **3.5** is installed, you can install the remaining dependencies using `pip install -r requirements.txt`.

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

**Note**: Students enrolled in the course will receive an email with their MuJoCo activation key. Please do **not** share this key.

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v2.pkl
* HalfCheetah-v2.pkl
* Hopper-v2.pkl
* Humanoid-v2.pkl
* Reacher-v2.pkl
* Walker2d-v2.pkl

The name of the pickle file corresponds to the name of the gym environment.

## Assignment
### 1. Installation

* Guidline - https://github.com/openai/mujoco-py/blob/master/README.md#requirements  
* Installation steps
```bash
mkdir ~/.mujoco
mv <your folder> ~/.mujoco/mjpro150
put mjkey.txt into ~/.mujoco/mjkey.txt
```

### 2. Exercise
#### 2.1 [Section 2] Behavior Cloning
##### Part 1
* Generation of expert behaviors $o_t$ and $a_t$
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 500 --only_expert_generate 0
```
Reward performance analysis:
1. average reward: $\frac{\sum_{t} r_{t}}{T}$ = 10357.515730550287, suppose $\left\lVert t \right\rVert = T$
2. standard deviation of reward: $\sigma(r_{t})$ = 686.9833818443245
function [expert_policy_out] in 266.8122 s

* Behavior Cloning
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 1 --only_expert_generate 1
```

##### Part 2
* Generation of expert behaviors $o_t$ and $a_t$
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 500 --only_expert_generate 0
```
mean return 10347.679984145881
std of return 721.6504464500745
function [expert_policy_out] in 669.2001 s

* Behavior Cloning [BC epochs=10]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 1
```
Epoch=0, loss=0.11469971
Epoch=5, loss=0.027275419
Final Epoch=9, loss=0.019465791
mean return 7647.900418809727
std of return 3303.701292393463
function [bc_policy_estimate] in  86.9479 s

```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 100 --only_expert_generate 1
```
Epoch=0, loss=0.11426015
Epoch=5, loss=0.027157418
Final Epoch=9, loss=0.019502923
mean return 4349.717127544281
std of return 3184.253907784524
function [bc_policy_estimate] in  97.6978 s

* Behavior Cloning [BC epochs=60]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 1
```
Epoch=0, loss=0.115757085
Epoch=5, loss=0.027166951
Epoch=10, loss=0.018360632
Epoch=15, loss=0.01457547
Epoch=20, loss=0.012546841
Epoch=25, loss=0.011068988
Epoch=30, loss=0.01018011
Epoch=35, loss=0.009535846
Epoch=40, loss=0.00901775
Epoch=45, loss=0.008615256
Epoch=50, loss=0.0082463585
Epoch=55, loss=0.008115762
Final Epoch=59, loss=0.0077691893
mean return 10138.193242617175
std of return 1338.987515723146
function [bc_policy_estimate] in 253.3601 s

* Behavior Cloning [BC epochs=80, full gradient descent]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 1
```
Epoch=0, loss=1.8137274
Epoch=5, loss=0.44362494
Epoch=10, loss=0.3252377
Epoch=15, loss=0.26969045
Epoch=20, loss=0.23329984
Epoch=25, loss=0.21012451
Epoch=30, loss=0.19155973
Epoch=35, loss=0.17868455
Epoch=40, loss=0.16747464
Epoch=45, loss=0.15881208
Epoch=50, loss=0.15144132
Epoch=55, loss=0.14515686
Epoch=60, loss=0.13950676
Epoch=65, loss=0.13464123
Epoch=70, loss=0.13017614
Epoch=75, loss=0.12608676
Final Epoch=79, loss=0.123026684
mean return 318.2433393156144
std of return 73.9023671335878
function [bc_policy_estimate] in 287.5026 s

* Behavior Cloning [BC epochs=80, 1024 batch gradient descent]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 1
```
Epoch=0, loss=0.11540921
Epoch=5, loss=0.027249534
Epoch=10, loss=0.018459626
Epoch=15, loss=0.0145938685
Epoch=20, loss=0.012500723
Epoch=25, loss=0.01119253
Epoch=30, loss=0.010436625
Epoch=35, loss=0.009746236
Epoch=40, loss=0.009210576
Epoch=45, loss=0.008742314
Epoch=50, loss=0.008551183
Epoch=55, loss=0.00812455
Epoch=60, loss=0.007905282
Epoch=65, loss=0.0077865347
Epoch=70, loss=0.007558631
Epoch=75, loss=0.007488649
Final Epoch=79, loss=0.0072339475
mean return 10176.550300507024
std of return 1367.6098174785814
function [bc_policy_estimate] in 309.3935 s

* Behavior Cloning [BC epochs=100, 1024 batch gradient descent]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 1
```
Epoch=0, loss=0.11540921
Epoch=5, loss=0.027249534
Epoch=10, loss=0.018459626
Epoch=15, loss=0.0145938685
Epoch=20, loss=0.012500723
Epoch=25, loss=0.01119253
Epoch=30, loss=0.010436625
Epoch=35, loss=0.009746236
Epoch=40, loss=0.009210576
Epoch=45, loss=0.008742314
Epoch=50, loss=0.008551183
Epoch=55, loss=0.00812455
Epoch=60, loss=0.007905282
Epoch=65, loss=0.0077865347
Epoch=70, loss=0.007558631
Epoch=75, loss=0.007488649
Final Epoch=79, loss=0.0072339475
mean return 10176.550300507024
std of return 1367.6098174785814
function [bc_policy_estimate] in 309.3935 s

* Behavior Cloning [BC epochs=80, full gradient descent, iteration=5 for full gradient descent]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 1
```
Epoch=0, loss=0.53019905
Epoch=5, loss=0.19239265
Epoch=10, loss=0.14397344
Epoch=15, loss=0.12132259
Epoch=20, loss=0.10594936
Epoch=25, loss=0.09446361
Epoch=30, loss=0.08573013
Epoch=35, loss=0.07880429
Epoch=40, loss=0.07269504
Epoch=45, loss=0.068423145
Epoch=50, loss=0.06358309
Epoch=55, loss=0.06048623
Epoch=60, loss=0.05686547
Epoch=65, loss=0.054783233
Epoch=70, loss=0.05154618
Epoch=75, loss=0.049049344
Final Epoch=79, loss=0.048039272
mean return 529.7355482636622
std of return 127.12297541557157
function [bc_policy_estimate] in 1390.5618 s

* Behavior Cloning [BC epochs=100, 2048 gradient descent]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 1
```
Epoch=0, loss=0.14715466
Epoch=5, loss=0.032662433
Epoch=10, loss=0.021775546
Epoch=15, loss=0.016877564
Epoch=20, loss=0.014108729
Epoch=25, loss=0.012313774
Epoch=30, loss=0.011078215
Epoch=35, loss=0.010290821
Epoch=40, loss=0.009271965
Epoch=45, loss=0.008937109
Epoch=50, loss=0.008514404
Epoch=55, loss=0.00820086
Epoch=60, loss=0.0077426825
Epoch=65, loss=0.007561854
Epoch=70, loss=0.0073805456
Epoch=75, loss=0.0071492125
Final Epoch=79, loss=0.0071405536
mean return 9985.341163047473
std of return 1806.3608494965674
function [bc_policy_estimate] in 266.3633 s

* Dagger [epochs=60, 1024 batch gradient descent]
```bash
python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 2
```
time 1: 
mean return 10267.833819902753
std of return 778.7349450080611
function [dagger_policy_estimate] in 6001.7666 s

time 2: 
mean return 10377.02161569598
std of return 61.91716123687015
function [dagger_policy_estimate] in 6263.4127 s