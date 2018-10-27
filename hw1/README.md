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
* Generation of expert behaviors $o_t$ and $a_t$
```bash
python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts 1
```
Performance Cost:
1. mean return 3781.212325396108
2. std of return 0.0