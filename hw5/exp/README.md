# CS294-112 HW 5a: Exploration

Dependencies:
 * Python **3.5**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * seaborn
 * tqdm==**4.26.0**

Before doing anything, first replace `gym/envs/mujoco/half_cheetah.py` with the provided `sparse_half_cheetah.py` file. It is always a good idea to keep a copy of the original `gym/envs/mujoco/half_cheetah.py` just in case you need it for something else.

You will implement `density_model.py`, `exploration.py`, and `train_ac_exploration_f18.py`.

See the hw5a.pdf in this folder for further instructions.
<!--See the [HW5 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5a.pdf) for further instructions-->

## 1. Preparaion of "sparse_half_cheetah.py" for SparseHalfCheetah-v1
   1) Customize ~/miniconda3/lib/python3.6/site-packages/gym/envs/__init__.py
   ```python
   ## Orlando's adding for hw5a of CS-Berkeley
   register(
       id='SparseHalfCheetah-v1',
       entry_point='gym.envs.mujoco:SparseHalfCheetahEnv',
       max_episode_steps=1000,
       reward_threshold=4800.0,
    )
   ```
   2) prepare 'SparseHalfCheetah-v1' and assert
   ```bash
   cp hw5/exp/sparse_half_cheetah.py ~/miniconda3/lib/python3.6/site-packages/gym/envs/mujoco/
   ```
   ```python
   # customize sparse_half_cheetah.py
   class SparseHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'sparse_half_cheetah.xml', 5)
        utils.EzPickle.__init__(self)
   ```
   ```bash
   cp assets/half_cheetah.xml assets/sparse_half_cheetah.xml
   ```
   ```python
   # customize ~/miniconda3/lib/python3.6/site-packages/gym/envs/mujoco/__init__.py
    from gym.envs.mujoco.mujoco_env import MujocoEnv
    # ^^^^^ so that user gets the correct error
    # message if mujoco is not installed correctly
    ...
    from gym.envs.mujoco.sparse_half_cheetah import SparseHalfCheetahEnv
    ...
   ```
   2) test if environment "SparseHalfCheetah-v1" is ready
   ```python
   import gym 
   env = gym.make("SparseHalfCheetah-v1") 
   ```
   3) check for "PointMass-v0"
   ```bash
   python pointmass.py test_pointmass
   ```