# CS294-112 HW 5c: Meta-Learning

Dependencies:
 * Python **3.5**
 * Numpy version 1.14.5
 * TensorFlow version 1.10.5
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.5**
 * seaborn
 * Box2D==2.3.2

See the [HW5c PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5c.pdf) for further instructions.

## 1. Implementation
### Problem 1: Context as Task ID
In point_mass_observed.py, state is fixed. In order to **augment the observation with a one-hot vector encoding the task ID**, do as follow:
* Change the dimension of the observation space, Line 29
```python
# Line 29
self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2+self._num_tasks,))
```
* Augment the observation with a one-hot vector that encodes the task ID, Line 61 and Line 33
```python
## Line 33
def onehot(num_task, task_id):
    """[using num_tasks and task_id to generate onehot vector for task identifier]
    
    Arguments:
        num_task {[int]} -- [number of task]
        task_id {[int]} -- [task identifier]
    """
    onehot_vector = np.zeros(num_task)
    onehot_vector[task_id] = 1
    return onehot_vector

idx = np.random.choice(len(self.tasks))
# task id from target scope
self._task = self.tasks[idx]
self._task_onehot = onehot(self._num_tasks, self._task)

## Line 61
return np.concatenate(np.copy(self._state), self._task_onehot)
```

### Problem 2: Meta-Learned Context

### Problem 3: Generalization

## 2. Deliverables
### 2.1 Problem 1 
```bash
# training with only Multilayer Perceptron for predicting result
python train_policy.py 'pm-obs' --exp_name pm_obs_average_return --history 1 -lr 5e-5 -n 100 --num_tasks 4
# result in data/pm_obs_average_return_pm-obs_17-12-2018_21-37-03
```
* Debugging for issue of average reward not -50
```bash
# need to use debugger port not so big, 8000 will be OK
tensorboard --logdir /tmp/logdir --host localhost --port 50000 --debugger_port 8000
python train_policy.py 'pm-obs' --exp_name pm_obs_average_return --history 1 -lr 5e-5 -n 100 --num_tasks 4 -debug True
```
* Result analysis
```bash
python plot.py data/pm_obs_average_return_pm-obs_17-12-2018_21-37-03 --legend pm-obs --value AverageReturn FinalReward StdReturn
```
1. AverageReturn Figure:  

<img src="data/Problem1/AverageReturn.png" width="60%"/>

2. FinalReward Mean:   

<img src="data/Problem1/FinalReward.png" width="60%"/>

3. StdReturn Mean:   

<img src="data/Problem1/StdReturn.png" width="60%"/>

### 2.2 Problem 2

* For single thread process [Too slow, then use multithread to speed-up]
```bash
## 1. feed-forward neural network
python train_policy.py 'pm' --exp_name pm_mlp_history100 --history 100 --discount 0.90 -lr 5e-4 -n 60
# result in data/pm_mlp_history100_pm_18-12-2018_13-05-16

## 2. recurrent neural network [Too slow, just skip then run when back to home]
python train_policy.py 'pm' --exp_name pm_recurrent_history100 --history 100 --discount 0.90 -lr 5e-4 -n 60 -rec
# result in 
```
* For multi-thread process
```bash
## 1. feed-forward neural network
python train_policy.py 'pm' --exp_name pm_mlp_history100_tnum4 --history 100 --discount 0.90 -lr 5e-4 -n 60 -tnum 4
# result in data/pm_mlp_history100_tnum4_pm_18-12-2018_14-42-13

## 2. recurrent neural network
python train_policy.py 'pm' --exp_name pm_recurrent_history100_tnum4 --history 100 --discount 0.90 -lr 5e-4 -n 60  -tnum 4 -rec
# result in 
```

* Result analysis
```bash
python plot.py data/pm_mlp_history100_tnum4_pm_18-12-2018_14-42-13 --legend pm_mlp_his100_tnum4 pm_recurrent_his100_tnum4 --value AverageReturn FinalReward StdReturn
```
1. AverageReturn Figure:  

<img src="data/Problem1/AverageReturn.png" width="60%"/>

2. FinalReward Mean:   

<img src="data/Problem1/FinalReward.png" width="60%"/>

3. StdReturn Mean:   

<img src="data/Problem1/StdReturn.png" width="60%"/>

### 2.3 Problem 3
