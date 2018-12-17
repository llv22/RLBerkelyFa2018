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
python train_policy.py 'pm-obs' --exp_name pm_obs_average_return --history 1 -lr 5e-5 -n 100 --num_tasks 4
```
* Debugging for issue of average reward not -50
```bash
# need to use debugger port not so big, 8000 will be OK
tensorboard --logdir /tmp/logdir --host localhost --port 50000 --debugger_port 8000
python train_policy.py 'pm-obs' --exp_name pm_obs_average_return --history 1 -lr 5e-5 -n 100 --num_tasks 4 -debug True
```

### 2.2 Problem 2
### 2.3 Problem 3
