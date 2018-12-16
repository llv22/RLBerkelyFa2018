import numpy as np
from gym import spaces
from gym import Env


class ObservedPointEnv(Env):
    """
    point mass on a 2-D plane
    four tasks: move to (-10, -10), (-10, 10), (10, -10), (10, 10)

    Problem 1: augment the observation with a one-hot vector encoding the task ID
     - change the dimension of the observation space
     - augment the observation with a one-hot vector that encodes the task ID
    """
    #====================================================================================#
    #                           ----------PROBLEM 1----------
    #====================================================================================#
    # YOUR CODE SOMEWHERE HERE
    def __init__(self, num_tasks=1):
        self.tasks = [0, 1, 2, 3][:num_tasks]
        # record current task num
        self._num_tasks = num_tasks
        self.reset_task()
        self.reset()

        ## Problem 1
        # Your Code
        # change the dimension of the observation space => state keep the same, but obs = [state, onehot for task ID]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2+self._num_tasks,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, is_evaluation=False):
        idx = np.random.choice(len(self.tasks))
        # task id from target scope
        self._task = self.tasks[idx]
        goals = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
        self._goal = np.array(goals[idx])*10

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        """[return current obs space, [2+self._num_tasks] just for encoding obs space]
        
        Returns:
            [np.array] -- [concatnate state+onehot for task ID]
        """
        return np.copy(self._state)

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        # check if task is complete
        done = abs(x) < 0.01 and abs(y) < 0.01
        # move to next state
        self._state = self._state + action
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def seed(self, seed):
        np.random.seed = seed
