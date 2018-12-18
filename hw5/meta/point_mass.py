import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    goals are sampled randomly from a square
    """

    def __init__(self, num_tasks=1):
        self._test_train_shift = False
        self._skew_to_train = True
        self.reset_task()
        self.reset()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    @property
    def skew_to_train(self):
        return self._skew_to_train

    @skew_to_train.setter
    def skew_to_train(self, skew_to_train_val: bool):
        if self._skew_to_train != skew_to_train_val:
            self._skew_to_train = skew_to_train_val

    @property
    def test_train_shift(self):
        return self._test_train_shift

    @test_train_shift.setter
    def test_train_shift(self, test_train_shift_val: bool):
        if self._test_train_shift != test_train_shift_val:
            self._test_train_shift = test_train_shift_val

    def reset_task(self, is_evaluation=False):
        '''
        sample a new task randomly

        Problem 3: make training and evaluation goals disjoint sets
        if `is_evaluation` is true, sample from the evaluation set,
        otherwise sample from the training set
        '''
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        #====================================================================================#
        # YOUR CODE HERE
        low, high = -10, 10
        if not self._test_train_shift:
            # if typical case, don't have test/train distribution shifting case
            cutoff = low + (high - low) // 2
            if is_evaluation:
                # for testing set : [0, 10)
                x = np.random.uniform(cutoff, high)
                y = np.random.uniform(cutoff, high)
            else:
                # for training set : [-10, 0)
                x = np.random.uniform(low, cutoff)
                y = np.random.uniform(low, cutoff)
        else:
            # test/train distribution shifting case
            # raise NotImplementedError
            upper_bound = 100.
            d1 = np.random.uniform(1., upper_bound)
            d2 = np.random.uniform(-upper_bound, -1.)
            x = np.random.uniform(low, high)
            if not self._skew_to_train:
                # if not skew to train, means testing set will larger boundary
                if is_evaluation:
                    # for testing set : not in the lower bound => linear geometry
                    if x <= 0:
                        y = np.random.uniform(max(low, d1*x), high)
                    else:
                        y = np.random.uniform(max(low, d2*x), high)
                else:
                    # for training set : in the lower bound
                    y = np.random.uniform(low, min([0, d1*x, d2*x]))
            else:
                if is_evaluation:
                    # for testing set : in the lower bound
                    y = np.random.uniform(low, min([0, d1*x, d2*x]))
                else:
                    # for training set : not in the lower bound => linear geometry
                    if x <= 0:
                        y = np.random.uniform(max(low, d1*x), high)
                    else:
                        y = np.random.uniform(max(low, d2*x), high)

        self._goal = np.array([x, y])

    def get_all_task_idx(self):
        return [0]

    def reset(self):
        self._state = np.array([0, 0], dtype=np.float32)
        return self._get_obs()

    def _get_obs(self):
        return np.copy(self._state)

    def reward_function(self, x, y):
        return - (x ** 2 + y ** 2) ** 0.5

    def step(self, action):
        x, y = self._state
        # compute reward, add penalty for large actions instead of clipping them
        x -= self._goal[0]
        y -= self._goal[1]
        # check if task is complete
        done = abs(x) < .01 and abs(y) < .01
        reward = self.reward_function(x, y)
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
