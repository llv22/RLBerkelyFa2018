import numpy as np
import tensorflow as tf

from logger import logger


############
### Data ###
############
def inSameSlice(start_idx, end_idx, slice_end_indices):
    """[check if (start_idx, end_indx) is in the same slice provied by slice_end_indices]
    
    Arguments:
        start_idx {[type]} -- [state start index]
        end_idx {[type]} -- [state end index]
        slice_end_indices {[type]} -- [end state tuples]
    """
    def firstLessThan(val, arr):
        """[binary search for index, whose value is the first element larger than val in arr]
        
        Arguments:
            val {int} -- [target value]
            arr {[int]} -- [int array with index]
        """
        l = 0; r = len(arr) - 1
        while l + 1 < r:
            m = l + (r-l) // 2
            if val > arr[m]:
                l = m
            else:
                # val <= arr[m], move to r part
                r = m
        # l + 1 == r
        if arr[l] >= val:
            return l
        elif arr[r] >= val:
            return r
        else:
            return -1

    return firstLessThan(start_idx, slice_end_indices) == firstLessThan(end_idx, slice_end_indices)

class Dataset(object):

    def __init__(self, steps_for_loss_train=1):
        """[using steps_for_loss_train to construct DataSet]
        
        Arguments:
            steps_for_loss_train {[int]} -- [how many action will form a real action for Neural Network]
        """

        self._states = []
        self._actions = []
        self._next_states = []
        self._rewards = []
        self._dones = []
        self._steps_for_loss_train = steps_for_loss_train

    @property
    def is_empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._states)

    ##################
    ### Statistics ###
    ##################

    @property
    def state_mean(self):
        return np.mean(self._states, axis=0)

    @property
    def state_std(self):
        return np.std(self._states, axis=0)

    @property
    def action_mean(self):
        if self._steps_for_loss_train == 1:
            return np.mean(self._actions, axis=0)
        else:
            nsteps_actions = np.asarray([np.asarray(self._actions[i:i+self._steps_for_loss_train]).reshape(-1, len(self._actions[0])*self._steps_for_loss_train)[0] for i in range(len(self._actions) - self._steps_for_loss_train)])
            assert nsteps_actions.shape[-1] == len(self._actions[0])*self._steps_for_loss_train
            return np.mean(nsteps_actions, axis=0)

    @property
    def action_std(self):
        if self._steps_for_loss_train == 1:
            return np.std(self._actions, axis=0)
        else:
            nsteps_actions = np.asarray([np.asarray(self._actions[i:i+self._steps_for_loss_train]).reshape(-1, len(self._actions[0])*self._steps_for_loss_train)[0] for i in range(len(self._actions) - self._steps_for_loss_train)])
            assert nsteps_actions.shape[-1] == len(self._actions[0])*self._steps_for_loss_train
            return np.std(nsteps_actions, axis=0)

    @property
    def delta_state_mean(self):
        return np.mean(np.array(self._next_states) - np.array(self._states), axis=0)

    @property
    def delta_state_std(self):
        return np.std(np.array(self._next_states) - np.array(self._states), axis=0)

    ###################
    ### Adding data ###
    ###################

    def add(self, state, action, next_state, reward, done):
        """
        Add (s, a, r, s') to this dataset
        """
        if not self.is_empty:
            # ensure the state, action, next_state are of the same dimension
            assert len(self._states[-1]) == len(np.ravel(state))
            assert len(self._actions[-1]) == len(np.ravel(action))
            assert len(self._next_states[-1]) == len(np.ravel(next_state))

        self._states.append(np.ravel(state))
        self._actions.append(np.ravel(action))
        self._next_states.append(np.ravel(next_state))
        self._rewards.append(reward)
        self._dones.append(done)

    def append(self, other_dataset):
        """
        Append other_dataset to this dataset
        """
        if not self.is_empty and not other_dataset.is_empty:
            # ensure the state, action, next_state are of the same dimension
            assert len(self._states[-1]) == len(other_dataset._states[-1])
            assert len(self._actions[-1]) == len(other_dataset._actions[-1])
            assert len(self._next_states[-1]) == len(other_dataset._next_states[-1])

        self._states += other_dataset._states
        self._actions += other_dataset._actions
        self._next_states += other_dataset._next_states
        self._rewards += other_dataset._rewards
        self._dones += other_dataset._dones

    ############################
    ### Iterate through data ###
    ############################

    def rollout_iterator(self):
        """
        Iterate through all the rollouts in the dataset sequentially
        """
        if self._steps_for_loss_train == 1:
            end_indices = np.nonzero(self._dones)[0] + 1

            states = np.asarray(self._states)
            actions = np.asarray(self._actions)
            next_states = np.asarray(self._next_states)
            rewards = np.asarray(self._rewards)
            dones = np.asarray(self._dones)

            start_idx = 0
            for end_idx in end_indices:
                indices = np.arange(start_idx, end_idx)
                yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]
                start_idx = end_idx
        else:
            ### PROBLEM Extra Credit (ii)
            ### YOUR CODE HERE
            #   We naturally convert At to At, At+1, ..., At+n-1 from input data part

            slice_indices = np.nonzero(self._dones)[0] + self._steps_for_loss_train
            slice_indices = [slice_index for slice_index in slice_indices if slice_index < len(self._dones)]
            slice_end_indices = np.nonzero(self._dones)[0]

            states = np.asarray(self._states)
            actions = np.asarray(self._actions)
            next_states = np.asarray(self._next_states)
            rewards = np.asarray(self._rewards)
            dones = np.asarray(self._dones)

            start_idx = 0
            for end_idx in slice_indices:
                # start_idx and end_idx must in the saome slice of end_indices
                if inSameSlice(start_idx, end_idx, slice_end_indices):
                    indices = np.arange(start_idx, end_idx)
                    yield states[np.arange(start_idx, start_idx+1)], actions[indices].reshape(-1), next_states[np.arange(end_idx, end_idx+1)], np.sum(rewards[indices]), dones[np.arange(end_idx, end_idx+1)]
                    start_idx = end_idx


    def random_iterator(self, batch_size):
        """
        Iterate once through all (s, a, r, s') in batches in a random order
        """

        if self._steps_for_loss_train == 1:
            all_indices = np.nonzero(np.logical_not(self._dones))[0]
            np.random.shuffle(all_indices)

            states = np.asarray(self._states)
            actions = np.asarray(self._actions)
            next_states = np.asarray(self._next_states)
            rewards = np.asarray(self._rewards)
            dones = np.asarray(self._dones)

            i = 0
            while i < len(all_indices):
                indices = all_indices[i:i+batch_size]

                yield states[indices], actions[indices], next_states[indices], rewards[indices], dones[indices]

                i += batch_size
        else:
            ### PROBLEM Extra Credit (ii)
            ### YOUR CODE HERE
            #   We naturally convert At to At, At+1, ..., At+n-1 from input data part
            all_indices = np.nonzero(np.logical_not(self._dones))[0]
            slice_end_indices = np.nonzero(self._dones)[0]
            all_valid_indices = [start_index for start_index in all_indices if start_index + self._steps_for_loss_train < len(self._dones) and inSameSlice(start_index, start_index + self._steps_for_loss_train, slice_end_indices)]

            np.random.shuffle(all_valid_indices)

            states = np.asarray(self._states)
            actions = np.asarray(self._actions)
            next_states = np.asarray(self._next_states)
            rewards = np.asarray(self._rewards)
            dones = np.asarray(self._dones)

            i = 0
            while i < len(all_valid_indices) and i+self._steps_for_loss_train+batch_size < len(all_valid_indices):
                state_indices = all_valid_indices[i:i+batch_size]
                next_state_indices = all_valid_indices[i+self._steps_for_loss_train: i+self._steps_for_loss_train+batch_size]
                # action from i:i+batch_size
                action_sel = [actions[np.arange(start, start+self._steps_for_loss_train)].reshape(-1) for start in state_indices]
                reward_sel = [np.sum(rewards[np.arange(start, start+self._steps_for_loss_train)]) for start in state_indices]

                yield states[state_indices], action_sel, next_states[next_state_indices], reward_sel, dones[next_state_indices]

                i += batch_size


    ###############
    ### Logging ###
    ###############

    def log(self):
        end_idxs = np.nonzero(self._dones)[0] + 1

        returns = []

        start_idx = 0
        for end_idx in end_idxs:
            rewards = self._rewards[start_idx:end_idx]
            returns.append(np.sum(rewards))

            start_idx = end_idx

        logger.record_tabular('ReturnAvg', np.mean(returns))
        logger.record_tabular('ReturnStd', np.std(returns))
        logger.record_tabular('ReturnMin', np.min(returns))
        logger.record_tabular('ReturnMax', np.max(returns))

##################
### Tensorflow ###
##################

def build_mlp(input_layer,
              output_dim,
              scope,
              n_layers=1,
              hidden_dim=500,
              activation=tf.nn.relu,
              output_activation=None,
              reuse=False):
    layer = input_layer
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            layer = tf.layers.dense(layer, hidden_dim, activation=activation)
        layer = tf.layers.dense(layer, output_dim, activation=output_activation)
    return layer

def normalize(x, mean, std, eps=1e-8):
    return (x - mean) / (std + eps)

def unnormalize(x, mean, std):
    return x * std + mean

################
### Policies ###
################

class RandomPolicy(object):

    def __init__(self, env, steps_for_loss_train=1):
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._steps_for_loss_train = steps_for_loss_train

    def get_action(self, state):
        return np.random.uniform(self._action_space_low, self._action_space_high)

