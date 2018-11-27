import tensorflow as tf
import numpy as np

import utils



def f(tuple_data):
    """
    extract data from tuple_data
    """
    i, actions_sequences, action_slice_size, _action_dim, _reuse, _dynamics_func, _cost_fn, _horizon, state_ph = tuple_data
    cost = 0
    for j in range(_horizon):
        # issue 1: slice api begin+size array
        # issue 2: in order to make state_ph (?,20) and action_for_state_i (6) has the same shape, need to make tf.reshape(action_for_state_i, shape=[-1,6])
        action_for_state_i = tf.reshape(tf.squeeze(tf.slice(actions_sequences, [i,j,0], action_slice_size)), shape=[-1, _action_dim])
        if j == 0:
            next_state_pred = _dynamics_func(state_ph, action_for_state_i, _reuse)
            cost += _cost_fn(state_ph, action_for_state_i, next_state_pred)
        else:
            s0 = next_state_pred
            next_state_pred = _dynamics_func(s0, action_for_state_i, _reuse)
            cost += _cost_fn(s0, action_for_state_i, next_state_pred)
    return cost

def threadf(i, actions_sequences, action_slice_size, _action_dim, _reuse, _dynamics_func, _cost_fn, _horizon, state_ph):
    """
    extract data for ThreadPoolExecutor
    """
    cost = 0
    for j in range(_horizon):
        # issue 1: slice api begin+size array
        # issue 2: in order to make state_ph (?,20) and action_for_state_i (6) has the same shape, need to make tf.reshape(action_for_state_i, shape=[-1,6])
        action_for_state_i = tf.reshape(tf.squeeze(tf.slice(actions_sequences, [i,j,0], action_slice_size)), shape=[-1, _action_dim])
        if j == 0:
            next_state_pred = _dynamics_func(state_ph, action_for_state_i, _reuse)
            cost += _cost_fn(state_ph, action_for_state_i, next_state_pred)
        else:
            s0 = next_state_pred
            next_state_pred = _dynamics_func(s0, action_for_state_i, _reuse)
            cost += _cost_fn(s0, action_for_state_i, next_state_pred)
    return cost

class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1,
                 use_cross_entropy=False,
                 steps_for_loss_train=1):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._init_dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3
        # if we can use cross-entropy method for action sampling
        self._use_cross_entropy = use_cross_entropy
        # loss steps for training
        self._steps_for_loss_train = steps_for_loss_train
        # only the first time is False, then using True for next time
        self._reuse = True

        self._sess, self._state_ph, self._action_ph, self._next_state_ph,\
            self._next_state_pred, self._loss, self._optimizer, self._best_action = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        state_ph = tf.placeholder(tf.float32, shape=[None, self._state_dim])
        next_state_ph = tf.placeholder(tf.float32, shape=[None, self._state_dim])
        ### Extra Credit(ii)
        # raise NotImplementedError
        # reformat action for At to At, ..., At+n-1
        action_ph = tf.placeholder(tf.float32, shape=[None, self._action_dim * self._steps_for_loss_train])
        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

            Extra Credit(ii):
                (a) St, At, ..., At+n-1, St+n, just change At to At, ..., At+n-1
                (b) need to reorganize the source code
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ## (a) normalize state and action
        normalized_state = utils.normalize(state, self._init_dataset.state_mean, self._init_dataset.state_std)
        normalized_action = utils.normalize(action, self._init_dataset.action_mean, self._init_dataset.action_std)
        ## (b) concatenate for normalized state and action and 
        ### Extra Credit(ii)
        # raise NotImplementedError
        normalized_state_action = tf.concat([normalized_state, normalized_action], axis=-1)
        ## (c) via neural network to build the normalized predicted difference between the next state and the current state
        normalized_next_state_diff_pred = utils.build_mlp(normalized_state_action, self._state_dim, scope="f_func", reuse=reuse)
        ## (d) Unnormalize the delta state prediction, add to current state to get next_state_pred
        next_state_pred = utils.unnormalize(normalized_next_state_diff_pred, self._init_dataset.state_mean, self._init_dataset.state_std) + state

        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

            Extra Credit(ii):
                (a) state_ph = St, next_state_ph=St+n
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ## (a) and (b) normalized actual and predicted state difference
        normalized_actual_state_diff = utils.normalize(state_ph - next_state_ph, self._init_dataset.state_mean, self._init_dataset.state_std)
        normalized_pred_state_diff = utils.normalize(state_ph - next_state_pred, self._init_dataset.state_mean, self._init_dataset.state_std)
        ## (c) loss for MSE of actual and predicted state difference
        loss = tf.losses.mean_squared_error(normalized_actual_state_diff, normalized_pred_state_diff)
        ## (d) optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(loss)

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            PROBLEM 2 implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences
            
            Extra Credit (i) implementation details:
                (a) Use cross-entropy method to sample action, instead of pure random_uniform
                (b) Still retrieve the best action for returning reference

            Extra Credit (ii) implementation details:
                (a) action convert to different dimension - self._action_dim to self._action_dim * self._steps_for_loss_train
        """
        if not self._use_cross_entropy:
            ### PROBLEM 2
            ### YOUR CODE HERE
            # raise NotImplementedError
            ## (b) Randomly sample action sequences = [self._num_random_action_selection, self._horizon]
            actions_sequences = tf.random_uniform([self._num_random_action_selection, self._horizon, self._action_dim * self._steps_for_loss_train], np.tile(self._action_space_low, self._steps_for_loss_train), np.tile(self._action_space_high, self._steps_for_loss_train), tf.float32)
            action_slice_size = [1,1,self._action_dim * self._steps_for_loss_train]

            ## 1. Parallel Model for cost_actions_decision : Not yest tested
            # Result: Can't be implemented easily, as processing isn't easy for tensorflow model for copyied
            # import multiprocessing
            # from tqdm import tqdm
            # cost_actions_decision = [None] * self._num_random_action_selection
            # with multiprocessing.Pool(12) as pool:
            #     # merge parameter as tuple
            #     args = [(i, actions_sequences, action_slice_size, self._action_dim, self._reuse, self._dynamics_func, self._cost_fn, self._horizon, state_ph) for i in range(self._num_random_action_selection)]
            #     with tqdm(pool.imap_unordered(f, args), total=self._num_random_action_selection) as pbar:
            #         for i, cost in pbar:
            #             cost_actions_decision[i] = cost

            ## 2. ThreadPool for cost_actions_decision
            # from concurrent.futures import ThreadPoolExecutor
            # cost_actions_decision = [None] * self._num_random_action_selection
            # with ThreadPoolExecutor(max_workers=12) as executor:
            #     for i in range(self._num_random_action_selection):
            #         future = executor.submit(threadf, i, actions_sequences, action_slice_size, self._action_dim, self._reuse, self._dynamics_func, self._cost_fn, self._horizon, state_ph)
            #         cost_actions_decision[i] = future.result()

            ## 3.1 Sequential Model for cost_actions_decision
            # Status: quite slow to run, as loop will be repeated for about self._num_random_action_selection times
            # Comments: will try to parallelize for [state_ph] * self._num_random_action_selection
            # cost_actions_decision = []
            # for i in range(self._num_random_action_selection):
            #     cost = 0
            #     for j in range(self._horizon):
            #         # issue 1: slice api begin+size array
            #         # issue 2: in order to make state_ph (?,20) and action_for_state_i (6) has the same shape, need to make tf.reshape(action_for_state_i, shape=[-1,6])
            #         action_for_state_i = tf.reshape(tf.squeeze(tf.slice(actions_sequences, [i,j,0], action_slice_size)), shape=[-1, self._action_dim])
            #         if j == 0:
            #             next_state_pred = self._dynamics_func(state_ph, action_for_state_i, self._reuse)
            #             cost += self._cost_fn(state_ph, action_for_state_i, next_state_pred)
            #         else:
            #             s0 = next_state_pred
            #             next_state_pred = self._dynamics_func(s0, action_for_state_i, self._reuse)
            #             cost += self._cost_fn(s0, action_for_state_i, next_state_pred)
            #     cost_actions_decision.append(cost)

            ## 3.2 Parallel rollout for [state_ph] * self._num_random_action_selection about self._horizon time for each rollout
            # Status: passed
            state_ph_sequences = tf.reshape([state_ph] * self._num_random_action_selection, shape=[-1, self._state_dim])
            cost_actions_decision = [] * self._num_random_action_selection
            for i in range(self._horizon):
                # vectorized, now only self._horizon = 10 will be sufficient
                next_state_pred_ph_sequences = self._dynamics_func(state_ph_sequences, actions_sequences[:,i], self._reuse)
                if i == 0:
                    cost_actions_decision = self._cost_fn(state_ph_sequences, actions_sequences[:,i], next_state_pred_ph_sequences)
                else:
                    cost_actions_decision = cost_actions_decision + self._cost_fn(state_ph_sequences, actions_sequences[:,i], next_state_pred_ph_sequences)
                state_ph_sequences = next_state_pred_ph_sequences
            best_action_index = tf.argmin(tf.convert_to_tensor(cost_actions_decision))
            best_action = tf.squeeze(tf.slice(actions_sequences, [best_action_index, 0, 0], action_slice_size))
        else:
            ### PROBLEM Extra Credit (i)
            ### YOUR CODE HERE
            # raise NotImplementedError
            ## Theory in https://en.wikipedia.org/wiki/Cross-entropy_method 
            ## refer to implementation of CEM in https://github.com/udacity/deep-reinforcement-learning/blob/master/cross-entropy/CEM.ipynb
            # In order to simplify logic, just use uniform distribution for sampling data, continue to estimate lower and upper bound via top 80% data for the lower bound and upper bound
            action0_sequences = tf.random_uniform([self._num_random_action_selection, self._action_dim * self._steps_for_loss_train], np.tile(self._action_space_low, self._steps_for_loss_train), np.tile(self._action_space_high, self._steps_for_loss_train), tf.float32)
            current_action_sequences = action0_sequences
            action_slice_size = [1,self._action_dim * self._steps_for_loss_train]
            state_ph_sequences = tf.reshape([state_ph] * self._num_random_action_selection, shape=[-1, self._state_dim])
            cost_actions_decision = [] * self._num_random_action_selection
            for i in range(self._horizon):
                next_state_pred_ph_sequences = self._dynamics_func(state_ph_sequences, current_action_sequences, self._reuse)
                one_step_rewards = self._cost_fn(state_ph_sequences, current_action_sequences, next_state_pred_ph_sequences)
                if i == 0:
                    cost_actions_decision = one_step_rewards
                else:
                    cost_actions_decision += one_step_rewards
                # use cross-entropy method to update action0_sequences, tf.math.top_k doesn't exist => tf.nn.top_k
                top80_values, top80_indices = tf.nn.top_k(one_step_rewards, int(self._num_random_action_selection * 0.95))
                current_action_sequences = tf.random_uniform([self._num_random_action_selection, self._action_dim * self._steps_for_loss_train], tf.minimum(np.tile(self._action_space_low, self._steps_for_loss_train), top80_values[-1]), tf.maximum(np.tile(self._action_space_high, self._steps_for_loss_train), top80_values[0]), tf.float32)
            best_action_index = tf.argmin(tf.convert_to_tensor(cost_actions_decision))
            best_action = tf.squeeze(tf.slice(action0_sequences, [best_action_index, 0], action_slice_size))
        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        sess = tf.Session()
        state_ph, action_ph, next_state_ph = self._setup_placeholders()

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ## Just 1 step for loss training, action_ph is 1-actions
        # for the first time to call reuse=False?
        # 
        ### PROBLEM Extra Credit (ii)
        ### YOUR CODE HERE
        #   We naturally convert At to At, At+1, ..., At+n-1 from input data part
        ### For multiple step for loss training, for multiple step (N-actions estimation), action_ph is N-actions
        next_state_pred = self._dynamics_func(state_ph, action_ph, not self._reuse)
        loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)

        ### PROBLEM 2
        ### YOUR CODE HERE
        best_action = self._setup_action_selection(state_ph)

        sess.run(tf.global_variables_initializer())

        return sess, state_ph, action_ph, next_state_ph, \
                next_state_pred, loss, optimizer, best_action

    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        ### PROBLEM Extra Credit (ii)
        ### YOUR CODE HERE
        #   We naturally convert At to At, At+1, ..., At+n-1 from input data part
        _, loss = self._sess.run([self._optimizer, self._loss], feed_dict={
            self._state_ph: np.array(states).reshape(-1, self._state_dim),
            self._action_ph: np.array(actions).reshape(-1, self._action_dim * self._steps_for_loss_train),
            self._next_state_ph: np.array(next_states).reshape(-1, self._state_dim)
        })

        return loss

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)

        ### PROBLEM 1
        ### YOUR CODE HERE
        # raise NotImplementedError
        # refer to https://stackoverflow.com/questions/18200052/how-to-convert-ndarray-to-array
        ### PROBLEM Extra Credit (ii)
        ### YOUR CODE HERE
        #   We naturally convert At to At, At+1, ..., At+n-1 from input data part
        next_state_pred = self._sess.run(self._next_state_pred, feed_dict={
            self._state_ph: np.array(state).reshape(-1, self._state_dim),
            self._action_ph: np.array(action).reshape(-1, self._action_dim * self._steps_for_loss_train)
        }).ravel()

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)

        ### PROBLEM 2
        ### YOUR CODE HERE
        # raise NotImplementedError
        best_action = self._sess.run(self._best_action, feed_dict={
            self._state_ph: np.array(state).reshape(-1, self._state_dim)
        }).ravel()

        assert np.shape(best_action) == (self._action_dim * self._steps_for_loss_train,)
        return best_action
