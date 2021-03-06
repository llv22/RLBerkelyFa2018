import os
import uuid
import time
import pickle
import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import inspect

## YOUR CODE - Problem 1.4 Evaluation, Question 1
import logz

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

class QLearner(object):

  def __init__(
    self,
    env,
    logdir,
    q_func,
    optimizer_spec,
    session,
    exploration=LinearSchedule(1000000, 0.1),
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    grad_norm_clipping=10,
    rew_file=None,
    double_q=True,
    lander=False):
    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    double_q: bool
        If True, then use double Q-learning to compute target values. Otherwise, use vanilla DQN.
        https://papers.nips.cc/paper/3964-double-q-learning.pdf
    """
    def setup_logger(logdir, locals_):
        """[setup local logger]
        
        Arguments:
            logdir {[str]} -- [logging directory]
            locals_ {[list]} -- [local variables]
        """
        # Configure output directory for logging
        logz.configure_output_dir(logdir)
        # Log experimental parameters
        args = inspect.getargspec(QLearner.__init__)[0]
        ## locals_ remove self, env, and unnecessary parameters setting
        locals_.pop("env")
        locals_.pop("self")
        locals_.pop("setup_logger")
        locals_.pop("exploration")
        locals_.pop("session")
        locals_.pop("stopping_criterion")
        locals_.pop("optimizer_spec")
        locals_.pop("q_func")
        ## append exp_name for locals_
        locals_["exp_name"] = env.spec.id
        params = {k: locals_[k] if k in locals_ else None for k in args}
        logz.save_params(params)
    
    assert logdir is not None
    setup_logger(logdir, locals())

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    self.target_update_freq = target_update_freq
    self.optimizer_spec = optimizer_spec
    self.batch_size = batch_size
    self.learning_freq = learning_freq
    self.learning_starts = learning_starts
    self.stopping_criterion = stopping_criterion
    self.env = env
    self.session = session 
    ## Equivalent to `with self.sess:` to avoid issue 
    #   Callstack: hw3/logz.py", line 81, in pickle_tf_vars
    #   "ValueError: Cannot evaluate tensor using `eval()`: No default session is registered. Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`"
    self.session.__enter__()
    self.exploration = exploration
    # put pkl of reward to logdir
    self.rew_file = os.path.join(logdir, str(uuid.uuid4()) + '.pkl' if rew_file is None else rew_file)
    # enable double-Q or not
    self.double_q = double_q

    ###############
    # BUILD MODEL #
    ###############

    if len(self.env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = self.env.observation_space.shape
    else:
        img_h, img_w, img_c = self.env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    self.num_actions = self.env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    ## Orlando's comments: smart for [None] + list(input_shape)
    self.obs_t_ph              = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    self.act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    self.rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    self.obs_tp1_ph            = tf.placeholder(
        tf.float32 if lander else tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    self.done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    if lander:
      obs_t_float = self.obs_t_ph
      obs_tp1_float = self.obs_tp1_ph
    else:
      obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
      obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0

    # Here, you should fill in your own code to compute the Bellman error. This requires
    # evaluating the current and next Q-values and constructing the corresponding error.
    # TensorFlow will differentiate this error for you, you just need to pass it to the
    # optimizer. See assignment text for details.
    # Your code should produce one scalar-valued tensor: total_error
    # This will be passed to the optimizer in the provided code below.
    # Your code should also produce two collections of variables:
    # q_func_vars
    # target_q_func_vars
    # These should hold all of the variables of the Q-function network and target network,
    # respectively. A convenient way to get these is to make use of TF's "scope" feature.
    # For example, you can create your Q-function network with the scope "q_func" like this:
    # <something> = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)
    # And then you can obtain the variables like this:
    # q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')
    # Older versions of TensorFlow may require using "VARIABLES" instead of "GLOBAL_VARIABLES"
    # Tip: use huber_loss (from dqn_utils) instead of squared error when defining self.total_error
    ######

    # YOUR CODE HERE - Problem 1.3 Implementation 
    ## 1. q_values network
    # qall_action_values.shape = (None, action_size), as self.qall_action_values will be used later
    qall_action_values = q_func(obs_t_float, self.num_actions, scope="q_func", reuse=False)
    self.max_action_for_qall = tf.argmax(qall_action_values, axis=-1)
    # select target action for q_values, filtered by self.act_t_ph
    q_action_values = tf.reduce_sum(qall_action_values * tf.one_hot(self.act_t_ph, self.num_actions), axis=-1)
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

    ## 2. target_q_value network
    # check if self.done_mask_ph == 1, then don't need to add gamma * tf.reduce_max(q_prime_values, axis=-1)
    q_prime_values = q_func(obs_tp1_float, self.num_actions, scope="target_q_func", reuse=False)
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')
    if self.double_q:
        # double-Q network
        # for max_action as index for input
        self.max_action_index_for_dQ = tf.placeholder(tf.int32,   [None])
        # q_prime_values * tf.one_hot(self.max_action_index_for_dQ, self.num_actions) is to mask "all non-max action index" for qall_action network.
        # tf.reduce_max(q_prime_values * tf.one_hot(self.max_action_index_for_dQ, self.num_actions), axis=-1) is to only first the max action index's Q value for target network of evaluated value
        # only need to input self.max_action_index_for_dQ with qall network will be sufficient for training and learning.
        y_values = self.rew_t_ph + gamma * (1 - self.done_mask_ph) * tf.reduce_max(q_prime_values * tf.one_hot(self.max_action_index_for_dQ, self.num_actions), axis=-1)
    else:
        # non-double-Q network
        y_values = self.rew_t_ph + gamma * (1 - self.done_mask_ph) * tf.reduce_max(q_prime_values, axis=-1)

    ## For double-Q network, it's very difficult to splitting for y_values, as argmax(qall_action_values) is needed to reuse "q_func" network.
    # for q_value and q_target_value's Bellman error
    self.total_error = huber_loss(q_action_values - y_values)

    ######

    # construct optimization op (with gradient clipping)
    self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
    self.train_fn = minimize_and_clip(optimizer, self.total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    self.update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    self.replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len, lander=lander)
    self.replay_buffer_idx = None

    ###############
    # RUN ENV     #
    ###############
    self.model_initialized = False
    self.num_param_updates = 0
    self.mean_episode_reward      = -float('nan')
    self.best_mean_episode_reward = -float('inf')
    self.last_obs = self.env.reset()
    self.log_every_n_steps = 10000

    self.start_time = None
    self.t = 0

  def stopping_criterion_met(self):
    return self.stopping_criterion is not None and self.stopping_criterion(self.env, self.t)

  def step_env(self):
    ### 2. Step the env and store the transition
    # At this point, "self.last_obs" contains the latest observation that was
    # recorded from the simulator. Here, your code needs to store this
    # observation and its outcome (reward, next observation, etc.) into
    # the replay buffer while stepping the simulator forward one step.
    # At the end of this block of code, the simulator should have been
    # advanced one step, and the replay buffer should contain one more
    # transition.
    # Specifically, self.last_obs must point to the new latest observation.
    # Useful functions you'll need to call:
    # obs, reward, done, info = env.step(action)
    # this steps the environment forward one step
    # obs = env.reset()
    # this resets the environment if you reached an episode boundary.
    # Don't forget to call env.reset() to get a new observation if done
    # is true!!
    # Note that you cannot use "self.last_obs" directly as input
    # into your network, since it needs to be processed to include context
    # from previous frames. You should check out the replay buffer
    # implementation in dqn_utils.py to see what functionality the replay
    # buffer exposes. The replay buffer has a function called
    # encode_recent_observation that will take the latest observation
    # that you pushed into the buffer and compute the corresponding
    # input that should be given to a Q network by appending some
    # previous frames.
    # Don't forget to include epsilon greedy exploration!
    # And remember that the first time you enter this loop, the model
    # may not yet have been initialized (but of course, the first step
    # might as well be random, since you haven't trained your net...)

    #####
    # raise NotImplementedError
    # YOUR CODE HERE - Problem 1.3 Implementation
    #####

    ## 1. initial action estimation
    self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)

    ## 2. action sampling
    if (np.random.random() < self.exploration.value(self.t)) or not self.model_initialized:
        # if model not initalized, then we can't use initialize_interdependent_variables() as feed_dict is required to fill (<_>|||, what is design?)
        # if model initialized, also eplsion greedy for random action
        cur_action = np.random.randint(0, self.num_actions)
        # print("random exploration cur_action:", cur_action)
    else:
        # self.model_initialized == True and (np.random.random() >= self.exploration.value(self.t)) , then I can run sess for action
        cur_obs = self.replay_buffer.encode_recent_observation()
        cur_action = self.session.run(self.max_action_for_qall, feed_dict={self.obs_t_ph: cur_obs[None, ]})
        # size == self.max_action_for_qall = tf.argmax(qall_action_values, axis=-1), but only for 1 self.last_obs, only for [1*cur_action]
        assert cur_action.shape == (1, )
        # print("via Q-network estimate cur_action: %s, shape = %s" %(np.asscalar(cur_action), cur_action.shape))
        cur_action = np.asscalar(cur_action)

    ## 3. make 1 step for simulation envrionment
    self.last_obs, reward, done, _ = self.env.step(cur_action)
    self.replay_buffer.store_effect(self.replay_buffer_idx, cur_action, reward, done)
    # reset self.last_obs to avoid next call for sel.step_env() to drop into inconsistent state
    if done:
        self.last_obs = self.env.reset()


  def update_model(self):
    ### 3. Perform experience replay and train the network.
    # note that this is only done if the replay buffer contains enough samples
    # for us to learn something useful -- until then, the model will not be
    # initialized and random actions should be taken
    if (self.t > self.learning_starts and \
        self.t % self.learning_freq == 0 and \
        self.replay_buffer.can_sample(self.batch_size)):
      # Here, you should perform training. Training consists of four steps:
      # 3.a: use the replay buffer to sample a batch of transitions (see the
      # replay buffer code for function definition, each batch that you sample
      # should consist of current observations, current actions, rewards,
      # next observations, and done indicator).
      # 3.b: initialize the model if it has not been initialized yet; to do
      # that, call
      #    initialize_interdependent_variables(self.session, tf.global_variables(), {
      #        self.obs_t_ph: obs_t_batch,
      #        self.obs_tp1_ph: obs_tp1_batch,
      #    })
      # where obs_t_batch and obs_tp1_batch are the batches of observations at
      # the current and next time step. The boolean variable model_initialized
      # indicates whether or not the model has been initialized.
      # Remember that you have to update the target network too (see 3.d)!
      # 3.c: train the model. To do this, you'll need to use the self.train_fn and
      # self.total_error ops that were created earlier: self.total_error is what you
      # created to compute the total Bellman error in a batch, and self.train_fn
      # will actually perform a gradient step and update the network parameters
      # to reduce total_error. When calling self.session.run on these you'll need to
      # populate the following placeholders:
      # self.obs_t_ph
      # self.act_t_ph
      # self.rew_t_ph
      # self.obs_tp1_ph
      # self.done_mask_ph
      # (this is needed for computing self.total_error)
      # self.learning_rate -- you can get this from self.optimizer_spec.lr_schedule.value(t)
      # (this is needed by the optimizer to choose the learning rate)
      # 3.d: periodically update the target network by calling
      # self.session.run(self.update_target_fn)
      # you should update every target_update_freq steps, and you may find the
      # variable self.num_param_updates useful for this (it was initialized to 0)
      #####
      # raise NotImplementedError        
      # YOUR CODE HERE - Problem 1.3 Implementation
      #####
      
      ## 3.a using self.replay_buffer to sample (obs_batch, act_batch, rew_batch, next_obs_batch, done_mask) with self.batch_size
      obs_t_batch, act_t_batch, rew_t_batch, next_obs_t_batch, done_t_mask = self.replay_buffer.sample(self.batch_size)

      ## 3.b initialize the model if it has not been initialized yet
      if not self.model_initialized:
          initialize_interdependent_variables(self.session, tf.global_variables(), {
              self.obs_t_ph: obs_t_batch,
              self.obs_tp1_ph: next_obs_t_batch,
          })
          self.model_initialized = True
      ## 3.c: train the model
      if self.double_q:
        # double q case, firstly calculate self.max_action_index_for_dQ's value
        max_action_index_for_dQ = self.session.run(self.max_action_for_qall, feed_dict={
            self.obs_t_ph: next_obs_t_batch
        })
        self.session.run([self.train_fn, self.total_error], feed_dict={
            self.obs_t_ph: obs_t_batch,
            self.act_t_ph: act_t_batch,
            self.rew_t_ph: rew_t_batch,
            self.obs_tp1_ph: next_obs_t_batch,
            self.max_action_index_for_dQ: max_action_index_for_dQ,
            self.done_mask_ph: done_t_mask,
            self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)
        })
      else:
        self.session.run([self.train_fn, self.total_error], feed_dict={
            self.obs_t_ph: obs_t_batch,
            self.act_t_ph: act_t_batch,
            self.rew_t_ph: rew_t_batch,
            self.obs_tp1_ph: next_obs_t_batch,
            self.done_mask_ph: done_t_mask,
            self.learning_rate: self.optimizer_spec.lr_schedule.value(self.t)
        })
      self.num_param_updates += 1

      ## 3.d periodically update the target network
      if self.num_param_updates % self.target_update_freq == 0:
          # just using var to update target_var via foreach tf.assign
          self.session.run(self.update_target_fn)

    self.t += 1

  def log_progress(self):
    """[logging progress of training and learning]
    """
    episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()

    if len(episode_rewards) > 0:
      self.mean_episode_reward = np.mean(episode_rewards[-100:])

    if len(episode_rewards) > 100:
      self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)

    if self.t % self.log_every_n_steps == 0 and self.model_initialized:
      ## YOUR CODE - Problem 1.4 Evaluation, Question 1
      # logging measure to logz folder, Iteration must be included, Iteration is used for storing Timestep
      logz.log_tabular("Iteration", self.t)
      logz.log_tabular("MeanRewardFor100Episodes", self.mean_episode_reward)
      logz.log_tabular("BestMeanEpisodeReward", self.best_mean_episode_reward)
      logz.log_tabular("Episodes", len(episode_rewards))

      print("Timestep %d" % (self.t,))
      print("mean reward (100 episodes) %f" % self.mean_episode_reward)
      print("best mean reward %f" % self.best_mean_episode_reward)
      print("episodes %d" % len(episode_rewards))
      print("exploration %f" % self.exploration.value(self.t))
      print("learning_rate %f" % self.optimizer_spec.lr_schedule.value(self.t))
      if self.start_time is not None:
        spent_time = (time.time() - self.start_time) / 60.
        print("running time %f" % (spent_time))
        # logging time to logz folder
        logz.log_tabular("TimeSpent", spent_time)
      else:
        ## logging time to logz folder
        # issue of log_tabular for the first time : AssertionError: Trying to introduce a new key TimeSpent that you didn't include in the first iteration
        logz.log_tabular("TimeSpent", None)

      # logging persisted
      logz.dump_tabular()
      logz.pickle_tf_vars()

      self.start_time = time.time()

      sys.stdout.flush()

      with open(self.rew_file, 'wb') as f:
        pickle.dump(episode_rewards, f, pickle.HIGHEST_PROTOCOL)

def learn(*args, **kwargs):
  alg = QLearner(*args, **kwargs)
  while not alg.stopping_criterion_met():
    alg.step_env()
    # at this point, the environment should have been advanced one step (and
    # reset if done was true), and self.last_obs should point to the new latest
    # observation
    alg.update_model()
    alg.log_progress()

