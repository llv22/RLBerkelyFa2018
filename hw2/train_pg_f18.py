"""
Original code from John Schulman for CS294 Deep Reinforcement Learning Spring 2017
Adapted for CS294-112 Fall 2017 by Abhishek Gupta and Joshua Achiam
Adapted for CS294-112 Fall 2018 by Michael Chang and Soroush Nasiriany
"""
import numpy as np
import tensorflow as tf
import gym
import logz
import os
import time
import inspect
from multiprocessing import Process

#============================================================================================#
# Utilities
#============================================================================================#

#========================================================================================#
#                           ----------PROBLEM 2----------
#========================================================================================#  
def build_mlp(input_placeholder, output_size, scope, n_layers, size, activation=tf.tanh, output_activation=None):
    """
        Builds a feedforward neural network
        
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            output_size: size of the output layer
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of the hidden layer
            activation: activation of the hidden layers
            output_activation: activation of the ouput layers

        returns:
            output placeholder of the network (the result of a forward pass) 

        Hint: use tf.layers.dense    
    """
    # YOUR CODE HERE - 4. Problem 2(a)
    with tf.variable_scope(scope):
        X = input_placeholder
        for _ in range(n_layers):
            X = tf.layers.dense(X, size, activation=activation)
        output_placeholder = tf.layers.dense(X, output_size, activation=output_activation)
    return output_placeholder

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

#============================================================================================#
# Policy Gradient
#============================================================================================#

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args, debugger_args, bonus_args):
        super(Agent, self).__init__()
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.discrete = computation_graph_args['discrete']
        self.size = computation_graph_args['size']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']

        self.gamma = estimate_return_args['gamma']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.nn_baseline = estimate_return_args['nn_baseline']
        self.normalize_advantages = estimate_return_args['normalize_advantages']
        
        self.ui_type = debugger_args['ui_type']
        self.debug = debugger_args['debug']
        self.tensorboard_debug_address = debugger_args['tensorboard_debug_address']

        self.tf_threads = bonus_args['tf_threads']
        self.gae_lambda = bonus_args['gae_lambda']
        self.gradient_descent_steps = bonus_args['gradient_descent_steps']

    def init_tf_sess(self):
        #
        ## Setup for tensorflow inter/intra threading number
        #
        ## YOUR CODE HERE - 8. Problem (a)
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=self.tf_threads, intra_op_parallelism_threads=self.tf_threads) 
        # Add debugging capacity with support --debug=True
        from tensorflow.python import debug as tf_debug
        self.sess = tf.Session(config=tf_config)
        if self.debug and self.tensorboard_debug_address:
            raise ValueError(
                "The --debug and --tensorboard_debug_address flags are mutually "
                "exclusive.")
        if self.debug:
            self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess, ui_type=self.ui_type)
        elif self.tensorboard_debug_address:
            self.sess = tf_debug.TensorBoardDebugWrapperSession(
                self.sess, self.tensorboard_debug_address)
        self.sess.__enter__() # equivalent to `with self.sess:`
        tf.global_variables_initializer().run() #pylint: disable=E1101

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def define_placeholders(self):
        """
            Placeholders for batch batch observations / actions / advantages in policy gradient 
            loss function.
            See Agent.build_computation_graph for notation

            returns:
                sy_ob_no: placeholder for observations
                sy_ac_na: placeholder for actions
                sy_adv_n: placeholder for advantages
        """
        # raise NotImplementedError
        sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)
        if self.discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
        else:
            sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32) 
        # YOUR CODE HERE - 4. Problem 2(b)(i)
        # sy_adv_n.shape = batch_size, only for Q(i,t) for samples
        # refer to https://github.com/Kelym/DeepRL-UCB2017-Homework/blob/master/hw2/train_pg.py 
        sy_adv_n = tf.placeholder(shape=[None], name='adv', dtype=tf.float32)
        return sy_ob_no, sy_ac_na, sy_adv_n


    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def policy_forward_pass(self, sy_ob_no):
        """ Constructs the symbolic operation for the policy network outputs,
            which are the parameters of the policy distribution p(a|s)

            arguments:
                sy_ob_no: (batch_size, self.ob_dim)

            returns:
                the parameters of the policy.

                if discrete, the parameters are the logits of a categorical distribution
                    over the actions
                    sy_logits_na: (batch_size, self.ac_dim)

                if continuous, the parameters are a tuple (mean, log_std) of a Gaussian
                    distribution over actions. log_std should just be a trainable
                    variable, not a network output.
                    sy_mean: (batch_size, self.ac_dim)
                    sy_logstd: (self.ac_dim,)

            Hint: use the 'build_mlp' function to output the logits (in the discrete case)
                and the mean (in the continuous case).
                Pass in self.n_layers for the 'n_layers' argument, and
                pass in self.size for the 'size' argument.
        """
        # raise NotImplementedError
        if self.discrete:
            # YOUR_CODE_HERE - 4. Problem 2(b)(ii)
            # shape (batch_size, self.ac_dim), not via logits 
            sy_logits_na = build_mlp(sy_ob_no, self.ac_dim, "discrete_mlp", self.n_layers, self.size, output_activation=tf.nn.relu)
            return sy_logits_na
        else:
            # YOUR_CODE_HERE - 4. Problem 2(b)(ii)
            # sy_mean = build_mlp(sy_ob_no, self.ac_dim, "continuous_mlp", self.n_layers, self.size, output_activation=tf.nn.relu)
            sy_mean = build_mlp(sy_ob_no, self.ac_dim, "continuous_mlp", self.n_layers, self.size)
            # sy_logstd = tf.get_variable(shape=[self.ac_dim,], dtype=tf.float32, name="sy_logstd")
            sy_logstd = tf.get_variable(shape=[self.ac_dim,], dtype=tf.float32, name="sy_logstd", initializer=tf.contrib.layers.xavier_initializer())
            # sy_logstd = tf.Variable(tf.zeros([1, self.ac_dim], name = 'logstd'))
            return (sy_mean, sy_logstd)

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def sample_action(self, policy_parameters):
        """ Constructs a symbolic operation for stochastically sampling from the policy
            distribution

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

            returns:
                sy_sampled_ac: 
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            Hint: for the continuous case, use the reparameterization trick:
                 The output from a Gaussian distribution with mean 'mu' and std 'sigma' is
        
                      mu + sigma * z,         z ~ N(0, I)
        
                 This reduces the problem to just sampling z. (Hint: use tf.random_normal!)
        """
        # raise NotImplementedError
        if self.discrete:
            sy_logits_na = policy_parameters
            # YOUR_CODE_HERE - 4. Problem 2(b)(iii)
            ## Deterministic sample action will lead to the same sampled_ac, add some noise for sampled action
            # sy_sampled_ac = tf.argmax(sy_logits_na, axis=-1)
            # [Learn] Use tf.multinomial to generate (batch_size, 1), then to use tf.squeence tp remove the last dimension to shape (batch_size,)
            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1), axis=-1)
        else:
            sy_mean, sy_logstd = policy_parameters
            # YOUR_CODE_HERE - 4. Problem 2(b)(iii) via boardcasting of tf.exp(sy_logstd) from (self.ac_dim,) to (batch_size, self.ac_dim)
            sy_sampled_ac = sy_mean + tf.exp(sy_logstd) * tf.random_normal(tf.shape(sy_mean))
        return sy_sampled_ac

    #========================================================================================#
    #                           ----------PROBLEM 2----------
    #========================================================================================#
    def get_log_prob(self, policy_parameters, sy_ac_na):
        """ Constructs a symbolic operation for computing the log probability of a set of actions
            that were actually taken according to the policy

            arguments:
                policy_parameters
                    if discrete: logits of a categorical distribution over actions 
                        sy_logits_na: (batch_size, self.ac_dim)
                    if continuous: (mean, log_std) of a Gaussian distribution over actions
                        sy_mean: (batch_size, self.ac_dim)
                        sy_logstd: (self.ac_dim,)

                sy_ac_na: 
                    if discrete: (batch_size,)
                    if continuous: (batch_size, self.ac_dim)

            returns:
                sy_logprob_n: (batch_size)

            Hint:
                For the discrete case, use the log probability under a categorical distribution.
                For the continuous case, use the log probability under a multivariate gaussian.
        """
        # raise NotImplementedError
        if self.discrete:
            sy_logits_na = policy_parameters
            # YOUR_CODE_HERE - 4. Problem 2(b)(iv)
            ## Difference between sparse_softmax_cross_entropy_with_logits and softmax_cross_entropy_with_logits,  refer to https://stackoverflow.com/questions/37312421/whats-the-difference-between-sparse-softmax-cross-entropy-with-logits-and-softm
            # tf.shape(sy_ac_na) = (batch_size,), tf.shape(sy_logits_na) = (batch_size, self.ac_dim)
            sy_logprob_n = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)
        else:
            sy_mean, sy_logstd = policy_parameters
            # YOUR_CODE_HERE - 4. Problem 2(b)(iv)
            ## Multivariate Nominal in tensorflow, refer to https://www.tensorflow.org/versions/r1.10/api_docs/python/tf/contrib/distributions/MultivariateNormalDiag
            tfd = tf.contrib.distributions
            ## Hint: Use the log probability under a multivariate gaussian.
            # Learned from github.com/mwhittaker
            ## [Learn] Do the negate here works; 
            #  Do it in loss / weighted_negative_likelihood not work.
            sy_logprob_n = tfd.MultivariateNormalDiag(loc=sy_ac_na, scale_diag=tf.exp(sy_logstd)).log_prob(sy_mean)
            # refer to https://github.com/fwtan/cs294-homework/blob/master/hw2/train_pg.py
            # tmp = tf.norm(sy_mean - sy_ac_na/(sy_logstd + 1e-4), axis=-1)
            # sy_logprob_n = -0.5 * tmp * tmp  # Hint: Use the log probability under a multivariate gaussian. 
            # sy_z = (sy_mean - sy_ac_na) / tf.exp(sy_logstd)
            # sy_logprob_n = - 0.5 * tf.reduce_sum(tf.square(sy_z), axis = -1)
        return sy_logprob_n

    def build_computation_graph(self):
        """
            Notes on notation:
            
            Symbolic variables have the prefix sy_, to distinguish them from the numerical values
            that are computed later in the function
            
            Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)
            
            Note: batch self.size /n/ is defined at runtime, and until then, the shape for that axis
            is None

            ----------------------------------------------------------------------------------
            loss: a function of self.sy_logprob_n and self.sy_adv_n that we will differentiate
                to get the policy gradient.
        """
        self.sy_ob_no, self.sy_ac_na, self.sy_adv_n = self.define_placeholders()

        # The policy takes in an observation and produces a distribution over the action space
        self.policy_parameters = self.policy_forward_pass(self.sy_ob_no)

        # We can sample actions from this action distribution.
        # This will be called in Agent.sample_trajectory() where we generate a rollout.
        self.sy_sampled_ac = self.sample_action(self.policy_parameters)

        # We can also compute the logprob of the actions that were actually taken by the policy
        # This is used in the loss function.
        self.sy_logprob_n = self.get_log_prob(self.policy_parameters, self.sy_ac_na)

        #========================================================================================#
        #                           ----------PROBLEM 2----------
        # Loss Function and Training Operation
        #========================================================================================#
        ## YOUR CODE HERE - 4. Problem 2(b)(v)
        # element-wise multiply for logpro_n and adv_n by position, Q: how to select out adv_n
        # have to share in order to let Line 558 access for loss value
        self.loss = -tf.reduce_mean(self.sy_logprob_n * self.sy_adv_n)
        self.update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        #========================================================================================#
        #                           ----------PROBLEM 6----------
        # Optional Baseline
        #
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 
        #========================================================================================#
        if self.nn_baseline:
            # raise NotImplementedError
            self.baseline_prediction = tf.squeeze(build_mlp(
                                    self.sy_ob_no, 
                                    1,
                                    "nn_baseline",
                                    n_layers=self.n_layers,
                                    size=self.size))
            ## YOUR CODE HERE - 6. Problem (a)
            # status: to verify status, V_pi(s) is invalid name, have to go for V_pi_s
            self.sy_target_n = tf.placeholder(shape=[None], name='V_pi_s', dtype=tf.float32)
            self.baseline_loss = tf.losses.mean_squared_error(labels=self.sy_target_n, predictions=self.baseline_prediction)
            self.baseline_update_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.baseline_loss)

    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > self.min_timesteps_per_batch:
                break
        return paths, timesteps_this_batch

    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()
        obs, acs, rewards = [], [], []
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)
            obs.append(ob)
            #====================================================================================#
            #                           ----------PROBLEM 3----------
            #====================================================================================#
            # raise NotImplementedError
            ## YOUR CODE HERE - 5. Problem 3.1
            # sample action based on Pi_{theta}(Observation)
            # print("NN(Pi) input ob.shape = %s, ob = %s" % (np.array(ob).reshape(-1, 4).shape, np.array(ob).reshape(-1, 4).tolist()))
            # ac = self.sess.run([self.sy_sampled_ac], feed_dict={self.sy_ob_no: np.array(ob).reshape(-1, 4)})
            # print("NN(Pi) input ob.shape = %s, ob = %s" % (ob[None].shape, ob[None].tolist()))
            ## issue: not [self.sy_sampled_ac], otherwise it will have [] as list for single parameter 
            # ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None]})
            ac = self.sess.run(self.sy_sampled_ac, feed_dict={self.sy_ob_no: ob[None]})
            ac = ac[0]
            # print("NN(Pi) output ac.shape = %s, ac = %s" % (ac.shape, ac))
            acs.append(ac)
            ob, rew, done, _ = env.step(ac)
            rewards.append(rew)
            steps += 1
            if done or steps > self.max_path_length:
                break
        path = {"observation" : np.array(obs, dtype=np.float32), 
                "reward" : np.array(rewards, dtype=np.float32), 
                "action" : np.array(acs, dtype=np.float32)}
        return path

    #====================================================================================#
    #                           ----------PROBLEM 3----------
    #====================================================================================#
    def sum_of_rewards(self, re_n):
        """
            Monte Carlo estimation of the Q function.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

            ----------------------------------------------------------------------------------
            
            Your code should construct numpy arrays for Q-values which will be used to compute
            advantages (which will in turn be fed to the placeholder you defined in 
            Agent.define_placeholders). 
            
            Recall that the expression for the policy gradient PG is
            
                  PG = E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * (Q_t - b_t )]
            
            where 
            
                  tau=(s_0, a_0, ...) is a trajectory,
                  Q_t is the Q-value at time t, Q^{pi}(s_t, a_t),
                  and b_t is a baseline which may depend on s_t. 
            
            You will write code for two cases, controlled by the flag 'reward_to_go':
            
              Case 1: trajectory-based PG 
            
                  (reward_to_go = False)
            
                  Instead of Q^{pi}(s_t, a_t), we use the total discounted reward summed over 
                  entire trajectory (regardless of which time step the Q-value should be for). 
            
                  For this case, the policy gradient estimator is
            
                      E_{tau} [sum_{t=0}^T grad log pi(a_t|s_t) * Ret(tau)]
            
                  where
            
                      Ret(tau) = sum_{t'=0}^T gamma^t' r_{t'}.
            
                  Thus, you should compute
            
                      Q_t = Ret(tau)
            
              Case 2: reward-to-go PG 
            
                  (reward_to_go = True)
            
                  Here, you estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting
                  from time step t. Thus, you should compute
            
                      Q_t = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            
            
            Store the Q-values for all timesteps and all trajectories in a variable 'q_n',
            like the 'ob_no' and 'ac_na' above. 
        """
        # YOUR_CODE_HERE - 5. Problem 3.3 (b) 
        # Question: do we need to concatenate to 1D array? or use 2D array, just refer to https://github.com/Kelym/DeepRL-UCB2017-Homework/blob/master/hw2/train_pg.py as 1D array
        if self.reward_to_go:
            # raise NotImplementedError
            # 1. for re_tau loop : each trajectories' reward to collect trajectory length
            # 2. for start loop : 
            #   1) (gamma ** np.arange(start, len(re_tau)) to generate descent discount factor [1, \gamma[0],  \gamma[0]^2, ....]
            #   2) re_tau[::-1][:len(re_tau)-start] to generate 1:T steps length rewards
            #   3) sum for each step length : sum() -> 1:T steps rewards
            ## issue: gamma ** np.arange(len(re_tau)-start) NOT gamma ** np.arange(start, len(re_tau)), power index is always generated from 0 to last len(re_tau) - start as last index end
            ## q_n's value summation is must-have for each t'=1:T of \sum_{t'=1}^{T} r(s_{i,t'}, a_{i,t'}) 
            q_n = np.concatenate([[sum(re_tau[start:] * (self.gamma ** np.arange(0, len(re_tau)-start))) \
                    for start in range(len(re_tau))] \
                for re_tau in re_n])
        else:
            # raise NotImplementedError
            ## trajectory-based PG, then only need to collect for [1, \gamma[0],  \gamma[0]^2, ...], via 1 line, need to copy for 1:T for each trajectory for T steps, in order to element-wise multiple to sum
            q_n = np.concatenate([[sum(re_tau * (self.gamma ** np.array(range(len(re_tau)))))] * len(re_tau) \
                    for re_tau in re_n])
        return q_n

    def compute_advantage(self, ob_no, q_n, re_n):
        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths

                ## Orlando's refinement to get shape of [[traj0], [traj1], ...]
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Computing Baselines
        #====================================================================================#
        if self.nn_baseline:
            # If nn_baseline is True, use your neural network to predict reward-to-go
            # at each timestep for each trajectory, and save the result in a variable 'b_n'
            # like 'ob_no', 'ac_na', and 'q_n'.
            #
            # Hint #bl1: rescale the output from the nn_baseline to match the statistics
            # (mean and std) of the current batch of Q-values. (Goes with Hint
            # #bl2 in Agent.update_parameters.
            # raise NotImplementedError
            ## YOUR CODE HERE - 6. Problem (b)
            # status: to verify status
            b_n = self.sess.run(self.baseline_prediction, feed_dict={self.sy_ob_no: ob_no}) 
            # normlize for b_n via Reparameterize from N(0, I) to N(np.mean(q_n), np.std(q_n))
            b_n = b_n * (np.std(q_n) + 1e-9) + np.mean(q_n)
            if self.gae_lambda < 0:
                #
                ## Disable GAE(lambda) option 
                #
                adv_n = q_n - b_n
            else:
                #
                ## Enable GAE(lambda) option 
                #
                # raise NotImplementedError
                ## YOUR CODE HERE - 8. Problem (b)
                # refer to https://github.com/daggertye/CS294_homework/blob/master/hw2/train_pg.py#L369
                adv_n = []
                for rewards in re_n:
                    adv = np.zeros(len(rewards))
                    adv[-1] = rewards[-1] - b_n[-1]
                    for i in range(len(rewards)-1)[::-1]:
                        delta = rewards[i] + self.gamma * b_n[i+1] - b_n[i]
                        adv[i] = adv[i+1] * self.gamma * self.gae_lambda + delta
                    # if not reward to go, just the same as the first adv[0], as from advantage function, they are the same.
                    if not self.reward_to_go:
                        adv = np.ones(len(rewards)) * adv[0]
                    adv_n.extend(adv)
        else:
            adv_n = q_n.copy()
        return adv_n

    def estimate_return(self, ob_no, re_n):
        """
            Estimates the returns over a set of trajectories. [Take Care: Don't use tensorflow for calculating estimate_return, as it's in numpy]

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from 
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                re_n: length: num_paths. Each element in re_n is a numpy array 
                    containing the rewards for the particular path

            returns:
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths
        """
        q_n = self.sum_of_rewards(re_n)
        # print("q_n:", q_n)
        adv_n = self.compute_advantage(ob_no, q_n, re_n)
        # print("adv_n:", adv_n)
        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Advantage Normalization
        #====================================================================================#
        if self.normalize_advantages:
            # On the next line, implement a trick which is known empirically to reduce variance
            # in policy gradient methods: normalize adv_n to have mean zero and std=1.
            # raise NotImplementedError
            # YOUR_CODE_HERE - 5. Problem 3.2
            # Option 1: 
            # using adv_mean = tf.reduce_mean(adv_n) and adv_std = tf.sqrt(tf.reduce_mean(tf.square(adv_n - adv_mean))), then adv_n = (adv_n - adv_mean) / adv_std  
            # Option 2:
            # check with https://www.tensorflow.org/api_docs/python/tf/nn/moments, we can calculate adv_mean and adv_std in a step
            # adv_mean, adv_variance = tf.nn.moments(tf.constant(adv_n), axes=[0])
            # adv_std = tf.sqrt(adv_variance)
            # adv_n = tf.cond(adv_std < 1e-9, lambda: (adv_n-adv_mean), lambda: (adv_n-adv_mean)/adv_std)
            # Option : must be implemented in numpy by value calculation
            # adv_mean, adv_std = np.mean(adv_n), np.std(adv_n)
            # adv_n = adv_n-adv_mean if adv_std < 1e-9 else (adv_n-adv_mean)/adv_std
            adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-9)
        return q_n, adv_n

    def update_parameters(self, ob_no, ac_na, q_n, adv_n):
        """ 
            Update the parameters of the policy and (possibly) the neural network baseline, 
            which is trained to approximate the value function.

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                ac_na: shape: (sum_of_path_lengths).
                q_n: shape: (sum_of_path_lengths). A single vector for the estimated q values 
                    whose length is the sum of the lengths of the paths
                adv_n: shape: (sum_of_path_lengths). A single vector for the estimated 
                    advantages whose length is the sum of the lengths of the paths

            returns:
                nothing

        """
        #====================================================================================#
        #                           ----------PROBLEM 6----------
        # Optimizing Neural Network Baseline
        #====================================================================================#
        if self.nn_baseline:
            # If a neural network baseline is used, set up the targets and the inputs for the 
            # baseline. 
            # 
            # Fit it to the current batch in order to use for the next iteration. Use the 
            # baseline_update_op you defined earlier.
            #
            # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
            # targets to have mean zero and std=1. (Goes with Hint #bl1 in 
            # Agent.compute_advantage.)

            ## YOUR CODE HERE - 6. Problem (c)
            # status: verify
            # raise NotImplementedError
            # normalize q_n to N(0, I)
            target_n = (q_n - np.mean(q_n)) / (np.std(q_n) + 1e-9)
            loss_base_val, _ = self.sess.run([self.baseline_loss, self.baseline_update_op], feed_dict={self.sy_ob_no: ob_no, 
                                                                                                       self.sy_target_n: target_n})
            logz.log_tabular("Baseline Loss", loss_base_val)

        #====================================================================================#
        #                           ----------PROBLEM 3----------
        # Performing the Policy Update
        #====================================================================================#

        # Call the update operation necessary to perform the policy gradient update based on 
        # the current batch of rollouts.
        # 
        # For debug purposes, you may wish to save the value of the loss function before
        # and after an update, and then log them below. 

        # YOUR_CODE_HERE - 5. Problem 3.3 (c) 
        # raise NotImplementedError
        # also refer to https://github.com/Kelym/DeepRL-UCB2017-Homework/blob/master/hw2/train_pg.py#L455
        ## issue: ValueError: setting an array element with a sequence. - debugging purpose, adv_n->q_n forget to sum for each q(t:1...T)
        # print("adv_n.shape:", adv_n.shape)
        # print("adv_n:", adv_n)
        # print("self.sy_adv_n.shape:", self.sy_adv_n.shape)
        for _ in range(self.gradient_descent_steps):
            # via multiple gradient descent steps
            loss_val, _ = self.sess.run([self.loss, self.update_op], feed_dict={self.sy_ob_no: ob_no, 
                                                                                self.sy_ac_na: ac_na, 
                                                                                self.sy_adv_n: adv_n})
        logz.log_tabular("Loss", loss_val)
        if type(self.policy_parameters) is tuple and len(self.policy_parameters) == 2:
            print("self.sy_logstd = ", self.sess.run(self.policy_parameters[1]))
        


def train_PG(
        exp_name,
        env_name,
        n_iter, 
        gamma, 
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate, 
        reward_to_go, 
        animate, 
        logdir, 
        normalize_advantages,
        nn_baseline, 
        seed,
        n_layers,
        size,
        ui_type,
        debug,
        tensorboard_debug_address,
        tf_threads,
        gae_lambda,
        gradient_descent_steps):

    start = time.time()

    #========================================================================================#
    # Set Up Logger
    #========================================================================================#
    setup_logger(logdir, locals())

    #========================================================================================#
    # Set Up Env
    #========================================================================================#

    # Make the gym environment
    env = gym.make(env_name)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    print("action space type:", "discrete" if discrete else "continous")

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    print("action dim:", ac_dim)

    #========================================================================================#
    # Initialize Agent
    #========================================================================================#
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'discrete': discrete,
        'size': size,
        'learning_rate': learning_rate,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'reward_to_go': reward_to_go,
        'nn_baseline': nn_baseline,
        'normalize_advantages': normalize_advantages,
    }

    debugger_args = {
        'ui_type': ui_type,
        'debug': debug,
        'tensorboard_debug_address': tensorboard_debug_address
    }

    bonus_args = {
        "tf_threads": tf_threads,
        "gae_lambda": gae_lambda,
        "gradient_descent_steps": gradient_descent_steps
    }

    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args, debugger_args, bonus_args)

    # build computation graph
    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    #========================================================================================#
    # Training Loop
    #========================================================================================#

    total_timesteps = 0
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]

        # First to collect estimate return for sampled trajectories
        q_n, adv_n = agent.estimate_return(ob_no, re_n)
        # Then to update parameters via self.update_op
        agent.update_parameters(ob_no, ac_na, q_n, adv_n)

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        logz.pickle_tf_vars()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Basic setting
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('--process_in_parallel', '-p', type=int, default=0)
    # Debugging capacity parameters
    parser.add_argument(
        "--ui_type",
        type=str,
        default="curses",
        help="Command-line user interface type (curses | readline)")
    parser.add_argument(
        "--debug",
        # type="bool",
        type=lambda x: (str(x).lower() == 'true'),
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training. "
        "Mutually exclusive with the --tensorboard_debug_address flag.")
    parser.add_argument(
        "--tensorboard_debug_address",
        type=str,
        default=None,
        help="Connect to the TensorBoard Debugger Plugin backend specified by "
        "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
        "--debug flag.")
    # Section 8 - Bonus part setting
    parser.add_argument('--tf_threads', '-t', type=int, default=1)
    # if gae_lambda < 0, it means that we close GAE(lambda)
    parser.add_argument('--gae_lambda', '-glambda', type=float, default=-1.0)
    # PG gradient descent steps
    parser.add_argument('--gradient_descent_steps', '-gds', type=int, default=1)

    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            print("args:", args)
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                ui_type=args.ui_type,
                debug=args.debug,
                tensorboard_debug_address=args.tensorboard_debug_address,
                tf_threads=args.tf_threads,
                gae_lambda=args.gae_lambda,
                gradient_descent_steps=args.gradient_descent_steps
                )
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block 
        # until this process finishes
        if not args.process_in_parallel:
            # if not run in parallel for processes, just run in sequence
            p.join()

    # otherwise, run in parallel; only finished, back to main process
    if args.process_in_parallel:
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
