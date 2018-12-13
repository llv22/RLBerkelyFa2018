import tensorflow as tf
import time

class SAC:
    """Soft Actor-Critic (SAC)
    Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

    References
    ----------
    [1] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine, "Soft
        Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
        with a Stochastic Actor," ICML 2018.
    """

    def __init__(self,
                 alpha=1.0,
                 batch_size=256,
                 discount=0.99,
                 epoch_length=1000,
                 learning_rate=3e-3,
                 reparameterize=False,
                 tau=0.01,
                 **kwargs):
        """
        Args:
        """

        self._alpha = alpha
        self._batch_size = batch_size
        self._discount = discount
        self._epoch_length = epoch_length
        self._learning_rate = learning_rate
        self._reparameterize = reparameterize
        self._tau = tau

        self._training_ops = []

    def build(self, env, policy, q_function, q_function2, value_function,
              target_value_function):

        self._create_placeholders(env)

        policy_loss = self._policy_loss_for(policy, q_function, q_function2, value_function)
        value_function_loss = self._value_function_loss_for(
            policy, q_function, q_function2, value_function)
        q_function_loss = self._q_function_loss_for(q_function,
                                                    target_value_function)
        if q_function2 is not None:
            q_function2_loss = self._q_function_loss_for(q_function2,
                                                        target_value_function)

        optimizer = tf.train.AdamOptimizer(
            self._learning_rate, name='optimizer')
        policy_training_op = optimizer.minimize(
            loss=policy_loss, var_list=policy.trainable_variables)
        value_training_op = optimizer.minimize(
            loss=value_function_loss,
            var_list=value_function.trainable_variables)
        q_function_training_op = optimizer.minimize(
            loss=q_function_loss, var_list=q_function.trainable_variables)
        if q_function2 is not None:
            q_function2_training_op = optimizer.minimize(
                loss=q_function2_loss, var_list=q_function2.trainable_variables)

        self._training_ops = [
            policy_training_op, value_training_op, q_function_training_op
        ]
        if q_function2 is not None:
            self._training_ops += [q_function2_training_op]
        self._target_update_ops = self._create_target_update(
            source=value_function, target=target_value_function)

        tf.get_default_session().run(tf.global_variables_initializer())

    def _create_placeholders(self, env):
        observation_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self._observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='observation',
        )
        self._next_observations_ph = tf.placeholder(
            tf.float32,
            shape=(None, observation_dim),
            name='next_observation',
        )
        self._actions_ph = tf.placeholder(
            tf.float32,
            shape=(None, action_dim),
            name='actions',
        )
        self._rewards_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='rewards',
        )
        self._terminals_ph = tf.placeholder(
            tf.float32,
            shape=(None, ),
            name='terminals',
        )

    def _policy_loss_for(self, policy, q_function, q_function2, value_function):
        if not self._reparameterize:
            ### Problem 1.3.A
            ### YOUR CODE HERE
            # raise NotImplementedError
            ###  J_pi = E_{s \in D}[ E_{a \sim \pi_{\phi}(a|S) [ \nabla_{\phi} log \pi(a|s) (\alpha log \pi_{\phi} (a|s) - Q_{\theta}(s, a)) + b(s) | s ] ]
            actions, log_pis = policy(self._observations_ph)
            E_sa = tf.reduce_mean(tf.stop_gradient(self._alpha * log_pis - q_function((self._observations_ph, self._actions_ph))) + value_function((self._observations_ph)), axis=-1)
            J_pi = tf.reduce_mean(E_sa)
        else:
            ### Problem 1.3.B
            ### YOUR CODE HERE
            # raise NotImplementedError
            actions, log_pis = policy(self._observations_ph)
            E_ea = tf.reduce_mean(self._alpha * policy((self._observations_ph, actions)) - q_function((self._observations_ph, actions)), axis=-1)
            J_pi = tf.reduce_mean(E_ea)
        
        return J_pi

    def _value_function_loss_for(self, policy, q_function, q_function2, value_function):
        ### Problem 1.2.A
        ### YOUR CODE HERE
        ### Comments: just temp.ly ignore q_function2
        ### e1: E_sa =  E_{a \in pi(a|s)}[Q(s, a) - temperature * log pi(a|s)] 
        ### e2: Jv(\phi) = E_{s \in D} [V(s) - E_sa ]
        # raise NotImplementedError
        ## 1. Sample actions from policy in internal expection in e1
        actions, log_pis = policy(self._observations_ph)
        ## 2. Calculate Jv(\phi) = E_{s \in D} [V(s) - E_sa], taking care state in D
        ## (q_funcion((self._observations_ph, self._actions_ph)) - self._alpha * log_pis).shape = |S| * |A|
        ## E_sa.shape = |S|
        E_sa = tf.reduce_mean(q_funcion((self._observations_ph, self._actions_ph)) - self._alpha * log_pis, axis=-1)
        Jv_s = tf.losses.mean_squared_error(E_sa, value_function(self._observations_ph))
        return Jv_s

    def _q_function_loss_for(self, q_function, target_value_function):
        ### Problem 1.1.A
        ### YOUR CODE HERE
        # raise NotImplementedError
        q_values = q_function((self._observations_ph, self._actions_ph))
        target_values = self._rewards_ph + self._discount * (1 - self._terminals_ph) * target_value_function((self._next_observations_ph))
        return tf.losses.mean_squared_error(q_values, target_values)


    def _create_target_update(self, source, target):
        """Create tensorflow operations for updating target value function."""

        return [
            tf.assign(target, (1 - self._tau) * target + self._tau * source)
            for target, source in zip(target.trainable_variables, source.
                                      trainable_variables)
        ]

    def train(self, sampler, n_epochs=1000):
        """Return a generator that performs RL training.

        Args:
            env (`rllab.Env`): Environment used for training
            policy (`Policy`): Policy used for training
            initial_exploration_policy ('Policy'): Policy used for exploration
                If None, then all exploration is done using policy
            pool (`PoolBase`): Sample pool to add samples to
        """
        self._start = time.time()
        for epoch in range(n_epochs):
            for t in range(self._epoch_length):
                sampler.sample()

                batch = sampler.random_batch(self._batch_size)
                feed_dict = {
                    self._observations_ph: batch['observations'],
                    self._actions_ph: batch['actions'],
                    self._next_observations_ph: batch['next_observations'],
                    self._rewards_ph: batch['rewards'],
                    self._terminals_ph: batch['terminals'],
                }
                tf.get_default_session().run(self._training_ops, feed_dict)
                tf.get_default_session().run(self._target_update_ops)

            yield epoch

    def get_statistics(self):
        statistics = {
            'Time': time.time() - self._start,
            'TimestepsThisBatch': self._epoch_length,
        }

        return statistics
