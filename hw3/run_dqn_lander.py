import os
# Install error for Pyglet on Mac OS X, refer to https://groups.google.com/forum/#!topic/pyglet-users/4H8luPX69zc
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = '/lib:/usr/lib:/usr/bin/lib:/' + os.environ['DYLD_FALLBACK_LIBRARY_PATH']
import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *

def lander_model(obs, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    assert nn_output_sizes_before_action[-1] > num_actions
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("action_value"):
            for output_size in nn_output_sizes_before_action:
                out = layers.fully_connected(out, num_outputs=output_size, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def lander_optimizer():
    return dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(1e-3),
        kwargs={}
    )

def lander_stopping_criterion(num_timesteps):
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
    return stopping_criterion

def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )

def lander_kwargs():
    return {
        'optimizer_spec': lander_optimizer(),
        'q_func': lander_model,
        'replay_buffer_size': 50000,
        'batch_size': 32,
        'gamma': 1.00,
        'learning_starts': 1000,
        'learning_freq': 1,
        'frame_history_len': 1,
        'target_update_freq': 3000,
        'grad_norm_clipping': 10,
        'lander': True
    }

def lander_learn(env,
                 session,
                 num_timesteps,
                 seed):

    optimizer = lander_optimizer()
    stopping_criterion = lander_stopping_criterion(num_timesteps)
    exploration_schedule = lander_exploration_schedule(num_timesteps)

    dqn.learn(
        env=env,
        logdir=logdir,
        session=session,
        exploration=lander_exploration_schedule(num_timesteps),
        stopping_criterion=lander_stopping_criterion(num_timesteps),
        double_q=double_q,
        **lander_kwargs()
    )
    env.close()

def set_global_seeds(i):
    tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=thread_num_for_tf,
        intra_op_parallelism_threads=thread_num_for_tf,
        device_count={'GPU': 0})
    # GPUs don't significantly speed up deep Q-learning for lunar lander,
    # since the observations are low-dimensional
    session = tf.Session(config=tf_config)
    return session

def get_env(seed):
    env = gym.make('LunarLander-v2')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    return env

def main():
    # Run training
    seed = 4565 # you may want to randomize this
    print('random seed = %d' % seed)
    env = get_env(seed)
    session = get_session()
    set_global_seeds(seed)
    lander_learn(env, session, num_timesteps, seed=seed)

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    # Basic setting
    parser.add_argument('--exp_name', type=str, default='lander')
    parser.add_argument('--tf_threads', '-t', type=int, default=1)
    parser.add_argument('--num_timesteps', '-n', type=int, default=500000)
    parser.add_argument("--enable_double_q", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable double-Q network or not.")
    parser.add_argument("--act_nn", "-nn", nargs='*', type=int, default=[64, 64], help='output size of Q(state, action) network by layer')
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    # in setup_logger will configure the logger directory: exists clean or update
    logdir = args.exp_name + '_LunarLander-v2_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    thread_num_for_tf = args.tf_threads
    num_timesteps = args.num_timesteps
    double_q = args.enable_double_q
    # constant lr-schedule
    nn_output_sizes_before_action = args.act_nn
    assert len(nn_output_sizes_before_action) > 0

    main()
