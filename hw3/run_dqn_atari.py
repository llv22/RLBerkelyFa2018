import os
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
from atari_wrappers import *


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    assert nn_output_sizes_before_action[-1] > num_actions
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            for output_size in nn_output_sizes_before_action:
                out = layers.fully_connected(out, num_outputs=output_size, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0

    ## 1. learning rate update
    # Why need to add PiecewiseSchedule for epsilon threshold estimation?
    # Reason: using scheduled learning rate for updates
    lr_schedule = PiecewiseSchedule([
                                         (0,                   lr_schedule_v[0] * lr_multiplier),
                                         (num_iterations / 10, lr_schedule_v[1] * lr_multiplier),
                                         (num_iterations / 2,  lr_schedule_v[2] * lr_multiplier),
                                    ],
                                    outside_value=lr_schedule_v[2] * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    ## 2. exploring schedule - pure exploring, then less exploring with exploiting
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.01),
        ], outside_value=0.01
    )

    dqn.learn(
        env=env,
        logdir=logdir,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10,
        double_q=double_q
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=thread_num_for_tf,
        intra_op_parallelism_threads=thread_num_for_tf)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env = gym.make('PongNoFrameskip-v4')

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)

    return env

def main():
    # Get Atari games.
    task = gym.make('PongNoFrameskip-v4')

    # Run training
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, num_timesteps)

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    # Basic setting
    parser.add_argument('--exp_name', type=str, default='atari')
    parser.add_argument('--tf_threads', '-t', type=int, default=1)
    parser.add_argument('--num_timesteps', '-n', type=int, default=int(2e8))
    parser.add_argument("--enable_double_q", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable double-Q network or not.")
    parser.add_argument("--lr_schedule", "-l", nargs=3, type=float, default=[1e-4, 1e-4, 5e-5], help='learning rate schedule for 3 steps')
    parser.add_argument("--act_nn", "-nn", nargs='*', type=int, default=[512], help='output size of Q(state, action) network by layer')
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    # in setup_logger will configure the logger directory: exists clean or update
    logdir = args.exp_name + '_PongNoFrameskip-v4_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    thread_num_for_tf = args.tf_threads
    num_timesteps = args.num_timesteps
    double_q = args.enable_double_q
    lr_schedule_v = args.lr_schedule
    assert len(lr_schedule_v) == 3
    nn_output_sizes_before_action = args.act_nn
    assert len(nn_output_sizes_before_action) > 0

    main()
