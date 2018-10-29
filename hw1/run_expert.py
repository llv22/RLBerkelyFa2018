#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 1
    python run_expert.py experts/HalfCheetah-v2.pkl HalfCheetah-v2 --render --num_rollouts 1

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

def expert_policy_out():
    """[Using the existing expert policy to generate expert trajectory to expert_data_dir folder]
    """
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    ## 1, run expert policy to generate roll-outs
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # trajectory sampling
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {
            'observations': np.array(observations), 
            'actions': np.array(actions)
        }

        with open(os.path.join(expert_data_dir, args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

def bc_policy_estimate(lr: float):
    """[Using behaviorial cloning to do policy estimate and use estimated policy to do task]
        Arguments:
            lr {[float]} -- [learning rate value]
    """
    from tensorflow.layers import dense, batch_normalization
    from tensorflow.losses import mean_squared_error
    from tensorflow.train import AdamOptimizer
    
    with open(os.path.join(expert_data_dir, args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)
    assert expert_data != None

    # construct full-connected layers
    O_shape = list(expert_data['observations'].shape)
    A_shape = list(expert_data['actions'].shape)
    assert O_shape[0] == A_shape[0]
    # to tensor shape
    A_fshape=A_shape; A_fshape[0]=-1
    O_shape[0]=None; A_shape[0]=None
    # sign
    is_training=tf.placeholder(tf.bool)
    # construct fully_connected layers
    O = tf.placeholder(tf.float32, shape=O_shape)
    A = tf.placeholder(tf.float32, shape=A_shape)
    # layer dimension
    layer_sizes = [8, 5, 3]
    X = batch_normalization(O, is_training)
    for layer_size in layer_sizes:
        X = dense(X, units=layer_size)
    X_out = tf.reshape(X, A_fshape)
    loss = mean_squared_error(X_out, A)
    optimizer = AdamOptimizer(learning_rate=lr)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # TODO : Missing epoch and iteration setting here
        sess.run([optimizer, loss], feed_dict={O: expert_data['observations'], A: expert_data['actions']})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--expert_data", type=str, default="expert_data", help="Export data folder")
    parser.add_argument("--skip_expert_generate", type=bool, default=True)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()
    expert_data_dir = args.expert_data

    # Preparation of expert policy to generate archived expert trajectory
    if not args.skip_expert_generate:
        # if not skip expert generate, then run it out
        expert_policy_out()

    # Comparsion using BC to esimate policy
    bc_policy_estimate()
