#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    1. Expert generation
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 200 --only_expert_generate 0
    2. Training for BC
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 1 --only_expert_generate 1
    3. Training for Dagger
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --num_rollouts 50 --only_expert_generate 2

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from tensorflow.layers import dense, batch_normalization
from tensorflow.contrib.layers import fully_connected
from tensorflow.losses import mean_squared_error
from tensorflow.train import AdamOptimizer

from time import time
from functools import wraps
from sklearn.utils import gen_batches

def timeit(func):
    @wraps(func)
    def timed(*args, **kwargs):
        startTime = time() 
        result = func(*args, **kwargs)
        print("function [{}] in {:8.4f} s".format(func.__name__, time() - startTime))
        return result
    return timed

def neuralNetwork(expert_data, lr):
    """[construct shared neural Network]
    
    Arguments:
        expert_data {[dictionary of expert data]} -- [exported expert data]
        lr {[float]} -- [learning rate]
    
    Returns:
        [Tensor tuple] -- [op-optimizatio operation, loss, O-obsevation tensor, A-action tensor, is_training, X_out-predicted action tensor]
    """
    # construct full-connected layers
    O_shape = list(expert_data['observations'].shape)
    A_shape = list(expert_data['actions'].shape)
    assert O_shape[0] == A_shape[0]
    # to tensor shape
    A_fshape=[-1 if i==0 else elem for i, elem in enumerate(A_shape)]
    O_shape[0]=None; A_shape[0]=None
    # sign
    is_training=tf.placeholder(tf.bool)
    # construct fully_connected layers
    O = tf.placeholder(tf.float32, shape=O_shape)
    A = tf.placeholder(tf.float32, shape=A_shape)
    # layer dimension
    layer_sizes = [int(elem * A_shape[-1]) for elem in [50, 30, 10, 1]]
    ## Why with batch_normal, we can't even reach the reward maximum?
    # X = batch_normalization(O, training=is_training)
    X = O
    for i, layer_size in enumerate(layer_sizes):
        if i != len(layer_sizes) - 1:
            X = dense(X, units=layer_size, activation='tanh')
            # X = fully_connected(X, num_outputs=layer_size, activation_fn=tf.nn.tanh)
        else:
            X = dense(X, units=layer_size, activation=None)
            # X = fully_connected(X, num_outputs=layer_size, activation_fn=None)
    X_out = tf.reshape(X, A_fshape)
    loss = mean_squared_error(predictions=X_out, labels=A)
    op = AdamOptimizer(learning_rate=lr).minimize(loss)
    return op, loss, O, A, is_training, X_out

def loadExpertOA():
    """[load expert data from data on disk]
    
    Returns:
        [dict] -- [expert data]
    """
    with open(os.path.join(expert_data_dir, args.envname + '.pkl'), 'rb') as f:
        expert_data = pickle.load(f)
    assert expert_data != None
    return expert_data

@timeit
def expert_policy_out():
    """[Using the existing expert policy to generate expert trajectory to expert_data_dir folder]
    """
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    ## 1, run expert policy to generate roll-outs
    with tf.Session():
        tf_util.initialize()

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

@timeit
def bc_policy_estimate(lr: float=1e-3, epoches: int=80, iterations: int=5):
    """[Using behaviorial cloning to do policy estimate and use estimated policy to do task]
        Arguments:
            lr {[float]} -- [learning rate value]
    """
    expert_data = loadExpertOA()
    # construct full-connected layers
    op, loss, O, A, is_training, X_out = neuralNetwork(expert_data, lr)

    with tf.Session() as sess:
        # 1. Training phase
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            indexes = np.arange(expert_data['observations'].shape[0])
            np.random.shuffle(indexes)
            # refer to https://github.com/tensorflow/tensorflow/issues/13847 and https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
            loss_vals = []
            batch_size = 2048
            # batch_size = len(indexes)
            for batch in gen_batches(len(indexes), batch_size):
                if batch_size == len(indexes):
                    for _ in range(iterations):
                        _, loss_val = sess.run([op, loss], feed_dict={O: expert_data['observations'][indexes[batch]], 
                                                    A: expert_data['actions'][indexes[batch]], 
                                                    is_training: True
                                                    })
                else: 
                    _, loss_val = sess.run([op, loss], feed_dict={O: expert_data['observations'][indexes[batch]], 
                                                                A: expert_data['actions'][indexes[batch]], 
                                                                is_training: True
                                                                })
                loss_vals.append(loss_val)
            if epoch % 5 == 0:
                print("Epoch=%s, loss=%s" % (epoch, np.average(loss_vals)))
        print("Final Epoch=%s, loss=%s" % (epoches-1, np.average(loss_vals)))

        # 2. Prediction phase
        returns = []

        # using the same rollout
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # trajectory sampling
                action = sess.run(X_out, feed_dict={O: obs[None,:], 
                                                    is_training: False
                                                    })
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

@timeit
def dagger_policy_estimate(lr: float=1e-3, epoches: int=60, iterations: int=5, dagger_loop: int=5):
    """[dataset aggregation algorithm]
    
    Keyword Arguments:
        lr {float} -- [learning rate value] (default: {1e-3})
        epoches {int} -- [training epoches] (default: {60})
        iterations {int} -- [iteration time for each epoch] (default: {100})
    """
    expert_data = loadExpertOA()
    # construct full-connected layers
    op, loss, O, A, is_training, X_out = neuralNetwork(expert_data, lr)
    # expert policy to use a human labeled action
    expert_fn = load_policy.load_policy(args.expert_policy_file)
    
    for dagger_i in range(dagger_loop):
        with tf.Session() as sess:
            # 1. Training phase to learn pi(theta)
            sess.run(tf.global_variables_initializer())
            tf_util.initialize()
            for epoch in range(epoches):
                indexes = np.arange(expert_data['observations'].shape[0])
                np.random.shuffle(indexes)
                loss_vals = []
                batch_size = 1024
                # batch_size = len(indexes)
                for batch in gen_batches(len(indexes), batch_size):
                    if batch_size == len(indexes):
                        for _ in range(iterations):
                            _, loss_val = sess.run([op, loss], feed_dict={O: expert_data['observations'][indexes[batch]], 
                                                        A: expert_data['actions'][indexes[batch]], 
                                                        is_training: True
                                                        })
                    else: 
                        _, loss_val = sess.run([op, loss], feed_dict={O: expert_data['observations'][indexes[batch]], 
                                                                    A: expert_data['actions'][indexes[batch]], 
                                                                    is_training: True
                                                                    })
                    loss_vals.append(loss_val)
                if epoch % 10 == 0:
                    print("Epoch=%s, loss=%s" % (epoch, np.average(loss_vals)))
            print("Final Epoch=%s, loss=%s" % (epoches-1, np.average(loss_vals)))
            
            # 2. Generate D_pi from D(pi(theta)) -> Neural Network : only increase 1 roll outs
            for i in range(args.num_rollouts):
                obs = env.reset()
                done = False
                steps = 0
                d_pi_obs = []; d_pi_action = []
                while not done:
                    # trajectory sampling
                    action = sess.run(X_out, feed_dict={O: obs[None,:], 
                                                        is_training: False})
                    obs, r, done, _ = env.step(action)
                    # using expert action by expert_fn
                    action_by_expert = expert_fn(obs[None,:])
                    d_pi_obs.append(obs)
                    d_pi_action.append(action_by_expert)
                    if steps >= max_steps:
                        break
            
                # 3. update expoert_data
                expert_data['observations'] = np.concatenate((expert_data['observations'], np.array(d_pi_obs)),axis=0)
                expert_data['actions'] = np.concatenate((expert_data['actions'], np.array(d_pi_action)),axis=0)
        
            if dagger_i == dagger_loop - 1:
                # the last loop to estimate rewards
                returns = []

                # using the same rollout
                for i in range(args.num_rollouts):
                    print('iter', i)
                    obs = env.reset()
                    done = False
                    totalr = 0.
                    steps = 0
                    while not done:
                        # trajectory sampling
                        action = sess.run(X_out, feed_dict={O: obs[None,:], 
                                                            is_training: False})
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



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--expert_data", type=str, default="expert_data", help="Export data folder")
    parser.add_argument("--only_expert_generate", type=int, default=0)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20, help='Number of expert roll outs')
    args = parser.parse_args()
    expert_data_dir = args.expert_data

    # gym env
    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    # Preparation of expert policy to generate archived expert trajectory
    if args.only_expert_generate == 0:
        # if not skip expert generate, then run it out
        expert_policy_out()
    elif args.only_expert_generate == 1:
        # Comparsion using BC to esimate policy
        bc_policy_estimate()
    elif args.only_expert_generate == 2:
        # Comparsion using Dagger to esimate policy
        dagger_policy_estimate()
    else:
        raise ValueError("Invalid argument parameter: args.only_expert_generate is only in [0, 1, 2]")
