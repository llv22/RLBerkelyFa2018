# CS294-112 HW 5b: Soft Actor Critic
Original code from Tuomas Haarnoja, Soroush Nasiriany, and Aurick Zhou for CS294-112 Fall 2018

Dependencies:
 * Python **3.4.5**
 * Numpy version **1.15.2**
 * TensorFlow version **1.10.0**
 * tensorflow-probability version **0.4.0**
 * OpenAI Gym version **0.10.8**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.59**
 * seaborn version **0.9.0**

You will implement `sac.py`, and `nn.py`.

See the [HW5 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5b.pdf) for further instructions.

## Problem 1: Loss Function
* Essential Equation

### 1.1 Policy Loss

* Case 1: **Reinforcement Learning Loss**

$$\nabla_{\phi} J_{\pi}(\phi) = \mathop{\mathbb{E}}_{s \sim D} \left[ \mathop{\mathbb{E}}_{a \sim \pi_{\phi}(a|S)} [\nabla_{\phi} \log \pi(a|s) (\alpha \cdot log \pi_{\phi} (a|s) - Q_{\theta}(s, a)) + b(s) | s]  \right]$$

* Case 2: **Reparameter Trick**

$$\nabla_{\phi} J_{\pi}(\phi) = \mathop{\mathbb{E}}_{s \sim D} \left[ \mathop{\mathbb{E}}_{\epsilon \sim N(0, I)} [\alpha \cdot \log \pi_{\phi}(f_{\phi}(\epsilon;s)|s) - Q_{\theta}(s, f_{\phi}(\epsilon; s)) | s] \right]$$

### 1.2 Bonus Task - Questions for Problem 1

1. In Task A, what was your choice of baseline, and why did you choose it?   
   Answer:
    - Baseline b(s): choose value function $V(s)$
    - Reason: via softmax Q-learning, target value $V(s')=softmax_{a'}Q_{\phi}(s',a')=\log \int exp(Q_{\phi}(s',a'))da'$, $\pi(a|s)=exp(Q_{\phi}(s,a)-V(s))=exp(A(s,a))$ means that we need to use independent $Q_{\phi}$ target value network to evaluate $V(s')$. In order to simpilify, use independent target V network for V(s')

2. What are the pros and cons of the two types of gradient estimators?   
   Answer:
    - REINFORCE $\nabla_{\phi} J_{\pi}(\phi) = \mathop{\mathbb{E}}_{s \sim D} \left[ \mathop{\mathbb{E}}_{a \sim \pi_{\phi}(a|S)} [\nabla_{\phi} \log \pi(a|s) (\alpha \cdot log \pi_{\phi} (a|s) - Q_{\theta}(s, a)) + b(s) | s]  \right]$
      * pros: use b(s) to guarantee value function NOT hardmax with Q-value learning(also pros of soft Q-learning)
      * cons:  
        I. Inner expectation require explicit sample action and stop gradient to external expectation, computation cost increased and expect avoid gradient backpropagation  
        II. NO prior assumpation of action sample distribution. (NOT obvious cons, but mabye have bigger sample-action cost)
    - Reparameterize tricks $\nabla_{\phi} J_{\pi}(\phi) = \mathop{\mathbb{E}}_{s \sim D} \left[ \mathop{{}\mathbb{E}}_{\epsilon \sim N(0, I)} [\alpha \cdot \log \pi_{\phi}(f_{\phi}(\epsilon;s)|s) - Q_{\theta}(s, f_{\phi}(\epsilon; s)) | s] \right]$
      * pros: With prior assumption of action sample distribution, NO stop gradient and addtional computation cost
      * cons: Without baseline b(s), it's a kind of hardmax, so Q value will fluctuate as policy improve, also high bias without b(s). (see also A(s) in policy graident)

3. Why can we not use the reparameterization trick with policy gradients?   
    $\nabla_{\phi} J_{\pi}(\phi) = \mathop{\mathbb{E}}_{s \sim D} \left[ \mathop{\mathbb{E}}_{a \sim \pi_{\phi}(a|S)} [\nabla_{\phi} \log \pi(a|s) (\alpha \cdot log \pi_{\phi} (a|s) - Q_{\theta}(s, a)) + b(s) | s]  \right]$
    Answer:
    As inner expection is based on a single-action for the policy, outer expectation is bbase don data in replay buffer. And action isn't dependent on the policy parameter $\phi$, so without **the same model parameter $\phi$**, we can't do reparameter trick

4. We can minimize the policy loss in Equation 9 using oﬀ-policy data. Why is this not the case for standard actor-critic methods based on policy gradients, which require on-policy data?   
   Answer:  
    - Similar to 3, $J_{\pi}(\phi) = \mathop{\mathbb{E}}_{s \sim D} \left[ \mathop{\mathbb{E}}_{a \sim \pi_{\phi}(a|S)} [\nabla_{\phi} \log \pi(a|s) (\alpha \cdot log \pi_{\phi} (a|s) - Q_{\theta}(s, a)) + b(s) | s]  \right]$, inner expectation is sampled from given s by actions NOT based on policy
    - For standard AC based on policy gradient $J(\theta)=\log \pi_{\theta}(a|s)(Q^{\pi}(s_t, a_t) - V^{\pi}(s_t))$, if just minimize this $J(\theta)$, the $\theta$ is only used "on-policy" data, not good for generalized data.

## Problem 2: Squashing

### 2.1 Math Reasoning  
Reference: https://tex.stackexchange.com/questions/74125/how-do-i-put-text-over-symbols

* **Invertible functions' chain rule**

$$Z^{(N)} = (f_N \circ \cdots \circ f_1)(z^0) \iff \log p(z^{(N)}) = log(p(z^{(0)})) - \sum_{i=1}^{N} \left| \det(\frac{\partial f_i(z^{(i-1)})}{\partial z^{(i-1)}}) \right|$$

where $\frac{\partial f_i(z^{(i-1)})}{\partial z^{(i-1)}}$ is Jacobian of $f_i$, and $\det$ is the determinant.

* **Squashing via tanh for action A**

<!--
$$ a = \tanh \left(b_{\phi}(s) + A_{\phi}(s)\epsilon \right) \iff z=f_1(\epsilon) \triangleq b(s) + A(s) \epsilon, a = f_2(z) \triangleq \tanh(z) $$

or 

$$ a = \tanh \left(b_{\phi}(s) + A_{\phi}(s)\epsilon \right) \iff z=f_1(\epsilon) \equiv b(s) + A(s) \epsilon, a = f_2(z) \equiv \tanh(z) $$

or 
-->

$$a = \tanh \left(b_{\phi}(s) + A_{\phi}(s)\epsilon \right) \iff z=f_1(\epsilon) \stackrel{\text{def}}{=} b(s) + A(s) \epsilon, a = f_2(z) \stackrel{\text{def}}{=}\tanh(z) $$

As making f=$\tanh$, we have the Jacobian is a diagonal matrix with $\frac{\partial \tanh(z_i)}{\partial z_i} = 1 - \tanh ^2(z_i)$, finally we get

$$\log \left|  det(\frac{\partial f_2(z)}{\partial z}) \right| = \sum_{i=1}^{|A|} \log \left(1 - \tanh^2(z_i)\right)$$

Bonus Task:

$$\log \left(1 - \tanh^2(z_i)\right) = 2 \log 2 + 2 z_i - softplus(2z_i), \text{where } softplus(x) = \log(1+e^x)$$

## Problem 3: SAC with Two Q-Functions

* Key points: using Q1 and Q2 with different parameter $\theta_1$ and $\theta_2$, then use $Q(s,a)=\min(Q_1(s,a), Q_2(s,a))$ to restrict the sampling upper bound.

## Problem 4: Experiments

### 4.1 Task A - REINFORCE and preparameterized policy gradient on HalfCheetah
* Fix-warning:   
1. Find root reason to find warning trace   
```bash
PYTHONWARNINGS='error::ImportWarning' python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 3
Traceback (most recent call last):
  File "train_mujoco.py", line 5, in <module>
    import logz
  File "/Users/i058959/Documents/algorithm/machine_learning/DL_RL/RLCS294_by_ucberkeley/homework/RLBerkelyFa2018/hw5/sac/logz.py", line 20, in <module>
    import tensorflow as tf
  File "/Users/i058959/miniconda3/lib/python3.6/site-packages/tensorflow/__init__.py", line 22, in <module>
    from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
  File "/Users/i058959/miniconda3/lib/python3.6/site-packages/tensorflow/python/__init__.py", line 63, in <module>
    from tensorflow.python.framework.framework_lib import *  # pylint: disable=redefined-builtin
  File "/Users/i058959/miniconda3/lib/python3.6/site-packages/tensorflow/python/framework/framework_lib.py", line 30, in <module>
    from tensorflow.python.framework.sparse_tensor import SparseTensor
  File "/Users/i058959/miniconda3/lib/python3.6/site-packages/tensorflow/python/framework/sparse_tensor.py", line 26, in <module>
    from tensorflow.python.framework import tensor_util
  File "/Users/i058959/miniconda3/lib/python3.6/site-packages/tensorflow/python/framework/tensor_util.py", line 32, in <module>
    from tensorflow.python.framework import fast_tensor_util
  File "tensorflow/python/framework/fast_tensor_util.pyx", line 4, in init tensorflow.python.framework.fast_tensor_util
ImportWarning: can't resolve package from __spec__ or __package__, falling back on __name__ and __path__
```
2. warnings filter   
Add following logic in sac.py (the first model, which is loaded for tensorflow)
```python
import warnings
warnings.filterwarnings('ignore', message="can't resolve package from __spec__ or __package__, falling back on __name__ and __path__", category=ImportWarning, lineno=219)
```

* Experiment  
As current gpu consumption isn't so big, just use process parallelization
```bash
# case 1: reparameterize = False
python train_mujoco.py --env_name HalfCheetah-v2 --exp_name reinf -e 3 -p True
# case 2: reparameterize = True
python train_mujoco.py --env_name HalfCheetah-v2 -re True --exp_name reparam -e 3 -p True
```
