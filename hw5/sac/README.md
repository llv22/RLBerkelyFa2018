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

* Policy Loss

    Case 1: **Reinforcement Learning Loss**

    $$ \nabla_{\phi} J_{\pi}(\phi) = \mathop{{}\mathbb{E}}_{s \sim D} \lbrace \mathop{{}\mathbb{E}}_{a \sim \pi_{\phi}(a|S)} [\nabla_{\phi} \log \pi(a|s) (\alpha \cdot log \pi_{\phi} (a|s) - Q_{\theta}(s, a)) + b(s) | s]  \rbrace $$ 

    Case 2: **Reparameter Trick**  

    $$ \nabla_{\phi} J_{\pi}(\phi) = \mathop{{}\mathbb{E}}_{s \sim D} \lbrace \mathop{{}\mathbb{E}}_{\epsilon \sim N(0, I)} [\alpha \cdot \log \pi_{\phi}(f_{\phi}(\epsilon;s)|s) - Q_{\theta}(s, f_{\phi}(\epsilon; s)) | s] \rbrace $$ 

* Bonus Task - Questions for Problem 1

    1. In Task A, what was your choice of baseline, and why did you choose it?

    2. What are the pros and cons of the two types of gradient estimators?

    3. Why can we not use the reparameterization trick with policy gradients?

    4. We can minimize the policy loss in Equation 9 using oﬀ-policy data. Why
    is this not the case for standard actor-critic methods based on policy
    gradients, which require on-policy data?

## Problem 2: Squashing
## Problem 3: SAC with Two Q-Functions
## Problem 4: Experiments


