# RLBerkelyFa2018
Homework of CS 294-112 at UC Berkeley Deep Reinforcement Learning, refer to http://rail.eecs.berkeley.edu/deeprlcourse/


## Summary

| Assignment | Descrirption         |  Code          | Key Points               | Summary   |
|:----------:|:--------------------:|:------------------------:|:------------------------:|:---------:|
| Homework 1 | [Imitation Learning](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw1.pdf)|hw1/| $\pi_{\theta}$ as expert, then using as human judgement for imitation learning| Still Not finished with logz and visualization |
| Homework 2 | [Policy Gradient](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw2.pdf)|hw2/| refer to hw2/answers/hw2_instructions_answer_by_orlando.tex for math and status summary, also refer to [result](https://github.com/llv22/RLBerkelyFa2018/tree/master/hw2)| Bonus implementation not so good as expected | 
| Homework 3 | [Q-Learning and Actor-Critic](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf)|hw3/| Actor/Critic update of network in [dqn.py](hw3/dqn.py) and using double-Q for value evaluation in the same network in [train_ac_f18.py](hw3/train_ac_f18.py) | Double-Q implementation still have very long startup plateau |
| Homework 4 | [Model-Based RL](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw4.pdf)|hw4/|Cross-Entropy Method and Multiple-Loss is a new approach for action exploration|Cross-Entropy Method and Multiple-Loss still don't have a better result than standard q3, just have a trail for result identification|
| Homework 5A | [Exploration](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5a.pdf)|hw5/sac|RBF, EX2 for better exploration|rbf somehow don't play so good, I just wondering if there exists bugs in code, need to review|
| Homework 5B | [Soft Actor-Critic](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5b.pdf)|hw5/exp|SAC with 2 Q funtions; softmax based on expectation of Q value/V value; stop gradient of REINFORCE policy graident; the reparameterization trick with policy gradients|Hybperparameter trial is still not finished; 2qf has more restricted boundary than 1qf, a little bit concious about this. but from math initution, somehow make sense|
| Homework 5C | [Meta Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw5c.pdf)|hw5/meta|Meta Reinforcement Learning via RNN |Just wonder if professor's idea about training/testing skew is the same/similiar|