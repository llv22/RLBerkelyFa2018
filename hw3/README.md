# CS294-112 HW 3: Q-Learning

Dependencies:
 * Python **3.6**
 * Numpy version **1.14.5**
 * TensorFlow version **1.10.5**
 * MuJoCo version **1.50** and mujoco-py **1.50.1.56**
 * OpenAI Gym version **0.10.8**
 * seaborn
 * Box2D==**2.3.2**
 * OpenCV
 * ffmpeg

Before doing anything, first replace `gym/envs/box2d/lunar_lander.py` with the provided `lunar_lander.py` file.

The only files that you need to look at are `dqn.py` and `train_ac_f18.py`, which you will implement.

See the [HW3 PDF](http://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw3.pdf) for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.

## 1. Installation
* For mac preparation
    ```bash
    brew install ffmpeg
    conda install opencv-python
    ```

* For windows version of gym[atari], refer to https://stackoverflow.com/questions/42605769/openai-gym-atari-on-windows

    1. atari-py installation for windows
    ```bash
    pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
    # If you have any distutils supported compiler you can install from sources:
    pip install git+https://github.com/Kojoley/atari-py.git
    ```
    2. ffmpeg installation for windows
    ```bash
    conda install -c conda-forge ffmpeg
    ```
    3. swig installation for windows (box2d dependency)
    ```bash
    conda install swig
    ```

## 2. Exercise