<h1 align=center> Baxter Environment </h1>
<h3><b>About the environment</b></h3>

This environment is made by our team from **Robotics-Club** of IIT BHU to tackle an open challenge in the field of robotics, which is *Simultaneous work*. Generally, Humans have a good efficiency in managing different things simultaneously using both hands. Hence, we want to explore how the <b>Reinforcement Learning</b> Algorithms are trying to achieve this task comparable to humans or even better. For this purpose **Baxter** humanoid robot is used as the Agent. It is a gym compatible environment made using simulator [PyBullet](https://pybullet.org/)<br>

<h3 align=center> (SMA) Simultaneous Multitasking Agent</h3>
<p align = 'center'>
<img width="700" height="300" src="media/env.png"><br>
<\p>

Baxter was an industrial humanoid robot built by [Rethink Robotics](https://www.rethinkrobotics.com/).We used it in our environment to get a good idea of how the huamnoid agent will behave when left in the same scenario as that of a human.For more info on baxter visit [this]()

<p>
<img width="300" height="300" src="media/baxter.png">
<img width="300" height="300" src="media/baxter_desc.png">
<h4 align='center'><b>Baxter</b></h4>
</p>

## Initial plannings
We have divided our work into several steps which we want to achieve one by one.

- [x] Creating experimental and environmental setup
- [ ] Finding **Joint Reward** Function
- [ ] Learn easy simultaneous tasks first
- [ ] Increase task complexity
- [ ] Increase interaction complexity

We made the required environment with aiming some basic works like simultaneous lifting, touching etc. Now we are working on our MDP formulation along with joint reward function.

##  **Installation Guidelines**
Anyone who want to conribute to this repo or want to modify it, will fork the repo and can create pull requests. One can also take this environment as a ready made setup for aiming the same challenges.
### Dependencies
- Python3
- PyBullet
- gym
- OpenCV
- numpy
- Matplotlib

After cloning the repo run following commands to quickly done with the required setup

- install the environment using `pip install -e baxter-env`. It will automatically install the required additional dependencies

- In case there are problems with the PyBullet installation, you can refer to this [guide](https://github.com/Robotics-Club-IIT-BHU/Robo-Summer-Camp-20/blob/master/Part1/Subpart%201/README.md).

- After that use few lines to import it in your workspace.

``` python
import gym
import baxter_env

env = gym.make('baxter_env-v0')
```

<h2>Functions in our environment</h2>

After importing the environment, there are different functions available in the environment which can be used for various purposes, some of the important ones is listed below, for more details on the functions and arguments, you can see [here]()

- `env.getImage()`<br>
    This will return an RGB image of the environment from baxter's eye view.
- `env.step()`<br>
    This will take list of actions and perform that action along with providing required information of states..
- `env.moveVC()`<br>
    This will move the joints to a specified position by taking the index of that joint.
- `env.roll_dice()`<br>
    This will simulate the rolling of the dice and will give the next shape and colour to which the car shall have to move.
- `env.move_husky()`<br>
    This will be used to give the motor velocity to each wheel individually of the car.
- `env.reset()`<br> 
    This will reset the whole environment and gives the information regarding the state.

<p float=left>
 <img width="360" src="media/baxter.gif">
 <img  width="200" src="media/top-view.png">
</p>

## References 
- [Pybullet URDFs](https://github.com/erwincoumans/pybullet_robots)
- Stable Baselines [Colab Tutorials](https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/1_getting_started.ipynb) &nbsp; [Docs](https://stable-baselines.readthedocs.io/en/master/)
