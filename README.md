<h1 align=center> Baxter Environment </h1>
<h3><b>About the environment</b></h3>

This environment is made by our team from **RoboReG** of Robotics Club,IIT BHU to tackle an open challenge in the field of robotics, which is *Simultaneous Work*. Generally, Humans have a good efficiency in managing different things simultaneously using both hands. Hence, we want to explore how the <b>Reinforcement Learning</b> Algorithms are trying to achieve this task comparable to humans or even better. For this purpose **Baxter** humanoid robot is used as the Agent. It is a gym compatible environment made using simulator [PyBullet](https://pybullet.org/)<br>

<h3 align=center> (SMA) Simultaneous Multitasking Agent</h3>
<img width="800" height="400" src="media/env.png"><br>

**Baxter** is an industrial humanoid robot built by [Rethink Robotics](https://www.rethinkrobotics.com/). We used it in our environment to get a good idea of how the humanoid agent will behave when left in the same scenario as that of a human.For more info on baxter visit [this](https://www.ohio.edu/mechanical-faculty/williams/html/pdf/BaxterKinematics.pdf)
<br>
<p align= 'center'>
<img width="327" height="300" src="media/baxter.png">
<img width="327" height="300" src="media/baxter_desc.png">
</p>

## Initial plannings
We have divided our work into several steps which we want to achieve one by one.

- [x] Creating experimental and environmental setup
- [ ] Finding **Joint Reward** Function
- [ ] Learn easy simultaneous tasks first
- [ ] Increase task complexity
- [ ] Increase interaction complexity

We made the required environment with aiming some basic works like simultaneous lifting, touching etc. Now we are working on our MDP (*Markov Decision Process*) formulation along with joint reward function.

##  **Installation Guidelines**
Anyone who want to conribute to this repo or want to modify it, can fork the repo and create pull requests [here](https://github.com/Robotics-Club-IIT-BHU/gym-Baxter_SMA/pulls). One can also take this environment as a ready made setup for aiming the same challenges.
### Dependencies
- python3
- pybullet
- gym
- opencv
- numpy
- matplotlib

After cloning the repo run the following commands to install the required setup

- Install the environment using `pip install -e baxter-env`. It will automatically install the required additional dependencies

- In case there are problems with the PyBullet installation, you can refer to this [guide](https://github.com/Robotics-Club-IIT-BHU/Robo-Summer-Camp-20/blob/master/Part1/Subpart%201/README.md).

- After that use the following lines to import the environment in your workspace.

``` python
import gym
import baxter_env

env = gym.make('baxter_env-v0')
```

<h2>Functions in our environment</h2>

After importing the environment, there are different functions available which can be used for various purposes, some of the important ones are listed below, for more details on the functions and arguments, you can see [here](baxter-env/baxter_env/envs/baxter_env.py)

- `env.getImage()`<br>
    This will return a tuple containing 128x128 RGB image along with the depth map of the environment from baxter's eye view.
- `env.step()`<br>
    This will take list of actions and perform that action along with providing required information of states.
- `env.moveVC()`<br>
    This will move the joints to a specified position by taking the index of that joint.
- `env.render()`<br>
    Function to render the eye view image of the environment.
- `env.reset()`<br> 
    This will reset the whole environment and gives the information regarding the state.

There are other functions also which we have made for our purpose like 
- `env.BeizerCurve()`<br> 
    This is an implementation of a family of curves which gives a trajectory passing through some points on the basis of some weights.
- `env.getReward()`<br>
    This will return the reward for trying simultaneous touching  (naive implementation, working on making it better).

Some implementation related to these functions are available [here](examples/)

<h2 align = 'center'>Snapshots</h2>
<p align='center'>
Few images and gifs of our environment and robot in action using some classical techniques.<br>
 <p float=left>
     <img width="400" height="250" src="media/baxter.gif">
     <img width="400" height="250" src="media/env-gif.gif">
 </p>
 <p align="center">
    <img height="250" width="300" src="media/image_data.png">
    <img  height="250" width="300" src="media/depth_data.png"><br>
    <b>Image data with depth map</b><br>
 </p>
</p>

## References 
- [Pybullet URDFs](https://github.com/erwincoumans/pybullet_robots)
- [Baxter](https://www.ohio.edu/mechanical-faculty/williams/html/pdf/BaxterKinematics.pdf)
- [Bezier Curves Visualisation](https://www.desmos.com/calculator/kiqc4acwwk)

## **Makers and Maintainers**

<table>
   <td align="center">
      <a href="https://github.com/monako2001">
         <img src="https://avatars2.githubusercontent.com/u/56964886?s=400&v=4" width="100px;" alt=""/>
         <br />
         <sub>
            <b>Mainak Samanta</b>
         </sub>
      </a>
      <br />
   </td>
   <td align="center">
      <a href="https://github.com/Akshatsood2249">
         <img src="https://avatars3.githubusercontent.com/u/68052998?s=400&u=d83d34a2596dc22bef460e3545e76469d2c72ad9&v=4" width="100px;" alt=""/>
         <br />
         <sub>
            <b>Akshat Sood</b>
         </sub>
      </a>
      <br />
   </td>
   <td align="center">
      <a href="https://github.com/Amshra267">
         <img src="https://avatars1.githubusercontent.com/u/60649720?s=460&u=9ea334300de5e3e7586af294904f4f76c24f5424&v=4" width="100px;" alt=""/>
         <br />
         <sub>
            <b>Aman Mishra</b>
         </sub>
      </a>
      <br />
   </td>
   <td align="center">
      <a href="https://github.com/arch-raven">
         <img src="https://avatars1.githubusercontent.com/u/55887731?s=400&u=7e70757a37e409360fda094f1f61fa1f261f8111&v=4" width="100px;" alt=""/>
         <br />
         <sub>
            <b>Aditya Kumar</b>
         </sub>
      </a>
      <br />
   </td>
</table>

## **Our Mentors**

<table>
   <td align="center">
      <a href="https://github.com/lok-i">
         <img src="https://avatars1.githubusercontent.com/u/54435909?s=400&u=29af076049dab351b2e43621e9a433919bf50fb1&v=4" width="100px;" alt=""/>
         <br />
         <sub>
            <b>Lokesh Krishna</b>
         </sub>
      </a>
      <br />
   </td>
   <td align="center">
      <a href="https://github.com/NiranthS">
         <img src="https://avatars3.githubusercontent.com/u/44475481?s=400&v=4" width="100px;" alt=""/>
         <br />
         <sub>
            <b>Niranth Sai</b>
         </sub>
      </a>
      <br />
   </td>
</table>
