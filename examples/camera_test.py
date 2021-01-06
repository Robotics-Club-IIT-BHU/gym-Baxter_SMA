import gym
import baxter_env
import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = gym.make('baxter_env-v0')
    s = env.reset()
    # Get the image and depth map from env
    img, depth= env.getImage()

    # Plot the image(image is in BGR so plt will plot it in RGB)
    plt.subplot(2, 1, 1)
    plt.title("Image")
    plt.imshow(img)

    # Plot the depth map of image
    plt.subplot(2, 1, 2)
    plt.title("Depth Image")
    plt.imshow(depth)

    plt.show()

    while True:
        p.stepSimulation()