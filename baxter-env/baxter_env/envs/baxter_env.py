import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import pybullet_data
import time
import cv2
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

class BaxterEnv(gym.Env):
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self):

        """
        Constructor Function:for creating the baxter environment
        Arguments:
            None
        Returns :  
            None   
        """
        try:
            p.connect(p.GUI)
            self.render_mode = p.ER_BULLET_HARDWARE_OPENGL
        except:
            p.connect(p.DIRECT)
            self.render_mode = p.ER_TINY_RENDERER
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)
        p.setGravity(0, 0, -0.2)
        
        self.baxter = p.loadURDF("data/baxter_common/baxter_description/urdf/toms_baxter.urdf", [0,0,0], useFixedBase=True)
        self.moveVC(self.baxter, [12, 34], [-0.2, 0.2]) # to prevent baxter hands from overlapping with the boxes
        
        self.moveable_joints = [5, 12, 13, 14, 15, 16, 18, 19, 27, 29, 34, 35, 36, 37, 38, 40, 41, 49, 51] # only revolute joints are allowed to move
        self.init_poses = [np.round_(p.getJointState(self.baxter, j)[0], decimals = 4) for j in self.moveable_joints] 
        time.sleep(0.1)

        self.bezier_right=p.getLinkState(self.baxter, 29)[0]
        self.bezier_left=p.getLinkState(self.baxter, 51)[0]
        
        self.huskypos = [0, 0, 1]
        self.table_pos = [0.7, 0, -0.9]
        
        self.boxes_num = [0.1, 0.2, 0.3, 0.4, 0.5] # for different type of boxes
        
        self.orn_t = p.getQuaternionFromEuler([0, 0, 1.58])
        self.table = p.loadURDF("data/table/table.urdf", self.table_pos, self.orn_t)
        self.orn = p.getBasePositionAndOrientation(self.table)
        
        box_1_num = np.random.choice(self.boxes_num)# choose one type box
        self.h1 = 0.1 if box_1_num == 0.1 else 0.2
        self.h1 = self.h1 + 0.08 # will be used to ensure touching and assigning task for box1
        
        box_2_num = np.random.choice(self.boxes_num)# choose one type box
        self.h2 = 0.1 if box_2_num == 0.1 else 0.2
        self.h2 = self.h2 + 0.08 # will be used to ensure touching and assigning task for box2
    
        self.box_t_poses = self.box_poses() # get random positions to load boxes
        
        self.box1 = p.loadURDF("data/blocks/block_1_" + str(box_1_num) + ".urdf", self.box_t_poses[0]) 
        self.box2 = p.loadURDF("data/blocks/block_2_" + str(box_2_num) + ".urdf", self.box_t_poses[1])
        
        self.orn1 = p.getBasePositionAndOrientation(self.box1)
        self.orn2 = p.getBasePositionAndOrientation(self.box2)
        
        print("Position and Orientation of Table: ", self.orn1, self.orn2)
        
        n1,n=0,1000
        
        while(n1<n):
            
            p.setJointMotorControl2(self.baxter, 12, p.VELOCITY_CONTROL, targetVelocity = 0, force= 0.2)# experimental
            p.setJointMotorControl2(self.baxter, 34, p.VELOCITY_CONTROL, targetVelocity = 0, force= 0.2)# experimental
            p.stepSimulation()
            
            n1 += 1
        
        self.discount = 0.9
        self.done = False

        self.bezier_timesteps = 0
        self.bezier_left = []
        self.bezier_right = []

    def box_poses(self):
        
        """
        Function to return random position of boxes
        Arguments:
            None
        Returns:
            Random positions of boxes
        """
        while True:
            poses_x = np.random.uniform(self.table_pos[0]-0.3, self.table_pos[0]+0.3, 2)
            if abs(poses_x[0]-poses_x[1])>0.3: # to ensure no overlap of both boxes in x
                break
                
        while True:
            poses_y = np.random.uniform(self.table_pos[1]-0.5, self.table_pos[1]+0.5, 2)
            if abs(poses_y[0]-poses_y[1])>0.3: # to ensure no overlap of both boxes in y
                break
                
        pos_1 = [np.round_(poses_x[0], decimals = 4), np.round_(poses_y[0], decimals = 4), 0.2]
        pos_2 = [np.round_(poses_x[1], decimals = 4), np.round_(poses_y[1], decimals = 4), 0.2]
        
        return [pos_1, pos_2]
    
    def reset(self):
        
        """
        Function to reset the whole simulation
        Arguments:
            None
        Returns:
            Dictionary containing information about the environment
        """
        for joint_index, target_pos in zip(self.moveable_joints, self.init_poses):
            p.resetJointState(self.baxter, joint_index, target_pos, 0.0)
            
        self.box_t_poses = self.box_poses()
        p.resetBasePositionAndOrientation(self.box1, self.box_t_poses[0], [0.0, 0.0, 0.0, 1.0])
        p.resetBasePositionAndOrientation(self.box2, self.box_t_poses[1], [0.0, 0.0, 0.0, 1.0])
        
        n1 = 0
        
        while(n1<1000): 
            
            p.stepSimulation()
            n1 += 1
            
        
        self.hand1_pos = p.getLinkState(self.baxter, 29)[0]
        self.hand2_pos = p.getLinkState(self.baxter, 51)[0]
        
        img, depth = self.getImage()
        
        self.state_dict = {"hand_ends": [self.hand1_pos, self.hand2_pos], "eye_view": img, "depth": depth} 
        return self.state_dict
    
    def render(self):
        
        """
        Function to render top view image of the Environment
        Argumets:
            None
        Returns:
            Image
        """
        img, _ = self.getImage()
        
        return img

    def getImage(self):
        
        """
        Function to extract eye view image of environemnt
        Arguments:
            None
        Returns:
            Image
        """
        
        cam_info = p.getLinkState(self.baxter, 7)
        
        cam_eye_pos = list(cam_info[0])
        eye_pos = [cam_eye_pos[0], cam_eye_pos[1], cam_eye_pos[2]]
        target_pos = [1.5, 0, -0.5]
        upvec = [cam_eye_pos[0], cam_eye_pos[1], cam_eye_pos[2]-0.8]

        self.view_matrix= p.computeViewMatrix(eye_pos, target_pos, upvec)
        self.width = 128
        self.height = 128
        self.fov = 90
        self.aspect_ratio = self.width/self.height
        self.near = 0.1
        self.far = 10 

        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect_ratio, self.near, self.far)
        self.image_info = p.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix, 
                                           shadow = True, renderer = self.render_mode)
        
        image = np.array(self.image_info[2]).reshape((self.height, self.width, 4))
        image = image[:, :, [0, 1, 2]]
        image = image.astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        depth_buffer_tiny = np.reshape(self.image_info[3], [self.width, self.height])
        depth_tiny = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_tiny)
        depth_tiny = depth_tiny.astype("float32")

        return image, depth_tiny

    def getImageUD(self):
        
        """
        Function to extract top view image of environemnt
        Arguments:
            None
        Returns:
            Image
        """
        self.var = 3
        
        eye_pos = [self.table_pos[0], self.table_pos[1], self.table_pos[2]+2]
        target_pos = [self.table_pos[0], self.table_pos[1], self.table_pos[2]+0.5]
        
        upvec = [1, 0, 0]
        
        self.view_matrix= p.computeViewMatrix(eye_pos, target_pos, upvec)
        self.width = 128
        self.height = 128
        self.fov = 60
        self.aspect_ratio = self.width/self.height
        self.near = 0.1
        
        self.far = self.var
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, self.aspect_ratio, self.near, self.far)
        self.image_info = p.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix, 
                                           shadow = True, renderer = self.render_mode)
        
        image = np.array(self.image_info[2]).reshape((self.height, self.width, 4))
        image = image[:, :, [0, 1, 2]]
        image = image.astype("uint8")
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        depth_buffer_tiny = np.reshape(self.image_info[3], [self.width, self.height])
        depth_tiny = self.far * self.near / (self.far - (self.far - self.near) * depth_buffer_tiny)
        depth_tiny = depth_tiny.astype("float32")

        return image, depth_tiny

    def moveVC(self, urdf, joint_indexes, target_poses): 
        
        """
        Function to move the joints to specified final positions
        Argumets:
            [URDF,List of Joint Indexes,Final Position]
        Returns:
            None
        """
        for joint_index, target_pos in zip(joint_indexes, target_poses):

            curr_joint = p.getJointState(urdf, joint_index)
            curr_joint_pos = np.round_(curr_joint[0], decimals = 4)

            min_pos, max_pos, f, mv = p.getJointInfo(urdf, joint_index)[8:12]
            
            if target_pos < min_pos :
                target_pos = min_pos
            if target_pos > max_pos :
                target_pos = max_pos
            if target_pos - curr_joint_pos < 0:
                b = -1
            else:
                b = 1

            c3 = b*(0.02)
           # c1 = 0
           # c2 = 0
            
            while (b*curr_joint_pos) < (b*target_pos):

                prev_curr_joint_pos = curr_joint_pos
                v = (target_pos-curr_joint_pos+c3)*mv
                
                if v > mv:
                    v = mv
                if v < -mv:
                    v = -mv

                p.setJointMotorControl2(urdf, joint_index, p.VELOCITY_CONTROL, targetVelocity = v, force = f)
                p.stepSimulation()
                
                curr_joint_pos = np.round_(p.getJointState(urdf, joint_index)[0], decimals = 4) 
                if (b*prev_curr_joint_pos) >= (b*curr_joint_pos):
                    break

            n1 = 0
            while n1<10:
                p.setJointMotorControl2(urdf, joint_index, p.VELOCITY_CONTROL, targetVelocity = 0, force = f)
                p.stepSimulation()
                n1 += 1

                
    def getReward(self, d1, d2):
        
        """
        Function to return the reward
        Arguments:
            Distances d1 and d2
        Returns:
            Reward
        """
        if 1-d1 < 0 and 1-d2 <0:
            sign = -1
        else:
            sign = 1
            
        reward_t = np.round_((1-d1)*(1-d2)*sign, decimals = 4) - 2
        
        return reward_t
    
    def vec_mag(self, v1, v2):
        
        """
        Function to calculate 
        """
        return ((v1[0]-v2[0])**2 + (v1[0]-v2[0])**2 + (v1[0]-v2[0])**2)**(0.5)
    
    def step(self, actions):
        
        """
        Environment Step Function
        Arguments:
            List of Actions(Points to go)
        Returns:
            Environment Information and Reward
        """
        reward = 0.0
        actions[actions<0] = 0
        for i in range(5):
            pts = self.BeizerCurve(actions)
            right_pos = pts[0]
            left_pos = pts[1]
            p1= 0
            while p1<5:
                joint_angles0 = p.calculateInverseKinematics(self.baxter, 29, right_pos)
                joint_angles1 = p.calculateInverseKinematics(self.baxter, 51, left_pos)
                move_actions = list(joint_angles0)[0:10] + list(joint_angles1)[10:]
                self.moveVC(self.baxter, self.moveable_joints, move_actions)
                p1 += 1

            box1_pos = list(p.getBasePositionAndOrientation(self.box1)[0])
            box2_pos = list(p.getBasePositionAndOrientation(self.box2)[0])
            boxes_poses = [box1_pos, box2_pos]

            hand1_pos = p.getLinkState(self.baxter, 29)[0]
            hand2_pos = p.getLinkState(self.baxter, 51)[0]

            d1 = np.amin([self.vec_mag(hand1_pos, box1_pos), self.vec_mag(hand2_pos, box1_pos)])
            d2 = np.amin([self.vec_mag(hand1_pos, box2_pos), self.vec_mag(hand2_pos, box2_pos)])
            
            reward_n = self.getReward(d1, d2)
            b1 = 0.5
            if  (boxes_poses[0][2] < self.box_t_poses[0][2]-b1 or boxes_poses[1][2] < self.box_t_poses[1][2]-b1):
                reward_n = reward_n-10
                episode_destroyed = 1
            else:
                episode_destroyed = 0
            reward = reward_n + self.discount*reward
        
            
        img, depth = self.getImage()

        if (d1<self.h1 and d2<self.h2 ):
            self.done = True
        else:
            self.done = False

        state_dict = {"hand_ends": [hand1_pos, hand2_pos], "eye_view": img, "depth": depth} 

        return (state_dict, reward, self.done, {"episode_destroyed":episode_destroyed})

    def encode(self, img, encoder):

        '''
        Function which uses the given model and image to generate latent vector of length 96
        Arguments:
        model : Encoder model
        image = image of shape (64,64,3)

        Return:
        Vector of length 96
        '''
        w,h,c = img.shape
        return encoder.predict(img.reshape(1, w, h, c))
    
    def BeizerCurve(self,weights):
        
    
        if self.bezier_timesteps == 0:
        
            list_one = [p.getBasePositionAndOrientation(self.box1)[0],p.getLinkState(self.baxter, 29)[0]]
            list_two = [p.getBasePositionAndOrientation(self.box2)[0],p.getLinkState(self.baxter, 51)[0]]

            x_one = [p[0] for p in list_one]
            y_one = [p[1] for p in list_one]
            z_one = [p[2] for p in list_one]

            x_one_mid = (x_one[0]+x_one[1])/2
            y_one_mid = (y_one[0]+y_one[1])/2
            z_one_mid = (z_one[0]+z_one[1])/2

            x_one.append(x_one_mid)
            y_one.append(y_one_mid)
            z_one.append(z_one_mid)

            self.bezier_right = [x_one, y_one, z_one]

            x_two = [p[0] for p in list_two]
            y_two = [p[1] for p in list_two]
            z_two = [p[2] for p in list_two]

            x_two_mid = (x_two[0]+x_two[1])/2
            y_two_mid = (y_two[0]+y_two[1])/2
            z_two_mid = (z_two[0]+z_two[1])/2

            x_two.append(x_two_mid)
            y_two.append(y_two_mid)
            z_two.append(z_two_mid)

            self.bezier_left = [x_two, y_two, z_two]
            
        points_one,points_two = [],[]

        n,t_i = 3,np.linspace(0.0,1.0,6)[1:]
        t=t_i[self.bezier_timesteps]

        polynomial_array_one = np.array([(comb(n,i)*(t**(i)) *(1-t)**(n-i))*w for i,w in zip(range(0,3),weights[:3])])
        polynomial_array_two = np.array([(comb(n,i)*(t**(i)) *(1-t)**(n-i))*w for i,w in zip(range(0,3),weights[3:])])

        x_one_values = np.dot(self.bezier_right[0], polynomial_array_one)
        y_one_values = np.dot(self.bezier_right[1], polynomial_array_one)
        z_one_values = np.dot(self.bezier_right[2], polynomial_array_one)

        x_two_values = np.dot(self.bezier_left[0], polynomial_array_two)
        y_two_values = np.dot(self.bezier_left[1], polynomial_array_two)
        z_two_values = np.dot(self.bezier_left[2], polynomial_array_two)

        points_one.extend([x_one_values,y_one_values,z_one_values])
        points_two.extend([x_two_values,y_two_values,z_two_values])

        self.bezier_timesteps+=1
        self.bezier_timesteps = self.bezier_timesteps % 5

        return points_one,points_two

    def sample_action(self):
        '''
        Function which gives random action
        Arguments:
        None

        Return:
        Vector of length 6
        '''
        action = np.random.uniform(-1, 1, (6,))
        return action
