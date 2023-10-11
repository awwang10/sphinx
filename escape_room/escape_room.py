import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import sys
import gym 
from gym.envs.registration import register
from gym.utils import seeding

class EscapeRoom(gym.Env):

    def __init__(self, width=1000, max_steps=200, door_width=50, num_pointers=4, render_width_pixels=400, render_height_pixels=400, pointer_size=10):
        super().__init__()

        self.observation_space = gym.spaces.Box(low=0,high=1,shape=(100, 100),dtype='uint8')
        self.action_space = gym.spaces.Box(low=-1,high=1,shape=(2,),dtype=np.float32)

        assert num_pointers == 4 or num_pointers == 9

        self.width = width #Width of environment in units
        self.height = 1000
        self.num_pointers = num_pointers 

        self.max_forward_velocity = 50.
        self.max_turn_velocity = 30.

        self.distance = None
        self.distance_threshold = 50
        self.goal_reward = 1

        self.render_width_pixels = render_width_pixels
        self.render_height_pixels = render_height_pixels
        
        # assert door_width < self.height
        # self.door_width = door_width
        # self.door_start = None
        # self.door_mid = None 
        self.pointers = []

        self.goal_pos = None
        self.step_count = 0
        self.max_steps = max_steps

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None
        self.agent_color = 255
        self.statue_color = 128

        self.pointer_size = pointer_size #The size of pointer
        
        # Rendering information
        self.item_render_size = 2 * int(self.pointer_size / self.height * self.render_height_pixels) + 1
        self.item_box_render_size = 2 * self.item_render_size
        self.img_padding  = self.item_box_render_size

        self.window = None
        self.background = None


    def step(self, action): #Expect between -1 and 1 for both dimensions
        forward, rotate = self.max_forward_velocity * action[0], self.max_turn_velocity * action[1]

        #Give reward based on distance to goal
        self.agent_dir = np.fmod(self.agent_dir - rotate, 360)
        self.agent_pos = self.agent_pos - (forward * np.cos(self.agent_dir / 360. * 2 * np.pi), forward * np.sin(self.agent_dir / 360. * 2 * np.pi))
        self.agent_pos = np.clip(self.agent_pos, 0, self.width)
        

        last_distance = self.distance
        self.distance = np.linalg.norm(self.goal_pos-self.agent_pos, ord=2)

        reward = (last_distance - self.distance) / self.width
        done = False

        #Check if we have reached goal
        if self.distance < self.distance_threshold:
            reward += self.goal_reward
            done = True

        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True

        obs = self.get_obs()

        return obs, reward, done, {}


    def get_obs(self):
        return np.expand_dims(self.render_agent_view("RGB") > 0, axis=0)

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    # State is represented as (agent_pos, agent_dir, goal_pos, statue_1_pos, statue_1_dir, statue_2_pos, statue_2_dir, statue_3_pos, statue_3_dir, statue_4_pos, statue_4_dir)
    def get_state(self):
        return np.concatenate([self.agent_pos / self.height, 
            [self.agent_dir / 360],
            self.goal_pos / self.height,
            self.pointers[0][0] / self.height,
            [self.pointers[0][2] / 360],
            self.pointers[1][0] / self.height,
            [self.pointers[1][2] / 360],
            self.pointers[2][0] / self.height,
            [self.pointers[2][2] / 360],
            self.pointers[3][0] / self.height,
            [self.pointers[3][2] / 360]
        ])

    def reset(self):
        # Set doors
        # self.door_start = np.random.random() * (self.height - self.door_width)

        # Set agent position
        self.agent_pos = np.random.random(2) * self.width
        self.agent_dir = np.random.random() * 360
        
        self.goal_pos = np.random.random(2) * self.width

        # Set number of pointers
        q1_pos = (np.random.random(2) * self.width//2) + (self.width//2,0)
        q2_pos = (np.random.random(2) * self.width//2) + (self.width//2, self.width//2)
        q3_pos = (np.random.random(2) * self.width//2) + (0, self.width//2)
        q4_pos = np.random.random(2) * self.width //2
        
        angle1 = np.arctan2(self.goal_pos[1] - q1_pos[1], self.goal_pos[0] - q1_pos[0]) * 180 / np.pi  + 180
        angle2 = np.arctan2(self.goal_pos[1] - q2_pos[1], self.goal_pos[0] - q2_pos[0]) * 180 / np.pi  + 180
        angle3 = np.arctan2(self.goal_pos[1] - q3_pos[1], self.goal_pos[0] - q3_pos[0]) * 180 / np.pi  + 180
        angle4 = np.arctan2(self.goal_pos[1] - q4_pos[1], self.goal_pos[0] - q4_pos[0]) * 180 / np.pi  + 180

        self.pointers = []
        self.pointers.append((q1_pos, self.statue_color, angle1))
        self.pointers.append((q2_pos, self.statue_color, angle2))
        self.pointers.append((q3_pos, self.statue_color, angle3))
        self.pointers.append((q4_pos, self.statue_color, angle4))

        # self.door_mid = self.door_start + self.door_width/2

        self.distance = np.linalg.norm(self.goal_pos-self.agent_pos, ord=2)

        self.step_count = 0

        self.background = self.render_background()

        obs = self.get_obs()
        return obs

    #Returns the rendered item as a square of even length
    def render_item(self, pointer_color, direction):
        box = np.zeros((self.item_render_size, self.item_render_size))
        box[-1, :] = pointer_color
        box[:, box.shape[0]//2] = pointer_color
        box = np.uint8(box)

        image_box = np.zeros((self.item_box_render_size, self.item_box_render_size))
        image_box[self.item_render_size//2:self.item_render_size//2+self.item_render_size, self.item_render_size//2:self.item_render_size//2+self.item_render_size] = box
        image_box = np.uint8(image_box) #Prevents weird bug

        image_object = Image.fromarray(image_box, 'L') #https://stackoverflow.com/questions/51275682/pil-open-tif-image-only-one-channel
        image_object = image_object.rotate(angle=direction, resample=Image.BICUBIC, expand=0, center=None, translate=None, fillcolor=None)

        image = np.asarray(image_object.getdata())
        return image.reshape(self.item_box_render_size,self.item_box_render_size)


    def render_goal(self, pointer_color):
        box = np.zeros((6,6))
        box[:, :] = pointer_color
        return box


    def _units_to_pixels(self, unit):
        return int(unit / self.width * self.render_width_pixels)


    def render_background(self):
        img_array = np.zeros((self.render_width_pixels, self.render_height_pixels))
        img_array = np.pad(img_array, self.img_padding) #Pad the size of the box we're about to overlay

        img_array[:, self.img_padding + self.render_width_pixels:] = 255
        img_array[:, :self.img_padding] = 255
        img_array[self.img_padding + self.render_width_pixels:, :] = 255
        img_array[:self.img_padding, :] = 255

        goal_box = self.render_goal(255)
        goal_pos = ( self._units_to_pixels(self.goal_pos[0]) + self.img_padding, self._units_to_pixels(self.goal_pos[1]) + self.img_padding)
        img_array[ goal_pos[0] - goal_box.shape[0]//2 : goal_pos[0] + goal_box.shape[0]//2, goal_pos[1] - goal_box.shape[1]//2 : goal_pos[1] + goal_box.shape[1]//2 ] = goal_box 

        # door_initial = int(self.door_start / self.height * self.render_height_pixels)
        # door_end = int((self.door_start + self.door_width) / self.height * self.render_height_pixels) 
        # img_array[door_initial:door_end, self.img_padding + self.render_width_pixels:] = 0

        for i, item in enumerate(self.pointers):
            statue_box = self.render_item(self.pointers[i][1], self.pointers[i][2])
            # Place image
            statue_pos = ( self._units_to_pixels(self.pointers[i][0][0]) + self.img_padding, self._units_to_pixels(self.pointers[i][0][1]) + self.img_padding) # Gets the position of the centroid of the statue in pixels
            img_array[ statue_pos[0] - statue_box.shape[0]//2 : statue_pos[0] + statue_box.shape[0]//2, statue_pos[1] - statue_box.shape[1]//2 : statue_pos[1] + statue_box.shape[1]//2] = statue_box # Overrides the image with statue position

        img_array = np.uint8(img_array)
        return img_array
        

    def render(self, mode='human', close=False):

        # np.set_printoptions(threshold=sys.maxsize)

        # plt.imshow(img_array, interpolation='nearest')
        # plt.show()

        # # self.render_pointers()
        # # self.render_door()

        # np.set_printoptions(threshold=sys.maxsize)
        # print(agent_box)


        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import window
            self.window = window.Window('EscapeRoom')
            self.window.show(block=False)

        agent_box = self.render_item(pointer_color=self.agent_color, direction=self.agent_dir)
        
        # Place image
        agent_img_pos = (int(self.agent_pos[0] / self.width * self.render_width_pixels) + self.img_padding, int(self.agent_pos[1] / self.height * self.render_height_pixels) + self.img_padding) # Gets the position of the centroid of the agent in pixels
        
        full_image = np.copy(self.background)
        full_image[ agent_img_pos[0] - agent_box.shape[0]//2 : agent_img_pos[0] + agent_box.shape[0]//2, agent_img_pos[1] - agent_box.shape[1]//2 : agent_img_pos[1] + agent_box.shape[1]//2] = agent_box # Overrides the image with agent position
        full_image = np.uint8(full_image)

        image_object = Image.fromarray(full_image, 'L') #https://stackoverflow.com/questions/51275682/pil-open-tif-image-only-one-channel
        image_object = np.asarray(image_object)
        # image_object.show()

        if mode == 'human':
            self.window.show_img(image_object)

        return image_object

    def render_agent_view(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import window
            self.window = window.Window('EscapeRoom')
            self.window.show(block=False)


        agent_box = self.render_item(pointer_color=self.agent_color, direction=self.agent_dir)
        
        # Place image
        agent_img_pos = (int(self.agent_pos[0] / self.width * self.render_width_pixels) + self.img_padding, int(self.agent_pos[1] / self.height * self.render_height_pixels) + self.img_padding) # Gets the position of the centroid of the agent in pixels
        agent_img_pos_no_pad = (int(self.agent_pos[0] / self.width * self.render_width_pixels), int(self.agent_pos[1] / self.height * self.render_height_pixels)) # Gets the position of the centroid of the agent in pixels

        
        full_image = np.copy(self.background)
        #print(agent_img_pos_no_pad[0] - agent_box.shape[0]//2, agent_img_pos_no_pad[0] + agent_box.shape[0]//2, agent_img_pos_no_pad[1] - agent_box.shape[1]//2,  agent_img_pos_no_pad[1] + agent_box.shape[1]//2)
        # full_image[ agent_img_pos[0] - agent_box.shape[0]//2 : agent_img_pos[0] + agent_box.shape[0]//2, agent_img_pos[1] - agent_box.shape[1]//2 : agent_img_pos[1] + agent_box.shape[1]//2] = agent_box # Overrides the image with agent position
        full_image = np.uint8(full_image)

        image_object = Image.fromarray(full_image, 'L')
        image_object = image_object.crop((agent_img_pos[1]-100, agent_img_pos[0]-100, agent_img_pos[1]+100, agent_img_pos[0]+100))
        image_object = image_object.rotate(angle=-self.agent_dir, resample=Image.BICUBIC, expand=False, center=None, translate=None, fillcolor=None)
        image_object = image_object.crop((50, 50, 150, 150))
        
        image_object = np.asarray(image_object)


        if mode == 'human':
            self.window.show_img(image_object)

        return image_object


    def close(self):
        if self.window:
            self.window.close()
        return

register(
    id='EscapeRoom-v0',
    entry_point='escape_room.escape_room:EscapeRoom'
)
