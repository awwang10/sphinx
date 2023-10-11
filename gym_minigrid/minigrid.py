import math
import hashlib
import gym
from enum import IntEnum
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from .rendering import *

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'brown'  : np.array([165, 42, 42]),
}

COLOR_NAMES = sorted(list(COLORS.keys()))

# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'brown' : 6
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
    'immovable_box' : 11,
    'genie'         : 12,
    'button'        : 13,
    'reward'        : 14,
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], 0)

    def encode_state(self):
        return (OBJECT_TO_IDX[self.type], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        #color = IDX_TO_COLOR[color_idx]
        color = 'blue'

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'reward':
            v = Reward()
        elif obj_type == 'lava':
            v = Lava()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError



class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Reward(WorldObj):
    def __init__(self):
        super().__init__('reward', 'brown')

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]
        fill_coords(img, point_in_circle(0.5, 0.5, 0.2), c)


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Door(WorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        #return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)
        return (OBJECT_TO_IDX[self.type], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class Box(WorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

class Button(WorldObj):
    def __init__(self, color, callback):
        super(Button, self).__init__('button', color)
        self.callback = callback

    def can_overlap(self):
        return True

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        fill_coords(img, point_in_circle(0.5, 0.5, 0.2), c)



class ImmovableBox(WorldObj):
    def __init__(self, color, contains=None):
        super(ImmovableBox, self).__init__('immovable_box', color)
        self.contains = contains

    def can_pickup(self):
        return False

    def can_overlap(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

    def encode(self):
        """Encode a description of this object as a 3-tuple of integers"""
        #return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)
        return (OBJECT_TO_IDX[self.type], 0)

    def encode_state(self):
        """Encode a fully-observable description of this object as a 3-tuple of integers"""
        if self.contains:
            # Just directly show what's in the box
            #return (OBJECT_TO_IDX[self.contains.type], COLOR_TO_IDX[self.contains.color], 0)
            return (OBJECT_TO_IDX[self.contains.type], 0)
        else:
            # Return box
            # return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)
            return (OBJECT_TO_IDX[self.type], 0)




class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        self.num_channels = 3

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1  = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def empty_horz(self, start_x, start_y, end_x):
        for i in range(start_x, end_x):
            self.set(i, start_y, None)

    def empty_vert(self, start_x, start_y, end_y):
        for i in range(start_y, end_y):
            self.set(start_x, i, None)

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def thick_wall_rect(self, x, y, w, h, thickness):
        for t in range(thickness):
            self.wall_rect(x+t, y+t, w-t*2, h-t*2)


    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        tile_size=TILE_PIXELS,
        subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5*math.pi*agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """


        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, agent_pos, agent_dir, mask_fn=None):
        """
        Produce a compact numpy encoding of the grid
        """

        array = np.zeros((self.width, self.height, self.num_channels), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):

                if mask_fn is not None:
                    mask_boolean = mask_fn(agent_pos, (i, j)) #True if we should mask
                else:
                    mask_boolean = True

                if not mask_boolean:
                    # If the cell is empty
                    array[i, j, 0] = OBJECT_TO_IDX['unseen']
                    array[i, j, 1] = 0
                    #array[i, j, 2] = 0
                else:
                    v = self.get(i, j)
                    if v is None:
                        # If the cell is empty
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        #array[i, j, 2] = 0
                    else:
                        array[i, j, :2] = v.encode()
                # (i, j, x, 1, agent_dir) only if the agent is at cell (i,j)
                if agent_pos[0] == i and agent_pos[1] == j:
                    array[i, j, 2] = agent_dir + 1
                else:
                    array[i, j, 2] = 0

        return array

    def encode_state(self, agent_pos, agent_dir):
        """
        Produce a compact fully-observable numpy encoding of the grid
        """

        array = np.zeros((self.width, self.height, self.num_channels), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                if v is None:
                    # If the cell is empty
                    array[i, j, 0] = OBJECT_TO_IDX['empty']
                    array[i, j, 1] = 0
                    #array[i, j, 2] = 0
                else:
                    array[i, j, :2] = v.encode_state()
                # (i, j, x, 1, agent_dir) only if the agent is at cell (i,j)
                if agent_pos[0] == i and agent_pos[1] == j:
                    array[i, j, 2] = agent_dir + 1
                else:
                    array[i, j, 2] = 0

        return array


    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """
        assert(False)
        width, height, channels = array.shape
        assert channels == 5

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state, agent_pos, agent_dir = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)

        return grid


class MiniGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

        # Pick up an object
        pickup = 3
        # Drop an object
        drop = 4
        # Toggle/activate an object
        toggle = 5

        # Done completing task
        done = 6

    class GenieActions(IntEnum):
        left = 0
        right = 1
        forward = 2

    class CookieActions(IntEnum):
        left = 0
        right = 1
        forward = 2



    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=100,
        see_through_walls=False,
        seed=1337,
    ):
        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        self.grid_size = grid_size

        # Action enumeration for this environment
        if "Genie" in self.__class__.__name__:
            self.actions = MiniGridEnv.GenieActions
        elif "Cookie" in self.__class__.__name__:
            self.actions = MiniGridEnv.CookieActions
        else:
            raise Exception("No actions set!")

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        self.normalization_factor = len(OBJECT_TO_IDX)

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size, self.grid_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

        # Number of possible objects on map
        self.num_objects = len(OBJECT_TO_IDX)


    def reset(self):
        # Current position and direction of the agent
        self.agent_pos = None
        self.agent_dir = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent_pos is not None
        assert self.agent_dir is not None

        # Check that the agent doesn't overlap with an object
        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def hash(self, size=16):
        """Compute a hash that uniquely identifies the current state of the environment.
        :param size: Size of the hashing
        """
        sample_hash = hashlib.sha256()

        to_encode = [self.grid.encode().tolist(), self.agent_pos, self.agent_dir]
        for item in to_encode:
            sample_hash.update(str(item).encode('utf8'))

        return sample_hash.hexdigest()[:size]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'F',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
            'immovable_box' : 'I',
            'genie'         : 'E',
            'button'        : 'b',
            'reward'        : 'R',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        return 1
        #return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """
        random_int = self.np_random.integers(low, high)
        return random_int

    def _rand_int_matrix(self, low, high, size):
        return self.np_random.integers(low, high, size)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.integers(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.integers(xLow, xHigh),
            self.np_random.integers(yLow, yHigh)
        )

    def place_obj(self,
        obj,
        top=None,
        size=None,
        reject_fn=None,
        max_tries=math.inf
    ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
        self,
        top=None,
        size=None,
        rand_dir=True,
        max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        self.agent_pos = None
        pos = self.place_obj(None, top, size, max_tries=max_tries)
        self.agent_pos = pos

        if rand_dir:
            self.agent_dir = self._rand_int(0, 4)

        return pos

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.agent_dir >= 0 and self.agent_dir < 4
        return DIR_TO_VEC[self.agent_dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.agent_pos + self.dir_vec


    def step(self, action):
        self.step_count += 1
        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if hasattr(self.actions, 'left') and action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif hasattr(self.actions, 'right') and action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif hasattr(self.actions, 'forward') and action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell != None and fwd_cell.type == 'reward':
                reward = self._reward()
                if "Cookie" in self.__class__.__name__:
                    # Regenerate button
                    self.place_button()
            if fwd_cell != None and fwd_cell.type == 'immovable_box':
                done = True
                if fwd_cell.contains is not None:
                    reward = 1
                else:
                    reward = 0
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True
            if fwd_cell != None and fwd_cell.type == "button": # Cookie button
                room = np.random.choice(a=list(range(3)))
                fwd_cell.callback(self.grid, room, self.cookie_positions)

            if "Genie" in self.__class__.__name__ and self.agent_pos in self.get_genie_trigger_locs():
                toggle_result = self.get_genie().talk(self, None)
                reward = -0.2

        # Pick up an object
        elif hasattr(self.actions, 'pickup') and action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif hasattr(self.actions, 'drop') and action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif hasattr(self.actions, 'toggle') and action == self.actions.toggle:
            if fwd_cell:
                if fwd_cell.type == "immovable_box":
                    if fwd_cell.contains is None:
                        done = True
                    else:
                        fwd_cell.toggle(self, fwd_pos)
                else:
                    fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif hasattr(self.actions, 'done') and action == self.actions.done:
            pass
        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # Reset the genie
        # if hasattr(self.actions, 'toggle') and action == self.actions.toggle:
        #     if fwd_cell:
        #         if fwd_cell.type == 'genie':
        #             fwd_cell.reset_hint()

        if "Genie" in self.__class__.__name__ and self.agent_pos in self.get_genie_trigger_locs():
            self.get_genie().reset_hint()

        return obs, reward, done, {}

    # Generate underlying state using current grid
    def get_state(self):
        grid = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode_state(agent_pos=self.agent_pos, agent_dir=self.agent_dir)
        image = image.astype(float)
        image[:, :, 0] = image[:, :, 0] / self.normalization_factor

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        state = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return state

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        grid = self.grid

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        if self.carrying:
            grid.set(*self.agent_pos, self.carrying)
        else:
            grid.set(*self.agent_pos, None)

        return grid

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(agent_pos=self.agent_pos, agent_dir=self.agent_dir, mask_fn=self.mask_fn)
        image = image.astype(float)
        image[:,:,0] = image[:,:,0] / self.normalization_factor

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            # 'direction': self.agent_dir,
            # 'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.grid_size // 2, self.grid_size - 1),
            agent_dir=3
        )

        return img

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return
