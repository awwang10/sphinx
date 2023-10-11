from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym_minigrid.minigrid import WorldObj, OBJECT_TO_IDX



class Genie(WorldObj):
    def __init__(self, color, item_location, can_lie=False, random_hint_prob=.5):
        super(Genie, self).__init__('genie', color)
        self.item_location = item_location

        # If we're giving out the hint right now, set to True, otherwise False
        # Use this to change if genie hint is perpetual
        self.give_hint = False

        self.can_lie = can_lie
        self.random_hint_prob = random_hint_prob

        self.env = None #Holds the environment this genie belongs to

    def can_pickup(self):
        return False

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


    def talk(self, env, pos):
        # env.grid.set(*pos, None)
        self.give_hint = True
        env.has_consulted_genie = True
        env.most_recent_consult_step = env.step_count
        return True

    def reset_hint(self):
        self.give_hint = False

    def set_env(self, env): # Need the env's random generator later
        self.env = env

    def encode(self):
        """Encode the description of this object as a 3-tuple of integers"""
        # State, -1: unknown, int: location
        if self.give_hint:

            # Vanilla Genie
            if not self.can_lie:
                return (OBJECT_TO_IDX[self.type], self.item_location)

            #Lying genie
            if self.env._rand_float(low=0, high=1) < self.random_hint_prob:
                hint = self.env._rand_int(low=0, high=self.env.num_boxes)
                return (OBJECT_TO_IDX[self.type], hint)
            else:
                return (OBJECT_TO_IDX[self.type], self.item_location)

        return (OBJECT_TO_IDX[self.type], 3)


class GenieEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        name,
        size=8,
        agent_start_pos=(3,2),
        agent_start_dir=0,
        agent_start_random=True,
        num_boxes=3,
    ):
        self.name = name
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_start_random = agent_start_random
        self.num_boxes = num_boxes
        self.has_aux_info = True
        self.has_consulted_genie = False
        self.most_recent_consult_step = -1
        self.max_steps = 100
        self.size = size

        self.genie_location = (1, 1)

        super().__init__(
            grid_size=size,
            max_steps=100,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def make_genie(self, goal_box_idx):
        g = Genie(color='blue', item_location=goal_box_idx)
        g.set_env(self)
        return g

    def get_genie_trigger_locs(self):
        """Walking by these coordinates talks to the genie"""
        return [(1, 2), (2, 1)]

    def generate_agent_start_loc(self):
        possible_locations = []
        for i in range(1, self.size):
            for j in range(1, self.size):
                if self.grid.get(i, j) is None and (i,j) not in self.get_genie_trigger_locs():
                    possible_locations.append((i, j))
        idx = self._rand_int(low=0, high=len(possible_locations))
        return possible_locations[idx]

    def get_genie(self):
        return self.grid.get(*self.genie_location)

    def aux_info_dim(self):
        return self.num_boxes

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # goal_box_idx = 0
        goal_box_idx = self._rand_int(low=0, high=self.num_boxes)
        locations = [(width-2, 1), (width-2, height-2), (1,height-2)]
        for i in range(self.num_boxes):
            if i == goal_box_idx:
                box = ImmovableBox(color='green', contains=Goal())
            else:
                box = ImmovableBox(color='green', contains=None)
            self.put_obj(box, locations[i][0], locations[i][1])

        #Change item_location to a cooordinate
        # item_location = locations[goal_box_idx][1] * width + locations[goal_box_idx][0]
        self.put_obj(self.make_genie(goal_box_idx), self.genie_location[0], self.genie_location[1])
        #(coodridnate, colour, integer)

        # Place the agent
        if self.agent_start_pos is not None:
            if self.agent_start_random:
                self.agent_pos = self.generate_agent_start_loc()
                self.agent_dir = self._rand_int(low=0, high=4)
            else:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
        else:
            raise NotImplementedError

        self.mission = "Hello"
        self.item_location = goal_box_idx

        self.has_consulted_genie = False
        self.most_recent_consult_step = -1

        # Used to mask observation
        self.mask_fn = None

    def getItemLocation(self):
        return self.item_location



class GenieEnv10x10(GenieEnv):
    def __init__(self, **kwargs):
        super().__init__(name="MiniGrid-Genie-10x10-v0", size=10, **kwargs)

class GenieEnv8x8(GenieEnv):
    def __init__(self, **kwargs):
        super().__init__(name="MiniGrid-Genie-8x8-v0", size=8, agent_start_pos=(4, 4), **kwargs)

class LyingGenieEnv8x8(GenieEnv):
    def __init__(self, **kwargs):
        super().__init__(name="MiniGrid-Lying-Genie-8x8-v0", size=8, agent_start_pos=(4, 4), **kwargs)

    def make_genie(self, goal_box_idx):
        g =  Genie(color='blue', item_location=goal_box_idx, can_lie=True, random_hint_prob=0.25)
        g.set_env(self)
        return g


class NoisyTVGenieEnv8x8(GenieEnv):
    def __init__(self, **kwargs):
        self.noisyTV = True
        super().__init__(name="MiniGrid-NoisyTV-Genie-8x8-v0", size=8, agent_start_pos=(4, 4), **kwargs)

    def gen_obs(self):
        """
        Produce a compact numpy encoding of the grid
        """
        grid = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(agent_pos=self.agent_pos, agent_dir=self.agent_dir, mask_fn=self.mask_fn)

        random_nums = self._rand_int_matrix(low=0, high=max(OBJECT_TO_IDX.values()), size=(image.shape[0], image.shape[1]))

        image[:, :, 0] = image[:, :, 0] * (image[:, :, 0] != 2) + random_nums * (image[:, :, 0] == 2)
        image = image.astype(float)
        image[:, :, 0] = image[:, :, 0] / self.normalization_factor

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

class GenieEnv6x6(GenieEnv):
    def __init__(self, **kwargs):
        super().__init__(name="MiniGrid-Genie-6x6-v0", size=6, agent_start_pos=(1, 1), **kwargs)


register(
    id='MiniGrid-Genie-10x10-v0',
    entry_point='gym_minigrid.genie:GenieEnv10x10'
)

register(
    id='MiniGrid-Genie-8x8-v0',
    entry_point='gym_minigrid.genie:GenieEnv8x8'
)

register(
    id='MiniGrid-NoisyTV-Genie-8x8-v0',
    entry_point='gym_minigrid.genie:NoisyTVGenieEnv8x8'
)

register(
    id='MiniGrid-Lying-Genie-8x8-v0',
    entry_point='gym_minigrid.genie:LyingGenieEnv8x8'
)

register(
    id='MiniGrid-Genie-6x6-v0',
    entry_point='gym_minigrid.genie:GenieEnv6x6'
)