from gym_minigrid.minigrid import *
from gym_minigrid.register import register


def place_cookie(grid, rand_int, cookie_positions):
    for pos in cookie_positions:
        x, y = pos
        grid.set(x, y, None)

    x, y = cookie_positions[rand_int]
    grid.set(x, y, Reward())


class BeliefTracker:
    # 1000 -- Checked No rooms
    # 1001 -- Checked W room
    # 1010 -- Checked S room
    # 1100 -- Checked E room
    # 1011 -- Checked WS room
    # 1101 -- Checked WE room
    # 1110 -- Checked SE room
    # 1111 -- Checked WSE room
    # 0000 -- Button not pressed
    def __init__(self, env):
        self.env = env
        self.base = 2

        #Start of episode
        self.reset()


    def button_pressed(self):
        return int(self.status, self.base) & (1 << 3)

    def has_checked_west_room(self):
        return (int(self.status, self.base) & 1)

    def has_checked_south_room(self):
        return (int(self.status, self.base) & (1 << 1))

    def has_checked_east_room(self):
        return (int(self.status, self.base) & (1 << 2))

    def has_seen_cookie(self):
        return int(self.cookie, self.base) != 0

    def cookie_in_east_room(self):
        east_coord = self.env.cookie_positions[2]
        return self.env.grid.get(*east_coord) is not None

    def cookie_in_west_room(self):
        west_coord = self.env.cookie_positions[1]
        return self.env.grid.get(*west_coord) is not None

    def cookie_in_south_room(self):
        south_coord = self.env.cookie_positions[0]
        return self.env.grid.get(*south_coord) is not None



    def update(self, env):
        if not self.button_pressed() and env.agent_pos == (env.top_corridor, 1): #Just pressed button
            self.status = "0b1000"
        elif env.grid.get(env.top_corridor, 1).__class__.__name__ == "Button": #Just ate cookie, if the tile at button location is "Button", else it would be "None"
            self.reset()
        elif self.button_pressed() and env.in_west_corridor(env.agent_pos) and not self.has_checked_west_room(): #Checked west room for first time
            self.status = bin(int(self.status, self.base) + int("001", self.base))
            if not self.has_seen_cookie() and self.cookie_in_west_room():
                self.cookie = bin(int(self.cookie, self.base) + int("001", self.base))
        elif self.button_pressed() and env.in_south_corridor(env.agent_pos) and not self.has_checked_south_room(): #Checked south room for first time
            self.status = bin(int(self.status, self.base) + int("010", self.base))
            if not self.has_seen_cookie() and self.cookie_in_south_room():
                self.cookie = bin(int(self.cookie, self.base) + int("010", self.base))
        elif self.button_pressed() and env.in_east_corridor(env.agent_pos) and not self.has_checked_east_room(): #Checked east room for first time
            self.status = bin(int(self.status, self.base) + int("100", self.base))
            if not self.has_seen_cookie() and self.cookie_in_east_room():
                self.cookie = bin(int(self.cookie, self.base) + int("100", self.base))


    def reset(self):
        if self.env.grid.get(self.env.top_corridor,1) is None: #Button not enabled
            self.status = "0b1000"
        else: #Button enabled
            self.status = "0b0000"

        self.cookie = "0b000"


class ModifiedCookieEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(4,4),
        agent_start_dir=0,
    ):

        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=200,
            # Set this to True for maximum speed
            see_through_walls=True
        )

        self.top_corridor = None
        self.bottom_corridor = None
        self.west_corridor = None
        self.east_corridor = None
        self.cookie_positions = None
        self.button_position = None

        self.name = "MiniGrid-Modified-Cookie-9x9-v0"


    def place_button(self):
        self.grid.set(self.top_corridor, 1, Button(color="blue", callback=place_cookie))

    def mask_fn(self, agent_pos, cell):
        # Returns true if we should display

        # If agent is in room
        if self.in_room(agent_pos):
            return self.in_room(cell)

        # If agent is north corridor
        if self.in_north_corridor(agent_pos):
            return self.in_north_corridor(cell)

        # If agent is south corridor
        if self.in_south_corridor(agent_pos):
            return self.in_south_corridor(cell)

        # If agent is west corridor
        if self.in_west_corridor(agent_pos):
            return self.in_west_corridor(cell)

        # If agent is east corridor
        if self.in_east_corridor(agent_pos):
            return self.in_east_corridor(cell)

        # Error
        raise Exception("Mask function not correctly specified.")

    def in_room(self, pos):
        x, y = pos
        return 3 <= x <= 5 and 3 <= y <= 5

    def in_west_corridor(self, pos):
        x, y = pos
        return 1 <= x <= 2 and y == self.west_corridor

    def in_east_corridor(self, pos):
        x, y = pos
        return 6 <= x <= 7 and y == self.east_corridor

    def in_north_corridor(self, pos):
        x, y = pos
        return 1 <= y <= 2 and x == self.top_corridor

    def in_south_corridor(self, pos):
        x, y = pos
        return 6 <= y <= 7 and x == self.bottom_corridor

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.thick_wall_rect(0, 0, width, height, 3)

        cookie_positions = []

        # Top corridor
        top_corridor = self._rand_int(low=3, high=6)
        self.grid.empty_vert(start_x=top_corridor, start_y=1, end_y=3)
        self.top_corridor = top_corridor
        self.button_position = (top_corridor, 1)

        # Bottom corridor
        bottom_corridor = self._rand_int(low=3, high=6)
        self.grid.empty_vert(start_x=bottom_corridor, start_y=6, end_y=8)
        cookie_positions.append((bottom_corridor, 7))
        self.bottom_corridor = bottom_corridor

        # West corridor
        west_corridor = self._rand_int(low=3, high=6)
        self.grid.empty_horz(start_x=1, start_y=west_corridor, end_x=3)
        cookie_positions.append((1, west_corridor))
        self.west_corridor = west_corridor

        # East corridor
        east_corridor = self._rand_int(low=3, high=6)
        self.grid.empty_horz(start_x=6, start_y=east_corridor, end_x=8)
        cookie_positions.append((7, east_corridor))
        self.east_corridor = east_corridor

        self.cookie_positions = cookie_positions

        start_config = self._rand_int(low=0, high=2)
        if start_config == 0: #Place button
            self.grid.set(top_corridor, 1, Button(color="blue", callback=place_cookie))
        else:
            room = np.random.choice(a=list(range(3)))
            place_cookie(self.grid, room, self.cookie_positions)


        #place_cookie(self.grid, 0, cookie_positions)

        # start_option = self._rand_int(low=0, high=3)
        # if start_option == 0:
        #     self.grid.set(6, 1, Button(color="blue", callback=place_cookie))
        # elif start_option == 1:
        #     place_cookie(self.grid, 0)
        # elif start_option == 2:
        #     place_cookie(self.grid, 1)


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            raise NotImplementedError

        self.mission = "Grab as many cookies as you can"

class ModifiedCookieEnv9x9(ModifiedCookieEnv):
    def __init__(self, **kwargs):
        super().__init__(size=9, **kwargs)

register(
    id='MiniGrid-Modified-Cookie-9x9-v0',
    entry_point='gym_minigrid.cookie_modified:ModifiedCookieEnv9x9'
)
