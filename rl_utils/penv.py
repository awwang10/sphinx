from multiprocessing import Process, Pipe
import gym

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            conn.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            conn.send(obs)
        elif cmd == "get_item_location":
            conn.send(env.getItemLocation())
        elif cmd == "has_consulted_genie":
            conn.send(env.has_consulted_genie)
        elif cmd == "get_step_count":
            conn.send(env.step_count)
        elif cmd == "get_most_recent_consult":
            conn.send(env.most_recent_consult_step)
        elif cmd == "get_state":
            conn.send(env.get_state())
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()


        if "Genie" in self.envs[0].__class__.__name__:
            self.genie_location = self.envs[0].genie_location
            self.num_boxes = self.envs[0].num_boxes

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        result = self.envs[0].step(actions[0])
        if len(result) == 4:
            obs, reward, done, info = result
        else:
            obs, reward, done, truncation, info = result
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])

        return results

    def get_state(self):
        for local in self.locals:
            local.send(("get_state", None))
        state = self.envs[0].get_state()
        results = [state] + [local.recv() for local in self.locals]
        return results

    def getItemLocation(self):
        """For genie environment only """
        for local in self.locals:
            local.send(("get_item_location", None))

        child_recv = []
        for local in self.locals:
            res = local.recv()
            child_recv.append(res)

        results = [self.envs[0].getItemLocation()] + child_recv
        return results

    def hasConsultedGenie(self):
        """For genie environment only """
        for local in self.locals:
            local.send(("has_consulted_genie", None))

        child_recv = []
        for local in self.locals:
            res = local.recv()
            child_recv.append(res)
        results = [self.envs[0].has_consulted_genie] + child_recv

        return results

    def getStepCount(self):
        """Gets the current step count of the env"""
        for local in self.locals:
            local.send(("get_step_count", None))

        child_recv = []
        for local in self.locals:
            res = local.recv()
            child_recv.append(res)
        results = [self.envs[0].step_count] + child_recv
        return results

    def mostRecentConsultStep(self):
        for local in self.locals:
            local.send(("get_most_recent_consult", None))

        child_recv = []
        for local in self.locals:
            res = local.recv()
            child_recv.append(res)
        results = [self.envs[0].most_recent_consult_step] + child_recv
        return results

    def render(self):
        raise NotImplementedError