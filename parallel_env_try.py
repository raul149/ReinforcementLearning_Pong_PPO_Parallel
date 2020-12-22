import gym
import numpy as np
import multiprocessing as mp


class SerialEnvs(object):
    def __init__(self, env_name, amount):
        self.envs = []
        self.env_name = env_name
        self.amount = amount
        for i in range(amount):
            if type(env_name) is str:
                env = gym.make(env_name)
            else:
                env = env_name()
            self.envs.append(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.current_steps = np.zeros(amount)
        self.rewards = np.zeros(amount)

    def step(self, actions):
        states, rewards, dones = [], [], []
        finished = []
        infos = []
        for i, a in enumerate(actions):
            try:
                state, reward, done, info = self.envs[i].step(a)
            except:
                print("Error while stepping env %d with a=%s. Resetting." % (i, a))
                reward, done = -10, True
                info = {}
            self.rewards[i] += reward
            infos.append(info)
            if done:
                total_rew = self.rewards[i]
                finished.append(total_rew)
                state = self.envs[i].reset()
                self.rewards[i] = 0
            states.append(state)
            rewards.append(reward)
            dones.append(done)
        rewards = np.stack(rewards)
        states = np.stack(states)
        return states, rewards, dones, {'finished': finished, 'infos': infos}

    def reset(self):
        states = []
        for e in self.envs:
            state = e.reset()
            states.append(state)
        states = np.stack(states)
        self.current_steps = np.zeros(self.amount)
        self.rewards = np.zeros(self.amount)
        return states

    def set_env_param(self, param, value):
        for e in self.envs:
            e.__setattr__(param, value)

    def env_call(self, function, args):
        ret = []
        for e in self.envs:
            if isinstance(e, gym.wrappers.time_limit.TimeLimit):
                retval = e.env.__getattribute__(function)(*args)
            else:
                retval = e.__getattribute__(function)(*args)
            ret.append(retval)
        return ret


class ParallelEnvs(object):
    base_seed = 123

    def __init__(self, env_name, processes, envs_per_process, action_limit=None):
        self.parent_pipes = []
        self.processes = []
        self.env_name = env_name
        self.num_processes = processes
        self.envs_per_process = envs_per_process
        self.action_limit = action_limit
        for i in range(self.num_processes):
            parent_pipe, child_pipe = mp.Pipe()
            process = mp.Process(target=ParallelEnvs.worker_proc, args=(self.env_name, child_pipe, i, envs_per_process))
            self.parent_pipes.append(parent_pipe)
            self.processes.append(process)

        for p in self.processes:
            p.start()

        if type(env_name) is str:
            env = gym.make(env_name)
        else:
            env = env_name()
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    @staticmethod
    def worker_proc(name, pipe, worker_id, num_envs):
        print("Creating %d %s environments"% (num_envs, name))
        env = SerialEnvs(name, num_envs)
        np.random.seed(ParallelEnvs.base_seed + worker_id)
        episode_reward = 0
        current_steps = 0
        while True:
            cmd, data = pipe.recv()
            if cmd == "step":
                a = data
                rew_to_send = np.array(env.rewards)
                state, reward, done, info = env.step(a)
                rew_to_send += reward
                current_steps += 1
                pipe.send((state, reward, done, info, rew_to_send))
            elif cmd == "reset":
                state = env.reset()
                pipe.send((state, 0, False, 0))
            elif cmd == "setparam":
                name, value = data
                try:
                    env.__setattr__(name, value)
                    pipe.send("ok")
                except AttributeError:
                    pipe.send("AttributeError")
            elif cmd == "call":
                function, args = data
                res = env.env_call(function, args)
                pipe.send(("ok", res))
            elif cmd == "quit":
                print("Worker quitting.")
                break
            else:
                raise ValueError("Unrecognized command:", cmd)

    def step(self, actions):
        finished = []
        idx, inc = 0, self.envs_per_process
        if self.action_limit is not None:
            actions = np.clip(actions, -self.action_limit, self.action_limit)
        # Send step commands
        for i in range(self.num_processes):
            self.parent_pipes[i].send(("step", actions[idx:(idx+inc)]))
            idx += inc

        # read resulting states
        states, rewards, dones, infos = [], [], [], []
        for i in range(self.num_processes):
            state, reward, done, info, total_rew = self.parent_pipes[i].recv()
            states.extend(state)
            rewards.extend(reward)
            dones.extend(done)
            infos.extend(info["infos"])
            for j, d in enumerate(done):
                if d:
                    finished.append((i*inc+j, total_rew[j]))
        rewards = np.stack(rewards)
        states = np.stack(states)
        return states, rewards, dones, {'finished': finished, 'infos': infos}

    def reset(self):
        # Send reset commands
        for i in range(self.num_processes):
            self.parent_pipes[i].send(("reset", 0))

        # Read states
        states = []
        for i in range(self.num_processes):
            state, _, _, _ = self.parent_pipes[i].recv()
            states.extend(state)
        states = np.stack(states)
        return states

    def set_env_param(self, param, value):
        for i in range(self.num_processes):
            self.parent_pipes[i].send(("setparam", (param, value)))

        for i in range(self.num_processes):
            res = self.parent_pipes[i].recv()
            if res != "ok":
                raise AttributeError("No such attribute: %s" % param)

    def env_call(self, function, args):
        ret = []
        for i in range(self.num_processes):
            self.parent_pipes[i].send(("call", (function, args)))

        for i in range(self.num_processes):
            res, retval = self.parent_pipes[i].recv()
            if res != "ok":
                raise AttributeError("Error calling: %s" % function)
            ret.append(retval)
        return ret
