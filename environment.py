import gym

class CartPole:
    def __init__(self):
        self.env = gym.make('CartPole-v0')
        self.action_space = self.env.action_space
    def reset(self):
        # returns state, {}
        return self.env.reset()

    def step(self, action):
        # returns state, reward, terminated, truncated, {}
        return self.env.step(action)
    