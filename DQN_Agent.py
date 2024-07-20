from DQN import QNetwork
from ReplayMemory import ReplayMemory
import torch
import random
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class DQN_Agent:

    def __init__(self, env, device, lr=5e-4, render=False):
        # Initialize the DQN Agent.
        self.device = device
        self.env = env
        self.lr = lr
        self.policy_net = QNetwork(self.env, self.lr).to(device)
        self.target_net = QNetwork(self.env, self.lr).to(device)
        self.target_net.net.load_state_dict(self.policy_net.net.state_dict())  # Copy the weight of the policy network
        self.rm = ReplayMemory(self.env)
        self.burn_in_memory()
        self.batch_size = 32
        self.gamma = 0.99
        self.c = 0

    def burn_in_memory(self):
        # Initialize replay memory with a burn-in number of episodes/transitions.
        cnt = 0
        terminated = False
        truncated = False
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Iterate until we store "burn_in" buffer
        while cnt < self.rm.burn_in:
            if (cnt % 1000 == 0):
                print(f"cnt = {cnt}") 
            # Reset environment if terminated or truncated
            if terminated or truncated:
                state, _ = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            
            # Randomly select an action (left or right) and take a step
            action = torch.tensor(random.sample([0, 1], 1)[0]).reshape(1, 1)
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                
            # Store new experience into memory
            transition = Transition(state, action, next_state, reward)
            self.rm.memory.append(transition)
            state = next_state
            cnt += 1

    def epsilon_greedy_policy(self, q_values, epsilon=0.05):
        # Implement an epsilon-greedy policy. 
        p = random.random()
        if p > epsilon:
            with torch.no_grad():
                return self.greedy_policy(q_values)
        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def greedy_policy(self, q_values):
        # Implement a greedy policy for test time.
        return torch.argmax(q_values)
        
    def train(self):
        # Train the Q-network using Deep Q-learning.
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        terminated = False
        truncated = False

        # Loop until reaching the termination state
        while not (terminated or truncated):
            with torch.no_grad():
                q_values = self.policy_net.net(state)

            # Decide the next action with epsilon greedy strategy
            action = self.epsilon_greedy_policy(q_values).reshape(1, 1)
            
            # Take action and observe reward and next state
            next_state, reward, terminated, truncated, _ = self.env.step(action.item())
            reward = torch.tensor([reward])
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Store the new experience
            transition = Transition(state, action, next_state, reward)
            self.rm.memory.append(transition)

            # Move to the next state
            state = next_state

            # Sample minibatch with size N from memory
            transitions = self.rm.sample_batch(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Get current and next state values
            state_action_values = self.policy_net.net(state_batch).gather(1, action_batch) # extract values corresponding to the actions Q(S_t, A_t)
            next_state_values = torch.zeros(self.batch_size)
            
            with torch.no_grad():
                # no next_state_value update if an episode is terminated (next_satate = None)
                # only update the non-termination state values (Ref: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
                next_state_values[non_final_mask] = self.target_net.net(non_final_next_states).max(1)[0] # extract max value
                
            # Update the model
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch
            criterion = torch.nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.policy_net.optimizer.zero_grad()
            loss.backward()
            self.policy_net.optimizer.step()

            # Update the target Q-network in each 50 steps
            self.c += 1
            if self.c % 50 == 0:
                self.target_net.net.load_state_dict(self.policy_net.net.state_dict())

    def test(self, model_file=None):
        # Evaluates the performance of the agent over 20 episodes.

        max_t = 1000
        state, _ = self.env.reset()
        rewards = []

        for t in range(max_t):
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net.net(state)
            action = self.greedy_policy(q_values)
            state, reward, terminated, truncated, _ = self.env.step(action.item())
            rewards.append(reward)
            if terminated or truncated:
                break

        return np.sum(rewards)
