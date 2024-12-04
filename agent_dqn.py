
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import dqn_model
import prioritized_replay_buffer

GRAPH_PATH = './graphs/'

class AgentDQN:
    def __init__(self, env, args, model_path, time_now):
        self.env = env
        self.num_actions = env.action_space.n
        self.model_path = model_path
        self.time_str = str(time_now)
        
        # Set DQN parameters here
        self.learning_rate = .00025
        self.gamma = .9
        self.epsilon_start = 1.0
        self.epsilon = self.epsilon_start
        self.decay = .999995
        self.epsilon_min = .1
        self.batch_size = 32
        self.replay_buffer_size = 100000
        self.train_start = 10000
        self.update_target_every = 10000
        
        # Replay buffer
        # self.memory = deque(maxlen=self.replay_buffer_size)
        # Try prioritized replay buffer
        self.memory = prioritized_replay_buffer.PrioritizedReplayBuffer(self.replay_buffer_size)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.online_net = dqn_model.DQN(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net = dqn_model.DQN(in_channels=4, num_actions=self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        
        if args.test:
            print("Loading model")
            self.online_net.load_state_dict(torch.load(self.model_path))
            self.online_net.eval()
    
    def take_action(self, observation, test=True):
        if test:
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            # observation = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(observation)
                action = torch.argmax(q_values).item()
        else:
            if np.random.rand() <= self.epsilon:
                action = random.randrange(self.num_actions)
            else:
                observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                # observation = torch.FloatTensor(observation).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
                with torch.no_grad():
                    q_values = self.online_net(observation)
                    action = torch.argmax(q_values).item()
        return action

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    '''
    def replay_buffer(self):
        if (len(self.memory) < self.train_start):
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.stack(states)).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        # states = torch.FloatTensor(np.stack(states)).permute(0, 3, 1, 2).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        # next_states = torch.FloatTensor(np.stack(next_states)).permute(0, 3, 1, 2).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        return states, actions, rewards, next_states, dones
    '''
    
    def replay_buffer(self, beta=0.4):
        if len(self.memory.memory) < self.train_start:
            return None
        transitions, indices, weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.FloatTensor(np.stack(states)).to(self.device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def train(self, iterations):
        state, _ = self.env.reset()
        score = 0
        episode = 0
        episode_rewards = [] # store rewards for graph
        avg_rewards = [] # store average rewards over time

        for t in range(iterations):
            action = self.take_action(state, test=False)
            print("Training iteration: " + str(t + 1))
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            score += reward
            self.push(state, action, reward, next_state, terminated or truncated)
            state = next_state

            if terminated or truncated:
                episode += 1
                episode_rewards.append(score)
                avg_rewards.append(np.mean(episode_rewards[-100:])) # avg last 100 episodes
                print("Episode:", episode, "Score:", score, "Epsilon:", self.epsilon)
                state, _ = self.env.reset()
                score = 0
                if episode % 10 == 0: # update graph every 10
                    self.plot_rewards(episode_rewards, avg_rewards, self.time_str)

            if t % self.update_target_every == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            if t > self.train_start:
                batch = self.replay_buffer()
                if batch is None:
                    continue

                states, actions, rewards, next_states, dones = batch

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken
                current_q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute V(s_{t+1}) for all next states.
                next_q_values = self.target_net(next_states).max(1)[0]

                # Compute the expected Q values
                expected_q_values = (next_q_values * self.gamma) * (~dones) + rewards

                # Compute Huber loss
                loss = F.smooth_l1_loss(current_q_values, expected_q_values)

                # Optimize the model
                self.optimizer.zero_grad()
                loss.backward()
                for param in self.online_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                self.optimizer.step()

                # Decay epsilon
                self.epsilon = max(self.epsilon_min, self.epsilon - (self.epsilon_start - self.epsilon_min) / self.decay)
        # plot last graph
        self.plot_rewards(episode_rewards, avg_rewards, self.time_str)
        # Save model
        torch.save(self.online_net.state_dict(), self.model_path)

    def plot_rewards(self, episode_rewards, avg_rewards, time_now_str):
        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label='Episode Reward')
        plt.plot(avg_rewards, label='Average Reward (100 Episodes)', color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward vs. Episodes')
        plt.legend()
        plt.grid()
        plt.savefig(GRAPH_PATH + 'training_' + time_now_str + '.png')
        plt.close()
