from . import RLAgent
import random
import numpy as np
from collections import deque
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.registry import Registry
import gym
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator, IntersectionVehicleGenerator
from torch.nn.utils import clip_grad_norm_
from agent import utils


@Registry.register_model('frap_lstm')
class LSTMFRAP_DQNAgent(RLAgent):
    '''
    FRAP_DQNAgent consists of FRAP and methods for training agents, communicating with environment, etc.
    '''
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        self.dic_agent_conf = Registry.mapping['model_mapping']['setting']
        self.dic_traffic_env_conf = Registry.mapping['world_mapping']['setting']
        
        self.gamma = self.dic_agent_conf.param["gamma"]
        self.grad_clip = self.dic_agent_conf.param["grad_clip"]
        self.epsilon = self.dic_agent_conf.param["epsilon"]
        self.epsilon_min = self.dic_agent_conf.param["epsilon_min"]
        self.epsilon_decay = self.dic_agent_conf.param["epsilon_decay"]
        self.learning_rate = self.dic_agent_conf.param["learning_rate"]
        self.batch_size = self.dic_agent_conf.param["batch_size"]
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.world = world
        self.sub_agents = 1
        self.rank = rank

        self.phase = self.dic_agent_conf.param['phase']
        self.one_hot = self.dic_agent_conf.param['one_hot']

        # get generator for each Agent
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)
        
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)

        map_name = self.dic_traffic_env_conf.param['network']

        # set valid action
        all_valid_acts = self.dic_traffic_env_conf.param['signal_config'][map_name]['valid_acts']
        if all_valid_acts is None:
            self.valid_acts = None
        else:
            if self.inter_id in all_valid_acts.keys():
                self.inter_name = self.inter_id
            else:
                if 'GS_' in self.inter_id:
                    self.inter_name = self.inter_id[3:]
                else:
                    self.inter_name = 'GS_' + self.inter_id
            self.valid_acts = all_valid_acts[self.inter_name]
        
        self.ob_order = None
        if 'lane_order' in self.dic_traffic_env_conf.param['signal_config'][map_name].keys():
            self.ob_order = self.dic_traffic_env_conf.param['signal_config'][map_name]['lane_order'][self.inter_name]
        
        # set phase_pairs
        self.phase_pairs = []
        all_phase_pairs = self.dic_traffic_env_conf.param['signal_config'][map_name]['phase_pairs']
        if self.valid_acts:
            for idx in self.valid_acts:
                self.phase_pairs.append([self.ob_order[x] for x in all_phase_pairs[idx]])
        else:
            self.phase_pairs = all_phase_pairs

        self.comp_mask = self.relation()
        self.num_phases = len(self.phase_pairs)
        self.num_actions = len(self.phase_pairs)
        
        # Initialize the model with LSTM
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=1e-7)
        self.criterion = nn.MSELoss(reduction='mean')

    def __repr__(self):
        return self.model.__repr__()

    def reset(self):
        '''
        reset
        Reset information, including ob_generator, phase_generator, queue, delay, etc.

        :param: None
        :return: None
        '''
        self.inter_id = self.world.intersection_ids[self.rank]
        self.inter_obj = self.world.id2intersection[self.inter_id]
        self.action_space = gym.spaces.Discrete(len(self.inter_obj.phases))
        self.ob_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                 ["lane_count"], in_only=True, average=None)
        self.phase_generator = IntersectionPhaseGenerator(self.world, self.inter_obj,
                                                          ['phase'], targets=['cur_phase'], negative=False)
        self.reward_generator = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True, average="all",
                                                     negative=True)
        self.queue = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_waiting_count"], in_only=True,
                                                     negative=False)
        self.delay = LaneVehicleGenerator(self.world, self.inter_obj,
                                                     ["lane_delay"], in_only=True, average="all",
                                                     negative=False)
    
    def relation(self):
        '''
        relation
        Get the phase competition relation between traffic movements.

        :param: None
        :return comp_mask: matrix of phase competition relation
        '''
        comp_mask = []
        for i in range(len(self.phase_pairs)):
            zeros = np.zeros(len(self.phase_pairs) - 1, dtype=np.int64)
            cnt = 0
            for j in range(len(self.phase_pairs)):
                if i == j: continue
                pair_a = self.phase_pairs[i]
                pair_b = self.phase_pairs[j]
                if len(list(set(pair_a + pair_b))) == 3: zeros[cnt] = 1
                cnt += 1
            comp_mask.append(zeros)
        comp_mask = torch.from_numpy(np.asarray(comp_mask))
        return comp_mask 

    def _build_model(self):
        '''
        _build_model
        Build a FRAP agent with LSTM.

        :param: None
        :return model: FRAP model with LSTM
        '''
        model = FRAP_LSTM(self.dic_agent_conf, self.num_actions, self.phase_pairs, self.comp_mask)
        return model

    def get_ob(self):
        '''
        get_ob
        Get observation from environment.

        :param: None
        :return x_obs: observation generated by ob_generator
        '''
        x_obs = []  # lane_nums
        tmp = self.ob_generator.generate()
        if self.ob_order != None:
            tt = []
            for i in range(12):
                # padding to 12 dims
                if i in self.ob_order.keys():
                    tt.append(tmp[self.ob_order[i]])
                else:
                    tt.append(0.)
            x_obs.append(np.array(tt))     
        else:
            x_obs.append(tmp)
        return x_obs

    def get_reward(self):
        '''
        get_reward
        Get reward from environment.

        :param: None
        :return rewards: rewards generated by reward_generator
        '''
        rewards = []
        rewards.append(self.reward_generator.generate())
        # TODO check whether to multiply 12
        rewards = np.squeeze(np.array(rewards))# * self.num_phases
        return rewards

    def get_phase(self):
        '''
        get_phase
        Get current phase of intersection(s) from environment.

        :param: None
        :return phase: current phase generated by phase_generator
        '''
        phase = []
        phase.append(self.phase_generator.generate())
        phase = (np.concatenate(phase)).astype(np.int8)
        return phase

    def get_action(self, ob, phase, test=False):
        '''
        get_action
        Generate action.

        :param ob: observation, the shape is (1,12)
        :param phase: current phase, the shape is (1,)
        :param test: boolean, decide whether is test process
        :return: action that has the highest score
        '''
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        if self.phase:
            if self.one_hot:
                feature_p = utils.idx2onehot(phase, self.action_space.n)
                feature = np.concatenate([feature_p, ob], axis=1)
            else:
                feature = np.concatenate([phase.reshape(1,-1), ob], axis=1)
        else:
            feature = ob
        observation = torch.tensor(feature, dtype=torch.float32)
        actions = self.model(observation, train=False)
        return actions.argmax().item()

    def sample(self):
        '''
        sample
        Generate sample action.

        :param: None
        :return action: sampled action
        '''
        if self.valid_acts:
            return random.choice(range(len(self.valid_acts)))
        return random.choice(range(self.action_space.n))

    def update_target_network(self):
        '''
        update_target_network
        Update target network.

        :param: None
        :return: None
        '''
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, ob, phase, action, reward, next_ob, next_phase, done):
        '''
        remember
        Store experience in replay buffer.

        :param ob: current observation
        :param phase: current phase
        :param action: action taken
        :param reward: reward received
        :param next_ob: next observation
        :param next_phase: next phase
        :param done: boolean, whether the episode is done
        :return: None
        '''
        self.replay_buffer.append((ob, phase, action, reward, next_ob, next_phase, done))

    def replay(self):
        '''
        replay
        Replay and train the model.

        :param: None
        :return: None
        '''
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        for ob, phase, action, reward, next_ob, next_phase, done in batch:
            target = reward
            if not done:
                if self.phase:
                    if self.one_hot:
                        feature_p = utils.idx2onehot(next_phase, self.action_space.n)
                        feature = np.concatenate([feature_p, next_ob], axis=1)
                    else:
                        feature = np.concatenate([next_phase.reshape(1,-1), next_ob], axis=1)
                else:
                    feature = next_ob
                target = reward + self.gamma * torch.max(self.target_model(torch.tensor(feature, dtype=torch.float32)).detach()).item()
            
            if self.phase:
                if self.one_hot:
                    feature_p = utils.idx2onehot(phase, self.action_space.n)
                    feature = np.concatenate([feature_p, ob], axis=1)
                else:
                    feature = np.concatenate([phase.reshape(1,-1), ob], axis=1)
            else:
                feature = ob
            q_values = self.model(torch.tensor(feature, dtype=torch.float32))
            target_f = q_values.clone()
            target_f[action] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# @Registry.register_network('frap_lstm')

class FRAP_LSTM(nn.Module):
    '''
    FRAP_LSTM: A FRAP model with LSTM.
    '''
    def __init__(self, dic_agent_conf, num_actions, phase_pairs, comp_mask):
        super(FRAP_LSTM, self).__init__()
        self.num_actions = num_actions
        self.phase_pairs = phase_pairs
        self.comp_mask = comp_mask
        self.hidden_dim = dic_agent_conf.param['hidden_dim']
        self.lstm_layers = dic_agent_conf.param['lstm_layers']
        
        # Feature extraction layers
        self.linear1 = nn.Linear(12, 100)
        self.linear2 = nn.Linear(100, self.hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.lstm_layers, batch_first=True)
        
        # Output layers
        self.comp_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, self.num_actions)

    def forward(self, x, train=True):
        '''
        forward
        Forward pass for the network.

        :param x: input tensor
        :param train: boolean, whether the model is in training mode
        :return: output tensor
        '''
        batch_size = x.size(0)
        
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        
        # Initialize LSTM hidden and cell states
        h0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.lstm_layers, batch_size, self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        x, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        x = x.squeeze(1)
        
        # Competition layer
        all_feats = []
        for i in range(self.num_actions):
            comp_feats = []
            for j in range(self.num_actions - 1):
                comp_feats.append(x * self.comp_mask[i][j])
            all_feats.append(torch.cat(comp_feats, dim=1))
        
        # Output layer
        out = F.relu(self.comp_layer(torch.cat(all_feats, dim=1)))
        out = self.out_layer(out)
        return out
