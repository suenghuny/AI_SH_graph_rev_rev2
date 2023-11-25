import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from GDN import NodeEmbedding
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from GAT.model import GAT
from GAT.layers import device
from cfg import get_cfg
import numpy as np
from GCRN.model import GCRN
cfg = get_cfg()
from torch.distributions import Categorical
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPONetwork(nn.Module):
    def __init__(self, state_size, state_action_size, layers=[8,12]):
        super(PPONetwork, self).__init__()
        self.state_size = state_size
        self.state_action_size = state_action_size
        self.NN_sequential = OrderedDict()
        self.fc_pi = nn.Linear(state_action_size, layers[0])
        self.bn_pi = nn.BatchNorm1d(layers[0])
        self.fc_v = nn.Linear(state_size, layers[0])
        self.bn_v = nn.BatchNorm1d(layers[0])
        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            #else:
        self.forward_cal = nn.Sequential(self.fcn)
        self.output_pi = nn.Linear(last_layer, 1)
        self.output_v = nn.Linear(last_layer, 1)




    def pi(self, x):
        x = self.fc_pi(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        pi = self.output_pi(x)
        return pi

    def v(self, x):
        x = self.fc_v(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        v = self.output_v(x)
        return v





class Agent:
    def __init__(self,
                 action_size,
                 feature_size_ship,
                 feature_size_missile,
                 n_node_feature_missile,


                 node_embedding_layers_ship=list(eval(cfg.ship_layers)),
                 node_embedding_layers_missile=list(eval(cfg.missile_layers)),

                 n_representation_ship = cfg.n_representation_ship,
                 n_representation_missile = cfg.n_representation_missile,
                 n_representation_action = cfg.n_representation_action,


                 learning_rate=cfg.lr,
                 learning_rate_critic=cfg.lr_critic,
                 gamma=cfg.gamma,
                 lmbda=cfg.lmbda,
                 eps_clip = cfg.eps_clip,
                 K_epoch = cfg.K_epoch,
                 layers=list(eval(cfg.ppo_layers))
                 ):

        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.data = []

        self.network = PPONetwork(state_size = n_representation_ship,
                               state_action_size = n_representation_ship+cfg.n_representation_action2 ,
                               layers = layers).to(device)
        self.node_representation_ship_feature = NodeEmbedding(feature_size=feature_size_ship,
                                                         n_representation_obs=n_representation_ship,
                                                         layers = node_embedding_layers_ship).to(device)  # 수정사항


        self.func_meta_path = GCRN(feature_size=feature_size_missile,
                                   embedding_size=cfg.n_representation_action,
                                   graph_embedding_size=cfg.hidden_size_meta_path,
                                   layers=node_embedding_layers_missile,
                                   num_node_cat=1,
                                   num_edge_cat=5).to(device)
        self.func_meta_path2 = GCRN(feature_size=cfg.n_representation_action,
                                    embedding_size=cfg.n_representation_action2,
                                    graph_embedding_size=cfg.hidden_size_meta_path2,
                                    layers=node_embedding_layers_missile,
                                    num_node_cat=1,
                                    num_edge_cat=5).to(device)



        self.eval_params = list(self.network.parameters()) + \
                           list(self.node_representation_ship_feature.parameters()) + \
                           list(self.func_meta_path.parameters()) + \
                           list(self.func_meta_path2.parameters())


        self.optimizer = optim.Adam(self.eval_params, lr=learning_rate)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)

        self.dummy_node = [[[0] * feature_size_missile for _ in range(i)] for i in range(n_node_feature_missile)]


        self.ship_feature_list= list()
        self.ship_feature_next_list = list()
        self.missile_node_feature_list= list()
        self.heterogeneous_edges_list= list()
        self.action_feature_list= list()
        self.action_blue_list= list()
        self.reward_list= list()
        self.prob_list= list()
        self.mask_list= list()
        self.done_list= list()
        self.avail_action_blue_list= list()
        self.a_index_list= list()

        self.dummy_ship_feature = torch.zeros((1, feature_size_ship)).to(device).float()

        #self.scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=self.learning_rate, total_steps=30000)
    def eval_check(self, eval):
        if eval == True:
            self.network.eval()
            self.node_representation_ship_feature.eval()
            self.func_meta_path.eval()
            self.func_meta_path2.eval()
        else:
            self.network.train()
            self.node_representation_ship_feature.train()
            self.func_meta_path.train()
            self.func_meta_path2.train()

    def get_node_representation(self,
                                ship_features,
                                missile_node_feature,
                                edge_index_missile,
                                mini_batch=False):
        if mini_batch == False:
            with torch.no_grad():

                ship_features = torch.tensor(ship_features, dtype=torch.float, device=device)
                node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
                missile_node_feature = torch.tensor(missile_node_feature, dtype=torch.float,device=device).clone().detach()
                node_representation_graph = self.func_meta_path(A=edge_index_missile,X=missile_node_feature, mini_batch=mini_batch)
                node_representation_graph = self.func_meta_path2(A=edge_index_missile, X=node_representation_graph,mini_batch=mini_batch)
                node_representation = torch.cat([node_embedding_ship_features], dim=1)
                return node_representation, node_representation_graph
        else:
            """ship feature 만드는 부분"""
            ship_features = torch.tensor(ship_features,dtype=torch.float).to(device).squeeze(1)
            node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
            max_len = np.max([len(mnf) for mnf in missile_node_feature])
            if max_len <= self.action_size:
                max_len = self.action_size
            temp = list()
            for mnf in missile_node_feature:
                temp.append(torch.cat([torch.tensor(mnf), torch.tensor(self.dummy_node[max_len - len(mnf)])], dim=0).tolist())

            missile_node_feature = torch.tensor(temp, dtype=torch.float).to(device)
            node_representation_graph = self.func_meta_path(A=edge_index_missile, X=missile_node_feature, mini_batch=mini_batch)
            node_representation_graph = self.func_meta_path2(A=edge_index_missile, X=node_representation_graph, mini_batch=mini_batch)
            node_representation = torch.cat([node_embedding_ship_features], dim=1)
            return node_representation, node_representation_graph


    def get_ship_representation(self, ship_features):
        """ship feature 만드는 부분"""
        ship_features = torch.tensor(ship_features,dtype=torch.float).to(device).squeeze(1)
        node_embedding_ship_features = self.node_representation_ship_feature(ship_features)
        return node_embedding_ship_features

    def put_data(self, transition):
        self.ship_feature_list.append(transition[0])
        self.missile_node_feature_list.append(transition[1])
        self.heterogeneous_edges_list.append(transition[2])
        self.action_feature_list.append(transition[3])
        self.action_blue_list.append(transition[4])
        self.reward_list.append(transition[5])
        self.prob_list.append(transition[6])
        self.done_list.append(transition[7])
        self.avail_action_blue_list.append(transition[8])
        self.a_index_list.append(transition[9])



    def make_batch(self):

        ship_feature=self.ship_feature_list
        missile_node_feature=self.missile_node_feature_list
        heterogeneous_edges = self.heterogeneous_edges_list
        action_feature = self.action_feature_list
        action_blue = self.action_blue_list
        reward = self.reward_list
        prob = self.prob_list
        mask = self.mask_list
        done = self.done_list
        avail_action_blue = self.avail_action_blue_list
        a_index = self.a_index_list
        #print(len(ship_feature), ship_feature)

        ship_feature = torch.tensor(ship_feature).to(device).float()
        ship_feature_next = torch.cat([ship_feature.clone().detach().squeeze(1), self.dummy_ship_feature], dim = 0)[1:].unsqueeze(1)

        #ship_feature_next = torch.tensor(ship_feature_next).to(device).float()

        action_blue = torch.tensor(action_blue).to(device).float()
        reward = torch.tensor(reward).to(device).float()
        prob = torch.tensor(prob).to(device).float()
        done = torch.tensor(done).to(device).float()
        avail_action_blue = torch.tensor(avail_action_blue).to(device).float()
        a_index = torch.tensor(a_index).to(device).long()


        self.ship_feature_list= list()
        self.ship_feature_next_list = list()
        self.missile_node_feature_list= list()
        self.heterogeneous_edges_list= list()
        self.action_feature_list= list()
        self.action_blue_list= list()
        self.reward_list= list()
        self.prob_list= list()
        self.mask_list= list()
        self.done_list= list()
        self.avail_action_blue_list= list()
        self.a_index_list= list()


        return ship_feature, ship_feature_next, missile_node_feature, heterogeneous_edges, action_feature, action_blue, reward, prob, mask, done, avail_action_blue, a_index

    @torch.no_grad()
    def sample_action(self, ship_features,node_features_missile, heterogenous_edges, possible_actions,action_feature):
        obs, act_graph = self.get_node_representation(ship_features,
                                                    node_features_missile,
                                                      heterogenous_edges,
                                                      mini_batch=False)

        act_graph = act_graph[0:self.action_size, :]

        obs = obs.expand([act_graph.shape[0], obs.shape[1]])
        obs_n_action = torch.cat([obs, act_graph], dim = 1)



        logit = [self.network.pi(obs_n_action[i].unsqueeze(0)) for i in range(obs_n_action.shape[0])]
        logit = torch.stack(logit).view(1, -1)
        action_size = obs_n_action.shape[0]
        if action_size >= self.action_size:
             action_size = self.action_size
        remain_action = torch.tensor([-1e8 for _ in range(self.action_size - action_size)], device=device).unsqueeze(0)
        logit = torch.cat([logit, remain_action], dim=1)
        #print(logit.shape, remain_action.shape)
        mask = torch.tensor(possible_actions, device=device).bool()
        #print(logit.shape, mask.shape)
        logit = logit.masked_fill(mask == 0, -1e8)
        prob = torch.softmax(logit, dim=-1)
        m = Categorical(prob)
        a = m.sample().item()


        a_index = a
        prob_a = prob.squeeze(0)[a]


        action_blue = action_feature[a]

        return action_blue, prob_a, mask, a_index

    def learn(self):
        ship_feature, \
        ship_feature_next, \
        missile_node_feature, \
        heterogeneous_edges, \
        action_feature, \
        action_blue, \
        reward, \
        prob, \
        _, \
        done, \
        avail_action_blue, \
        a_index = self.make_batch()
        avg_loss = 0.0
        a_indices = a_index.unsqueeze(1)
        self.eval_check(eval = False)
        for i in range(self.K_epoch):
            obs, act_graph = self.get_node_representation(ship_feature,missile_node_feature,heterogeneous_edges,mini_batch=True)
            obs_next = self.get_ship_representation(ship_feature_next)
            v_s = self.network.v(obs)
            td_target = reward.unsqueeze(1) + self.gamma * self.network.v(obs_next) * (1-done).unsqueeze(1)
            delta = td_target - v_s
            delta = delta.cpu().detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[ : :-1]:
                advantage = self.gamma * self.lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            mask = avail_action_blue

            obs_expand = obs.unsqueeze(1).expand([obs.shape[0], act_graph.shape[1], obs.shape[1]])
            obs_n_act = torch.cat([obs_expand, act_graph], dim= 2)
            logit = torch.stack([self.network.pi(obs_n_act[:, i]) for i in range(self.action_size)])
            logit = torch.einsum('ijk->jik', logit).squeeze(2)

            mask = mask.squeeze(1)
            logit = logit.masked_fill(mask == 0, -1e8)
            pi = torch.softmax(logit, dim=-1)
            pi_a = pi.gather(1, a_indices)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            if cfg.entropy == True:
                entropy = -torch.sum(torch.exp(pi) * pi, dim=1)
                loss = - torch.min(surr1, surr2) + 0.5 * F.smooth_l1_loss(v_s, td_target.detach()) -0.01*entropy.mean()# 수정 + 엔트로피
            else:
                loss = - torch.min(surr1, surr2) + 0.5 * F.smooth_l1_loss(v_s, td_target.detach())

            #print(-torch.sum(torch.exp(pi) * pi, dim=1))



            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.eval_params, cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            #avg_loss += loss.mean().item()

        return avg_loss / self.K_epoch

    def save_network(self, e, file_dir):

        # self.eval_params = list(self.network.parameters()) + \
        #                    list(self.node_representation_ship_feature.parameters()) + \
        #                    list(self.func_meta_path.parameters()) + \
        #                    list(self.func_meta_path2.parameters())


        torch.save({"episode": e,
                    "network": self.network.state_dict(),
                    "node_representation_ship_feature": self.node_representation_ship_feature.state_dict(),
                    "func_meta_path": self.func_meta_path.state_dict(),
                    "func_meta_path2": self.func_meta_path2.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   file_dir + "episode%d.pt" % e)



# action_encoding = np.eye(action_size, dtype = np.float)
#
# cumul = list()
# for n_epi in range(10000):
#     s = env.reset()
#     done = False
#     epi_reward = 0
#     s = s[0]
#     step =0
#     while not done:
#         a, prob, mask = agent.get_action(s)
#         s_prime, r, done, info, _ = env.step(a)
#         mask = [True, True]
#         epi_reward+= r
#         step+=1
#         agent.put_data((s, action_encoding[a], r, s_prime, prob[a].item(), mask, done))
#         s = s_prime
#     cumul.append(epi_reward)
#     n = 100
#     if n_epi > n:
#         print(np.mean(cumul[-n:]))
#     agent.train()
#     #print(r)