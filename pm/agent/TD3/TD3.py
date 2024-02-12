import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor
from types import MethodType
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np

from pm.registry import AGENT
from pm.registry import NET
from pm.registry import CRITERION
from pm.registry import OPTIMIZER
from pm.utils import ReplayBuffer, build_storage, get_optim_param, discretized_actions
from pm.metrics import ARR, VOL, DD, MDD, SR, CR, SOR

@AGENT.register_module()
class AgentTD3():
    def __init__(self,
                 act_net: dict = None,
                 cri_net: dict = None,
                 criterion: dict = None,
                 optimizer: dict = None,
                 if_use_per: bool = False,
                 num_envs: int = 1,
                 max_step: int = 1e4,
                 transition_shape: dict = None,
                 gamma: float = 0.99,
                 reward_scale: int = 2**0,
                 repeat_times: float = 1.0,
                 batch_size: int = 512,
                 clip_grad_norm: float = 3.0,
                 soft_update_tau: float = .0,
                 state_value_tau: float = 5e-3,
                 explore_noise_std: float = 0.05,
                 policy_noise_std: float = 0.1,
                 update_freq: int = 2,
                 device: torch.device = torch.device("cuda")
                 ):
        self.if_use_per = if_use_per

        self.num_envs = num_envs
        self.device = torch.device("cuda") if not device else device
        self.max_step = max_step

        self.transition_shape = transition_shape

        self.gamma = gamma
        self.reward_scale = reward_scale
        self.repeat_times = repeat_times
        self.batch_size = batch_size

        self.clip_grad_norm = clip_grad_norm
        self.soft_update_tau = soft_update_tau
        self.state_value_tau = state_value_tau

        self.explore_noise_std = explore_noise_std
        self.policy_noise_std = policy_noise_std
        self.update_freq = update_freq

        self.last_state = None

        self.act = self.act_target = NET.build(act_net).to(self.device)
        self.cri = self.cri_target = NET.build(cri_net).to(self.device) if cri_net else self.act
        self.cri_target = deepcopy(self.cri).to(self.device)

        self.act.explore_noise_std = explore_noise_std

        optimizer.update(dict(params=self.act.parameters()))
        self.act_optimizer = OPTIMIZER.build(optimizer)
        optimizer.update(dict(params=self.cri.parameters()))
        self.cri_optimizer = OPTIMIZER.build(optimizer) if cri_net else self.act_optimizer
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        if self.num_envs == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

        self.if_use_per = if_use_per
        if self.if_use_per:
            criterion.update(dict(reduction = "none"))
            self.get_obj_critic = self.get_obj_critic_per
        else:
            criterion.update(dict(reduction="mean"))
            self.get_obj_critic = self.get_obj_critic_raw
        self.criterion = CRITERION.build(criterion)

        self.target_entropy = transition_shape["action"]["shape"][-1]  # action dim

    def get_state_dict(self):
        print("get state dict")
        state_dict = {
            "act": self.act.state_dict(),
            "cri": self.cri.state_dict(),
            "act_target": self.act_target.state_dict(),
            "cri_target": self.cri_target.state_dict(),
            "act_optimizer": self.act_optimizer.state_dict(),
            "cri_optimizer": self.cri_optimizer.state_dict(),
        }
        print("get state dict success")
        return state_dict

    def set_state_dict(self, state_dict):
        print("set state dict")
        self.act.load_state_dict(state_dict["act"])
        self.cri.load_state_dict(state_dict["cri"])
        self.act_target.load_state_dict(state_dict["act_target"])
        self.cri_target.load_state_dict(state_dict["cri_target"])
        self.act_optimizer.load_state_dict(state_dict["act_optimizer"])
        self.cri_optimizer.load_state_dict(state_dict["cri_optimizer"])
        print("set state dict success")

    def explore_one_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:
        states = build_storage((horizon_len, *self.transition_shape["state"]["shape"]),
                                self.transition_shape["state"]["type"], self.device)
        actions = build_storage((horizon_len, *self.transition_shape["action"]["shape"]),
                                self.transition_shape["action"]["type"], self.device)
        rewards = build_storage((horizon_len, *self.transition_shape["reward"]["shape"]),
                                self.transition_shape["reward"]["type"], self.device)
        dones = build_storage((horizon_len, *self.transition_shape["done"]["shape"]),
                                self.transition_shape["done"]["type"], self.device)
        next_states = build_storage((horizon_len, *self.transition_shape["next_state"]["shape"]),
                                    self.transition_shape["next_state"]["type"], self.device)

        state = self.last_state

        get_action = self.act.get_action

        for t in range(horizon_len):
            action = get_action(state)

            if len(action) > 1:
                action = action[0]

            states[t] = state

            ary_action = action.detach().cpu().numpy()
            _, _, reward, done, next_state = env.step(ary_action)
            ary_state = env.reset() if done else next_state
            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            actions[t] = action
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state

        self.last_state = state  # state.shape == (1, state_dim) for a single env.

        rewards *= self.reward_scale
        dones = dones.type(torch.float32)
        return states, actions, rewards, dones, next_states

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()

    def optimizer_update_amp(self, optimizer: torch.optim, objective: Tensor):  # automatic mixed precision
        """minimize the optimization objective via update the network parameters

        amp: Automatic Mixed Precision

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        amp_scale = torch.cuda.amp.GradScaler()  # write in __init__()

        optimizer.zero_grad()
        amp_scale.scale(objective).backward()  # loss.backward()
        amp_scale.unscale_(optimizer)  # amp

        # from torch.nn.utils import clip_grad_norm_
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        amp_scale.step(optimizer)  # optimizer.step()
        amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net: torch.nn.Module, current_net: torch.nn.Module, tau: float):
        """soft update target network via current network

        target_net: update target network via current network to make training more stable.
        current_net: current network update via an optimizer
        tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def update_net(self, buffer: ReplayBuffer) -> dict:
        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(buffer.add_size * self.repeat_times)
        assert update_times >= 1

        for update_c in tqdm(range(update_times), bar_format="update net batch " + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
            obj_critic, state = self.get_obj_critic(buffer, self.batch_size)
            obj_critics += obj_critic.item()
            self.optimizer_update(self.cri_optimizer, obj_critic)
            self.soft_update(self.cri_target, self.cri, self.soft_update_tau)

            if update_c % self.update_freq == 0:  # delay update
                action_pg = self.act(state)  # policy gradient
                obj_actor = self.cri_target(state, action_pg).mean()  # use cri_target is more stable than cri
                obj_actors += obj_actor.item()
                self.optimizer_update(self.act_optimizer, -obj_actor)
                self.soft_update(self.act_target, self.act, self.soft_update_tau)

        stats = {
            "obj_critics":obj_critics / update_times,
            "obj_actors":obj_actors / update_times,
        }

        return stats

    @torch.no_grad()
    def validate_net(self, environment):

        state = environment.reset()
        rets = []
        while True:
            tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            action = self.act.get_action(tensor_state)

            if len(action) > 1:
                action = action[0]

            action = F.softmax(action, dim=1)

            ary_action = action.detach().cpu().numpy()

            _, _, reward, done, _ = environment.step(ary_action)

            rets.append(reward)
            if done:
                break

        rets = np.array(rets)

        # calculate total return
        arr = ARR(rets)
        # calculate volatility
        vol = VOL(rets)
        # calculate downside deviation
        dd = DD(rets)
        # calculate maximum drawdown
        mdd = MDD(rets)
        # calculate sharpe ratio
        sr = SR(rets)
        # calculate calmar ratio
        cr = CR(rets, mdd)
        # calculate sortino ratio
        sor = SOR(rets, dd)

        metrics = {
            "ARR%": arr * 100,
            "SR": sr,
            "CR": cr,
            "MDD%": mdd * 100,
            "VOL": vol,
            "DD": dd,
            "SOR": sor,
        }

        return metrics

    def get_obj_critic_raw(self, buffer, batch_size: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            states, actions, rewards, dones, next_states = buffer.sample(batch_size)  # next_ss: next states

            if states.device != self.device:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_states = next_states.to(self.device)

            next_actions = self.act_target.get_action_noise(next_states, self.policy_noise_std)  # next actions
            next_qvalues = self.cri_target.get_q_min(next_states, next_actions)  # next q values
            q_labels = rewards + (1 - dones) * self.gamma * next_qvalues

        q1, q2 = self.cri.get_q1_q2(states, actions)
        obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)  # twin critics
        return obj_critic, states

    def get_obj_critic_per(self, buffer: ReplayBuffer, batch_size: int) -> Tuple[Tensor, Tensor]:

        with torch.no_grad():
            states, actions, rewards, dones, next_states, is_weights, is_indices = buffer.sample_for_per(batch_size)

            if states.device != self.device:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                dones = dones.to(self.device)
                next_states = next_states.to(self.device)
                is_weights = is_weights.to(self.device)
                is_indices = is_indices.to(self.device)

            next_actions = self.act_target.get_action_noise(next_states, self.policy_noise_std)
            next_qvalues = self.cri_target.get_q_min(next_states, next_actions)
            q_labels = rewards + (1 - dones) * self.gamma * next_qvalues

        q1, q2 = self.cri.get_q1_q2(states, actions)
        td_errors = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
        obj_critic = (td_errors * is_weights).mean()

        buffer.td_error_update_for_per(is_indices.detach(), td_errors.detach())
        return obj_critic, states