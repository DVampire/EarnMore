import torch
from typing import Tuple
from copy import deepcopy
from torch import Tensor
from types import MethodType
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from tqdm.auto import tqdm

import numpy as np
from einops import rearrange
from pm.registry import AGENT
from pm.registry import NET
from pm.registry import CRITERION
from pm.registry import OPTIMIZER
from pm.registry import SCHEDULER
from pm.utils import ReplayBuffer, build_storage, get_optim_param
from pm.metrics import ARR, VOL, DD, MDD, SR, CR, SOR

@AGENT.register_module()
class AgentPPO():
    def __init__(self,
                 act_lr: float = None,
                 cri_lr: float = None,
                 act_net: dict = None,
                 cri_net: dict = None,
                 criterion: dict = None,
                 optimizer: dict = None,
                 scheduler: dict = None,
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
                 ratio_clip: float = 0.25,
                 lambda_gae_adv: float = 0.95,
                 lambda_entropy: float = 0.01,
                 device: torch.device = torch.device("cuda"),
                 action_wrapper_method: str = "reweight",
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

        self.ratio_clip = ratio_clip
        self.lambda_gae_adv = lambda_gae_adv
        self.lambda_entropy = lambda_entropy
        self.lambda_entropy = torch.tensor(self.lambda_entropy, dtype=torch.float32, device=self.device)

        self.clip_grad_norm = clip_grad_norm
        self.soft_update_tau = soft_update_tau
        self.state_value_tau = state_value_tau

        self.last_state = None

        self.act = NET.build(act_net).to(self.device)
        self.cri = NET.build(cri_net).to(self.device)

        # build optimizer
        act_optimizer = deepcopy(optimizer)
        act_optimizer.update(dict(params=self.act.parameters(), lr=act_lr))
        self.act_optimizer = OPTIMIZER.build(act_optimizer)
        self.act_optimizer.parameters = MethodType(get_optim_param, self.act_optimizer)

        cri_optimizer = deepcopy(optimizer)
        cri_optimizer.update(dict(params=self.cri.parameters(), lr=cri_lr))
        self.cri_optimizer = OPTIMIZER.build(cri_optimizer) if cri_net else self.act_optimizer
        self.cri_optimizer.parameters = MethodType(get_optim_param, self.cri_optimizer)

        self.convert_action = self.covert_action_wrapper(method=action_wrapper_method)

        # build scheduler
        scheduler.update(dict(optimizer=self.act_optimizer))
        self.act_scheduler = SCHEDULER.build(scheduler)
        scheduler.update(dict(optimizer=self.cri_optimizer))
        self.cri_scheduler = SCHEDULER.build(scheduler)

        self.if_use_per = if_use_per
        if self.if_use_per:
            criterion.update(dict(reduction = "none"))
        else:
            criterion.update(dict(reduction="mean"))
        self.criterion = CRITERION.build(criterion)

        self.global_step = 0

    def covert_action_wrapper(self,
                              method="softmax",  # softmax, reweight
                              ):
        def convert_action(
                       preds,
                       mask=None,
                       mask_value=1e6,
                       **kwargs):
            if mask is not None:
                mask_tensor = torch.from_numpy(mask).to(preds.device)
                mask_bool = torch.concat(
                    [torch.zeros((preds.shape[0], 1), dtype=torch.bool, device=preds.device), mask_tensor], dim=1).float()
                mask_bool = mask_bool * mask_value

            if method == "softmax":
                if mask is not None:
                    preds = preds - mask_bool

                weight = F.softmax(preds, dim=-1)

            elif method == "reweight":

                if mask is not None:
                    preds = preds - mask_bool

                indices = torch.sort(preds)[1]
                soft_pred = preds * torch.log(indices + 1)

                weight = F.softmax(soft_pred, dim=-1).squeeze(-1)
            else:
                raise NotImplementedError
            return weight

        return convert_action

    def get_state_dict(self):
        print("get state dict")
        state_dict = {
            "act": self.act.state_dict(),
            "cri": self.cri.state_dict(),
            "act_optimizer": self.act_optimizer.state_dict(),
            "cri_optimizer": self.cri_optimizer.state_dict(),
            "act_scheduler": self.act_scheduler.state_dict(),
            "cri_scheduler": self.cri_scheduler.state_dict(),
        }
        print("get state dict success")
        return state_dict

    def set_state_dict(self, state_dict):
        print("set state dict")
        self.act.load_state_dict(state_dict["act"])
        self.cri.load_state_dict(state_dict["cri"])
        self.act_optimizer.load_state_dict(state_dict["act_optimizer"])
        self.cri_optimizer.load_state_dict(state_dict["cri_optimizer"])
        self.act_scheduler.load_state_dict(state_dict["act_scheduler"])
        self.cri_scheduler.load_state_dict(state_dict["cri_scheduler"])
        print("set state dict success")

    def explore_env(self, env, horizon_len: int) -> Tuple[Tensor, ...]:

        states = build_storage((horizon_len, * self.transition_shape["state"]["shape"]),
                               self.transition_shape["state"]["type"], self.device)
        actions = build_storage((horizon_len, *self.transition_shape["action"]["shape"]),
                               self.transition_shape["action"]["type"], self.device)
        logprobs = build_storage((horizon_len, *self.transition_shape["logprob"]["shape"]),
                                self.transition_shape["logprob"]["type"], self.device)
        rewards = build_storage((horizon_len, *self.transition_shape["reward"]["shape"]),
                                self.transition_shape["reward"]["type"], self.device)
        dones = build_storage((horizon_len, *self.transition_shape["done"]["shape"]),
                                self.transition_shape["done"]["type"], self.device)
        next_states = build_storage((horizon_len, *self.transition_shape["next_state"]["shape"]),
                              self.transition_shape["next_state"]["type"], self.device)

        state = self.last_state

        for t in range(horizon_len):
            b, e, n, d, f = state.shape

            preds, logprob = self.act.get_action_logprob(rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e))

            states[t] = state

            ary_action = self.convert_action(preds).detach().cpu().numpy()
            next_state, reward, done, _ = env.step(ary_action)

            ary_state = env.reset() if np.sum(done) > 0 else next_state  # reset if done

            state = torch.as_tensor(ary_state, dtype=torch.float32, device=self.device).unsqueeze(0)  # next state
            reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = torch.as_tensor(done, dtype=torch.float32, device=self.device).unsqueeze(0)

            actions[t] = preds.unsqueeze(0)
            logprobs[t] = logprob.unsqueeze(0)
            rewards[t] = reward
            dones[t] = done
            next_states[t] = state

        self.last_state = state  # state.shape == (1, state_dim) for a single env.

        rewards *= self.reward_scale
        dones = dones.type(torch.float32)
        return states, actions, logprobs, rewards, dones, next_states

    def optimizer_update(self, optimizer: torch.optim, objective: Tensor, lr_scheduler=None, step=None):
        """minimize the optimization objective via update the network parameters

        optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]["params"], max_norm=self.clip_grad_norm)
        optimizer.step()
        lr_scheduler.step_update(step)

    def optimizer_update_amp(self, optimizer: torch.optim, objective: Tensor, lr_scheduler=None,
                             step=None):  # automatic mixed precision
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
        lr_scheduler.step_update(step)

    def get_advantages(self, rewards: Tensor, dones: Tensor, values: Tensor) -> Tensor:

        b, e, n, d, f = self.last_state.shape
        horizon_len = rewards.shape[0] // e

        rewards = rearrange(rewards, "(b e) -> b e", b=horizon_len, e=e)
        dones = rearrange(dones, "(b e) -> b e", b=horizon_len, e=e)
        values = rearrange(values, "(b e) -> b e", b=horizon_len, e=e)

        advantages = torch.empty_like(values)  # advantage value

        masks = (1 - dones) * self.gamma

        next_value = self.cri(rearrange(self.last_state, "b e n d f -> (b e) n d f", b=b, e=e)).detach()

        advantage = torch.zeros_like(next_value)  # last advantage value by GAE (Generalized Advantage Estimate)
        for t in range(horizon_len - 1, -1, -1):
            next_value = rewards[t] + masks[t] * next_value
            advantages[t] = advantage = next_value - values[t] + masks[t] * self.lambda_gae_adv * advantage
            next_value = values[t]

        advantages = rearrange(advantages, "b e -> (b e)")

        return advantages

    def update_net(self, buffer: ReplayBuffer) -> dict:

        with torch.no_grad():
            states, actions, logprobs, rewards, dones, next_states  = buffer.get_data(self.batch_size)

            values = self.cri(states)

            advantages = self.get_advantages(rewards, dones, values)
            reward_sums = advantages + values
            del rewards, dones, values

            advantages = (advantages - advantages.mean()) / (advantages.std(dim=0) + 1e-4)

        '''update network'''
        obj_critics = 0.0
        obj_actors = 0.0

        update_times = int(self.repeat_times)

        assert update_times >= 1
        for _ in tqdm(range(update_times), bar_format="update net batch " + "{bar:50}{percentage:3.0f}%|{elapsed}/{remaining}{postfix}"):
            ids = torch.randint(states.shape[0], size=(states.shape[0],), requires_grad=False)

            state = states[ids]
            action = actions[ids]
            logprob = logprobs[ids]
            advantage = advantages[ids]
            reward_sum = reward_sums[ids]

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state

            obj_critic = self.criterion(value, reward_sum)
            self.optimizer_update(self.cri_optimizer, obj_critic, lr_scheduler=self.cri_scheduler, step=self.global_step)

            new_logprob, obj_entropy = self.act.get_logprob_entropy(state, action)
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = torch.min(surrogate1, surrogate2).mean()

            obj_actor = obj_surrogate + obj_entropy.mean() * self.lambda_entropy
            self.optimizer_update(self.act_optimizer, -obj_actor, lr_scheduler=self.act_scheduler, step=self.global_step)

            obj_critics += obj_critic.item()
            obj_actors += obj_actor.item()

            self.global_step += 1

        act_lr = self.act_optimizer.param_groups[0]["lr"]
        cri_lr = self.cri_optimizer.param_groups[0]["lr"]
        a_std_log = self.act.action_std_log.mean() if hasattr(self.act, 'action_std_log') else torch.zeros(1)

        stats = {
            "obj_critics":obj_critics / update_times,
            "obj_actors":obj_actors / update_times,
            "a_std_log": a_std_log.item(),
            "act_lr":act_lr,
            "cri_lr":cri_lr,
        }

        return stats

    @torch.no_grad()
    def validate_net(self, environment, if_visualize=False):
        state = environment.reset()

        num_envs = environment.num_envs
        rets = [[] for _ in range(num_envs)]

        # visualize = None
        fig_list = None
        # last_ary_action = None
        # date = datetime.strptime(environment.get_current_date(), "%Y-%m-%d")
        # if if_visualize:
        #     visualize = {
        #         'return': [],
        #         'cost': [],
        #         'bench': [],
        #         'turnover': [],
        #         'date':[]
        #     }

        aux_stocks = environment.envs[0].aux_stocks
        masks = []
        for i in range(num_envs):
            mask = aux_stocks[i]["mask"]
            masks.append(mask)
        masks = np.array(masks)

        while True:
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            b, e, n, d, f = state.shape

            preds = self.act(x=rearrange(state, "b e n d f -> (b e) n d f", b=b, e=e))

            ary_action = self.convert_action(preds=preds, mask = masks).detach().cpu().numpy()
            state, reward, done, info = environment.step(ary_action)  # next_state

            # if if_visualize:
            #     if last_ary_action is not None:
            #         turnover = np.sum(np.maximum(ary_action - last_ary_action, 0))
            #         visualize['turnover'].append(turnover)
            #         cost = turnover * 1e-4
            #         visualize['cost'].append(cost)
            #     else:
            #         visualize['turnover'].append(0)
            #         visualize['cost'].append(0)
            #
            #     visualize['bench'].append(0)
            #     visualize['return'].append(reward)
            #     visualize['date'].append(date)
            #
            #     date = datetime.strptime(environment.get_current_date(), "%Y-%m-%d")
            #     last_ary_action = ary_action

            for i in range(num_envs):
                rets[i].append(info[i]['portfolio_ret'])

            if np.sum(done) > 0:  # if any done
                break

        # if if_visualize:
        #     visualize = pd.DataFrame(visualize, index=visualize['date'])
        #     visualize.index.name = 'date'
        #     fig_list = report_graph(visualize, show_notebook=False)

        metrics = {}

        for i in range(num_envs):
            # calculate total return
            rets_ary = np.array(rets[i])

            arr = ARR(rets_ary)
            # calculate volatility
            vol = VOL(rets_ary)
            # calculate downside deviation
            dd = DD(rets_ary)
            # calculate maximum drawdown
            mdd = MDD(rets_ary)
            # calculate sharpe ratio
            sr = SR(rets_ary)
            # calculate calmar ratio
            cr = CR(rets_ary, mdd)
            # calculate sortino ratio
            sor = SOR(rets_ary, dd)

            metrics.update({
                "ARR%_env{}".format(i): arr * 100,
                "SR_env{}".format(i): sr,
                "CR_env{}".format(i): cr,
                "MDD%_env{}".format(i): mdd * 100,
                "VOL_env{}".format(i): vol,
                "DD_env{}".format(i): dd,
                "SOR_env{}".format(i): sor,
            })

        if if_visualize:
            return metrics, fig_list
        else:
            return metrics