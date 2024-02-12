import torch
import torch.nn.functional as F
import numpy as np
def discretized_actions(action, discretized_low, discretized_high):
    scaled_tensor = action * (discretized_high - discretized_low) + discretized_low

    discretized_tensor = torch.floor(scaled_tensor)
    diff = torch.tensor([discretized_high - discretized_low - torch.sum(row) for row in discretized_tensor])

    for row_idx, row_diff in enumerate(diff):
        if row_diff > 0:
            indices_to_increment = torch.topk(scaled_tensor[row_idx] - discretized_tensor[row_idx], int(row_diff))[1]
            discretized_tensor[row_idx, indices_to_increment] += 1
        elif row_diff < 0:
            indices_to_decrement = torch.topk(discretized_tensor[row_idx] - (scaled_tensor[row_idx] - 1), int(abs(row_diff)))[1]
            discretized_tensor[row_idx, indices_to_decrement] -= 1

    return discretized_tensor.long()

def get_optim_param(optimizer: torch.optim) -> list:
    params_list = []
    for params_dict in optimizer.state_dict()["state"].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list

def get_action_wrapper(func,
                       method = "softmax", # softmax, reweight
                       T = 1.0,
                       ):
    def get_action(x,
                   mask = None,
                   mask_value = 1e6,
                   **kwargs):
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).to(x.device)
            mask_bool = torch.concat(
                [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), mask_tensor], dim=1).float()
            mask_bool = mask_bool * mask_value

        if method == "softmax":
            pred = func(x, **kwargs)
            pred = pred / T

            if mask is not None:
                pred = pred - mask_bool

            weight = F.softmax(pred, dim=-1)

        elif method == "reweight":
            pred = func(x, **kwargs)

            if mask is not None:
                pred = pred - mask_bool

            indices = torch.sort(pred)[1]
            soft_pred = pred * torch.log(indices + 1)

            weight = F.softmax(soft_pred, dim=-1).squeeze(-1)
        else:
            raise NotImplementedError
        return weight
    return get_action

def get_action_logprob_wrapper(func,
                       method = "softmax", # softmax, reweight
                       T = 1.0,
                       ):
    def get_action(x,
                   mask = None,
                   mask_value = 1e6,
                   **kwargs):
        if mask is not None:
            mask_tensor = torch.from_numpy(mask).to(x.device)
            mask_bool = torch.concat(
                [torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), mask_tensor], dim=1).float()
            mask_bool = mask_bool * mask_value

        if method == "softmax":
            pred, logprob = func(x, **kwargs)
            pred = pred / T

            if mask is not None:
                pred = pred - mask_bool

            weight = F.softmax(pred, dim=-1)

        elif method == "reweight":
            pred, logprob = func(x, **kwargs)

            if mask is not None:
                pred = pred - mask_bool

            indices = torch.sort(pred)[1]
            soft_pred = pred * torch.log(indices + 1)

            weight = F.softmax(soft_pred, dim=-1).squeeze(-1)

        else:
            raise NotImplementedError
        return weight, logprob
    return get_action

def forward_action_wrapper(func,
                    method = "softmax", # softmax, reweight
                    T = 1.0,
                    ):
    def forward_action(x,
                       mask = None,
                       mask_value = 1e6,
                       **kwargs):

        if mask is not None:
            mask_tensor = torch.from_numpy(mask).to(x.device)
            mask_bool = torch.concat([torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device), mask_tensor], dim=1).float()
            mask_bool = mask_bool * mask_value

        if method == "softmax":
            pred = func(x, **kwargs)
            pred = pred / T

            if mask is not None:
                pred = pred - mask_bool

            weight = F.softmax(pred, dim=-1)

        elif method == "reweight":
            pred = func(x, **kwargs)

            if mask is not None:
                pred = pred - mask_bool

            indices = torch.sort(pred)[1]
            soft_pred = pred * torch.log(indices + 1)

            weight = F.softmax(soft_pred, dim=-1).squeeze(-1)
        else:
            raise NotImplementedError
        return weight

    return forward_action