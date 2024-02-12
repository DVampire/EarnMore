# base parameters (do not modify)
root = None
workdir = "workdir"
tag = "TD3"
num_stocks = 420
num_envs = 1
discretized_low = 0
discretized_high = 10
num_features = 102 # num features name + num temporals name
temporal_dim = 3 # weekday, day, month
train_start_date = "2007-09-26"
val_start_date = "2019-07-22"
test_start_date = "2021-01-08"
test_end_date = None
if_use_per = False
if_norm = True
if_norm_temporal = False
save_freq = 20

# train parameters (adjust mainly)
num_episodes = 1000
days = 10
batch_size = 64
buffer_size = 4096
horizon_len = 64
embed_dim = 64
depth = 1 # 2 mlp
lr = 1e-3
explore_noise_std = 0.05
policy_noise_std = 0.1
seed = 10

transition = ["state", "action", "reward", "done", "next_state"]
transition_shape = dict(
    state = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32"),
    action = dict(shape = (num_envs, num_stocks + 1), type = "float32"),
    reward = dict(shape = (num_envs, ), type = "float32"),
    done = dict(shape = (num_envs, ), type = "float32"),
    next_state  = dict(shape = (num_envs, num_stocks, days, num_features), type = "float32")
)

dataset = dict(
    type = "PortfolioManagementDataset",
    root = root,
    data_path= "datasets/sp500/features",
    stocks_path = "datasets/sp500/stocks.txt",
    features_name = [
        'open',
        'high',
        'low',
        'close',
        'kmid2',
        'kup2',
        'klow',
        'klow2',
        'ksft2',
        'roc_5',
        'roc_10',
        'roc_20',
        'roc_30',
        'roc_60',
        'ma_5',
        'ma_10',
        'ma_20',
        'ma_30',
        'ma_60',
        'std_5',
        'std_10',
        'std_20',
        'std_30',
        'std_60',
        'beta_5',
        'beta_10',
        'beta_20',
        'beta_30',
        'beta_60',
        'max_5',
        'max_10',
        'max_20',
        'max_30',
        'max_60',
        'min_5',
        'min_10',
        'min_20',
        'min_30',
        'min_60',
        'qtlu_5',
        'qtlu_10',
        'qtlu_20',
        'qtlu_30',
        'qtlu_60',
        'qtld_5',
        'qtld_10',
        'qtld_20',
        'qtld_30',
        'qtld_60',
        'rank_5',
        'rank_10',
        'rank_20',
        'rank_30',
        'rank_60',
        'imax_5',
        'imax_10',
        'imax_20',
        'imax_30',
        'imax_60',
        'imin_5',
        'imin_10',
        'imin_20',
        'imin_30',
        'imin_60',
        'imxd_5',
        'imxd_10',
        'imxd_20',
        'imxd_30',
        'imxd_60',
        'cntp_5',
        'cntp_10',
        'cntp_20',
        'cntp_30',
        'cntp_60',
        'cntn_5',
        'cntn_10',
        'cntn_20',
        'cntn_30',
        'cntn_60',
        'cntd_5',
        'cntd_10',
        'cntd_20',
        'cntd_30',
        'cntd_60',
        'sump_5',
        'sump_10',
        'sump_20',
        'sump_30',
        'sump_60',
        'sumn_5',
        'sumn_10',
        'sumn_20',
        'sumn_30',
        'sumn_60',
        'sumd_5',
        'sumd_10',
        'sumd_20',
        'sumd_30',
        'sumd_60',
    ],
    temporals_name = [
        "weekday",
        "day",
        "month",
    ],
    labels_name = [
        'ret1',
        'mov1',
    ]
)

environment = dict(
    type = "Environment",
    dataset = None,
    mode = "train",
    if_norm = if_norm,
    if_norm_temporal = if_norm_temporal,
    scaler = None,
    days = days,
    start_date = None,
    end_date = None,
    initial_amount = 1e3,
    transaction_cost_pct = 1e-3,
)

act_net = dict(
        type = "ActorTD3",
        embed_type = "TimesEmbed",
        input_dim = num_features,
        temporal_dim = temporal_dim,
        in_chans = 1,
        embed_dim = embed_dim,
        depth = depth,
        cls_embed = True,
        explore_noise_std = explore_noise_std
    )

cri_net = dict(
        type = "CriticTD3",
        embed_type="TimesEmbed",
        input_dim=num_features,
        temporal_dim=temporal_dim,
        in_chans = 1,
        embed_dim = embed_dim,
        depth = depth,
        cls_embed = True,
    )

criterion = dict(type='MSELoss', reduction="none")
optimizer = dict(type='AdamW',
                 params = None,
                 lr=lr)

agent = dict(
    type = "AgentTD3",
    act_net = act_net,
    cri_net = cri_net,
    criterion = criterion,
    optimizer = optimizer,
    if_use_per = if_use_per,
    num_envs = num_envs,
    transition_shape = transition_shape,
    max_step = 1e4,
    gamma = 0.99,
    reward_scale = 2**0,
    repeat_times = 1.0,
    batch_size = batch_size,
    clip_grad_norm = 3.0,
    soft_update_tau = 0,
    state_value_tau = 5e-3,
    explore_noise_std = explore_noise_std,
    policy_noise_std = policy_noise_std,
    device = None
)