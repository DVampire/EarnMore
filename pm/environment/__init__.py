from .pm_based_portfolio_value import EnvironmentPV
from .pm_based_portfolio_return import EnvironmentRET
from gym.envs.registration import register

register(id = "PortfolioManagement-v0", entry_point = "pm.environment.wrapper:EnvironmentWrapper")