from mmengine.registry import Registry

DATASET = Registry('dataset', locations=['pm.dataset'])
NET = Registry('net', locations=['pm.net'])
AGENT = Registry('agent', locations=['pm.agent'])
OPTIMIZER = Registry('optimizer', locations=['pm.optimizer'])
SCHEDULER = Registry('scheduler', locations=['pm.scheduler'])
CRITERION = Registry('criterion', locations=['pm.criterion'])
ENVIRONMENT = Registry('environment', locations=['pm.environment'])
EMBED = Registry('embed', locations=['pm.embed'])