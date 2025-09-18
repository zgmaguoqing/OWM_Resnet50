from yacs.config import CfgNode as CfgNode

_C = CfgNode()
cfg = _C

# ==================== official variable ====================
_C.LOGGER_PATH = 'outputs'
_C.GPUID = [1]

# ----- Scenario of continual learning ------
# 10split
_C.DOMAIN_INCR = False

# ------------ Agent ------------------------
_C.AGENT = CfgNode()

# # OWM
_C.AGENT.TYPE = 'trainer_OWM'
_C.AGENT.NAME = 'OWMTrainer'

_C.AGENT.MODEL_TYPE = 'resnet'
_C.AGENT.MODEL_NAME = 'resnet50'

_C.AGENT.REG_COEF = 100
_C.AGENT.FIX_BN = False
_C.AGENT.FIX_HEAD = True # only for domain incremental

_C.AGENT.ALPHA_ZERO = 1e-5
_C.AGENT.ALPHA_ONE = 1e-5
_C.AGENT.ALPHA_TWO = 1e-4
_C.AGENT.ALPHA_THREE = 1e-1
_C.AGENT.STRIDE = 2
# ---------- Dataset --------------------------
_C.DATASET = CfgNode()
# 10split
_C.DATASET.ROOT = '/data0/user/lxguo/Code/data/10splitTasks'
_C.DATASET.NAME = '10splitTasks'
_C.DATASET.NUM_CLASSES = 100
_C.DATASET.NUM_TASKS = 10

# 4split
# _C.DATASET.ROOT = '/data0/user/lxguo/Code/data/4splitDomains'
# _C.DATASET.NAME = '4splitDomains'
# _C.DATASET.NUM_CLASSES = 60
# _C.DATASET.NUM_TASKS = 4



# ==================== customer variable ====================
_C.SEED = 0
_C.PRINT_FREQ = 100

# ---------- Dataset --------------------------
_C.DATASET.BATCHSIZE = 64
_C.DATASET.NUM_WORKERS = 4

# --------- Optimizer --------------------------
# For OWM
_C.OPT = CfgNode()
_C.OPT.NAME = 'SGD'
_C.OPT.LR = 0.05
_C.OPT.MOMENTUM = 0.0
_C.OPT.WEIGHT_DECAY = 0.0
_C.OPT.SCHEDULE = [20, 2]
_C.OPT.GAMMA = 0.0


# For NSCL test
_C.OPT = CfgNode()
_C.OPT.NAME = 'Adam'
_C.OPT.LR = 2.5e-4
_C.OPT.MOMENTUM = 0.0
_C.OPT.WEIGHT_DECAY = 0.0
_C.OPT.SCHEDULE = [1]
_C.OPT.GAMMA = 0.5
_C.OPT.MODEL_LR = _C.OPT.LR
_C.OPT.SVD_LR = _C.OPT.LR
_C.OPT.HEAD_LR = _C.OPT.LR
_C.OPT.BN_LR = _C.OPT.LR
_C.OPT.SVD_THRES = 10

def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    return config