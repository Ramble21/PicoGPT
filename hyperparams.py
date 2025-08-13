import torch

# Hyperparameters
CONTEXT_LEN = 8 # maximum context length for predictions
BATCH_SIZE = 32 # number of sequences being processed by the transformer in parallel
N_EMBD = 32 # number of embedding dimensions
LR = 1e-3 # learning rate parameter for optimization

# Other various magic numbers
NUM_STEPS = 20000 # number of steps used in model training
LOSS_BENCH = 5000 # number of steps inbetween loss benchmarking prints during optimization

# Misc
DATASET_LINK = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # device used for computations
