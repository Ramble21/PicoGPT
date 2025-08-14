# Imports
import random
import requests
from torchgen.native_function_generation import self_to_out_signature

from transformer import Transformer

# Hyperparameters
from hyperparams import *


# Seed for consistent random numbers
SEED = 8675309
random.seed(SEED)
torch.manual_seed(SEED)

# Open dataset
response = requests.get(DATASET_LINK)
response.raise_for_status()
with open('input.txt', 'w', encoding='utf-8') as f:
    f.write(response.text)
with open('input.txt', 'r', encoding='utf-8') as f:
  data_raw = f.read()
print(f"Dataset size: {len(data_raw):,} chars")

# Character-level tokenization
chars = sorted(list(set(data_raw)))
vocab_size = len(chars)

s_to_i = {ch:i for i,ch in enumerate(chars)}
i_to_s = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [s_to_i[c] for c in s] # use lambda functions to make encoding/decoding more clear
decode = lambda l: ''.join([i_to_s[i] for i in l])

data = torch.tensor(encode(data_raw), dtype=torch.long)

# Training vs validation/dev splits (skipping over test which would be used in practice)
n = int(0.9*len(data))
data_tr =  data[:n]
data_dev = data[n:]

# Helper functions to generalize and clean up the code
def sample(m, num_chars, initial_context='\n'):
  encoded = encode(initial_context)
  initial_context = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
  generated = m.generate(initial_context, num_chars)
  print(decode(generated[0].tolist()))

def get_batch(split):
  data_x = data_tr if split == 'train' else data_dev
  index = torch.randint(len(data_x) - CONTEXT_LEN, (BATCH_SIZE, ))
  x = torch.stack([data[i:i+CONTEXT_LEN] for i in index])
  y = torch.stack([data[i+1:i+CONTEXT_LEN+1] for i in index])
  x, y = x.to(DEVICE), y.to(DEVICE)
  return x, y

def train(m):
  # Pytorch optimizer object, used for training models in production
  # torch.optim.SGD represents the classic stochastic gradient descent optimization, but AdamW is more advanced and popular
  optimizer = torch.optim.AdamW(m.parameters(), lr=LR)
  # Train our model
  loss_log = []
  for i in range(NUM_STEPS):
    X_b, Y_b = get_batch('train')
    # Forward pass
    logits, loss = m(X_b, Y_b)
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # Log
    loss_log.append(loss.item())
    if i % LOSS_BENCH == 0:
      print(f"{i} / {NUM_STEPS}: approx. loss={loss.item():.4f}")
  print(f"Final approx. loss: {loss.item():.4f}")
  return loss_log

# ---------------- Transformer ---------------------------------

m = Transformer(vocab_size)
m.to(DEVICE)
log = train(m)
print("Training finished!")

NUM_CHARS = 2000
sample(m, NUM_CHARS)