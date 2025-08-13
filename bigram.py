import torch.nn as nn
from torch.nn import functional as F
from hyperparams import *

# Simple bigram model to show how nn.Module can be incorporated
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, N_EMBD)
        self.position_emb_table = nn.Embedding(CONTEXT_LEN, N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, batch, targets=None):

        B, T = batch.shape
        tok_emb = self.token_emb_table(batch) # (B,T,C)
        pos_emb = self.position_emb_table(torch.arange(T, device=DEVICE)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x) # (B,T,C)

        # If no targets are given, avoid an error by nulling loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # change shape of logits to match Pytorch specifications
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, batch, max_new_tokens):
        # batch is (B,T) array of indices in the current context
        for i in range(max_new_tokens):
            logits, loss = self(batch)
            logits = logits[:, -1, :]  # (B*T, C) -> (B, C)
            probs = F.softmax(logits, dim=-1)
            next_batch = torch.multinomial(probs, num_samples=1)  # (B, 1)
            batch = torch.cat((batch, next_batch), dim=1)  # (B, T+1)
        return batch