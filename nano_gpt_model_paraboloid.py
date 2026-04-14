import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import geondpt as gpt
except ImportError:
    import geondptfree as gpt

# model architecture
class AttentionHead(nn.Module):
  """a single head of self attention"""
  
  def __init__(self, n_embed, head_size, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    B, T, C = x.shape
    K = self.key(x) # (B, T, C)
    Q = self.query(x) # (B, T, C)
    
    wei = Q @ K.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, H, C) -> (B, T, T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)    
    V = self.value(x) # (B, T, C)
    
    out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)
    return out
  
class MultiHeadAttention(nn.Module):
  """a multi-head self attention layer"""
  
  def __init__(self, n_embed, n_heads, head_size, block_size, dropout):
    super().__init__()
    self.heads = nn.ModuleList([AttentionHead(n_embed, head_size, block_size, dropout) for _ in range(n_heads)])
    self.fc = nn.Linear(head_size * n_heads, n_embed)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads*C)
    
    
    
    #out = self.fc(out) # (B, T, C)
    out = self.dropout(out) 
    return out
  
class FeedForward(nn.Module):
  def __init__(self, n_embed, n_hidden, dropout):
    super().__init__()
#    self.net = nn.Sequential(
#      nn.Linear(n_embed, n_hidden),
#      nn.ReLU(),
#      nn.Linear(n_hidden, n_embed),
#      nn.Dropout(dropout)
#    )
    self.pb = gpt.Paraboloid(n_embed, n_hidden, lr_factor = 10., input_factor = 0.1, wd_factor = 0.01)
    self.rl = nn.ReLU()
    self.ln = nn.Linear(n_hidden, n_embed)
    self.dr = nn.Dropout(dropout)
    
  def forward(self, x):
   
   x_shape = x.shape
   x_reshaped = x.reshape(-1, x_shape[2])
   out = self.pb(x_reshaped)
   out = out.reshape(x_shape[0],x_shape[1],self.pb.output_features)
   out = self.rl(out)
   out = self.ln(out)
   out = self.dr(out)

   
   #return self.net(x)
   return out
  
class Block(nn.Module):
  def __init__(self, n_embed, n_heads, block_size, dropout):
    super().__init__()
    self.sa_heads = MultiHeadAttention(n_embed, n_heads, n_embed // n_heads, block_size, dropout)
    #self.ffwd = FeedForward(n_embed, n_embed*4, dropout)
    self.ffwd = FeedForward(n_embed, n_embed, dropout)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)
    
    
  def forward(self, x):
    x = x + self.sa_heads(self.ln1(x)) #  [batch_size, block_size, n_embed]
    x = x + self.ffwd(self.ln2(x)) # [batch_size, block_size, n_embed]
    return x

class NanoGPT_paraboloid(nn.Module):
  def __init__(self, hyperparameters, device="cpu"):
    super().__init__()
    
    # hyperparameters
    vocab_size = hyperparameters['vocab_size']
    block_size = hyperparameters['block_size']
    n_embed = hyperparameters['n_embed']
    n_heads = hyperparameters['n_heads']
    n_layers = hyperparameters['n_layers']
    dropout = hyperparameters['dropout']
    
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.position_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_heads, block_size, dropout) for _ in range(n_layers)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)
    #self.lm_head = gpt.ParaboloidOutput(n_embed, vocab_size, input_factor = 0.5, lr_factor = 2.) # 0.46147534251213074
    #self.lm_head = gpt.ParaboloidOutput(n_embed, vocab_size, input_factor = 0.5, lr_factor = 2., grad_factor = 0.5)
    #self.lm_head = gpt.ParaboloidOutput(n_embed, vocab_size, input_factor = 0.5, lr_factor = 1., wd_factor = 1.)
    
    self.device = device
    self.block_size = block_size
      
  def forward(self, idx, targets=None):
    # idx and target are both [batch_size, block_size]
    B, T = idx.shape
    
    tok_emb = self.token_embedding_table(idx) # [batch_size, block_size, n_embed]
    pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # [block_size, n_embed]
    x = tok_emb + pos_emb # [batch_size, block_size, n_embed]
    x = self.blocks(x)
    x = self.ln_f(x)
    
    logits = self.lm_head(x) # [batch_size, block_size, vocab_size]
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    
    return logits, loss
    # return 0, 0
      
  def generate(self, idx, max_new_tokens=100):
    # idx is (B, T)
    for _ in range(max_new_tokens):
      # get the last block_size tokens
      idx_cond = idx[:, -self.block_size:] # (B, T)
      # get the predictions
      logits, _ = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=1) # (B, C)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
      
    return idx
