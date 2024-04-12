import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 64 #how many independent sequences will we process in parallel
block_size = 256 #how many characters will we look back at to predict the next character
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# reading the data
with open('input.txt',  'r', encoding='utf-8') as f:
  text = f.read()

#all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

#mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder - take a string an output list of integers
decode = lambda l: ''.join([itos[i] for i in l]) #decoder - take a list of integers and output text

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i : i + block_size] for i in ix])
  y = torch.stack([data[i+1 : i + block_size + 1] for i in ix])
  x, y = x.to(device) , y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out 


class Head(nn.Module):
  """single head of self attention"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias = False)
    self.query = nn.Linear(n_embed, head_size, bias =False)
    self.value = nn.Linear(n_embed, head_size, bias = False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.head_size = head_size
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) #B,T, head_size
    q = self.query(x) #B,T, head_size
    v = self.value(x)

    #compute attention / affinities
    wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5) #B,T,head_size @ B,head_size, T = > B, T, T
    # wei = wei.masked_fill(self.tril == 0, float('-inf')) #assuming block_size = T (which is the case)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #because during generation there could be arbitrary number of input tokens which can break as tril will be of max block size
    wei = F.softmax(wei, dim = -1) #B,T,T
    wei = self.dropout(wei)
    out = wei @ v # B,T, head_size

    return out
  

class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out

class FeedForwardMLP(nn.Module):
  """ a simple linear layer followed by a non linearity"""
  def __init__(self, n_embed):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed), #scaled up as per the transformers paper
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed), #projection layer
      nn.Dropout(dropout)
    )

  def forward(self,x):
    return self.net(x)


class Block(nn.Module):
  """Transformer block - communication followed by computation"""
  def __init__(self, n_embed, num_heads):
    super().__init__()
    head_size = n_embed // num_heads
    self.sa = MultiHeadAttention(num_heads, head_size)
    self.ffwd = FeedForwardMLP(n_embed)
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    #applying pre norm (modern way) and addition of gradient super highway for better initialisation
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):

  def __init__(self):
    super().__init__()
    #each token directly reads off the logits for the next token from a lookup table
    self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.pos_embedding_table = nn.Embedding(block_size, n_embed) # max position can be upto block size
    # self.sa_heads = MultiHeadAttention(4, n_embed // 4 ) 
    # self.ffwd = FeedForwardMLP(n_embed)
    # self.blocks = nn.Sequential(
    #   Block(n_embed, num_heads=4),
    #   Block(n_embed, num_heads=4),
    #   Block(n_embed, num_heads=4),
    # )
    self.blocks = nn.Sequential(*[Block(n_embed, num_heads=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed) #there should be a layernorm right before the lm head as well
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    # idx and targets are both (B,T) tensor of integers
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) # B, T, C/n_embed
    pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) #T,C
    x = tok_emb + pos_emb # (B, T, C) + (T, C) = (B, T, C)
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # B, T, C/Vocab size

    if targets is None:
      loss = None
    else :
      #loss = F.cross_entropy(logits, targets) # this won't work as pytorch expets in the form of B, C, T and we have logits in the form of B, T, C
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    #idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      #crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      #get the predictions
      logits ,loss = self(idx_cond)
      #focus only on the last time step
      logits = logits[:, -1, :] # becomes (B,C) # taking only the last character to predict the next one as we are making a bigram model
      #apply softmax to get probs
      probs = F.softmax(logits, dim=-1) #(B,C)
      #get the prediction from the sample
      idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
    return idx

model = BigramLanguageModel()
m = model.to(device)

#create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) #typical is 3e-4 - but this 1e-3 high rate would work for our small nn

for iter in range(max_iters):
    #every once in a while evaluate the loss of train and val data
    if iter % eval_interval == 0 :
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

	#sample a batch of data
    xb, yb = get_batch('train')

	#evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

#generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device) #sending in the newline character (a 1*1 input tensor containing zero (batch size = 1 and input/ T = 1 as well containing 0))
print(decode(m.generate(context, max_new_tokens=3000)[0].tolist())) 