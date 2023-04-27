import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)


# hyperparams
batch_size = 32
block_size = 8
training_steps = 30000
n_embed = 32
num_heads = 4
learning_rate = 1e-3
num_blocks = 4

train_split_ratio = 0.8
eval_interval = 500
eval_samples = 300
device = 'GPU' if torch.cuda.is_available() else 'cpu'
#-----



text_file = 'text-sample.txt'
with open(text_file, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda int_list: ''.join([itos[i] for i in int_list])

data = torch.tensor(encode(text), dtype=torch.long, device=device)

train_split_index = int(train_split_ratio*len(data))
train_data = data[:train_split_index]
val_data = data[train_split_index:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, device=device)))
    
    def forward(self, x):
        B,T,C  = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)

        tril = torch.tril(torch.ones(T, T, device=device))
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        
        # TODO after implementing the whole self-attention, try to use x instead of v and compare the results
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embed)

    def forward(self, x):
        x = self.sa(x) + x
        x = self.ffwd(x) + x
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_blocks = nn.Sequential(
            Block(n_embed, num_heads),
            Block(n_embed, num_heads),
            Block(n_embed, num_heads),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_input = idx[:, -block_size:]
            logits, loss = self(idx_input)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel()
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
logits, loss = m.forward(xb, yb)


def decode_tensor(x_encoded):
    b, t = x_encoded.shape
    decoded_strings = []
    for i in range(b):
        int_list = x_encoded[i, :].tolist()
        decoded_string = decode(int_list)
        decoded_strings.append(decoded_string)
    return decoded_strings

def estimate_loss():
    results = {}
    for split in ['train', 'val']:
        batch_losses = []
        for i in range(eval_samples):
            x, y = get_batch(split)
            _, loss = m(x, y)
            batch_losses.append(loss)
        results[split] = torch.mean(torch.tensor(batch_losses, device=device))
    return results

# training
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for steps in range(training_steps):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"steps {steps}/{training_steps} Training loss: {losses['train']}, val loss: {losses['val']}")


# generation
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode_tensor(m.generate(idx, 100))

print('Generated text:')
print(generated_text)
