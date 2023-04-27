import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(42)


# hyperparams
batch_size = 32
block_size = 8
training_steps = 10000

train_split_ratio = 0.8
eval_interval = 500
eval_samples = 300
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

data = torch.tensor(encode(text), dtype=torch.long)

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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
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
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
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
        results[split] = torch.mean(torch.tensor(batch_losses))
    return results

# training
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

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
idx = torch.zeros((1, 1), dtype=torch.long)
generated_text = decode_tensor(m.generate(idx, 300))

print('Generated text:')
print(generated_text)
