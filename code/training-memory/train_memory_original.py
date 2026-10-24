import torch
from model import GPT, GPTconfig

torch.manual_seed(0)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

B,T = 8, 1024

def mib():
    return torch.cuda.memory_allocated()/2**20

tensor_a = torch.rand((1024,1024), device = "cuda")
tensor_b = torch.rand((1024, 1024), device = "cuda")

res = torch.matmul(tensor_a, tensor_b)

del tensor_a
del tensor_b
del res
print(f"[R0] after warmup: {mib():8.1f} MiB")
model = GPT(GPTconfig()).cuda()

model.train()
print(f"[R1] after model.cuda(): {mib():8.1f} MiB")
para = 0
for p in model.parameters():
    para += p.numel()
print("Parameter count",  para)
print(model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr())

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
print(f"[R2] after optimizer: {mib():8.1f} MiB")
buffer = torch.randint(0, GPTconfig.vocab_size, (B, T+1), device='cuda')
x, y = buffer[:,:-1], buffer[:,1:]
print(x.shape, y.shape)


for step in (1, 2):
    optimizer.zero_grad(set_to_none = True)
    torch.cuda.reset_peak_memory_stats()
    logits, loss = model(x, y)
    print(f"[s{step}] loss: {loss.item():.4f}")
    print(f"[s{step}] after forward:   {mib():8.1f} MiB")
    loss.backward()
    print(f"[s{step}] after backward:   {mib():8.1f} MiB")
    del logits
    print(f"[s{step}] after deleting logits:   {mib():8.1f} MiB")
    optimizer.step()
    print(f"[s{step}] after optimizer step:   {mib():8.1f} MiB")
    optimizer.zero_grad(set_to_none=True)
    print(f"[s{step}] after zero_grad:   {mib():8.1f} MiB")
    print(f"[s{step}] peak: {torch.cuda.max_memory_allocated()/2**20:8.1f} MiB")
    