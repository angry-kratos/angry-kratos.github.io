"""One fresh-process, two-step memory measurement. No OOM retries."""
import argparse
import json
import torch
from model import GPT, GPTconfig
MIB = 2**20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    if args.batch < 1:
        ap.error('--batch must be positive')
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result = dict(batch=args.batch, status='OK', step=0)
    try:
        a = torch.rand((1024,1024), device='cuda')
        b = torch.rand_like(a)
        c = a @ b
        del a, b, c
        model = GPT(GPTconfig()).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        batch = torch.randint(0, GPTconfig.vocab_size, (args.batch,1025), device='cuda')
        x, y = batch[:,:-1], batch[:,1:]
        for step in (1,2):
            result['step'] = step
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            logits, loss = model(x,y)
            loss.backward()
            del logits
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            del loss
            torch.cuda.synchronize()
        # Success reports step 2, not a maximum mixed across both steps.
    except torch.cuda.OutOfMemoryError:
        result['status'] = 'OOM'
        # Exit this process after reporting; never retry or reuse its allocator.
    result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    print('batch,status,peak_allocated_MiB,peak_reserved_MiB', flush=True)
    print(f"{args.batch},{result['status']},{result['peak_allocated_bytes']/MIB:.3f},{result['peak_reserved_bytes']/MIB:.3f}", flush=True)
    print('RESULT_JSON=' + json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
