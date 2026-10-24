# Training memory experiments

Run these commands from `code/training-memory/` in this repository.
Run date: 2026-10-24. PyTorch 2.5.1, CUDA 12.1, NVIDIA A40 (physical GPU 1).
Model: GPT-2 small, fp32, TF32 disabled, 1024 tokens, AdamW.
`model.py` is unchanged. The original worker is saved as `train_memory_original.py`.

## Reproduce

From the remote folder, using the project environment:

```bash
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 python train_memory.py --batch 20
env -u PYTHONPATH -u PYTHONHOME PYTHONNOUSERSITE=1 python batch_sweep.py --gpu 1
```

The driver queries nvidia-smi before running, selects the GPU by UUID, launches
one subprocess per batch, saves stdout/stderr logs, and flushes each CSV row.
There are no retries after OOM. Success reports step 2 peaks. An OOM row reports
partial peaks in the failed step, identified by `step`; these are not completed
step measurements and are excluded from the plotted curves and fitted slope.

Capacity: 46,068 MiB as reported by nvidia-smi. B16, B18, B20 pass; B21/B22 OOM
in step 2; B23/B24 OOM in step 1. Allocated slope over completed runs:
1,941.09 MiB per sequence (not forced to the expected 1,945).
The capacity line is physical capacity, not all usable allocator space.
Allocated and reserved peaks are independent high-water marks, so their gap
is not an instantaneous measurement of fragmentation.

## Figures

1. `results/step2/fig1.png` and `.svg`: main stacked chart, grouped phase labels,
   annotated high-water jump, purple vocab payload separated from green residual.
2. `results/step2/fig2.png` and `.svg`: what the hooks didn't see; allocation trace.
3. `results/batch_sweep/fig3.png` and `.svg`: allocated/reserved vs batch size.

The styling matches post 1's warm paper palette and xkcd sketch settings.
Optional handwriting fonts are loaded from `fonts/` or `../kv-cache/fonts/`;
otherwise it uses the same DejaVu Sans fallback as post 1.

Render without rerunning GPU work:

```bash
python plot_step2.py
python inspect_snapshot.py
python plot_batch_sweep.py
```

## Step-2 measurement and interpretation

`profile_step2.py` records 55 scalar checkpoints using Block forward and full
backward hooks, plus pre-hooks and explicit loss/delete/step/zero-grad points.
Run it with the same clean environment and CUDA_VISIBLE_DEVICES=1 to regenerate.
It records history for step 2 only, then disables recording in a finally block.
The trusted locally generated snapshot is `results/step2/step2.pickle`.
Open https://pytorch.org/memory_viz and select it. Do not unpickle untrusted files.

Fixed buckets deduplicate tied weight storage and use live parameter gradient
and optimizer storage sizes. The residual includes activations, logits,
temporaries and allocator slack, not solely saved layer activations.
The vocab split is a ledger calculation from exact B*T*V*4 payloads and known
lifetimes: logits from lm_head until del logits; saved log-softmax after loss
until loss backward (before the first Block backward hook). It is not an
additional instrumented allocation measurement. The row decomposition is in
`vocab_ledger.json`. Each vocab tensor is 1,570.53125 MiB; the snapshot contains
four simultaneous allocations of that size at the peak, versus two after loss.

The true peak is 17,053.9 MiB in log-softmax backward; highest sampled checkpoint
is 13,912.9 MiB. Snapshot replay rounds request sizes to 512-byte allocator units
for this run and asserts exact agreement with peak and final allocated totals.
`peak_event.json` holds the peak stack. The final snapshot audit finds two
8,519,680-byte persistent allocations outside the tensor ledger, consistent
with library workspaces; their stack traces predate history recording.

The figures and results accompany the training-memory blog post.
