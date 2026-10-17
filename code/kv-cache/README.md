# KV cache experiments (post 1)

From-scratch GPT-2 small (`model.py`) and the script behind every number and figure in
[1000× Less Math, 6× Less Time](https://angry-kratos.github.io/posts/2026-10-kv-cache/).

```bash
pip install torch transformers matplotlib numpy
python kv_experiments.py               # all experiments (CUDA GPU), then figures
python kv_experiments.py --memory-only # re-run only the memory experiment, keep timings
python kv_experiments.py --plots-only  # redraw figures from results/results.json
```

Hand-drawn figures: put `xkcd-script.ttf` (github.com/ipython/xkcd-font) in `fonts/` next to the script.
Measured on an NVIDIA A40, fp32 with TF32 off, batch 1, median of 5 runs per position.
