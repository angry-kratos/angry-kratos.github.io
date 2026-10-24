"""Replay our own trusted snapshot to locate the peak missed by block hooks."""
import json
import pickle
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('results/step2')
meta = json.loads((out / 'step2_memory.json').read_text())
with (out / 'step2.pickle').open('rb') as f:
    snapshot = pickle.load(f)
events = next(t for t in snapshot['device_traces'] if t)
current = peak = meta['baseline_bytes']
curve = [current]
peak_event = None
for i, event in enumerate(events):
    # Trace sizes are requested bytes; this run uses default 512-byte rounding.
    rounded = (event['size'] + 511) // 512 * 512
    if event['action'] == 'alloc':
        current += rounded
    elif event['action'] == 'free_requested':
        current -= rounded
    curve.append(current)
    if current > peak:
        peak = current
        peak_event = {'event_index': i, **event}
assert peak == meta['peak_bytes']
assert current == meta['rows'][-1]['allocated']
(out / 'peak_event.json').write_text(json.dumps(peak_event, indent=2))
from figure_style import style, axes, save
with style():
    fig, ax = plt.subplots(figsize=(12,5.5)); axes(ax)
    ax.plot([v/2**20 for v in curve],color='#1baf7a',linewidth=1.4)
    ax.scatter([peak_event['event_index']+1],[peak/2**20],color='#b23a48',zorder=3)
    ax.annotate(f'log-softmax backward\\n{peak/2**20:,.1f} MiB peak'.replace('\\n','\n'),
        (peak_event['event_index']+1,peak/2**20),xytext=(25,-15),textcoords='offset points',fontsize=11)
    ax.set(xlabel='Allocator event index (not elapsed time)',ylabel='CUDA allocated memory (MiB)',
        title="Figure 2 · what the hooks didn't see")
    fig.tight_layout(); save(fig,out,'fig2'); save(fig,out,'step2_allocator_trace'); plt.close(fig)
print(f'Snapshot replay matches peak {peak/2**20:.1f} MiB and final {current/2**20:.1f} MiB')
