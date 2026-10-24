"""Plot completed measurements only; OOM peaks are incomplete, not data points."""
import csv,json
from pathlib import Path
import numpy as np
from figure_style import style, axes, save, plt
out=Path('results/batch_sweep')
rows=list(csv.DictReader((out/'batch_sweep.csv').open()))
device=json.loads((out/'device.json').read_text())
ok=[r for r in rows if r['status']=='OK']
x=np.array([int(r['batch']) for r in ok]); a=np.array([float(r['peak_allocated_mib']) for r in ok]); r=np.array([float(r['peak_reserved_mib']) for r in ok])
slope,intercept=np.polyfit(x,a,1) if len(x)>1 else (float('nan'),float('nan'))
(out/'fit.json').write_text(json.dumps(dict(allocated_slope_mib_per_sequence=slope,intercept_mib=intercept,expected_slope=1945,capacity_mib=device['capacity_mib']),indent=2))
with style():
    fig,ax=plt.subplots(figsize=(11,6)); axes(ax)
    ax.plot(x,a,'o-',color='#2a78d6',label='Peak allocated · completed step 2')
    ax.plot(x,r,'s-',color='#eb6834',label='Peak reserved · completed step 2')
    ax.fill_between(x,a,r,color='#eb6834',alpha=.12,path_effects=[])
    cap=device['capacity_mib']; ax.axhline(cap,color='#7a776f',ls='--',label=f'Card capacity: {cap:,} MiB')
    for row in rows:
        if row['status']=='OOM':
            b=int(row['batch']); ax.plot(b,cap,'x',color='#b23a48',ms=10); ax.annotate('OOM',(b,cap),xytext=(0,12),textcoords='offset points',ha='center')
    ax.set(xticks=[int(row['batch']) for row in rows],xlabel='Batch size (1024 tokens per sequence)',ylabel='Peak memory (MiB)',
        title=f'Figure 3 · Peak memory vs batch size\nFresh process per batch · allocated slope: {slope:,.1f} MiB / sequence',ylim=(0,cap*1.12))
    ax.legend(loc='lower right',fontsize=10,frameon=False)
    fig.tight_layout(); save(fig,out,'fig3'); plt.close(fig)
print(json.dumps(dict(slope_mib_per_sequence=slope,capacity_mib=cap)))
