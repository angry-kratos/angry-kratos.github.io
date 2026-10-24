"""Post 1 palette and sketch settings, with optional local handwriting font."""
from contextlib import contextmanager
from pathlib import Path
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
PAPER='#faf6ee'; INK='#2b2a27'; GRID='#e6dfd1'
@contextmanager
def style():
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    font=None
    for folder in (Path(__file__).parent/'fonts',Path(__file__).parent.parent/'kv-cache'/'fonts'):
        for path in sorted(folder.glob('*.ttf')):
            font_manager.fontManager.addfont(str(path))
            font=font_manager.FontProperties(fname=str(path)).get_name()
            break
        if font: break
    with plt.xkcd(scale=.7,length=160,randomness=1.6):
        plt.rcParams.update({'font.family':[font,'DejaVu Sans'] if font else ['DejaVu Sans'],
            'font.size':12,'path.effects':[patheffects.withStroke(linewidth=3,foreground=PAPER)],
            'figure.facecolor':PAPER,'axes.facecolor':PAPER,'savefig.facecolor':PAPER,
            'axes.edgecolor':INK,'axes.labelcolor':INK,'text.color':INK,
            'xtick.color':INK,'ytick.color':INK,'axes.linewidth':1.4})
        yield

def axes(ax):
    for side in ('top','right'): ax.spines[side].set_visible(False)
    ax.grid(axis='y',color=GRID,linewidth=1)
    ax.set_axisbelow(True)

def save(fig,out,name):
    out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):
        fig.savefig(out/f'{name}.{ext}',dpi=200,bbox_inches='tight')
