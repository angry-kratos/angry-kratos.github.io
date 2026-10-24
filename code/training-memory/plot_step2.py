"""Figure 1. Vocab payload uses exact shapes and known tensor lifetimes.
This is a ledger decomposition, not a claim that all residual bytes are activations.
"""
import json
from pathlib import Path
import numpy as np
from figure_style import style, axes, save, plt, INK
MIB=2**20

def vocab_payload(label, batch, tokens, vocab):
    # logits persist from lm_head through backward until explicit del logits.
    # Saved log-softmax output persists after loss until loss backward, before B11.
    n=2 if label=='after loss' else 1 if (label=='logits / before loss' or label.startswith('B') or label=='after backward') else 0
    return n*batch*tokens*vocab*4

def plot(meta,out):
    rows=meta['rows']; ix=np.arange(len(rows))
    vocab=np.array([vocab_payload(r['label'],meta['batch'],meta['tokens'],50257) for r in rows])
    rest=np.array([r['residual'] for r in rows])-vocab
    assert (rest>=-4096).all()
    (out/'vocab_ledger.json').write_text(json.dumps({'method':'Exact fp32 B*T*V payload and model tensor lifetimes; remainder includes allocator slack and other temporaries.',
        'bytes_per_vocab_tensor':meta['batch']*meta['tokens']*50257*4,
        'rows':[{'label':r['label'],'vocab_bytes':int(v),'layer_and_other_bytes':int(a)} for r,v,a in zip(rows,vocab,rest)]},indent=2))
    with style():
        fig,ax=plt.subplots(figsize=(13,7)); axes(ax)
        keys=['weights','adam','buffers_inputs','persistent_other','grads']
        data=[np.array([r[k] for r in rows])/MIB for k in keys]+[rest/MIB,vocab/MIB]
        labels=['Weights','Adam state','Buffers + inputs','Persistent baseline remainder','Parameter gradients','Layer activations + other temporaries/slack','Vocab tensors (logits + saved log-softmax)']
        ax.stackplot(ix,*data,labels=labels,colors=['#486b9c','#80a8ca','#c7b58a','#998d7b','#dd9761','#1baf7a','#b783cf'],step='post',path_effects=[])
        ax.plot(ix,[r['allocated']/MIB for r in rows],color=INK,lw=1,drawstyle='steps-post')
        ax.plot(ix,[r['peak_so_far']/MIB for r in rows],'--',color='#b23a48',lw=1.8,drawstyle='steps-post',label='Allocator high-water mark')
        loss=next(i for i,r in enumerate(rows) if r['label']=='after loss')
        ax.annotate('log-softmax backward: +2 × 1,570 MiB,\ninvisible to hooks.',xy=(loss+1,meta['peak_bytes']/MIB),xytext=(3,18500),fontsize=12,
            arrowprops=dict(arrowstyle='->',color=INK,connectionstyle='arc3,rad=-.12'))
        ax.set_xticks([12.5,26,38.5,52.5],['forward · 12 blocks','loss','backward · 12 blocks','step'])
        for x in (24.5,26.5,50.5): ax.axvline(x,color='#7a776f',alpha=.35,lw=.8)
        ax.set(xlim=(-1,55),ylim=(0,20300),ylabel='CUDA allocated memory (MiB)',xlabel='Step 2 phases (event order, not elapsed time)',title='Figure 1 · Where training memory goes\nGPT-2 small · A40 · batch 8 × 1024 · fp32 · AdamW')
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.17),ncol=2,fontsize=10,frameon=False)
        fig.subplots_adjust(bottom=.25)
        save(fig,out,'fig1'); save(fig,out,'step2_memory'); plt.close(fig)

if __name__=='__main__':
    import sys
    out=Path(sys.argv[1] if len(sys.argv)>1 else 'results/step2')
    plot(json.loads((out/'step2_memory.json').read_text()),out)
