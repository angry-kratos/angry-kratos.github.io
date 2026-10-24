"""Each subprocess owns a new CUDA context. OOM rows are partial observations."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', nargs='+', type=int, default=[16,18,20,21,22,23,24])
    ap.add_argument('--gpu', type=int, default=1, help='Physical nvidia-smi index')
    ap.add_argument('--out', default='results/batch_sweep')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    capacity = subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv'], text=True)
    (out/'capacity.txt').write_text(capacity)
    info = subprocess.check_output(['nvidia-smi','-i',str(args.gpu),'--query-gpu=uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'],text=True)
    uuid, name, total, free = [v.strip() for v in info.strip().split(',')]
    (out/'device.json').write_text(json.dumps(dict(gpu=args.gpu,uuid=uuid,name=name,capacity_mib=int(total),initial_free_mib=int(free)),indent=2))
    env = os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME'):
        env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES=uuid)
    fields = ['batch','status','step','peak_allocated_mib','peak_reserved_mib','peak_allocated_bytes','peak_reserved_bytes','returncode']
    with (out/'batch_sweep.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); f.flush()
        for batch in args.batches:
            run = subprocess.run([sys.executable,str(Path(__file__).with_name('train_memory.py')),'--batch',str(batch)],env=env,capture_output=True,text=True)
            (out/f'batch_{batch}.log').write_text(run.stdout+'\nSTDERR:\n'+run.stderr)
            records=[line.removeprefix('RESULT_JSON=') for line in run.stdout.splitlines() if line.startswith('RESULT_JSON=')]
            row=json.loads(records[-1]) if records else dict(batch=batch,status='ERROR',step='')
            if run.returncode:
                row['status']='ERROR'
            for kind in ('allocated','reserved'):
                val=row.get(f'peak_{kind}_bytes')
                row[f'peak_{kind}_mib']=val/2**20 if val is not None else ''
            row['returncode']=run.returncode
            writer.writerow(row); f.flush()
            print(json.dumps(row),flush=True)

if __name__ == '__main__':
    main()
