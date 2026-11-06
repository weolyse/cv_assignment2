"""Compare exported masks to human annotations by matching PNG filenames."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--references',type=Path,required=True);args=p.parse_args()
rows=[]
for reference in sorted(args.references.glob('*.png')):
    prediction=args.predictions/reference.name
    if not prediction.exists(): raise FileNotFoundError(prediction)
    a=np.asarray(Image.open(reference).convert('L'))>127;b=np.asarray(Image.open(prediction).convert('L'))>127
    if a.shape!=b.shape: raise ValueError(f'Shape mismatch: {reference.name}')
    intersection=(a&b).sum();union=(a|b).sum();total=a.sum()+b.sum()
    rows.append({'image':reference.name,'iou':float(intersection/union) if union else 1.,'dice':float(2*intersection/total) if total else 1.})
if not rows: raise ValueError('No reference PNG masks found')
print(json.dumps({'images':rows,'mean_iou':float(np.mean([r['iou'] for r in rows])),'mean_dice':float(np.mean([r['dice'] for r in rows]))},indent=2))
