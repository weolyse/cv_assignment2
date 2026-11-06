"""Evaluate the same held-out images with and without real MedSAM2 masking."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image,ImageOps
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from app.models import make_model
from app.imaging import preprocessing
from app.segmentation import MedSAM2Segmenter
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
transform=preprocessing(); segmenter=MedSAM2Segmenter('cpu')
split=json.loads((ROOT/'checkpoints/split.json').read_text())
rows=[]; failures=[]
for path,label in split['test']:
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB');im.thumbnail((1024,1024))
    start=time.perf_counter()
    try: mask,score,box=segmenter.segment(im)
    except ValueError as exc:
        failures.append({'image':path,'reason':str(exc)});print(f'Segmentation failed: {Path(path).name}',flush=True);continue
    roi=np.asarray(im).copy();roi[~mask]=[124,116,104]
    rows.append((transform(im),transform(Image.fromarray(roi)),label,time.perf_counter()-start,float(mask.mean())))
    print(f'Masked {len(rows)}/{len(split["test"])}',flush=True)
if not rows: raise RuntimeError("No successful segmentations; comparison unavailable")
report={}
for name in ['simple_cnn','resnet18']:
    model=make_model(name);model.load_state_dict(torch.load(ROOT/f'checkpoints/{name}.pt',weights_only=True,map_location='cpu')['state_dict']);model.eval()
    truth=[r[2] for r in rows]; report[name]={}
    for index,mode in enumerate(['full_image','medsam2_roi']):
        predictions=[];start=time.perf_counter()
        with torch.no_grad():
            for r in rows: predictions.append(int(model(r[index].unsqueeze(0)).argmax(1)))
        report[name][mode]={'accuracy':accuracy_score(truth,predictions),'macro_f1':f1_score(truth,predictions,average='macro'),'confusion_matrix':confusion_matrix(truth,predictions,labels=[0,1]).tolist(),'mean_classifier_ms':1000*(time.perf_counter()-start)/len(rows)}
    report[name]['delta_macro_f1']=report[name]['medsam2_roi']['macro_f1']-report[name]['full_image']['macro_f1']
report['metadata']={'count':len(rows),'total_test_count':len(split['test']),'failures':failures,'comparison_scope':'Both modes evaluated on the same successfully segmented subset; failures excluded and explicitly counted.','mean_segmentation_ms':1000*np.mean([r[3] for r in rows]),'prompt':'2% inset full-frame box','device':'cpu','mask_iou':None,'reason':'No human reference masks provided; segmentation scores are model estimates, not measured IoU.'}
(ROOT/'artifacts/roi_evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
