import json, logging, os, threading, time
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from app.models import make_model
from app.imaging import decode, png, gradcam, heat_overlay
from app.segmentation import MedSAM2Segmenter, CHECKPOINT

ROOT=Path(__file__).resolve().parents[1]
DEVICE=os.getenv('PLANT_DEVICE','cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
torch.set_num_threads(4)
app=FastAPI(title='LeafLens',description='Plant classification with Grad-CAM and MedSAM2')
app.mount('/static',StaticFiles(directory=ROOT/'app/static'),name='static')
models={}; lock=threading.Lock(); segmenter=MedSAM2Segmenter(DEVICE)

@app.get('/')
def index(): return FileResponse(ROOT/'app/static/index.html')

@app.get('/api/status')
def status():
    return {'device':DEVICE,'models':{n:(ROOT/f'checkpoints/{n}.pt').is_file() for n in ['simple_cnn','resnet18']},'medsam2':CHECKPOINT.is_file(),'classes':['Diseased','Healthy']}

@app.get('/api/roi-metrics')
def roi_metrics():
    path=ROOT/'artifacts/roi_evaluation.json'
    return json.loads(path.read_text()) if path.is_file() else {}

@app.get('/api/metrics')
def metrics():
    return {n:json.loads(p.read_text()) for n in ['simple_cnn','resnet18'] if (p:=ROOT/f'checkpoints/{n}_metrics.json').is_file()}

def load_model(name):
    if name not in models:
        path=ROOT/f'checkpoints/{name}.pt'
        if not path.is_file(): raise HTTPException(503,'Classifier not trained. Run scripts/train.py first.')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['classes']!=['Diseased','Healthy']: raise RuntimeError('Checkpoint class mapping mismatch')
        model=make_model(name); model.load_state_dict(checkpoint['state_dict']); model.to(DEVICE).eval(); models[name]=model
    return models[name]

@app.post('/api/analyze')
def analyze(file: UploadFile=File(...),model: str=Form('simple_cnn'),segment: bool=Form(True),box: Optional[str]=Form(None)):
    if model not in ('simple_cnn','resnet18'): raise HTTPException(422,'Unknown classifier.')
    data=file.file.read(10*1024*1024+1)
    if len(data)>10*1024*1024: raise HTTPException(413,'Image must be smaller than 10 MB.')
    try: image=decode(data)
    except (ValueError,UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc: raise HTTPException(400,str(exc))
    coordinates=None
    if box:
        try:
            coordinates=json.loads(box)
            if len(coordinates)!=4 or not all(type(v) in (int,float) and np.isfinite(v) and 0<=v<=1 for v in coordinates): raise ValueError()
            if coordinates[0]>=coordinates[2] or coordinates[1]>=coordinates[3]: raise ValueError()
        except (ValueError,TypeError): raise HTTPException(422,'Box must contain normalized x1,y1,x2,y2 coordinates.')
    with lock:
        try:
            classifier=load_model(model)
            started=time.perf_counter(); mask=None; quality=None
            if segment:
                mask,quality,coordinates=segmenter.segment(image,coordinates)
                roi=np.asarray(image).copy(); roi[~mask]=np.array([124,116,104],dtype=np.uint8)
                classifier_image=Image.fromarray(roi)
            else: classifier_image=image
            segmented=time.perf_counter()
            probs,heat,active=gradcam(classifier,classifier_image,DEVICE)
            finished=time.perf_counter()
        except HTTPException: raise
        except ValueError as exc: raise HTTPException(422,str(exc))
        except Exception:
            logging.exception('Inference failed')
            raise HTTPException(503,'Model inference failed. Check the server log and MedSAM2 installation; no substitute prediction was generated.')
    prediction=int(np.argmax(probs)); images={'original':png(image),'input':png(classifier_image),'gradcam':png(heat_overlay(image,heat))}
    if mask is not None:
        overlay=np.asarray(image).copy().astype(float); overlay[mask]=overlay[mask]*.55+np.array([62,210,145])*.45
        images['mask_overlay']=png(Image.fromarray(overlay.astype('uint8')))
        images['mask']=png(Image.fromarray(mask.astype('uint8')*255))
    return {'label':['Diseased','Healthy'][prediction],'confidence':probs[prediction],'probabilities':dict(zip(['Diseased','Healthy'],probs)),'model':model,'segmentation':'MedSAM2' if segment else 'Disabled','mask_score':quality,'mask_coverage':float(mask.mean()) if mask is not None else None,'box':coordinates,'gradcam_active':active,'device':DEVICE,'latency_ms':{'segmentation':round((segmented-started)*1000),'classification_and_gradcam':round((finished-segmented)*1000),'total':round((finished-started)*1000)},'images':images,'notes':['Softmax confidence is not calibrated.','Grad-CAM shows class influence, not a lesion diagnosis.']+(['MedSAM2 is medical-domain pretrained; leaf masks require visual review. ROI inputs differ from full-image classifier training.'] if segment else [])}
