"""Official MedSAM2 image predictor, prompted by a user box or full-frame inset."""
import os, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
CHECKPOINT=Path(os.getenv('MEDSAM2_CHECKPOINT',str(ROOT/'checkpoints/MedSAM2_latest.pt')))
CONFIG=os.getenv('MEDSAM2_CONFIG','configs/sam2.1_hiera_t512.yaml')
REPO=Path(os.getenv('MEDSAM2_REPO',str(ROOT/'vendor/MedSAM2')))

class MedSAM2Segmenter:
    def __init__(self,device): self.device=device; self.predictor=None
    def load(self):
        if not CHECKPOINT.is_file(): raise RuntimeError('MedSAM2 checkpoint missing. Run scripts/setup_medsam2.sh.')
        if REPO.is_dir() and str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.predictor=SAM2ImagePredictor(build_sam2(CONFIG,str(CHECKPOINT),device=self.device,apply_postprocessing=False))
    def segment(self,image,box=None):
        if self.predictor is None: self.load()
        w,h=image.size
        if box is None: box=[.02,.02,.98,.98]
        pixels=np.array(box,dtype=np.float32)*[w,h,w,h]
        with torch.inference_mode():
            self.predictor.set_image(np.array(image,copy=True))
            masks,scores,_=self.predictor.predict(box=pixels,multimask_output=True)
        index=int(np.argmax(scores)); mask=masks[index].astype(bool)
        if mask.sum()==0: raise ValueError('MedSAM2 returned an empty mask. Draw a tighter box around the leaf.')
        return mask,float(scores[index]),box
