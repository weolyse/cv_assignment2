"""Reproducible training: content-deduplicated, stratified split; test used only after selection."""
import argparse, hashlib, json, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from app.models import make_model
from app.imaging import preprocessing

ROOT=Path(__file__).resolve().parents[1]
class Leaves(Dataset):
    def __init__(self,rows,augment): self.rows=rows; self.transform=preprocessing(augment)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        path,label=self.rows[i]
        with Image.open(path) as im: image=ImageOps.exif_transpose(im).convert('RGB')
        return self.transform(image),label

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True); p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--models',nargs='+',default=['simple_cnn','resnet18'],choices=['simple_cnn','resnet18'])
    p.add_argument('--no-augmentation',action='store_true'); p.add_argument('--output',type=Path,default=ROOT/'checkpoints')
    p.add_argument('--device',default='auto'); args=p.parse_args()
    torch.set_num_threads(4)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device=('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') if args.device=='auto' else args.device
    rows=[]; seen={}; duplicates=0
    for label,cls in enumerate(['Diseased','Healthy']):
        for path in sorted((args.data/cls).glob('*')):
            if path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'): continue
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                key=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
            if key in seen:
                if seen[key]!=label: raise ValueError(f'Conflicting labels for {path}')
                duplicates+=1; continue
            seen[key]=label; rows.append((str(path.resolve()),label))
    train,held=train_test_split(rows,test_size=.30,random_state=42,stratify=[r[1] for r in rows])
    val,test=train_test_split(held,test_size=.50,random_state=42,stratify=[r[1] for r in held])
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={'seed':42,'classes':['Diseased','Healthy'],'duplicates_removed':duplicates,'train':train,'validation':val,'test':test}
    (args.output/'split.json').write_text(json.dumps(manifest,indent=2))
    loaders=[DataLoader(Leaves(r,i==0 and not args.no_augmentation),batch_size=16,shuffle=i==0,num_workers=0) for i,r in enumerate([train,val,test])]
    print(f'Device {device}; unique {len(rows)}, duplicates {duplicates}; split {len(train)}/{len(val)}/{len(test)}',flush=True)
    for name in args.models:
        torch.manual_seed(42)
        model=make_model(name).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001 if name=='simple_cnn' else .0001)
        loss_fn=torch.nn.CrossEntropyLoss(); history=[]; best=float('inf'); selected=0
        for epoch in range(args.epochs):
            result={'epoch':epoch+1}
            for phase,loader in zip(['train','validation'],loaders[:2]):
                model.train(phase=='train'); total=correct=0; loss_total=0.
                for x,y in loader:
                    x,y=x.to(device),y.to(device)
                    with torch.set_grad_enabled(phase=='train'):
                        logits=model(x); loss=loss_fn(logits,y)
                        if phase=='train': optimizer.zero_grad(); loss.backward(); optimizer.step()
                    total+=len(y); correct+=(logits.argmax(1)==y).sum().item(); loss_total+=loss.item()*len(y)
                result[phase+'_loss']=loss_total/total; result[phase+'_accuracy']=correct/total
            history.append(result); print(name,result,flush=True)
            if result['validation_loss']<best:
                best=result['validation_loss']; selected=epoch+1
                torch.save({'model':name,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'classes':['Diseased','Healthy'],'initialization':'random','epoch':selected},args.output/f'{name}.pt')
        ckpt=torch.load(args.output/f'{name}.pt',weights_only=True,map_location=device); model.load_state_dict(ckpt['state_dict']); model.eval()
        truth=[]; pred=[]
        with torch.no_grad():
            for x,y in loaders[2]: truth.extend(y.tolist()); pred.extend(model(x.to(device)).argmax(1).cpu().tolist())
        report={'model':name,'initialization':'random','augmentation':not args.no_augmentation,'device':device,'epochs':args.epochs,'selected_epoch':selected,'train_count':len(train),'validation_count':len(val),'test_count':len(test),'duplicates_removed':duplicates,'accuracy':accuracy_score(truth,pred),'macro_f1':f1_score(truth,pred,average='macro'),'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist(),'classification_report':classification_report(truth,pred,target_names=['Diseased','Healthy'],zero_division=0,output_dict=True),'history':history}
        (args.output/f'{name}_metrics.json').write_text(json.dumps(report,indent=2)); print(report,flush=True)
if __name__=='__main__': main()
