import base64
from io import BytesIO
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

MEAN=(.485,.456,.406)
STD=(.229,.224,.225)

def preprocessing(augment=False):
    steps=[transforms.Resize((224,224))]
    if augment: steps += [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(.3),transforms.RandomRotation(20),transforms.ColorJitter(brightness=.2,contrast=.2)]
    return transforms.Compose(steps+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])

def decode(data):
    with Image.open(BytesIO(data)) as im:
        if im.format not in ('JPEG','PNG','WEBP'): raise ValueError('Use a JPEG, PNG, or WebP image.')
        if im.width*im.height>20_000_000: raise ValueError('Image exceeds 20 megapixels.')
        im=ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((1024,1024))
        return im.copy()

def png(image):
    buf=BytesIO(); image.save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()

def gradcam(model, image, device):
    activations=[]
    handle=model.cam_layer.register_forward_hook(lambda m,i,o: activations.append(o))
    try:
        with torch.enable_grad():
            x=preprocessing()(image).unsqueeze(0).to(device)
            logits=model(x)
            target=int(logits.argmax(1).item())
            features=activations[0]
            gradients=torch.autograd.grad(logits[0,target],features)[0]
            heat=(gradients.mean((2,3),keepdim=True)*features).sum(1).relu()[0]
            heat=heat.detach().cpu().numpy()
            maximum=float(heat.max())
            heat=heat/maximum if maximum>1e-10 else np.zeros_like(heat)
            heat=np.asarray(Image.fromarray(heat).resize(image.size,Image.Resampling.BILINEAR))
            return logits.softmax(1)[0].detach().cpu().tolist(),heat,maximum>1e-10
    finally: handle.remove()

def heat_overlay(image,heat):
    # Blue -> cyan -> yellow -> red; blend proportional to activation.
    rgb=np.stack([np.clip(1.5-abs(4*heat-3),0,1),np.clip(1.5-abs(4*heat-2),0,1),np.clip(1.5-abs(4*heat-1),0,1)],-1)*255
    alpha=(heat*.65)[...,None]
    return Image.fromarray((np.asarray(image)*(1-alpha)+rgb*alpha).astype('uint8'))
