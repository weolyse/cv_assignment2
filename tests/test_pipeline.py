from io import BytesIO
import numpy as np
import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
from app.models import make_model
from app.imaging import gradcam, decode

client=TestClient(app)
def payload():
    out=BytesIO();Image.new('RGB',(80,60),(54,130,42)).save(out,format='PNG');return out.getvalue()

@pytest.mark.parametrize('name',['simple_cnn','resnet18'])
def test_model_and_gradcam(name):
    torch.manual_seed(4);model=make_model(name).eval()
    image=Image.fromarray(np.random.default_rng(4).integers(0,256,(70,90,3),dtype=np.uint8))
    probs,heat,active=gradcam(model,image,'cpu')
    assert len(probs)==2 and sum(probs)==pytest.approx(1)
    assert heat.shape==(70,90) and np.isfinite(heat).all()
    assert heat.min()>=0 and heat.max()<=1.001
    assert not model.cam_layer._forward_hooks

def test_image_validation():
    assert decode(payload()).size==(80,60)
    assert client.post('/api/analyze',files={'file':('bad.png',b'not an image','image/png')}).status_code==400
    assert client.post('/api/analyze',files={'file':('leaf.png',payload(),'image/png')},data={'model':'unknown'}).status_code==422
    for box in ('[1,0,0,1]','[0,0,2,1]','null','[0,0,NaN,1]'):
        assert client.post('/api/analyze',files={'file':('leaf.png',payload(),'image/png')},data={'box':box}).status_code==422

def test_upload_limit():
    r=client.post('/api/analyze',files={'file':('large.png',b'x'*(10*1024*1024+1),'image/png')})
    assert r.status_code==413

def test_status_and_home():
    assert client.get('/').status_code==200
    assert set(client.get('/api/status').json()['models'])=={'simple_cnn','resnet18'}
