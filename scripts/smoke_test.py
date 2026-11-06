"""Integration checks against real installed checkpoints, no mocked inference."""
import base64,json,sys
from io import BytesIO
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app
ROOT=Path(__file__).resolve().parents[1]
path=Path(json.loads((ROOT/'checkpoints/split.json').read_text())['test'][0][0])
client=TestClient(app); summary=[]
for model in ['simple_cnn','resnet18']:
    for segment in [False,True]:
        response=client.post('/api/analyze',files={'file':(path.name,path.read_bytes(),'image/jpeg')},data={'model':model,'segment':str(segment).lower(),'box':'[0.13,0.32,0.8,0.85]'})
        assert response.status_code==200,response.text
        data=response.json();assert abs(sum(data['probabilities'].values())-1)<1e-5
        sizes={}
        for name,url in data['images'].items():
            image=Image.open(BytesIO(base64.b64decode(url.split(',')[1])));sizes[name]=image.size
        assert len(set(sizes.values()))==1
        assert ('mask' in sizes)==segment
        summary.append({k:v for k,v in data.items() if k!='images'})
        print(model,segment,data['label'],data['latency_ms'],flush=True)
(ROOT/'artifacts/integration_checks.json').write_text(json.dumps(summary,indent=2))
