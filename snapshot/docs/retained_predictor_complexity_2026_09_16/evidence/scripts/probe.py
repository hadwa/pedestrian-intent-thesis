import sys,os,json,zipfile,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
import torch,joblib
R=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean')
def exp(n):return next((R/'experiments').glob(f'{n:03d}_*'))
for n,stem in [(53,'G'),(53,'I'),(95,'J'),(84,'A'),(93,'R'),(89,'A'),(91,'N')]:
 p=exp(n)/f'checkpoints/{stem}_seed42.best.pt';c=torch.load(p,map_location='cpu',weights_only=False)
 print(n,stem,p.stat().st_size,list(c.keys()))
 print('metadata', {k:v for k,v in c.items() if not isinstance(v,(dict,list)) and not torch.is_tensor(v)})
 sd=c['model_state']; print('tensor elements',sum(t.numel() for t in sd.values()),'sample',[(k,list(v.shape)) for k,v in list(sd.items())[:3]])
 print('zip compression',set(i.compress_type for i in zipfile.ZipFile(p).infolist()))
p=exp(96)/'models/v2_seed42.joblib';m=joblib.load(p);print('XGB',type(m),m.named_steps, 'bytes',p.stat().st_size,'header',p.read_bytes()[:8].hex());print(m.named_steps['model'].get_booster().save_config())
