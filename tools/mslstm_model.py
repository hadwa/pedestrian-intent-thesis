"""Inference architecture from Exp050/053/095/120; no data or training imports."""
import torch
from torch import nn
class StreamAttention(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(512,128),nn.Tanh(),nn.Linear(128,1))
 def forward(self,sequence):
  weights=torch.softmax(self.net(sequence).squeeze(2),dim=1)
  return torch.sum(sequence*weights.unsqueeze(2),dim=1)
class Model(nn.Module):
 def __init__(self,streams,dimensions,vectors):
  super().__init__();self.streams=streams;self.vectors=set(vectors)
  self.lstms=nn.ModuleDict();self.attentions=nn.ModuleDict();self.vector_projections=nn.ModuleDict()
  for name in streams:
   if name in self.vectors:self.vector_projections[name]=nn.Sequential(nn.Linear(dimensions[name],128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,512))
   else:
    self.lstms[name]=nn.LSTM(dimensions[name],256,2,batch_first=True,dropout=.3,bidirectional=True)
    self.attentions[name]=StreamAttention()
  self.context_normalization=nn.LayerNorm(512,elementwise_affine=False)
  self.fusion_weights=nn.Parameter(torch.ones(len(streams)))
  self.dropout=nn.Dropout(.3);self.fc1=nn.Linear(512,256);self.relu=nn.ReLU();self.fc2=nn.Linear(256,2)
 def forward(self,inputs):
  contexts=[]
  for name in self.streams:
   if name in self.vectors:context=self.vector_projections[name](inputs[name])
   else:
    sequence,_=self.lstms[name](inputs[name]);context=self.attentions[name](sequence)
   contexts.append(self.context_normalization(context))
  weights=torch.softmax(self.fusion_weights,dim=0).view(1,-1,1)
  fused=torch.sum(torch.stack(contexts,dim=1)*weights,dim=1)
  return self.fc2(self.dropout(self.relu(self.fc1(self.dropout(fused)))))
