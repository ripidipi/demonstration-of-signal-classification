import torch
from torchviz import make_dot
from model import SOTAClassifier


model = SOTAClassifier(10)
model.eval()
iq1 = torch.randn(1, 2, 1024, requires_grad=True)
iq2 = torch.randn(1, 2, 1024, requires_grad=True)
const = torch.randn(1, 64 * 64, requires_grad=True)
snr = torch.randn(1, 1, requires_grad=True)
output = model(iq1, iq2, const, snr)
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render('model_graph') 
