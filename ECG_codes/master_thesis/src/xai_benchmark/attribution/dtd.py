"""
Deep Taylor Decomposition with the z+ rule.

CNN models as we discused in the thesis are decomposed layer by layer back to the input.
For the CNN-LSTM model the recurrent gates have no clean DTD rule, so we fall back
to gradient x input at the CNN/LSTM boundary and only apply z+ inside the CNN.

"""

import numpy as np
import torch
import torch.nn.functional as _F

from ..models import ECGCNN1D_Baseline, ECGCNN1D_Deeper, ECGCNNLSTMAttn


def _lrp_conv1d(layer, x_saved, R_out, eps=1e-9):
    # z+ rule for nn.Conv1d
    x_p = x_saved.clamp(min=0)
    W_p = layer.weight.data.clamp(min=0)
    stride  = layer.stride[0]  if isinstance(layer.stride,  (tuple, list)) else layer.stride
    padding = layer.padding[0] if isinstance(layer.padding, (tuple, list)) else layer.padding
    z = _F.conv1d(x_p, W_p, bias=None, stride=stride, padding=padding) + eps
    s = R_out / z
    c = _F.conv_transpose1d(s, W_p, bias=None, stride=stride, padding=padding)
    return x_p * c


def _lrp_linear(layer, x_saved, R_out, eps=1e-9):
    # z+ rule for nn.Linear
    x_p = x_saved.clamp(min=0)
    W_p = layer.weight.data.clamp(min=0)
    z   = (x_p @ W_p.t()) + eps
    s   = R_out / z
    return x_p * (s @ W_p)


def _lrp_maxpool1d(layer, x_saved, R_out, eps=1e-9):
    #Send all relevance to the winning position (max_unpool)
    ks = layer.kernel_size
    st = layer.stride if layer.stride is not None else ks
    pd = layer.padding if hasattr(layer, 'padding') else 0
    _, idx = _F.max_pool1d_with_indices(x_saved, kernel_size=ks, stride=st, padding=pd)
    return _F.max_unpool1d(R_out, idx, kernel_size=ks, stride=st, padding=pd,
                            output_size=x_saved.shape)


def _lrp_adaptive_avg_pool1d(x_saved, R_out, eps=1e-9):
    #Also Redistribute relevance over positive pre-pool activations (z+ rule)
    x_p = x_saved.clamp(min=0)
    z   = x_p.sum(dim=-1, keepdim=True) + eps
    return x_p / z * R_out


def _dtd_baseline_cnn(model, x_t, target_class, device, eps):
    
    #z+ DTD for ECGCNN1D_Baseline.
    
    
    acts = {}
    hooks = []
    for i in range(7):
        hooks.append(model.net[i].register_forward_hook(
            lambda m, inp, out, i=i: acts.update({f'net_{i}': inp[0].detach()})))
    hooks.append(model.fc.register_forward_hook(
        lambda m, inp, out: acts.update({'fc': inp[0].detach()})))

    with torch.no_grad():
        out = model(x_t)
    for h in hooks:
        h.remove()

    R = torch.zeros_like(out)
    R[0, target_class] = out[0, target_class].clamp(min=eps)

    R = _lrp_linear(model.fc, acts['fc'], R, eps)           
    R = R.unsqueeze(-1)                                      
    R = _lrp_adaptive_avg_pool1d(acts['net_6'], R)           
    R = _lrp_maxpool1d(model.net[5], acts['net_5'], R, eps)  
    R = _lrp_conv1d(model.net[3], acts['net_3'], R, eps)     
    R = _lrp_maxpool1d(model.net[2], acts['net_2'], R, eps)  
    R = _lrp_conv1d(model.net[0], acts['net_0'], R, eps)     
    return R[0].cpu().numpy().astype(np.float32)


def _dtd_deeper_cnn(model, x_t, target_class, device, eps):
    
    #z+ DTD for ECGCNN1D_Deeper. BatchNorm is a pass-through in eval mode.
    
    
    acts = {}
    hooks = []
    for i in range(12):
        hooks.append(model.features[i].register_forward_hook(
            lambda m, inp, out, i=i: acts.update({f'feat_{i}': inp[0].detach()})))
    hooks.append(model.classifier[-1].register_forward_hook(
        lambda m, inp, out: acts.update({'fc': inp[0].detach()})))

    with torch.no_grad():
        out = model(x_t)
    for h in hooks:
        h.remove()

    R = torch.zeros_like(out)
    R[0, target_class] = out[0, target_class].clamp(min=eps)

    R = _lrp_linear(model.classifier[-1], acts['fc'], R, eps)        
    R = R.unsqueeze(-1)                                               
    R = _lrp_adaptive_avg_pool1d(acts['feat_11'], R)                  
    R = _lrp_conv1d(model.features[8], acts['feat_8'], R, eps)       
    R = _lrp_maxpool1d(model.features[7], acts['feat_7'], R, eps)    
    R = _lrp_conv1d(model.features[4], acts['feat_4'], R, eps)       
    R = _lrp_maxpool1d(model.features[3], acts['feat_3'], R, eps)    
    R = _lrp_conv1d(model.features[0], acts['feat_0'], R, eps)       
    return R[0].cpu().numpy().astype(np.float32)


def _dtd_cnn_lstm(model, x_t, target_class, device, eps):
    """
    Hybrid DTD for ECGCNNLSTMAttn: gradient x input at the CNN/LSTM boundary,
    then z+ back through the CNN.
    
    """
    acts = {}
    hooks = []
    for i in range(11):
        hooks.append(model.cnn[i].register_forward_hook(
            lambda m, inp, out, i=i: acts.update({f'cnn_{i}': inp[0].detach()})))

    with torch.no_grad():
        cnn_out = model.cnn(x_t)          # (1, 128, T')
    for h in hooks:
        h.remove()

    # as we know cuDNN needs the LSTM in train mode to keep the activations for its backward pass
    # BN still uses running stats and dropout stays off
    model.lstm.train()
    cnn_req = cnn_out.detach().requires_grad_(True)
    perm = cnn_req.permute(0, 2, 1)                               
    lstm_out, _ = model.lstm(perm)                                 
    attn_w = torch.softmax(model.attn_mlp(lstm_out), dim=1)
    ctx = torch.sum(attn_w * lstm_out, dim=1)                    
    logit = model.fc(model.dropout(ctx))                         

    grad = torch.autograd.grad(logit[0, target_class].sum(), cnn_req,
                                create_graph=False)[0]
    model.lstm.eval()
    # seed relevance with the positive part of gradient x activation
    R = (grad * cnn_req.detach()).clamp(min=0)                    

    R = _lrp_conv1d(model.cnn[8], acts['cnn_8'], R, eps)         
    R = _lrp_maxpool1d(model.cnn[7], acts['cnn_7'], R, eps)      
    R = _lrp_conv1d(model.cnn[4], acts['cnn_4'], R, eps)         
    R = _lrp_maxpool1d(model.cnn[3], acts['cnn_3'], R, eps)      
    R = _lrp_conv1d(model.cnn[0], acts['cnn_0'], R, eps)         
    return R[0].cpu().numpy().astype(np.float32)


def deep_taylor_decomposition(
    model,
    x: np.ndarray,
    target_class: int,
    device,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    DTD dispatcher. Picks the implementation from the model type and returns a
    (C, T) attribution map in the same format as IG.
    """
    model.eval()
    x_t = torch.tensor(x[None], dtype=torch.float32, device=device)

    if isinstance(model, ECGCNNLSTMAttn):
        return _dtd_cnn_lstm(model, x_t, target_class, device, eps)
    elif isinstance(model, ECGCNN1D_Deeper):
        return _dtd_deeper_cnn(model, x_t, target_class, device, eps)
    else:   # ECGCNN1D_Baseline
        return _dtd_baseline_cnn(model, x_t, target_class, device, eps)
