import torch

def get_grid_coords(h, w, stride, device):
    shifts_x = torch.arange(0, w, dtype=torch.float32, device=device) * stride + stride // 2
    shifts_y = torch.arange(0, h, dtype=torch.float32, device=device) * stride + stride // 2
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
    return shift_x.reshape(-1), shift_y.reshape(-1)

def decode_fcos_boxes(bbox_reg, coords):
    l, t, r, b = bbox_reg.unbind(-1)
    x, y = coords.unbind(-1)
    
    x1 = x - l
    y1 = y - t
    x2 = x + r
    y2 = y + b
    
    return torch.stack([x1, y1, x2, y2], dim=-1)