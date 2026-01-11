import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 유틸리티 레이어 ---
class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32)) # 학습 가능한 스케일 파라미터

    def forward(self, x):
        return x * self.scale

# --- Conv + GN + ReLU 블록 ---
def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.GroupNorm(4, out_channels), 
        nn.ReLU(inplace=True)
    )

# --- Tiny Backbone + FPN + Anchor Free Head ---
class TinyBackbone(nn.Module):
    def __init__(self):
        super(TinyBackbone, self).__init__()
        # Input: (3, 256, 256)
        self.stem = conv_bn_relu(3, 32, stride=2) 
        self.layer1 = nn.Sequential(conv_bn_relu(32, 64, stride=2), conv_bn_relu(64, 64))
        self.layer2 = nn.Sequential(conv_bn_relu(64, 128, stride=2), conv_bn_relu(128, 128), conv_bn_relu(128, 128))
        self.layer3 = nn.Sequential(conv_bn_relu(128, 256, stride=2), conv_bn_relu(256, 256), conv_bn_relu(256, 256))
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        return [c2, c3, c4] # Strides: [4, 8, 16]

class TinyFPN(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256], out_channels=64):
        super(TinyFPN, self).__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, inputs):
        p4 = self.lateral_convs[2](inputs[2])
        p3_in = self.lateral_convs[1](inputs[1])
        p3 = p3_in + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2_in = self.lateral_convs[0](inputs[0])
        p2 = p2_in + F.interpolate(p3, scale_factor=2, mode="nearest")
        return [self.fpn_convs[0](p2), self.fpn_convs[1](p3), self.fpn_convs[2](p4)]

class FCOSHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes
        
        # Classification Branch
        self.cls_convs = nn.Sequential(
            conv_bn_relu(in_channels, in_channels),
            conv_bn_relu(in_channels, in_channels),
            nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        )
        
        # Regression Branch
        self.reg_convs = nn.Sequential(
            conv_bn_relu(in_channels, in_channels),
            conv_bn_relu(in_channels, in_channels),
            nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        )
        
        # Centerness Branch (노이즈라고 생각 들어 제거)
        #self.cnt_head = nn.Conv2d(4, 1, kernel_size=3, padding=1)

        # 레벨별 스케일 (P2, P3, P4)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(3)])

        self._init_weights()

    def _init_weights(self):
        # focal loss 안정화를 위해 초기 확률이 0.01이 되도록 설정
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_convs[-1].bias, bias_value)
        
        nn.init.normal_(self.reg_convs[-1].weight, std=0.01)
        nn.init.constant_(self.reg_convs[-1].bias, 0)

    def forward(self, features):
        cls_logits = []
        bbox_reg = []

        for l, feature in enumerate(features):
            # Cls
            cls_score = self.cls_convs(feature)
            cls_logits.append(cls_score)

            # Reg
            reg_feat = self.reg_convs[0](feature)
            reg_feat = self.reg_convs[1](reg_feat)
            reg_map = self.reg_convs[2](reg_feat)
            
            # Scale 적용, EXP(거리는 항상 양수)
            reg_map = torch.exp(self.scales[l](reg_map))
            bbox_reg.append(reg_map)

            # Centerness (노이즈라고 생각 들어 제거)
            #cnt_score = self.cnt_head(reg_map)
            #centerness.append(cnt_score)

        return cls_logits, bbox_reg

class InhaDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(InhaDetector, self).__init__()
        self.backbone = TinyBackbone()
        self.fpn = TinyFPN([64, 128, 256], out_channels=64)
        self.head = FCOSHead(in_channels=64, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        fpn_feats = self.fpn(features)
        return self.head(fpn_feats)