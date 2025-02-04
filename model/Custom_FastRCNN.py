import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def custom_FRCNN(num_classes):
    # # 백본으로 ResNet50을 사용하고 FPN을 추가, 사전 가중치 사용
    # backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)  

    # # 백본으로 ResNet50을 사용하되, 사전 학습된 가중치를 사용하지 않음
    backbone = torchvision.models.resnet50(weights=None)
    
    # 마지막 두 레이어 제거: (분류를 위한 fully connected layers)를 제거
    backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))  
    backbone.out_channels = 2048  # FPN의 출력 채널 수

    # 앵커 생성기 설정
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # Faster R-CNN 모델 생성
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

    return model
