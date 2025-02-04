import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from model.Custom_FastRCNN import custom_FRCNN  # 사용자 정의 모델 임포트
import json
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 사전 훈련된 모델 다운로드 캐시 경로 설정
cache_dir = './cache'
os.makedirs(cache_dir, exist_ok=True)  # 디렉토리가 없으면 생성
os.environ['TORCH_HOME'] = cache_dir

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
        
        # 레이블 인코딩을 위한 클래스 맵 생성
        self.label_map = {label: idx for idx, label in enumerate(set(obj['name'] for img in self.data['images'] for obj in img['objects']))}

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        img_info = self.data['images'][idx]
        img_path = os.path.join(self.img_dir, img_info['filename'])
        
        # 파일 존재 여부 확인
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # 객체 정보 추출
        boxes = []
        labels = []
        image_ids = []
        iscrowd = []
        for obj in img_info['objects']:
            boxes.append([obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']])
            labels.append(self.label_map[obj['name']])  # 이름을 정수 레이블로 변환
            image_ids.append(idx)  # 현재 이미지의 인덱스를 image_id로 사용
            iscrowd.append(0)  # 모든 객체가 군중이 아님

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)  # 레이블을 텐서로 변환
        image_ids = torch.tensor(image_ids)  # image_id를 텐서로 변환
        iscrowd = torch.tensor(iscrowd, dtype=torch.int64)  # iscrowd를 텐서로 변환

        if self.transform:
            image = self.transform(image)

        target = {
            'boxes': boxes,
            'labels': labels,  # 정수 레이블
            'image_id': image_ids,  # image_id 추가
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),  # 면적 계산
            'iscrowd': iscrowd  # iscrowd 추가
        }

        return image, target
    
class CocoDataset(CocoDetection):
    def __init__(self, img_dir, annotation_file, transform=None):
        super().__init__(img_dir, annotation_file)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if self.transform:
            img = self.transform(img)
        else:
            img = F.to_tensor(img)  # 기본 변환 적용
            
        return img, target

class ResizeAndPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # 이미지 크기 조정
        img = F.resize(img, self.size)
        # 패딩 추가
        pad_height = self.size[0] - img.size[1]
        pad_width = self.size[1] - img.size[0]
        img = F.pad(img, (0, 0, pad_width, pad_height), fill=0)  # 오른쪽과 아래쪽에 패딩 추가
        return img

# 모델 로드 및 준비
def get_model(num_classes):
    # 사전 훈련된 Faster R-CNN 모델 로드
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name='resnet50',  # 키워드 인자로 변경
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    )
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model

# COCO 주석 파일에서 categories 정보를 읽는 함수
def load_coco_categories(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    categories = {category['id']: category['name'] for category in data['categories']}
    return categories

def load_custom_categories(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # 고유한 레이블을 추출하여 categories 리스트 생성
    categories = list(set(obj['name'] for img in data['images'] for obj in img['objects']))
    print(f'\ncategories = {categories}')
    return categories
    
# COCO 데이터셋의 클래스 수를 자동으로 읽어오는 함수
def get_num_coco_classes(annotation_file):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return len(data['categories'])  # coco 데이터는 categories의 길이를 반환

# # custom 데이터셋의 수를 자동으로 읽어오는 함수
# def get_num_custom_classes(annotation_file):
#     with open(annotation_file, 'r') as f:
#         data = json.load(f)
#     return len(data['images'])  # custom 데이터는 images의 길이를 반환

# collate_fn 수정: 배치 내에서 이미지 크기 일치시키기
def collate_fn(batch, categories):
    # batch는 [(image, target), (image, target), ...] 형태입니다.

    images, targets = zip(*batch)
    
    # # 이미지 크기 일치시키기 (배치의 모든 이미지를 동일한 크기로 패딩/리사이즈 처리)
    # images = [F.to_tensor(image) for image in images]
    
    # 이미지를 이미 텐서로 변환했으므로 다시 F.to_tensor 호출은 불필요
    images = [image for image in images]  # 이미 텐서인 이미지들만 배치로 묶음
    
    num_classes = len(categories)  # 모델에서 처리할 수 있는 클래스 수
    
    target_dicts = []
    for target in targets:
        if len(target) == 0:  # target이 비어있는 경우 (예: 객체가 없는 이미지)
            target_dict = {
                'boxes': torch.empty(0, 4),  # 빈 tensor로 처리
                'labels': torch.empty(0, dtype=torch.int64),
                'image_id': torch.tensor([0]),  # 해당 image_id를 0으로 설정 (또는 적절한 값으로 설정)
                'area': torch.empty(0),
                'iscrowd': torch.empty(0),
                'category_names': []
            }
        else:
            # target 안의 각 객체에 대해 적절한 키를 찾아서 변환
            boxes = torch.stack([torch.tensor(item['bbox']) for item in target])  # 'bbox'를 'boxes'로 변환
            labels = torch.tensor([item['category_id'] for item in target])  # 'category_id'를 'labels'로 변환
            area = torch.tensor([item['area'] for item in target])
            iscrowd = torch.tensor([item['iscrowd'] for item in target])
            
            # 바운딩 박스 유효성 검사 및 수정 (텐서 수준에서 처리)
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                if x2 <= x1:
                    boxes[i][2] = x1 + 1  # x2를 x1보다 크도록 수정
                if y2 <= y1:
                    boxes[i][3] = y1 + 1  # y2를 y1보다 크도록 수정
            
            # 'labels'가 범위를 벗어나지 않도록 처리
            labels = torch.clamp(labels, 0, num_classes - 1)  # 범위 내 값으로 클램핑
            
            # 각 target을 dict로 변환
            target_dict = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([item['image_id'] for item in target]),  # image_id를 리스트에서 가져옴
                'area': area,
                'iscrowd': iscrowd,
                'category_names': [categories[label.item()] for label in labels]  # category_names 자동 변환
            }
        target_dicts.append(target_dict)
        
    return images, target_dicts

def custom_collate_fn(batch, categories):
    images, targets = zip(*batch)

    # 이미지를 이미 텐서로 변환했으므로 다시 F.to_tensor 호출은 불필요
    images = [image for image in images]  # 이미 텐서인 이미지들만 배치로 묶음
    
    target_dicts = []
    
    for target in targets:
        boxes = target['boxes']  # 'boxes'를 사용
        labels = target['labels']  # 'labels'를 사용
        
        # 바운딩 박스 유효성 검사 및 수정 (텐서 수준에서 처리)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            if x2 <= x1:
                boxes[i][2] = x1 + 1  # x2를 x1보다 크도록 수정
            if y2 <= y1:
                boxes[i][3] = y1 + 1  # y2를 y1보다 크도록 수정
        
        # 'labels'가 범위를 벗어나지 않도록 처리
        labels = torch.clamp(labels, 0, len(categories) ) 
        
        # 각 target을 dict로 변환
        target_dict = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([0]),  # image_id를 적절하게 설정
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),  # 면적 계산
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)  # 모든 객체가 군중이 아님
        }
        
        target_dicts.append(target_dict)
        
    return images, target_dicts

# coco 훈련 데이터셋 로드 함수
def get_coco_train_data_loader(img_dir, annotation_file, batch_size, image_size, categories):
    transform = transforms.Compose([
        ResizeAndPad(image_size),  # 사용자 정의 패딩 변환
        transforms.ToTensor(),  # 이미지를 텐서로 변환
    ])
    dataset = CocoDataset(img_dir, annotation_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch_size: collate_fn(batch_size, categories))

# coco 검증 데이터셋 로드 함수
def get_coco_val_data_loader(img_dir, annotation_file, batch_size, image_size, categories):
    transform = transforms.Compose([
        ResizeAndPad(image_size),  # 사용자 정의 패딩 변환
        transforms.ToTensor(),  # 이미지를 텐서로 변환
    ])
    dataset = CocoDataset(img_dir, annotation_file, transform=transform)  # CustomCocoDataset 사용
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch_size: collate_fn(batch_size, categories))

# custom 훈련 데이터셋 로드 함수
def get_custom_train_data_loader(img_dir, annotations, batch_size, image_size, categories):
    transform = transforms.Compose([
        # transforms.Resize(image_size),  # 모든 이미지를 지정된 크기로 조정
        ResizeAndPad(image_size),  # 사용자 정의 패딩 변환
        transforms.ToTensor(),  # 이미지를 텐서로 변환
    ])
    dataset = CustomDataset(img_dir, annotations, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch_size: custom_collate_fn(batch_size, categories))

# custom 검증 데이터셋 로드 함수
def get_custom_val_data_loader(img_dir, annotations, batch_size, image_size, categories):
    transform = transforms.Compose([
        # transforms.Resize(image_size),  # 모든 이미지를 지정된 크기로 조정
        ResizeAndPad(image_size),  # 사용자 정의 패딩 변환
        transforms.ToTensor(),  # 이미지를 텐서로 변환
    ])
    dataset = CustomDataset(img_dir, annotations, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda batch_size: custom_collate_fn(batch_size, categories))

# 학습 함수
def train(model, data_loader, num_epochs, device, optimizer):
    model.train()
    for epoch in range(num_epochs):
        # for images, targets in data_loader:
        for i, (images, targets) in enumerate(tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')):
            # # for debug
            # print(f'Image sizes: {[image.size() for image in images]}')  # 이미지 크기 확인

            images = [image.to(device) for image in images]  # 텐서로 변환
            
            # 각 target에 대해 to(device) 호출
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            
            # 모델에 이미지를 입력하고 손실 계산
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # 역전파 및 최적화
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # # 메모리 정리 및 사용량 출력 (100 배치마다)
            # if (i + 1) % 100 == 0:
            #     del outputs, losses   # 메모리 정리: Tensor 객체 삭제
                
            #     if device.type == 'cuda':
            #         torch.cuda.empty_cache()    # 캐시된 메모리 정리
            
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}')
        
        # 학습 완료 후 모델 저장
        torch.save(model.state_dict(), 'trained_model.pth')
        print('모델이 저장되었습니다.')

# 검증 함수
def validate(model, val_data_loader, device, categories):
    model.eval()  # 평가 모드로 설정
    total_loss = 0
    with torch.no_grad():  # 그라디언트 계산을 방지
        for images, targets in val_data_loader:
            images = torch.stack(images).to(device)  # images가 list라면 Tensor로 변환
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            ############################################################################
            # IoU 계산
            ############################################################################
            # 모델을 통해 예측값을 계산
            predictions = model(images)

            # # 각 이미지에 대해 타겟과 모델 출력 비교
            # for i, (target, prediction) in enumerate(zip(targets, predictions)):
            #     # print(f"Comparing target vs prediction for image {i}:")

            #     # # 타겟의 'boxes'와 'labels' 출력
            #     target_boxes = target['boxes']
            #     # target_labels = target['labels']
            #     # # print(f"Target Boxes: {target_boxes}")
            #     # # print(f"Target Labels: {target_labels}")

            #     # # 예측값의 'boxes', 'labels', 'scores' 출력
            #     pred_boxes = prediction['boxes']
            #     # pred_labels = prediction['labels']
            #     # pred_scores = prediction['scores']
            #     # # print(f"Predicted Boxes: {pred_boxes}")
            #     # # print(f"Predicted Labels: {pred_labels}")
            #     # # print(f"Predicted Scores: {pred_scores}")

            #     # # 정확도 측정 (간단한 예시)
            #     # # 예측된 'labels'가 타겟 'labels'와 일치하는지 확인
            #     # correct_predictions = (pred_labels == target_labels).sum().item()
            #     # print(f"Correct Predictions: {correct_predictions}/{len(target_labels)}")

            #     # 예측된 'boxes'가 타겟 'boxes'와 얼마나 일치하는지 확인
            #     # 예를 들어, IoU(Intersection over Union) 계산을 할 수 있습니다
            #     iou = calculate_iou(pred_boxes, target_boxes)
            #     print(f"IoU: {iou}")
            ############################################################################

            # 모델을 통해 손실값을 계산
            loss_dict = model(images, targets)

            # # loss_dict의 내용을 출력
            # print("loss_dict:", loss_dict)

            # loss_dict가 딕셔너리인지 확인하고, 그에 맞는 처리
            if isinstance(loss_dict, dict):
                # 딕셔너리라면 각 손실 값을 합산
                losses = sum(loss.item() if isinstance(loss, torch.Tensor) else 0 for loss in loss_dict.values())
            elif isinstance(loss_dict, list):
                # 리스트라면 각 손실 값을 합산
                losses = sum(loss.item() if isinstance(loss, torch.Tensor) else 0 for loss in loss_dict)
            else:
                raise ValueError(f"Unknown loss format: {type(loss_dict)}")
            
            # # for debug
            # print(f"Batch loss: {losses}")
            
            total_loss += losses            
    
    print(f"Total validation loss: {total_loss}")
    
    visualize_predictions(images, targets, predictions, category_names=categories)
    
# IoU 계산 함수 (Intersection over Union)
def calculate_iou(pred_boxes, target_boxes):
    # 각 예측 박스와 타겟 박스 간의 IoU 계산
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # 교차 영역 (intersection)
    intersection_area = torch.max(x2 - x1, torch.tensor(0)) * torch.max(y2 - y1, torch.tensor(0))

    # 예측 박스와 타겟 박스의 면적
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])

    # 합집합 영역 (union)
    union_area = pred_area + target_area - intersection_area

    # IoU 계산
    iou = intersection_area / union_area
    return iou.mean().item()   

def visualize_predictions(images, targets, predictions, category_names=None):
    """
    Visualize predictions on images.
    
    Parameters:
    - images: A list of image tensors.
    - targets: A list of target dictionaries with ground truth information.
    - predictions: A list of prediction dictionaries from the model.
    - category_names: A list of category names, to map label indices to class names (optional).
    """
    for i, (image, target, prediction) in enumerate(zip(images, targets, predictions)):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(F.to_pil_image(image))  # 이미지 그리기

        # 타겟 바운딩 박스 그리기 (Ground Truth)
        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{category_names[label.item()]}' if category_names else f'Label {label.item()}', color='g')

        # 예측된 바운딩 박스 그리기
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score < 0.5:  # 예측된 신뢰도가 낮으면 제외
                continue
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'{category_names[label.item()]}: {score:.2f}' if category_names else f'Label {label.item()}: {score:.2f}', color='r')

        plt.show()
        
def main():
    # 설정
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu') # cuda 사용시 blue screen 발생
    
    print("\n=== 시스템 정보 ===")
    print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
    print(f"현재 PyTorch의 CUDA 버전: {torch.version.cuda}")
    print(f"PyTorch 버전: {torch.__version__}") # 예: '2.0.0+cu121'는 CUDA 12.1 버전을 지원
    print(f'사용 중인 장치: {device}')
    
    # # coco 데이터 사용
    # train_annotation_file = 'coco_data/train/annotations/instances_train2017.json'  # COCO 형식의 주석 파일 경로
    # train_img_dir = 'coco_data/train/train2017'  # train 이미지 디렉토리 경로
    # categories = load_coco_categories(train_annotation_file)  # COCO 카테고리 정보 로드
    
    # val_annotation_file = 'coco_data/val/annotations/instances_val2017.json'  # COCO 형식의 주석 파일 경로
    # val_img_dir = 'coco_data/val/val2017'  # validation 이미지 디렉토리 경로
    
    # custom 데이터 사용
    train_annotation_file = 'custom_data/train/annotations/annotations.json'  # COCO 형식의 주석 파일 경로
    train_img_dir = 'custom_data/train/train'  # train 이미지 디렉토리 경로
    categories = load_custom_categories(train_annotation_file)

    val_annotation_file = 'custom_data/val/annotations/annotations.json'  # COCO 형식의 주석 파일 경로
    val_img_dir = 'custom_data/val/val'  # validation 이미지 디렉토리 경로
    
    #######################################################
    # 하이퍼 파라미터 설정
    num_epochs = 1
    batch_size = 5
    image_size = (640, 640)  # 이미지 크기 설정    
    learning_rate = 0.005
    momentum = 0.9
    weight_decay = 0.0005
    
    # num_classes = get_num_coco_classes(train_annotation_file) + 1  # COCO는 0번 클래스를 포함하므로 +1
    num_classes = len(categories) # category가 class임
    print(f'num_classes = [{num_classes}]')
    #######################################################
    
    # # 모델 및 데이터 로더 준비
    # model = get_model(num_classes).to(device)
    
    # # cocodata loader 준비
    # # 데이터 로더에서 collate_fn을 사용하여 데이터를 로드할 때 `categories`를 넘겨줍니다.
    # train_data_loader = get_coco_train_data_loader(train_img_dir, train_annotation_file, batch_size, image_size, categories)
    # val_data_loader = get_coco_val_data_loader(val_img_dir, val_annotation_file, batch_size, image_size, categories )
  
    #####################################################
    # custom 모델 사용
    #####################################################
    model = custom_FRCNN(num_classes).to(device)
    
    # custom 데이터 로더 준비
    train_data_loader = get_custom_train_data_loader(train_img_dir, train_annotation_file, batch_size, image_size, categories)
    val_data_loader = get_custom_val_data_loader(val_img_dir, val_annotation_file, batch_size, image_size, categories)

    # 옵티마이저 설정
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    # 학습 시작
    print("학습 시작")
    train(model, train_data_loader, num_epochs, device, optimizer)
    
    # 검증 시작
    print("검증 시작")
    validate(model, val_data_loader, device, categories)

if __name__ == "__main__":
    main()
