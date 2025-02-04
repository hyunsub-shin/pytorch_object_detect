import os
import requests
import zipfile

def download_coco_data(data_dir):
    # COCO 데이터셋 URL
    urls = {
        "train_images": "http://images.cocodataset.org/zips/train2017.zip",
        "val_images": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }

    # 데이터 디렉토리 생성
    os.makedirs(data_dir, exist_ok=True)

    for name, url in urls.items():
        print(f"Downloading {name}...")
        response = requests.get(url)
        zip_file_path = os.path.join(data_dir, f"{name}.zip")

        # ZIP 파일 저장
        with open(zip_file_path, 'wb') as f:
            f.write(response.content)

        # ZIP 파일 압축 해제
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # ZIP 파일 삭제
        os.remove(zip_file_path)
        print(f"{name} downloaded and extracted.")

# 사용 예시
if __name__ == "__main__":
    data_dir = "coco_data"  # 데이터 저장 디렉토리
    download_coco_data(data_dir)
