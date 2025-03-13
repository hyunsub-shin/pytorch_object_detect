import sys
import os
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QRect
import pandas as pd
from PIL import Image
import json

class ImageLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(ImageLabel, self).__init__(parent)
        self.drawing = False
        self.start_point = QtCore.QPoint()
        self.end_point = QtCore.QPoint()
        self.rect = QRect()

    def set_current_pixmap(self, pixmap):
        """MainWindow에서 current_pixmap을 전달받아 설정"""
        self.current_pixmap = pixmap
        self.setPixmap(pixmap)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            
            # 이미지를 벗어나지 않도록 rect 좌표 제한
            max_x = self.current_pixmap.width()
            max_y = self.current_pixmap.height()

            # Rect의 x, y가 이미지 크기를 넘지 않도록 제한
            self.end_point.setX(min(self.end_point.x(), max_x))
            self.end_point.setY(min(self.end_point.y(), max_y))
            
            # 시작점과 끝점을 이용하여 rect 생성
            self.rect = QRect(self.start_point, self.end_point)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            
            # 이미지를 벗어나지 않도록 rect 좌표 제한
            max_x = self.current_pixmap.width()
            max_y = self.current_pixmap.height()

            # Rect의 x, y가 이미지 크기를 넘지 않도록 제한
            self.end_point.setX(min(self.end_point.x(), max_x))
            self.end_point.setY(min(self.end_point.y(), max_y))
            
            self.rect = QRect(self.start_point, self.end_point)
            self.update()
            
            # rect를 MainWindow로 전달
            self.parent().update_coordinates(self.rect)

    def paintEvent(self, event):
        super(ImageLabel, self).paintEvent(event)
        painter = QPainter(self)
        pen = QPen(Qt.red, 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.drawRect(self.rect.normalized())

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # uic.loadUi('label_box_maker.ui', self)
        #####################################################
        # pyinstaller에서 ui file 가져올수 있게 수정
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ui_file = os.path.join(BASE_DIR, 'label_box_maker.ui')

        # 운영체제 확인 및 경로 설정
        if os.name == 'nt':  # Windows
            ui_file = ui_file.replace('/', '\\')
        else:  # Linux/Unix
            ui_file = ui_file.replace('\\', '/')

        ui = uic.loadUi(ui_file, self)
        ui.setWindowTitle("Label Box Maker")
        ######################################################
        
        self.pushButton_img_dir.clicked.connect(self.select_image_directory)
        self.tableWidget_img_list.clicked.connect(self.display_selected_image)
        self.pushButton_save.clicked.connect(self.save_data)

        self.image_dir = ""
        self.image_list = []
        self.current_image_path = ""
        self.current_pixmap = QPixmap()
        self.data = []
        self.excel_path = "annotations.xlsx"
        self.load_existing_data()

        # Replace QLabel with ImageLabel to handle drawing
        self.label_img = ImageLabel(self)
        self.label_img.setGeometry(self.findChild(QtWidgets.QLabel, "label_img").geometry())
        self.label_img.setAutoFillBackground(True)
        self.label_img.setText("")
        
        # 이미지를 표시할 때 current_pixmap을 설정
        self.label_img.set_current_pixmap(self.current_pixmap)

        # tableWidget_img_list None인지 확인
        if self.tableWidget_img_list is None:
            print("Error: 'tableWidget_img_list' not found in the UI file.")
        else:
            self.tableWidget_img_list.clicked.connect(self.display_selected_image)

        # 원본 이미지 크기 저장을 위한 변수 추가
        self.original_width = 0
        self.original_height = 0
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        
        # 클래스 이름과 ID 매핑 딕셔너리 추가
        self.class_mapping = {}
        self.next_class_id = 0

    def load_existing_data(self):
        if os.path.exists(self.excel_path):
            self.df = pd.read_excel(self.excel_path)
            if not self.df.empty:
                self.image_id = self.df['image_id'].max() + 1
            else:
                self.image_id = 1
        else:
            self.df = pd.DataFrame(columns=['filename', 'image_id', 'label', 'xmin', 'ymin', 'xmax', 'ymax', 'iscrowd'])
            self.image_id = 1

    def select_image_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "이미지 디렉토리 선택", "")
        if directory:
            self.image_dir = directory
            self.lineEdit_img_dir.setText(self.image_dir)
            self.load_images()

    def load_images(self):
        supported_formats = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
        self.image_list = [f for f in os.listdir(self.image_dir) if os.path.splitext(f)[1].lower() in supported_formats]
        
        if self.tableWidget_img_list is not None:  # None 체크 추가
            self.tableWidget_img_list.setRowCount(len(self.image_list))  # 행 수 설정
            self.tableWidget_img_list.setColumnCount(1)  # 열 수 설정 (1열로 설정)
            for row, image_name in enumerate(self.image_list):
                item = QTableWidgetItem(image_name)  # QTableWidgetItem 생성
                self.tableWidget_img_list.setItem(row, 0, item)  # 첫 번째 열에 아이템 설정
        else:
            print("Error: 'tableWidget_img_list' is None.")

    def display_selected_image(self, index):
        image_name = self.image_list[index.row()]
        self.current_image_path = os.path.join(self.image_dir, image_name)
        self.current_pixmap = QPixmap(self.current_image_path)
        
        # 원본 이미지 크기 저장
        self.original_width = self.current_pixmap.width()
        self.original_height = self.current_pixmap.height()
        
        # 최대 크기 설정 (640x640)
        max_size = 640
        
        # 이미지 크기가 640보다 큰 경우, 크기를 줄여서 표시
        if self.original_width > max_size or self.original_height > max_size:
            # 비율을 유지하며 크기 조정
            scaled_pixmap = self.current_pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio)
            # 스케일 팩터 계산
            self.scale_factor_x = scaled_pixmap.width() / self.original_width
            self.scale_factor_y = scaled_pixmap.height() / self.original_height
            self.current_pixmap = scaled_pixmap
        else:
            self.scale_factor_x = 1.0
            self.scale_factor_y = 1.0
            self.current_pixmap = self.current_pixmap.scaled(self.original_width, self.original_height, Qt.KeepAspectRatio)
        
        # 이미지 크기 비율에 맞게 QLabel 크기 조정
        self.label_img.setFixedSize(self.current_pixmap.width(), self.current_pixmap.height())  # QLabel의 크기를 이미지 크기에 맞게 설정
        
        # 이미지를 QLabel에 표시 (이미지 크기 조정)
        # ImageLabel에 current_pixmap을 전달
        self.label_img.set_current_pixmap(self.current_pixmap)
        
        # ImageLabel 내의 rect를 초기화하여 이전 박스 지우기
        self.label_img.rect = QRect()  # ImageLabel의 rect 초기화
        
        # 박스 그림, 좌표 및 레이블 값 리셋        
        self.label_xmin_val.setText("0")  # xmin 리셋
        self.label_ymin_val.setText("0")  # ymin 리셋
        self.label_xmax_val.setText("0")  # xmax 리셋
        self.label_ymax_val.setText("0")  # ymax 리셋
        self.findChild(QtWidgets.QLineEdit, "lineEdit_label").clear()  # 레이블 입력 필드 리셋
        
    def update_coordinates(self, rect):
        if not rect.isNull():
            # 화면에 표시된 좌표
            xmin = max(rect.left(), 0)
            ymin = max(rect.top(), 0)
            xmax = max(rect.right(), 0)
            ymax = max(rect.bottom(), 0)
            
            # 좌상단과 우하단의 좌표가 항상 정확하도록 수정
            xmin = min(xmin, xmax)
            xmax = max(xmin, xmax)
            ymin = min(ymin, ymax)
            ymax = max(ymin, ymax)
            
            # 원본 이미지 크기에 맞게 좌표 변환
            xmin = int(xmin / self.scale_factor_x)
            ymin = int(ymin / self.scale_factor_y)
            xmax = int(xmax / self.scale_factor_x)
            ymax = int(ymax / self.scale_factor_y)
            
            self.label_xmin_val.setText(str(xmin))
            self.label_ymin_val.setText(str(ymin))
            self.label_xmax_val.setText(str(xmax))
            self.label_ymax_val.setText(str(ymax))

    def save_data(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "경고", "저장할 이미지를 선택해주세요.")
            return

        label = self.findChild(QtWidgets.QLineEdit, "lineEdit_label").text()
        if not label:
            QMessageBox.warning(self, "경고", "레이블을 입력해주세요.")
            return

        # 클래스 ID 할당
        if label not in self.class_mapping:
            self.class_mapping[label] = self.next_class_id
            self.next_class_id += 1
        class_id = self.class_mapping[label]

        xmin = int(self.label_xmin_val.text())
        ymin = int(self.label_ymin_val.text())
        xmax = int(self.label_xmax_val.text())
        ymax = int(self.label_ymax_val.text())

        # YOLO 포맷으로 변환 (중심점 x, y, 너비, 높이)
        x_center = ((xmin + xmax) / 2) / self.original_width
        y_center = ((ymin + ymax) / 2) / self.original_height
        width = (xmax - xmin) / self.original_width
        height = (ymax - ymin) / self.original_height

        # YOLO 포맷으로 txt 파일 저장
        image_name = os.path.basename(self.current_image_path)
        txt_name = os.path.splitext(image_name)[0] + '.txt'
        txt_path = os.path.join(self.image_dir, txt_name)

        with open(txt_path, 'a') as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # 클래스 매핑 정보 저장
        classes_file = os.path.join(self.image_dir, 'classes.txt')
        with open(classes_file, 'w') as f:
            for label, class_id in sorted(self.class_mapping.items(), key=lambda x: x[1]):
                f.write(f"{label}\n")

        # 기존 엑셀 저장 로직
        new_entry = {
            'filename': os.path.basename(self.current_image_path),
            'image_id': self.image_id,
            'label': label,
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'iscrowd': 0
        }

        new_df = pd.DataFrame([new_entry])
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df.to_excel(self.excel_path, index=False)
        
        QMessageBox.information(self, "성공", "데이터가 저장되었습니다.")
        self.image_id += 1
        
        self.create_json_from_excel(self.excel_path, self.image_dir)

    def create_json_from_excel(self, excel_path, image_dir, json_path="annotations.json"):
        """
        엑셀 파일에서 이미지 파일명, 클래스 이름, 박스 좌표를 읽어 JSON 라벨 파일을 생성합니다.

        Args:
            excel_path: 엑셀 파일 경로
            image_dir: 이미지 파일이 있는 디렉토리 경로
            json_path: JSON 파일 경로 (기본값: "annotations.json")
        """

        try:
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            print(f"Error: Excel file not found at {excel_path}")
            return

        data = {"images": []}

        for index, row in df.iterrows():
            image_name = row["filename"]  # 엑셀 파일의 'filename' 열
            image_id = row["image_id"]
            class_name = row["label"]  # 엑셀 파일의 'class' 열
            xmin = row["xmin"]  # 엑셀 파일의 'xmin' 열
            ymin = row["ymin"]  # 엑셀 파일의 'ymin' 열
            xmax = row["xmax"]  # 엑셀 파일의 'xmax' 열
            ymax = row["ymax"]  # 엑셀 파일의 'ymax' 열
            iscrowd = row['iscrowd']    # 엑셀 파일의 'iscrowd' 열

            image_path = os.path.join(image_dir, image_name)

            try:
                image = Image.open(image_path)
                width, height = image.size
            except FileNotFoundError:
                print(f"Warning: Image file not found at {image_path}")
                continue  # 이미지가 없으면 해당 이미지 정보는 건너뜀

            image_info = {
                "filename": image_name,
                "width": width,
                "height": height,
                "objects": []
            }

            obj = {
                "name": class_name,
                "image_id": image_id,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "iscrowd": iscrowd
            }
            image_info["objects"].append(obj)
            data["images"].append(image_info)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)    
    window = MainWindow()
    window.show()
    
    # CI 환경에서 실행 중인지 확인
    if os.environ.get('CI'):
        # 3초 후 앱 종료
        QtCore.QTimer.singleShot(3000, app.quit)
    
    sys.exit(app.exec_()) 