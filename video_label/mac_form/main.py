import os
import sys
import json
import concurrent.futures
import urllib.parse
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QTextOption
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QMessageBox, QLineEdit,
    QSizePolicy, QTextEdit, QSpinBox
)
import threading
from boto3.session import Session
import shutil  # To remove files from cache

from video_player import VideoPlayer


# S3 download function
def download_single_file_from_s3(aws_key, aws_secret_key, bucket, region_name, file_key, download_directory):
    session = Session(aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key, region_name=region_name)
    s3 = session.client("s3")

    os.makedirs(download_directory, exist_ok=True)
    file_name = os.path.basename(file_key)
    download_path = os.path.join(download_directory, file_name)

    try:
        s3.download_file(Bucket=bucket, Key=file_key, Filename=download_path)
        print(f"Downloaded {file_key} to {download_path}")
        return download_path
    except Exception as e:
        print(f"Error downloading {file_key}: {e}")
        return None


# class VideoPlayer(QWidget):
#     def __init__(self, video_path, cache_dir, aws_credentials, max_cache_size=5, fixed_height=300, parent=None):
#         super().__init__(parent)
#         self.video_path = video_path
#         self.cache_dir = cache_dir
#         self.aws_credentials = aws_credentials
#         self.max_cache_size = max_cache_size
#         self.cap = self.init_video_capture(self.video_path)
#         self.video_label = QLabel(self)
#         self.timer = QTimer(self)
#         self.fixed_height = fixed_height
#
#         layout = QVBoxLayout()
#         layout.addWidget(self.video_label)
#         self.setLayout(layout)
#
#         self.timer.timeout.connect(self.update_frame)
#         self.start_video()
#
#     def init_video_capture(self, video_path):
#         if self.is_url(video_path):
#             return cv2.VideoCapture(video_path)
#         elif not os.path.exists(video_path):
#             video_path = self.handle_s3_key(video_path)
#             if video_path is None:
#                 QMessageBox.warning(self, "Error", f"Cannot download or locate video: {self.video_path}")
#                 return None
#         return cv2.VideoCapture(video_path)
#
#     def is_url(self, path):
#         parsed = urllib.parse.urlparse(path)
#         return bool(parsed.scheme and parsed.netloc)
#
#     def handle_s3_key(self, s3_key):
#         cache_path = os.path.join(self.cache_dir, os.path.basename(s3_key))
#         if os.path.exists(cache_path):
#             return cache_path
#
#         if len(os.listdir(self.cache_dir)) >= self.max_cache_size:
#             self.clean_cache()
#
#         # Download from S3
#         download_path = download_single_file_from_s3(
#             self.aws_credentials['aws_key'],
#             self.aws_credentials['aws_secret_key'],
#             self.aws_credentials['bucket'],
#             self.aws_credentials['region'],
#             s3_key,
#             self.cache_dir
#         )
#         return download_path
#
#     def clean_cache(self):
#         cache_files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)]
#         cache_files.sort(key=lambda x: os.path.getmtime(x))  # Sort by modification time
#         if cache_files:
#             oldest_file = cache_files[0]
#             os.remove(oldest_file)
#             print(f"Removed oldest cached video: {oldest_file}")
#
#     def start_video(self):
#         if not self.cap or not self.cap.isOpened():
#             QMessageBox.warning(self, "Error", f"Cannot open video file: {self.video_path}")
#             return
#         self.timer.start(33)
#
#     def update_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h, w, ch = frame.shape
#             scale_factor = self.fixed_height / h
#             new_width = int(w * scale_factor)
#             frame = cv2.resize(frame, (new_width, self.fixed_height))
#             bytes_per_line = ch * new_width
#             image = QImage(frame.data, new_width, self.fixed_height, bytes_per_line, QImage.Format_RGB888)
#             self.video_label.setPixmap(QPixmap.fromImage(image))
#         else:
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#
#     def stop_video(self):
#         self.timer.stop()
#         if self.cap:
#             self.cap.release()
#
#     def closeEvent(self, event):
#         self.stop_video()
#         event.accept()
#

class VideoLabelButton(QPushButton):
    def __init__(self, label_name, video_num, initial_state=None):
        super().__init__(label_name)
        self.label_name = label_name
        self.state = initial_state  # True, False, or None
        self.update_style()
        self.video_num = video_num
        self.clicked.connect(self.toggle_state)

    def toggle_state(self):
        if self.state is None:
            self.state = True
        elif self.state is True:
            self.state = False
        elif self.state is False:
            self.state = True
        else:
            self.state = None
        self.update_style()
        self.parent().update_label(self.label_name, self.state, self.video_num)

    def update_style(self):
        if self.state is True:
            self.setStyleSheet("background-color: green")
            self.setText(f"{self.label_name}: True")
        elif self.state is False:
            self.setStyleSheet("background-color: red")
            self.setText(f"{self.label_name}: False")
        else:
            self.setStyleSheet("background-color: none")
            self.setText(f"{self.label_name}: Null")


class AnnotationTool(QWidget):
    def __init__(self, json_data, json_file_path, aws_credentials):
        super().__init__()
        self.setWindowTitle("视频标注工具")
        self.json_data = json_data
        self.current_index = 0  # Initialize current index
        self.json_file_path = json_file_path
        self.cache_dir = "./cache"  # Directory for caching videos
        self.aws_credentials = aws_credentials
        self.max_cache_size = 100
        self.download_executor = concurrent.futures.ThreadPoolExecutor()  # For concurrent downloads
        self.lock = threading.Lock()  # To prevent concurrent access issues

        os.makedirs(self.cache_dir, exist_ok=True)

        self.init_ui()

        # Start cache management and preloading tasks when initializing
        self.manage_cache_and_download()

    def manage_cache_and_download(self):
        # Generate video list based on current index
        video_list = self.generate_video_list(self.current_index)

        # Clean the cache based on this video list
        self.clean_cache(video_list)

        # Start downloading missing videos concurrently
        self.download_missing_videos(video_list)

    def generate_video_list(self, current_index):
        video_list = []
        total_videos = len(self.json_data)

        # Calculate the 80% forward and 20% backward range
        start_index = max(0, current_index - int(self.max_cache_size * 0.1))
        end_index = min(total_videos, current_index + int(self.max_cache_size * 0.9))

        # Collect the video paths within the range
        for i in range(start_index, end_index):
            video_list.append(self.json_data[i]["video0_body"]["video_path"])
            video_list.append(self.json_data[i]["video1_body"]["video_path"])

        return video_list

    def clean_cache(self, video_list):
        # Get the list of cached files
        cache_files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)]

        # Remove cached files not in the video list
        for cache_file in cache_files:
            # Map cached file back to the S3 key
            cache_file_name = os.path.basename(cache_file)
            if not any(cache_file_name == os.path.basename(video_path) for video_path in video_list):
                os.remove(cache_file)
                print(f"Removed cached video: {cache_file}")

    def download_missing_videos(self, video_list):
        missing_videos = []

        # Identify missing videos that are not in the cache
        for video_path in video_list:
            cache_path = os.path.join(self.cache_dir, os.path.basename(video_path))
            if not os.path.exists(cache_path):
                missing_videos.append(video_path)

        # Download missing videos concurrently
        for video in missing_videos:
            self.download_executor.submit(self.download_single_file_from_s3, video)

    def download_single_file_from_s3(self, s3_key):
        # Reuse the download function to download from S3
        with self.lock:  # Ensure no race conditions with shared resources
            download_single_file_from_s3(
                self.aws_credentials['aws_key'],
                self.aws_credentials['aws_secret_key'],
                self.aws_credentials['bucket'],
                self.aws_credentials['region'],
                s3_key,
                self.cache_dir
            )
        print(f"Downloaded and cached {s3_key}")

    def next_video_pair(self):
        if self.current_index < len(self.json_data) - 1:
            self.current_index += 1
            self.load_current_pair()

            # Preload cache and download new videos in background
            threading.Thread(target=self.manage_cache_and_download).start()
        else:
            QMessageBox.information(self, "提示", "已经是最后一个视频对了。")

    def prev_video_pair(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_pair()

            # Preload cache and download new videos in background
            threading.Thread(target=self.manage_cache_and_download).start()
        else:
            QMessageBox.information(self, "提示", "已经是第一个视频对了。")

    def init_ui(self):
        main_layout = QVBoxLayout()

        caption_layout = QHBoxLayout()
        self.caption_edit = QTextEdit(self.json_data[self.current_index]["caption"])
        self.caption_edit.setPlaceholderText("Caption")
        self.caption_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.caption_edit.setMaximumHeight(100)  # 设置最大高度，防止过高
        self.caption_edit.setMinimumHeight(50)  # 设置最小高度
        self.caption_edit.setWordWrapMode(QTextOption.WordWrap)  # 允许自动换行

        subset_label = QLabel(f"Subset: {self.json_data[self.current_index]['subset']}")
        subset_label.setAlignment(Qt.AlignLeft)

        caption_layout.addWidget(QLabel("Caption:"))
        caption_layout.addWidget(self.caption_edit)
        caption_layout.addWidget(subset_label)
        caption_layout.setAlignment(Qt.AlignTop)
        main_layout.addLayout(caption_layout)

        videos_layout = QHBoxLayout()
        fixed_height = 300

        self.video0_player = VideoPlayer(
            self.get_current_video_path(0),
            self.cache_dir,
            self.aws_credentials,
            max_cache_size=self.max_cache_size,
            fixed_height=fixed_height
        )
        self.video0_player.setMinimumSize(400, fixed_height)
        self.video1_player = VideoPlayer(
            self.get_current_video_path(1),
            self.cache_dir,
            self.aws_credentials,
            max_cache_size=self.max_cache_size,
            fixed_height=fixed_height
        )
        self.video1_player.setMinimumSize(400, fixed_height)

        videos_layout.addWidget(self.video0_player)
        videos_layout.addWidget(self.video1_player)
        main_layout.addLayout(videos_layout)

        labels_layout = QHBoxLayout()
        self.labels_video0 = self.create_labels(self.get_current_label(0), 0)
        self.labels_video1 = self.create_labels(self.get_current_label(1), 1)
        labels_layout.addLayout(self.labels_video0)
        labels_layout.addLayout(self.labels_video1)
        main_layout.addLayout(labels_layout)

        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("上一个")
        self.next_button = QPushButton("下一个")
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        main_layout.addLayout(nav_layout)

        self.setLayout(main_layout)

        self.prev_button.clicked.connect(self.prev_video_pair)
        self.next_button.clicked.connect(self.next_video_pair)

        # 在导航布局中添加索引输入框
        self.index_input = QSpinBox()
        self.index_input.setRange(0, len(self.json_data) - 1)  # 设置范围，限制在数据长度之内
        self.index_input.setValue(self.current_index)  # 默认值为当前索引

        # 添加跳转按钮
        self.jump_button = QPushButton("跳转")
        self.jump_button.clicked.connect(self.jump_to_index)  # 连接跳转函数

        nav_layout.addWidget(self.index_input)
        nav_layout.addWidget(self.jump_button)

    def jump_to_index(self):
        new_index = self.index_input.value()  # 获取输入框中的值
        if new_index != self.current_index:
            self.current_index = new_index
            self.load_current_pair()  # 加载新的视频对
        else:
            QMessageBox.information(self, "提示", "已经是当前视频对。")

    def create_labels(self, labels, video_num):
        layout = QVBoxLayout()
        for label_name, state in labels.items():
            btn = VideoLabelButton(label_name, video_num=video_num, initial_state=state)
            layout.addWidget(btn)
        return layout

    def get_current_video_path(self, video_num):
        key = f"video{video_num}_body"
        return self.json_data[self.current_index][key]["video_path"]

    def get_current_label(self, video_num):
        key = f"video{video_num}_body"
        return self.json_data[self.current_index][key]["label"]

    def update_label(self, label_name, state, video_num):
        key = f"video{video_num}_body"
        self.json_data[self.current_index][key]["label"][label_name] = state
        self.save_data()

    def update_caption(self, new_caption):
        self.json_data[self.current_index]["caption"] = new_caption
        self.save_data()

    def prev_video_pair(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_pair()
        else:
            QMessageBox.information(self, "提示", "已经是第一个视频对了。")

    def next_video_pair(self):
        if self.current_index < len(self.json_data) - 1:
            self.current_index += 1
            self.load_current_pair()
        else:
            QMessageBox.information(self, "提示", "已经是最后一个视频对了。")

    def load_current_pair(self):
        # self.video0_player.stop_video()
        # self.video1_player.stop_video()

        videos_layout = self.layout().itemAt(1).layout()
        for i in reversed(range(videos_layout.count())):
            widget = videos_layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()


        fixed_height = 300
        self.video0_player = VideoPlayer(self.get_current_video_path(0), self.cache_dir, self.aws_credentials,
                                         max_cache_size=self.max_cache_size, fixed_height=fixed_height)
        self.video1_player = VideoPlayer(self.get_current_video_path(1), self.cache_dir, self.aws_credentials,
                                         max_cache_size=self.max_cache_size, fixed_height=fixed_height)
        self.video0_player.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video1_player.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        videos_layout.addWidget(self.video0_player)
        videos_layout.addWidget(self.video1_player)


        labels_layout = self.layout().itemAt(2).layout()
        for i in reversed(range(labels_layout.count())):
            label_layout = labels_layout.itemAt(i).layout()
            if label_layout is not None:
                while label_layout.count():
                    widget = label_layout.takeAt(0).widget()
                    if widget is not None:
                        widget.deleteLater()
                labels_layout.removeItem(label_layout)

        self.labels_video0 = self.create_labels(self.get_current_label(0), 0)
        self.labels_video1 = self.create_labels(self.get_current_label(1), 1)
        labels_layout.addLayout(self.labels_video0)
        labels_layout.addLayout(self.labels_video1)

        self.caption_edit.setText(self.json_data[self.current_index]["caption"])

    def save_data(self):
        try:
            with open(self.json_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.json_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {e}")


def load_json_file():
    options = QFileDialog.Options()
    fileName, _ = QFileDialog.getOpenFileName(None, "选择标注数据文件", "", "JSON Files (*.json);;All Files (*)",
                                              options=options)
    if fileName:
        try:
            with open(fileName, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data, fileName
        except Exception as e:
            QMessageBox.warning(None, "错误", f"加载JSON失败: {e}")
            sys.exit()
    else:
        sys.exit()


def main():
    app = QApplication(sys.argv)

    aws_credentials = {
        'aws_key': "AKIAZAI4HFAL2ABLMM6J",
        'aws_secret_key': "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",
        'bucket': "mjbench2.0",
        'region': "us-east-2"
    }

    json_data, json_file_path = load_json_file()

    if not isinstance(json_data, list):
        json_data = [json_data]

    window = AnnotationTool(json_data, json_file_path, aws_credentials)
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
