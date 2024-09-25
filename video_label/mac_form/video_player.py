import os
import sys
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication, QMessageBox
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl

from boto3.session import Session

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

class VideoPlayer(QWidget):
    def __init__(self, video_path, cache_dir, aws_credentials, max_cache_size=5, fixed_height=300, parent=None):
        super().__init__(parent)

        self.cache_dir = cache_dir
        self.aws_credentials = aws_credentials
        self.max_cache_size = max_cache_size
        self.fixed_height = fixed_height

        # 初始化视频播放器组件
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()
        # 设置高度

        # 设置视频输出
        self.mediaPlayer.setVideoOutput(self.videoWidget)

        # 创建布局并添加视频窗口
        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        self.setLayout(layout)

        # 加载并自动播放视频
        self.load_and_play_video(video_path)

        # 连接stateChanged信号以处理视频播放结束时的重播
        self.mediaPlayer.stateChanged.connect(self.handle_state_changed)

    def load_and_play_video(self, video_path):
        if not os.path.exists(video_path):
            video_path = self.handle_s3_key(video_path)
            if video_path is None:
                QMessageBox.warning(self, "Error", f"Cannot download or locate video: {self.video_path}")
                return None
        video_path = os.path.join(os.getcwd(), video_path)
        # 加载并设置媒体内容
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        # 自动播放视频
        self.mediaPlayer.play()

    def handle_s3_key(self, s3_key):
        cache_path = os.path.join(self.cache_dir, os.path.basename(s3_key))
        if os.path.exists(cache_path):
            return cache_path

        if len(os.listdir(self.cache_dir)) >= self.max_cache_size:
            self.clean_cache()

        # Download from S3
        download_path = download_single_file_from_s3(
            self.aws_credentials['aws_key'],
            self.aws_credentials['aws_secret_key'],
            self.aws_credentials['bucket'],
            self.aws_credentials['region'],
            s3_key,
            self.cache_dir
        )
        return download_path

    def handle_state_changed(self, state):
        # 检查是否播放完成并重播视频
        if state == QMediaPlayer.StoppedState:
            self.mediaPlayer.play()


def main():
    app = QApplication(sys.argv)
    video_path = "/Users/cishoon/WorkPlace/thb-video-annotation-tool/videos/alignment_image1_3.mp4"  # 替换为实际的视频文件路径
    player = VideoPlayer(video_path)
    player.setWindowTitle("Video Player")
    player.resize(800, 600)
    player.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
