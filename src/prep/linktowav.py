import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ を import パスに追加
from paths import MY_WAV_DIR
from yt_dlp import YoutubeDL

def download_youtube_as_wav(url, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"ディレクトリを作成しました: {output_directory}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_directory, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'cookiesfrombrowser': ('chrome',),
        'remote_components': ['ejs:github'],
        'extractor_args': {
            'youtube': {
                'player_client': ['web']
            }
        }
    }

    print("ダウンロードと変換を開始")
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("完了しました！")

# 使い方
youtube_url = "https://www.youtube.com/watch?v=lw51UR6Qoy8"
output_dir = MY_WAV_DIR  # プロジェクト直下の my_wav_files

download_youtube_as_wav(youtube_url, output_dir)