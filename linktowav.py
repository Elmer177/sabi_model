import os
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
youtube_url = "https://youtu.be/QcpjV43ALCA?si=s0YnZs6q5GNOkH4K"
output_dir = "./my_wav_files"

download_youtube_as_wav(youtube_url, output_dir)