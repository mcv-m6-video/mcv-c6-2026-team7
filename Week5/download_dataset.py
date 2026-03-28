import os
import zipfile
from SoccerNet.Downloader import SoccerNetDownloader

# 1. Download
data_dir = "./data/soccernetball"
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=data_dir)
password = "" # PUT YOUR PASSWORD HERE
splits = ["train", "valid", "test"]

mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2025", split=splits, password=password)

