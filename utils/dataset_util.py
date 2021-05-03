import os
import cv2
import csv
from pathlib import Path
from argparse import ArgumentParser

"""
dataset folder structure
ROOT_DIR
    |__Label1
    |  |__vid1
    |  |__vid2
    |  |__vid3
    |
    |__Label2
    |  |__vid1
    |  |__vid2
    |  |__vid3
    |
    |__Label3
       |__vid1
       |__vid2
"""


def dataset_to_csv(opts):
    root_data_dir = Path(opts.root_dir)
    with open('../data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for label in os.listdir(root_data_dir.as_posix()):
            label_path = root_data_dir.joinpath(label)
            for video in os.listdir(label_path.as_posix()):
                video_path = label_path.joinpath(video)
                vid = cv2.VideoCapture(video_path.as_posix())
                num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
                writer.writerow([video_path.as_posix(), label, num_frames])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--root_dir", help="Root directory of dataset", required=True)
    args = parser.parse_args()
    dataset_to_csv(args)
