import csv
import cv2
import numpy
import torch
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, CenterCrop, ToTensor


class VideoDataset(Dataset):
    def __init__(self, csv_file, sequence_length, stride, transforms):
        self.csv_file = csv_file
        self.labels, self.data = self._parse_csv()
        self.sequence_len = sequence_length
        self.stride = stride
        self.transforms = transforms

    def _parse_csv(self):
        data = []
        labels = []
        with open(self.csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                labels.append(row[1])
                data.append((row[0], row[1], row[2]))
        labels = set(labels)
        labels = {l: idx for idx, l in enumerate(labels)}
        return labels, data

    def __getitem__(self, index):
        video_path, label, num_frames = self.data[index]
        num_seq = int((int(num_frames) - self.sequence_len) / self.stride) + 1
        start_idx = random.choice(range(num_seq))
        frames_idx = list(range(0, int(num_frames)))[start_idx::self.stride]
        frames = self._get_frames(video_path, int(num_frames), frames_idx)
        return frames, self.labels[label]

    def _get_frames(self, video_path, num_frames, frames_idx):
        frames = []
        cap = cv2.VideoCapture(video_path)
        for i in range(num_frames):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i in frames_idx:
                pil_image = Image.fromarray(frame)
                pil_image = self.transforms(pil_image)
                frames.append(numpy.asarray(pil_image))
            if len(frames) == self.sequence_len:
                break
        return torch.FloatTensor(frames)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    transforms = transforms.Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor()
    ])
    dataset = VideoDataset('./data.csv', sequence_length=16, stride=2, transforms=transforms)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for idx, (images, labels) in enumerate(train_loader):
        print(images.shape, labels.shape)

