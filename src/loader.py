import os
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader


# Searches through every file in the provided search directory
# and gets all the files which matches a pattern.
def get_files_matching(search_dir: str, rx: str):
    result = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if re.match(rx, file):
                file_path = os.path.join(root, file)
                result.append(file_path)
    return result


# Transform the image to a predefined size by adding padding to bottom and left.
class PaddingTransform:
    def __init__(self, padded_width: int = 5000, padded_height: int = 5000):
        self.padded_width = padded_width
        self.padded_height = padded_height

    def apply(self, im_mic, im_markers):
        height_diff, width_diff = self.padded_height - im_mic.shape[0], self.padded_width - im_mic.shape[1]
        padded_im_mic = np.pad(im_mic, ((0, height_diff), (0, width_diff), (0, 0)), 'constant')
        padded_im_markers = np.pad(im_markers, ((0, height_diff), (0, width_diff)), 'constant')
        return padded_im_mic, padded_im_markers


# Implementing the dataset loader for the provided dataset's file structure.
class ClassificationDataset(Dataset):
    def __init__(self, search_dir: str, same_size_transform = PaddingTransform()):
        self.files = get_files_matching(search_dir, '.*_seg.npz$')
        self.same_size_transform = same_size_transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int):
        seg = np.load(self.files[idx])
        im_mic = np.array(seg['im_mic'], dtype=np.uint8)
        im_markers = np.array(seg['im_markers'], dtype=np.uint8)
        return self.same_size_transform.apply(im_mic, im_markers)


# Creates a PyTorch dataloader from the provided dataset.
def create_dataloader(data: Dataset, batch_size=1, shuffle=True):
    return DataLoader(data, batch_size, shuffle)

