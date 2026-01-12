import os
import torch
from PIL import Image, ImageSequence
# from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import functional as F
import json
import numpy as np
import cv2
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

CACHE_DIR = os.environ.get('VBENCH_CACHE_DIR')
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'vbench')

def transform_image(images):
    # 输入: images (Tensor) - 形状为 [B, C, H, W] 输出: 归一化后的 [B, C, 1280, 720] Tensor
    b, c, h, w = images.size()
    if h * 1280 == w * 720:
        images = transforms.Resize(size=(720, 1280), antialias=False)(images)
        return images / 255.0
    else:
        scale = max(1280.0/w, 720.0/h)
        images = transforms.Resize(size=(round(h*scale), round(w*scale)), antialias=False)(images)
        images = F.center_crop(images, (720, 1280))
        return images / 255.0

def min_max_normalization(data):
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(value - min_value) / (max_value - min_value) for value in data]
    return normalized_data

def save_video(tensor, output_path):
    frames = tensor.permute(0, 2, 3, 1).numpy()
    frames = (frames * 255).astype(np.uint8)
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
    
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    cv2.destroyAllWindows()

def expand_to_batch_dim(tensor, batch_size):
    return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

from torchvision.io import read_video
def load_sample_video(video_path, mark_time, total_time, max_time = 200) -> torch.Tensor:
    video_length = total_time - mark_time
    frames,_,_ = read_video(video_path, pts_unit='sec')
    frames = frames.permute(0, 3, 1, 2)[:min(max_time, video_length)]
    # print("sample video length:", len(frames))
    return transform_image(frames)

def load_gt_video(video_path, mark_time, total_time, max_time = 200) -> torch.Tensor:
    start_time = (mark_time-12) / 24
    frames,_,_ = read_video(video_path, pts_unit='sec', start_pts=start_time)
    
    frames = frames.permute(0, 3, 1, 2)
    video_length = total_time - mark_time
    frames = frames[len(frames) - video_length:]
    if video_length > max_time:
        frames = frames[:max_time]
    return transform_image(frames)

def load_time_from_json(json_path):
    with open(json_path, 'r') as f:
        item = json.load(f)
    return item['mark_time'], item['total_time']

def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
