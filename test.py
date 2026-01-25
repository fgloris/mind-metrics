import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

from transformers import AutoModel, AutoImageProcessor
from transformers.image_utils import load_image

# 本地模型路径
model_path = "/home/wjp/Documents/Metric/dinov3/dinov3_vitb16"

# 加载本地图片
image_path = "/home/wjp/Documents/Metric/pipeline-cat-chonk.jpeg"
image = load_image(image_path)

# 加载模型和processor（使用AutoModel而不是pipeline）
print("加载DINOv3模型...")
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
model = model.cuda()
model.eval()

# 处理图像
inputs = processor(images=image, return_tensors="pt").to(model.device)

# 打印图像信息
print(f"\n{'='*60}")
print(f"Image loaded: {image_path}")
print(f"Image size: {image.size}")
print(f"{'='*60}")

# 获取最后一层transformer输出特征
print(f"\n提取Transformer特征...")
with torch.inference_mode():
    outputs = model(**inputs, output_hidden_states=True)

print(f"\n{'='*60}")
print("Transformer输出结构")
print(f"{'='*60}")

# 1. 查看outputs的所有属性
print(f"\n1. Outputs可用属性:")
for attr in dir(outputs):
    if not attr.startswith('_'):
        print(f"   - {attr}")

# 2. 获取最后一层hidden states
last_hidden_state = outputs.last_hidden_state  # [1, 201, 768]
print(f"\n2. 最后一层Hidden States:")
print(f"   Shape: {last_hidden_state.shape}")
print(f"   [batch_size={last_hidden_state.shape[0]}, seq_len={last_hidden_state.shape[1]}, hidden_dim={last_hidden_state.shape[2]}]")

# 3. 获取所有层的hidden states
all_hidden_states = outputs.hidden_states
print(f"\n3. 所有层Hidden States:")
print(f"   层数: {len(all_hidden_states)}")
for i, hidden_state in enumerate(all_hidden_states):
    print(f"   Layer {i}: {hidden_state.shape}")

# 4. 提取最后一层特征
features = last_hidden_state.squeeze(0)  # [201, 768]
features = features[5:, :]  # [196, 768]
feature_map_minmax = (features - features.min()) / (features.max() - features.min())

# 5. 将 [196, 768] 特征矩阵直接当作图像显示
feature_img = feature_map_minmax.cpu().numpy()  # [196, 768]

plt.figure(figsize=(15, 8))
plt.imshow(feature_img, cmap='gray', aspect='auto')
plt.colorbar(label='Normalized Feature Value')
plt.xlabel('Feature Dimension (768)')
plt.ylabel('Patch Tokens (196)')
plt.title('Feature Map Visualization [196 patches × 768 dimensions]')
plt.tight_layout()
plt.savefig('/home/wjp/Documents/Metric/feature_map_196x768.png', dpi=150, bbox_inches='tight')
print(f"\n可视化已保存到: /home/wjp/Documents/Metric/feature_map_196x768.png")
print(f"特征图形状: {feature_img.shape}")
