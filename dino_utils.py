import torch

from transformers import AutoModel, AutoImageProcessor

# 全局变量保存模型和processor
_dino_model = None
_dino_processor = None
_model_path = "/home/wjp/Documents/Metric/dinov3/dinov3_vitb16"

def load_dinov3_model(device='cuda:0'):
    global _dino_model, _dino_processor

    if _dino_model is None:
        print("加载DINOv3模型...")
        _dino_processor = AutoImageProcessor.from_pretrained(_model_path)
        _dino_model = AutoModel.from_pretrained(_model_path)
        _dino_model = _dino_model.to(device)
        _dino_model.eval()

    return _dino_model, _dino_processor

def extract_dinov3_features(frames: torch.Tensor, model=None, processor=None, device='cuda:0', batch_size=8) -> torch.Tensor:
    """
    从视频帧中提取DINOv3特征

    Args:
        frames: 视频帧张量，形状 [f, c, h, w]，范围 [0, 1]
        model: DINOv3模型（如果为None，会自动加载）
        processor: DINOv3 processor（如果为None，会自动加载）
        device: 计算设备
        batch_size: 批处理大小

    Returns:
        features: DINOv3特征张量，形状 [f, 196, 768]
    """
    if model is None or processor is None:
        model, processor = load_dinov3_model(device)

    f = frames.shape[0]
    features_list = []

    # DINOv3期望输入范围:[0,1]

    with torch.inference_mode():
        for i in range(0, f, batch_size):
            batch_frames = frames[i:i+batch_size]  # [batch, c, h, w]

            # 直接使用tensor，processor会自动处理归一化
            inputs = processor(images=batch_frames, return_tensors="pt").to(device)

            # 提取特征
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # [batch, 201, 768]

            # 提取patch tokens (跳过前5个token)
            patch_tokens = last_hidden_state[:, 5:, :]  # [batch, 196, 768]

            # L2 normalization
            # patch_tokens_norm = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)

            features_list.append(patch_tokens)

    features = torch.cat(features_list, dim=0)  # [f, 196, 768]

    return features