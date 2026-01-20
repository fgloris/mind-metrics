import lpips
import torch
from pyiqa.archs.musiq_arch import MUSIQ
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils import load_gt_video, load_sample_video, load_time_from_json, print_gpu_memory, get_musiq_spaq_path, get_vitl_path, get_aes_path, clip_transform_Image, extract_actions_from_json
from vipe_utils import extract_traj
import vipe_utils
from tqdm import tqdm
import torch.nn.functional as F
import json
import clip
import os
import numpy as np
from pathlib import Path
import subprocess
import hashlib

def crop_video_frames(video_path, max_frames, cache_dir, start_frame=0):
    """
    Crop video to first max_frames frames using ffmpeg.
    Returns path to cropped video (from cache if exists).
    
    Args:
        video_path: Path to original video
        max_frames: Number of frames to keep
        cache_dir: Directory for caching cropped videos
        start_frame: Starting frame position for cropping (default: 0)
    
    Returns:
        Path to cropped video file
    """
    video_path = Path(video_path).resolve()
    cache_dir = Path(cache_dir).resolve()
    
    # Generate cache filename based on original path and max_frames
    video_hash = hashlib.sha1(str(video_path).encode()).hexdigest()[:12]
    cache_filename = f"{video_path.stem}_{video_hash}_start{start_frame}_frames{max_frames}.mp4"
    cache_path = cache_dir / "cropped_videos" / cache_filename
    
    # Check if cached version exists
    if cache_path.exists():
        print(f"Using cached cropped video: {cache_path}")
        return cache_path
    
    # Create cache directory
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg to crop video
    print(f"Cropping video {video_path.name} to {max_frames} frames starting from frame {start_frame}...")
    
    # If start_frame is 0, we can use -vframes directly
    # Otherwise, we need to use -ss (seek) and -vframes
    if start_frame == 0:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vframes", str(max_frames),
            "-c:v", "libx264", "-crf", "18",
            "-y", str(cache_path)
        ]
    else:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"select=gte(n\,{start_frame})",  # 从指定帧开始
            "-vframes", str(max_frames),
            "-c:v", "libx264",
            "-crf", "18",
            "-y", str(cache_path)
        ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        print(f"Cropped video saved to: {cache_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error cropping video: {e.stderr}")
        raise
    
    return cache_path

def lcm_metric(pred, gt, requested_metrics, 
               lpips_metric, ssim_metric, psnr_metric, batch_size=100):
    f, c, h, w = pred.size()
    assert(torch.all(pred >= 0.0) and torch.all(pred <= 1.0))
    result_dict = {'length': f}

    # ------------------- MSE (无需模型) -------------------
    if 'mse' in requested_metrics:
        diff = (pred - gt) ** 2
        mse_per_sample = diff.reshape(f, -1).mean(dim=1).cpu().tolist()
        result_dict['mse'] = mse_per_sample
        result_dict['avg_mse'] = sum(mse_per_sample) / len(mse_per_sample)
        del diff

    # ------------------- PSNR (按需加载) -------------------
    if 'psnr' in requested_metrics:
        torch.cuda.synchronize()
        with torch.no_grad():
            psnr_list = []
            for i in range(0, f, batch_size):
                psnr_list.extend(psnr_metric(pred[i:i+batch_size], gt[i:i+batch_size]).cpu().tolist())
            result_dict['psnr'] = psnr_list
            result_dict['avg_psnr'] = sum(psnr_list) / len(psnr_list)

    # ------------------- SSIM (按需加载) -------------------
    if 'ssim' in requested_metrics:
        torch.cuda.synchronize()
        with torch.no_grad():
            ssim_list = []
            for i in range(0, f, batch_size):
                ssim_list.extend(ssim_metric(pred[i:i+batch_size], gt[i:i+batch_size]).cpu().tolist())
            result_dict['ssim'] = ssim_list
            result_dict['avg_ssim'] = sum(ssim_list) / len(ssim_list)

    # ------------------- LPIPS (按需加载) -------------------
    if 'lpips' in requested_metrics:
        torch.cuda.synchronize()
        with torch.no_grad():
            lpips_list = []
            for i in range(0, f, batch_size):
                # LPIPS 输入需要 [-1, 1] 范围
                lpips_batch = lpips_metric((pred[i:i+batch_size] * 2 - 1), (gt[i:i+batch_size] * 2 - 1)).cpu().tolist()
                lpips_batch = [item[0][0][0] for item in lpips_batch]
                lpips_list.extend(lpips_batch)
            result_dict['lpips'] = lpips_list
            result_dict['avg_lpips'] = sum(lpips_list) / len(lpips_list)

    return result_dict

def visual_quality_metric(images, imaging_model, aesthetic_model, clip_model, batch_size=8):
    result_dict = {}
    image_transform = clip_transform_Image(224)
    scores = []
    for i in range(0, len(images), batch_size):
        frame = images[i:i+batch_size]
        imaging_batch = imaging_model(frame).cpu().tolist()
        imaging_batch = [item[0] for item in imaging_batch]
        scores.extend(imaging_batch)
    result_dict["imaging"] = scores
    result_dict["avg_imaging"] = sum(scores) / len(scores)

    scores = []
    for i in range(0, len(images), batch_size):
        batch = images[i: i+batch_size]
        batch = image_transform(batch)
        image_feats = clip_model.encode_image(batch).to(torch.float32)
        image_feats = F.normalize(image_feats, dim=-1, p=2)
        aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)
        scores.extend(aesthetic_scores.cpu().tolist())
    result_dict["aesthetic"] = scores
    result_dict["avg_aesthetic"] = sum(scores) / len(scores)
    return result_dict

def action_accuracy_metric(pred_vid_path, gt_vid_path, mark_time, actions, cache_dir = Path("./cache_vipe"),
                           delta = 1, max_rot_deg = 179.0, max_trans = 1e9, min_valid_steps = 8, per_action = True, max_frames = 200):
    """
    Compute action accuracy metrics for a single video pair.

    Args:
        pred_vid_path: Path to predicted/generated video
        gt_vid_path: Path to ground truth video
        actions: List of action strings
        cache_dir: Directory for caching ViPE outputs
        delta: Frame delta for RPE computation
        max_rot_deg: Maximum rotation error threshold (filter outliers)
        max_trans: Maximum translation error threshold (filter outliers)
        min_valid_steps: Minimum number of valid steps required
        per_action: Whether to include per-action breakdown
        max_frames: Maximum number of frames to use from videos (None = use all frames)

    Returns:
        Dictionary with action accuracy metrics in JSON format, or None if failed
    """

    # Crop videos to max_frames if specified
    print(f"Cropping videos to {max_frames} frames...")
    pred_vid_path = crop_video_frames(pred_vid_path, max_frames, cache_dir, start_frame=mark_time)
    gt_vid_path = crop_video_frames(gt_vid_path, max_frames, cache_dir)

    gt_T = extract_traj(
        gt_vid_path, cache_dir,
    )
    gen_T = extract_traj(
        pred_vid_path, cache_dir,
    )

    # Align lengths to actions (+delta poses)
    n_steps = min(len(actions), len(gt_T) - delta, len(gen_T) - delta)
    if n_steps <= 0:
        print(f"Not enough steps after alignment for data {pred_vid_path}: n_steps={n_steps}")
        return None
    gt_use = gt_T[:(n_steps + delta)]
    gen_use = gen_T[:(n_steps + delta)]
    actions = actions[:n_steps]

    # Sim(3) align gen->gt (positions)
    s, R, t = vipe_utils.umeyama_sim3(gen_use[:, :3, 3], gt_use[:, :3, 3])
    gen_aligned = vipe_utils.apply_sim3_to_poses(gen_use, s, R, t)

    # RPE
    te, re = vipe_utils.compute_rpe(gt_use, gen_aligned, delta=delta)

    # Filter obvious pose failures
    valid = np.ones_like(te, dtype=bool)
    if max_rot_deg > 0:
        valid &= (re <= max_rot_deg)
    if max_trans > 0:
        valid &= (te <= max_trans)

    te = te[valid]
    re = re[valid]
    actions_valid = [a for (a, ok) in zip(actions, valid.tolist()) if ok]

    if te.size < min_valid_steps:
        print(f"Too few valid RPE samples for data {pred_vid_path}: te.size={te.size}, min_valid_steps={min_valid_steps}")
        return None

    # Helper function to compute statistics
    def compute_stats(te_arr, re_arr):
        return {
            "count": int(te_arr.size),
            "rpe_trans_mean": float(te_arr.mean()),
            "rpe_trans_median": float(np.median(te_arr)),
            "rpe_rot_mean_deg": float(re_arr.mean()),
            "rpe_rot_median_deg": float(np.median(re_arr)),
        }

    # Create result dictionary
    result = {}

    # Overall
    result["__overall__"] = compute_stats(te, re)

    # Group buckets (translation/rotation/other)
    groups = [vipe_utils.bucket_for_action(a) for a in actions_valid]
    for grp in {"translation", "rotation", "other"}:
        idx = [i for i, g in enumerate(groups) if g == grp]
        if idx:
            result[grp] = compute_stats(te[idx], re[idx])

    # Optional per-action breakdown
    if per_action:
        uniq = sorted(set(actions_valid))
        for a in uniq:
            idx = [i for i, aa in enumerate(actions_valid) if aa == a]
            result[f"act:{a}"] = compute_stats(te[idx], re[idx])
            
    print(result)
    return result

def compute_metrics(gt_root, test_root, requested_metrics=['mse','psnr','ssim','lpips'], video_max_time=100, process_batch_size=10, device='cuda:0'):
    lpips_metric = lpips.LPIPS(net='alex', spatial=False).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1, 2, 3]).to(device)
    imaging_model = MUSIQ(pretrained_model_path=get_musiq_spaq_path())
    aesthetic_model = torch.nn.Linear(768, 1)
    clip_model, preprocess = clip.load(get_vitl_path(), device=device)

    s = torch.load(get_aes_path(), weights_only=False)
    aesthetic_model.load_state_dict(s)
    aesthetic_model.to(device)
    aesthetic_model.eval()

    imaging_model.to(device)
    imaging_model.training = False
    result_dict = {
        '1st_data': {
            'mem_test': {},
            'action_space_test': {}
        },
        '3rd_data': {
            'mem_test': {},
            'action_space_test': {}
        }
    }

    for perspective in ['1st_data', '3rd_data']:
        for test_type in ['mem_test', 'action_space_test']:
            gt_dir = os.path.join(gt_root, perspective, 'test', test_type)
            test_dir = os.path.join(test_root, perspective, test_type)
            pbar = tqdm(os.listdir(gt_dir))
            pbar.set_description(f"Computing {test_type} on {perspective}")
            if test_type == 'mem_test':
                # 为了测试action, 暂时跳过
                continue
                for data in pbar:
                    if not os.path.exists(os.path.join(test_dir, data)):
                        continue
                    pbar.set_postfix({"file": data})
                    mark_time, total_time = load_time_from_json(os.path.join(gt_dir, data, 'action.json'))

                    # 读入视频
                    gt_imgs = load_gt_video(os.path.join(gt_dir, data, 'video.mp4'), mark_time, total_time, video_max_time)
                    gt_imgs = gt_imgs.to(device)

                    sample_imgs = load_sample_video(os.path.join(test_dir, data, 'video.mp4'), mark_time, total_time, video_max_time)
                    sample_imgs = sample_imgs.to(device)

                    # 计算LCM指标
                    lcm = lcm_metric(sample_imgs, gt_imgs, requested_metrics, lpips_metric, ssim_metric, psnr_metric, process_batch_size)
                    result_dict[perspective][test_type][data] = lcm

                    # 计算image_quality指标
                    vq = visual_quality_metric(sample_imgs, imaging_model, aesthetic_model, clip_model)
                    result_dict[perspective][test_type][data] = vq #.update(vq)
                    
                    # 清理内存
                    del sample_imgs, gt_imgs
                    torch.cuda.empty_cache()
                    break
            elif test_type == 'action_space_test':
                # 使用 action_accuracy_metric
                for data in pbar:
                    mark_time, total_time = load_time_from_json(os.path.join(gt_dir, data, 'action.json'))
                    actions = extract_actions_from_json(os.path.join(gt_dir, data, 'action.json'), mark_time, video_max_time)
                    action = action_accuracy_metric(
                        os.path.join(test_dir, data, 'video.mp4'),
                        os.path.join(gt_dir, data, 'video.mp4'),
                        mark_time,
                        actions,
                        max_frames = video_max_time
                    )
                    result_dict[perspective][test_type][data] = action
                    break # 先测试一组

    return result_dict

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io._video_deprecation_warning")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    video_results = compute_metrics('../MIND-Data', '/media/wjp/gingerBackup/mind/structured_baselines/i2v')
    from datetime import datetime
    with open(f'result_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.json', 'w') as f:
        json.dump(video_results, f, indent=2)
    