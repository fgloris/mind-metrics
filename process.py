import lpips
import torch
from pyiqa.archs.musiq_arch import MUSIQ
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils import load_gt_video, load_sample_video, load_time_from_json, print_gpu_memory, get_musiq_spaq_path, get_vitl_path, get_aes_path, clip_transform_Image, extract_actions_from_json, crop_video_frames, ensure_all_models_downloaded
from vipe_utils import extract_traj
from dino_utils import load_dinov3_model, extract_dinov3_features
import vipe_utils
from tqdm import tqdm
import torch.nn.functional as F
import json
import clip
import os
import numpy as np
from pathlib import Path
import torch.multiprocessing as mp
import warnings


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
    scores = []
    for i in range(0, len(images), batch_size):
        frame = images[i:i+batch_size]
        imaging_batch = imaging_model(frame).cpu().tolist()
        imaging_batch = [item[0] for item in imaging_batch]
        scores.extend(imaging_batch)
    scores = [i/100 for i in scores]
    result_dict["imaging"] = scores
    result_dict["avg_imaging"] = sum(scores) / len(scores)

    scores = []
    image_transform = clip_transform_Image(224)
    for i in range(0, len(images), batch_size):
        batch = images[i: i+batch_size]
        batch = image_transform(batch)
        image_feats = clip_model.encode_image(batch).to(torch.float32)
        image_feats = F.normalize(image_feats, dim=-1, p=2)
        aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)
        scores.extend(aesthetic_scores.cpu().tolist())
    scores = [i/10 for i in scores]
    result_dict["aesthetic"] = scores
    result_dict["avg_aesthetic"] = sum(scores) / len(scores)
    return result_dict

def dino_mse_metric(pred_frames, gt_frames, dino_model=None, dino_processor=None, device='cuda:0', batch_size=8):
    """
    计算DINOv3特征的MSE指标

    Args:
        pred_frames: 预测视频帧，形状 [f, c, h, w]，范围 [0, 1]
        gt_frames: ground truth视频帧，形状 [f, c, h, w]，范围 [0, 1]
        dino_model: DINOv3模型（如果为None，会自动加载）
        dino_processor: DINOv3 processor（如果为None，会自动加载）
        device: 计算设备
        batch_size: 批处理大小

    Returns:
        result_dict: 包含dino_mse相关指标的字典
    """
    result_dict = {}

    # 提取GT帧的DINOv3特征
    gt_features = extract_dinov3_features(
        gt_frames,
        model=dino_model,
        processor=dino_processor,
        device=device,
        batch_size=batch_size
    )  # [f, 196, 768]

    # 提取预测帧的DINOv3特征
    pred_features = extract_dinov3_features(
        pred_frames,
        model=dino_model,
        processor=dino_processor,
        device=device,
        batch_size=batch_size
    )  # [f, 196, 768]

    # 计算每帧的MSE
    mse_per_frame = ((pred_features - gt_features) ** 2).reshape(pred_features.shape[0], -1).mean(dim=1)  # [f]

    result_dict['dino_mse'] = mse_per_frame.cpu().tolist()
    result_dict['avg_dino_mse'] = float(mse_per_frame.mean().cpu())

    return result_dict

def action_accuracy_metric(pred_vid_path, gt_vid_path, mark_time, actions, cache_dir = Path("./cache_vipe"),
                           delta = 1, max_rot_deg = 179.0, max_trans = 1e9, min_valid_steps = 8, per_action = True, max_frames = 200, gt_data_dir = None):
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
        gt_data_dir: GT data directory (for caching images.txt)

    Returns:
        Dictionary with action accuracy metrics in JSON format, or None if failed
    """

    # Crop videos to max_frames if specified
    tqdm.write(f"Cropping videos to {max_frames} frames...")
    pred_vid_path = crop_video_frames(pred_vid_path, max_frames, cache_dir)
    gt_vid_path = crop_video_frames(gt_vid_path, max_frames, cache_dir, start_frame=mark_time)

    # Extract GT trajectory with caching
    gt_T = extract_traj(
        gt_vid_path, cache_dir,
        gt_cache_path=gt_data_dir
    )
    # Extract generated trajectory (no caching)
    gen_T = extract_traj(
        pred_vid_path, cache_dir,
    )

    # Align lengths to actions (+delta poses)
    n_steps = min(len(actions), len(gt_T) - delta, len(gen_T) - delta)
    if n_steps <= 0:
        tqdm.write(f"Not enough steps after alignment for data {pred_vid_path}: n_steps={n_steps}")
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
        tqdm.write(f"Too few valid RPE samples for data {pred_vid_path}: te.size={te.size}, min_valid_steps={min_valid_steps}")
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
            
    return result

def compute_metrics_single_gpu(data_list, gt_root, test_root,
                               requested_metrics, video_max_time, process_batch_size, device, gpu_id):
    import warnings
    # 在子进程中设置warnings过滤
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")
    """
    在单个GPU上计算指定数据列表的metrics

    Args:
        data_list: 要处理的数据文件夹名称列表
        gt_root: ground truth数据根目录
        test_root: 测试数据根目录
        perspective: '1st_data' or '3rd_data'
        test_type: 'mem_test' or 'action_space_test'
        requested_metrics: 要计算的指标列表
        video_max_time: 视频最大帧数
        process_batch_size: 批处理大小
        device: GPU设备ID
        gpu_id: GPU编号(用于显示)

    Returns:
        字典，包含所有处理的数据的结果
    """
    # 初始化模型
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

    # 加载DINOv3模型
    # dino_model, dino_processor = load_dinov3_model(device)

    results = []
    pbar = tqdm(data_list, position=gpu_id, desc=f"GPU{gpu_id}")

    for data in pbar:
        data_path = data['path']
        gt_dir = os.path.join(gt_root, data['perspective'], 'test', data['test_type'])
        test_dir = os.path.join(test_root, data['perspective'], data['test_type'])
        if not os.path.exists(os.path.join(test_dir, data_path)):
            continue

        pbar.set_postfix({"file": data_path})

        try:
            mark_time, total_time = load_time_from_json(os.path.join(gt_dir, data_path, 'action.json'))
            result = data
            result['error'] = None
            result['mark_time'] = mark_time
            result['total_time'] = total_time

            # 读入视频
            gt_imgs = load_gt_video(os.path.join(gt_dir, data_path, 'video.mp4'), mark_time, total_time, video_max_time)
            gt_imgs = gt_imgs.to(device)

            sample_imgs = load_sample_video(os.path.join(test_dir, data_path, 'video.mp4'), mark_time, total_time, video_max_time)
            sample_imgs = sample_imgs.to(device)

            # 计算LCM指标
            lcm = lcm_metric(sample_imgs, gt_imgs, requested_metrics, lpips_metric, ssim_metric, psnr_metric, process_batch_size)
            result['lcm']= lcm

            # 计算image_quality指标
            vq = visual_quality_metric(sample_imgs, imaging_model, aesthetic_model, clip_model)
            result['visual_quality'] = vq

            # 计算dino_MSE指标
            # dino_mse = dino_mse_metric(sample_imgs, gt_imgs, dino_model, dino_processor, device, batch_size=process_batch_size)
            # result['dino'] = dino_mse

            # 清理内存
            del sample_imgs, gt_imgs
            torch.cuda.empty_cache()

            # 计算action accuracy
            actions = extract_actions_from_json(os.path.join(gt_dir, data_path, 'action.json'), mark_time, video_max_time)
            action = action_accuracy_metric(
                os.path.join(test_dir, data_path, 'video.mp4'),
                os.path.join(gt_dir, data_path, 'video.mp4'),
                mark_time,
                actions,
                max_frames = video_max_time,
                gt_data_dir = os.path.join(gt_dir, data_path)
            )
            if action is not None:
                result['action'] = action

        except KeyboardInterrupt:
            tqdm.write(f"\nGPU{gpu_id}: Interrupted by user. Returning partial results...")
            raise
        except Exception as e:
            tqdm.write(f"Error processing {data_path} on GPU{gpu_id}: {e}")
            result['error'] = str(e)
        
        results.append(result)
        break

    return results

def compute_metrics(gt_root, test_root, requested_metrics=['mse','psnr','ssim','lpips'],
                   video_max_time=100, process_batch_size=10, num_gpus=1):
    """
    计算视频metrics，支持多GPU并行（统一架构，单GPU也使用multiprocessing）

    Args:
        gt_root: ground truth数据根目录
        test_root: 测试数据根目录
        requested_metrics: 要计算的指标列表
        video_max_time: 视频最大帧数
        process_batch_size: 批处理大小
        num_gpus: 使用的GPU数量（>=1）

    Returns:
        包含所有结果的字典
    """
    result_dict = {'data':[], 'video_max_time':video_max_time}

    # 统一使用multiprocessing架构
    mp.set_start_method('spawn', force=True)

    all_data = []
    for perspective in ['1st_data', '3rd_data']:
        for test_type in ['mem_test', 'action_space_test']:
            gt_dir = os.path.join(gt_root, perspective, 'test', test_type)
            test_dir = os.path.join(test_root, perspective, test_type)

            # 获取所有数据文件夹
            all_data += [{'path': d, 'perspective': perspective, 'test_type': test_type} 
                for d in os.listdir(gt_dir) if os.path.exists(os.path.join(test_dir, d))]

    if len(all_data) == 0:
        tqdm.write(f"No data found!")
        return

    # 将数据分配到各个GPU
    data_per_gpu = len(all_data) // num_gpus
    data_splits = []
    for i in range(num_gpus):
        start_idx = i * data_per_gpu
        end_idx = start_idx + data_per_gpu if i < num_gpus - 1 else len(all_data)
        data_splits.append(all_data[start_idx:end_idx])

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Total data: {len(all_data)}, Using {num_gpus} GPU(s)")
    for i, split in enumerate(data_splits):
        tqdm.write(f"  GPU{i}: {len(split)} videos")
    tqdm.write(f"{'='*60}\n")

    # 在并行前预先下载所有模型文件，避免多进程冲突
    ensure_all_models_downloaded()

    # 创建进程池并行处理
    try:
        with mp.Pool(processes=num_gpus) as pool:
            worker_args = [
                (data_splits[i], gt_root, test_root,
                    requested_metrics, video_max_time, process_batch_size, f'cuda:{i}', i)
                for i in range(num_gpus)
            ]

            result_list = pool.starmap(compute_metrics_single_gpu, worker_args)
            for result in result_list:
                result_dict['data'] += result

    except KeyboardInterrupt:
        tqdm.write("\n\nInterrupted by user! Saving partial results...")
        pool.terminate()
        pool.join()
        raise

    return result_dict

if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore", message=".*pretrained.*")
    warnings.filterwarnings("ignore", message=".*Weights.*")
    warnings.filterwarnings("ignore", message=".*video.*deprecated.*")

    parser = argparse.ArgumentParser(description='Compute video metrics with multi-GPU support')
    parser.add_argument('--gt_root', type=str, default='../MIND-Data', help='Ground truth data root directory')
    parser.add_argument('--test_root', type=str, default='/media/wjp/gingerBackup/mind/structured_baselines/i2v',
                       help='Test data root directory')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
    parser.add_argument('--video_max_time', type=int, default=100, help='Maximum video frames (default: 100)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')

    args = parser.parse_args()

    # 检查GPU数量
    available_gpus = torch.cuda.device_count()
    if args.num_gpus > available_gpus:
        tqdm.write(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        args.num_gpus = available_gpus

    # 准备输出路径
    if args.output:
        output_path = args.output
    else:
        from datetime import datetime
        output_path = f'result_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.json'

    tqdm.write(f"Starting computation with {args.num_gpus} GPU(s)...")
    tqdm.write(f"Output will be saved to: {output_path}")

    video_results = None
    try:
        video_results = compute_metrics(
            args.gt_root,
            args.test_root,
            video_max_time=args.video_max_time,
            num_gpus=args.num_gpus
        )
    except KeyboardInterrupt:
        tqdm.write("\n\n" + "="*60)
        tqdm.write("INTERRUPTED BY USER (Ctrl+C)")
        tqdm.write("="*60)
        if video_results is None:
            tqdm.write("No results to save (computation was interrupted too early)")
            exit(1)
    finally:
        # 保存结果(即使被中断)
        if video_results is not None:
            with open(output_path, 'w') as f:
                json.dump(video_results, f, indent=2)
            tqdm.write(f"\nResults saved to: {output_path}")

            # 打印统计信息
            tqdm.write(f"{len(video_results)} videos are processed")
        else:
            tqdm.write("\nNo results to save.")
    