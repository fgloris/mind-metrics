import lpips
import torch
from pyiqa.archs.musiq_arch import MUSIQ
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils import load_gt_video, load_sample_video, load_time_from_json, print_gpu_memory, get_musiq_spaq_path, get_vitl_path, get_aes_path, clip_transform_Image
from tqdm import tqdm
import torch.nn.functional as F
import json
import clip
import os

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

def action_accuracy_metrix(pred, gt):
    pass

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

    for perspective in ['1st_data', ]:
        for test_type in ['mem_test']:
            gt_dir = os.path.join(gt_root, perspective, 'test', test_type)
            test_dir = os.path.join(test_root, perspective, 'test', test_type)
            pbar = tqdm(os.listdir(gt_dir))
            pbar.set_description(f"Computing {test_type} on {perspective}")
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
                #lcm = lcm_metric(sample_imgs, gt_imgs, requested_metrics, lpips_metric, ssim_metric, psnr_metric, process_batch_size)
                #result_dict[perspective][test_type][data] = lcm

                # 计算image_quality指标
                vq = visual_quality_metric(sample_imgs, imaging_model, aesthetic_model, clip_model)
                result_dict[perspective][test_type][data] = vq #.update(vq)
                
                # 清理内存
                del sample_imgs, gt_imgs
                torch.cuda.empty_cache()
                break

    return result_dict

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io._video_deprecation_warning")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    video_results = compute_metrics('../MIND-Data', '../MIND-Data')
    with open('result_3rd.json', 'w') as f:
        json.dump(video_results, f, indent=2)
    