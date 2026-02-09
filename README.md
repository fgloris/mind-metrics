# mind-metrics

## Environment setup

First follow vipe's instruction to build conda envrionment, until ```vipe``` command is avilable

https://github.com/nv-tlabs/vipe


then install our requirements under the same conda env:
```bash
pip install -r requirements.txt 
```
in the same env.

## Multi-GPU Support

The metrics computation now supports multi-GPU parallel processing for faster evaluation.

### Usage

```bash
python src/process.py --gt_root /path/to/MIND-Data --test_root /path/to/test/videos --num_gpus 8 --metrics lcm,visual,action
```

### Command Line Arguments

- `--gt_root`: Ground truth data root directory (required)
- `--test_root`: Test data root directory (required)
- `--dino_path`: DINOv3 model weights directory (default: `./dinov3_vitb16`)
- `--num_gpus`: Number of GPUs to use for parallel processing (default: 1)
- `--video_max_time`: Maximum video frames to process (default: `None` = use all frames)
- `--output`: Output JSON file path (default: `result_{test_root}_{timestamp}.json`)
- `--metrics`: Comma-separated metrics to compute (default: `lcm,visual,dino,action,gsc`)

### How Multi-GPU Works

- Videos are put into a task queue.
- Each GPU process take one task from the queue when vacant.
- If failed, the task will be put back into the queue.
- Progress bars show accumulation for all results.
- Every time when a task is finished, the result file is updated. You can obtain intermediate results from the file.

## action.json

```
{
    "mark_time": [int] the divider of memory context and expected perdiction; the start frame index of the expected prediction
    "total_time": [int] the total frames of the ground truth video
    "data": [
        {
            "time": [int] frame index of the recorded state
            "ws": [int] 0: move forward, 1: move backward
            "ad": [int] 0: move left, 1: move right
            "ud": [int] 0: look up, 1: look down
            "lr": [int] 0: look left, 1: look right
            "actor_pos": {
                "x": [float] the x-coordinate of the character
                "y": [float] the y-coordinate of the character
                "z": [float] the z-coordinate of the character
            },
            "actor_rpy": {
                "x": [float] the roll angle of the character (Euler angles)
                "y": [float] the pitch angle of the character
                "z": [float] the yaw angle of the character
            },
            "camera_pos": {
                "x": [float] the x-coordinate of the camera (only exists in 3rd-person mode)
                "y": [float] the y-coordinate of the camera
                "z": [float] the z-coordinate of the camera
            },
            "camera_rpy": {
                "x": [float] the roll angle of the camera (Euler angles)
                "y": [float] the pitch angle of the camera
                "z": [float] the yaw angle of the camera
            }
        },
        ...
    ]
}
```

## result.json

```
{
  "video_max_time": [int] video_max_time given in cmd parameters; max frames of the sample video to compute metrics (except action accuracy).
  "data": [
    {
      "path": [string] the directory name of the video data.
      "perspective": [string] 1st_data/3rd_data, the perspective of the video data.
      "test_type": [string] mem_test/action_space_test, the test set of the video data.
      "error": [string] the error occur when computing metrics
      "mark_time": [int] the divider of memory context and expected perdiction; the start frame index of the expected prediction.
      "total_time": [int] the total frames of the ground truth video.
      "sample_frames": [int ]the total frames of the video to be tested.
      "gsc": { the general scene consistency metric result.
        "length": [int] length of the origin prediction and the mirrored prediction.
        "mse": [list[float]] the per-frame mean square error.
        "avg_mse": [float] the average of mse.
        "lpips": [list[float]] the per-frame Learned Perceptual Image Patch Similarity.
        "avg_lpips": [float] the average of lpips.
        "ssim": [list[float]] the per-frame Structural Similarity Index Measure.
        "avg_ssim": [float] the average of ssim.
        "psnr": [list[float]] the per-frame Peak Signal-to-Noise Ratio.
        "avg_psnr": [float] the average of psnr.
      },
      "lcm": { the long context memory metric result.
        "mse": [list[float]] the per-frame mean square error.
        "avg_mse": [float] the average of mse.
        "lpips": [list[float]] the per-frame Learned Perceptual Image Patch Similarity.
        "avg_lpips": [float] the average of lpips.
        "ssim": [list[float]] the per-frame Structural Similarity Index Measure.
        "avg_ssim": [float] the average of ssim.
        "psnr": [list[float]] the per-frame Peak Signal-to-Noise Ratio.
        "avg_psnr": [float] the average of psnr.
      },
      "visual_quality": { the visual quality metric result.
        "imaging": [list[float]] the per-frame imaging quality.
        "avg_imaging": [float] the average of imaging quality. 
        "aesthetic": [list[float]] the per-frame aesthetic quality.
        "avg_imaging": [float] the average of aesthetic quality. 
      },
      "action": { the action accuracy metric result. computed by ViPE pose estimation and trajectory alignment.
        "__overall__": { the overall statistics of all valid frames after outlier filtering.
          "count": [int] number of valid samples used for statistics.
          "rpe_trans_mean": [float] mean of Relative Pose Error for translation (in meters).
          "rpe_trans_median": [float] median of RPE translation.
          "rpe_rot_mean_deg": [float] mean of RPE rotation in degrees.
          "rpe_rot_median_deg": [float] median of RPE rotation.
        },
        "translation": { the statistics of pure translation actions (forward/backward/left/right).
          "count": [int] number of valid samples for translation actions.
          "rpe_trans_mean": [float] mean RPE translation for translation actions.
          "rpe_trans_median": [float] median RPE translation for translation actions.
          "rpe_rot_mean_deg": [float] mean RPE rotation for translation actions.
          "rpe_rot_median_deg": [float] median RPE rotation for translation actions.
        },
        "rotation": { the statistics of pure rotation actions (cam_left/cam_right/cam_up/cam_down).
          "count": [int] number of valid samples for rotation actions.
          ...
        },
        "other": { the statistics of combined actions (e.g., forward+look_right).
          "count": [int] number of valid samples for other actions.
          ...
        },
        "act:forward": { the statistics of specific action "forward".
          "count": [int] number of valid samples for this action.
          "rpe_trans_mean": [float] mean RPE translation.
          "rpe_trans_median": [float] median RPE translation.
          "rpe_rot_mean_deg": [float] mean RPE rotation.
          "rpe_rot_median_deg": [float] median RPE rotation.
        },
        "act:look_right": { the statistics of specific action "look_right".
          ...
        },
        ...
      },
      "dino": { the dino mse metric result.
        "dino_mse": [list[float]] the per-frame mse of dino features.
        "avg_dino_mse": [float] the average of dino_mse. 
      }
    },
    ...
  ]
}
```

## How to order files

### the structure of our ground truth videos (both for training and for testing) looks like:
```bash
MIND-Data
├── 1st_data
│   ├── test
│   │   ├── action_space_test
│   │   │   ├── {gt data name}
│   │   │   │   ├── action.json
│   │   │   │   ├── images.txt
│   │   │   │   └── video.mp4
|   |   |   ...
|   |   |    
│   │   └── mem_test
│   │       ├── {gt data name}
│   │       │   ├── action.json
│   │       │   ├── images.txt
│   │       │   └── video.mp4
|   |       ...
|   └── train
|       ├── {gt data name}
|       │   ├── action.json
|       │   └── video.mp4
|       ...
|
├── 3rd_data
│   ├── test
│   │   ├── action_space_test
│   │   │   ├── {gt data name}
│   │   │   │   ├── action.json
│   │   │   │   ├── images.txt
│   │   │   │   └── video.mp4
|   |   |   ...
|   |   |    
│   │   └── mem_test
│   │       ├── {gt data name}
│   │       │   ├── action.json
│   │       │   ├── images.txt
│   │       │   └── video.mp4
|   |       ...
|   └── train
|       ├── {gt data name}
|       │   ├── action.json
|       │   └── video.mp4
|       ...
```
### suppose the structure your test videos loos like:
```
{model_name}
├── 1st_data
│   ├── action_space_test
│   │   ├── {corresponding data name}
│   │   │   └── video.mp4
|   |   ...
|   |    
│   ├── mirror_test
│   |   ├── {arbitrary data name}
│   │   │   ├── path-1.mp4
│   │   │   ├── path-2.mp4
│   │   │   ├── path-3.mp4
│   │   │   ...
│   │   │   └── path-10.mp4
|   |   ...
|   |
|   └── mem_test
│       ├── {corresponding data name}
│       │   └── video.mp4
|       ...
|
├── 3rd_data
│   ├── action_space_test
│   │   ├── {corresponding data name}
│   │   │   └── video.mp4
|   |   ...
|   |    
│   ├── mirror_test
│   |   ├── {carbitrary data name}
│   │   │   ├── path-1.mp4
│   │   │   ├── path-2.mp4
│   │   │   ├── path-3.mp4
│   │   │   ...
│   │   │   └── path-10.mp4
|   |   ...
|   |
│   └── mem_test
│       ├── {corresponding data name}
│       │   └── video.mp4
|       ...
```