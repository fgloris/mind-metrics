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
- `--metrics`: Comma-separated metrics to compute (default: `lcm,visual,dino,action`)

### How Multi-GPU Works

- Videos are put into a task queue.
- Each GPU process take one task from the queue when vacant.
- If failed, the task will be put back into the queue.
- Progress bars show accumulation for all results.
- Every time when a task is finished, the result file is updated. You can obtain intermediate results from the file.

## How to order files

### the structure of our ground truth videos (both for training and for testing) looks like:
```bash
MIND-Data
├── 1st_data
│   ├── test
│   │   ├── action_space_test
│   │   │   ├── {gt data name}
│   │   │   │   ├── action.json
│   │   │   │   └── video.mp4
|   |   |   ...
|   |   |    
│   │   └── mem_test
│   │       ├── {gt data name}
│   │       │   ├── action.json
│   │       │   └── video.mp4
|   |       ...
|   └── train
|       ├── 2025-10-18-17-25-BoardcastStudio_B-0m37s
|       │   ├── action.json
|       │   └── video.mp4
|       ...
|
├── 3rd_data
│   ├── test
│   │   ├── action_space_test
│   │   │   ├── {gt data name}
│   │   │   │   ├── action.json
│   │   │   │   └── video.mp4
|   |   |   ...
|   |   |    
│   │   └── mem_test
│   │       ├── {gt data name}
│   │       │   ├── action.json
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
│   └── mem_test
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
│   └── mem_test
│       ├── {corresponding data name}
│       │   └── video.mp4
|       ...
```