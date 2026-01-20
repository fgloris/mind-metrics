# mind-metrics

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