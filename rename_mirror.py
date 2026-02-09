import os
import shutil
from collections import defaultdict

source_dir = '/storage/v-jinpewang/az_workspace/eason/worldmodel_brenchmark_dataset/inference/yx_dev/V2V_3rd_mirror/checkpoint_model_002000'
target_base = '/storage/v-jinpewang/az_workspace/eason/v2v/3rd_data/mirror_test'

files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]

groups = defaultdict(list)
for f in files:
  prefix = f.split('_-')[0]
  groups[prefix].append(f)

for prefix, file_list in groups.items():
  target_dir = os.path.join(target_base, prefix)
  os.makedirs(target_dir, exist_ok=True)

  for idx, filename in enumerate(file_list, 1):
    src_path = os.path.join(source_dir, filename)
    dst_name = f'path-{idx}.mp4'
    dst_path = os.path.join(target_dir, dst_name)
    shutil.copy(src_path, dst_path)
    print(f'Moved: {filename} -> {prefix}/{dst_name}')
