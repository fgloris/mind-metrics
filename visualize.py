import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt


def create_metrics_plot(data, frame_index, total_frames, output_size):
    """
    实时生成包含两个指标（mse和dino_mse）对比图的图像。
    在每个图上，绘制一条垂直线表示当前帧，并标记交点。
    返回一个 NumPy 数组（图像）。
    """
    plt.switch_backend('Agg')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    metrics = ["mse", "dino_mse"]
    metric_names = ["MSE", "DINO MSE"]

    for i, metric_key in enumerate(metrics):
        ax = axes[i]

        # 获取指标数据
        if metric_key in data:
            metric_list = data[metric_key]
            x_indices = list(range(len(metric_list)))

            # 绘制折线
            ax.plot(x_indices, metric_list, linestyle='-',
                    linewidth=2, color='royalblue', alpha=0.7, label=metric_key)

            # 在当前帧位置标记交点
            if 0 <= frame_index < len(metric_list):
                current_value = metric_list[frame_index]
                ax.plot(frame_index, current_value, marker='o', markersize=10,
                        color='red', markeredgecolor='black', markeredgewidth=1.5,
                        zorder=5)

            # 绘制表示当前帧的垂直线
            ax.axvline(x=frame_index, color='gray', linestyle='--', linewidth=2,
                       label=f'Frame {frame_index}', zorder=4)

        # 设置子图属性
        ax.set_title(f'{metric_names[i]} across Frames', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10)

        # 设置x轴刻度
        ax.set_xlim(0, total_frames - 1)
        if total_frames > 10:
            ax.set_xticks(range(0, total_frames, max(1, total_frames // 10)))
        else:
            ax.set_xticks(range(total_frames))

    plt.tight_layout()

    # 将 Matplotlib 图表渲染成 NumPy 数组
    fig.canvas.draw()
    buf = fig.canvas.tostring_argb()
    plot_image = np.frombuffer(buf, dtype=np.uint8)
    plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # 转换为BGR格式
    plot_image_bgr = plot_image[:, :, [3, 2, 1]]

    # 调整图像大小
    resized_plot = cv2.resize(plot_image_bgr, output_size)

    return resized_plot


def combine_videos_with_metrics(gt_video_path, pred_video_path, output_path,
                                metrics_data, scene_name, mark_time, video_max_time):
    """
    拼接GT视频和预测视频（左右布局），并在底部显示指标图。

    Args:
        gt_video_path: Ground truth视频路径
        pred_video_path: 预测视频路径
        output_path: 输出视频路径
        metrics_data: 包含mse和dino_mse的字典
        scene_name: 场景名称（用于显示）
        mark_time: GT视频开始帧号
        video_max_time: 最大视频帧数
    """
    # 打开视频
    gt_cap = cv2.VideoCapture(gt_video_path)
    pred_cap = cv2.VideoCapture(pred_video_path)

    if not gt_cap.isOpened() or not pred_cap.isOpened():
        print(f"Error: Cannot open videos for scene '{scene_name}'")
        return

    # 获取视频参数
    fps = int(gt_cap.get(cv2.CAP_PROP_FPS))
    gt_total_frames = int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pred_total_frames = int(pred_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = video_max_time

    # 定义输出分辨率（左右布局）
    video_size = (1280, 360)  # 视频部分
    plot_size = (1280, 360)   # 指标图部分
    final_output_size = (video_size[0], video_size[1] + plot_size[1])  # (1280, 1080)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, final_output_size)

    # 视频窗口尺寸 (640, 720)
    frame_size = (640, 360)

    # 视频叠加文字参数
    font_scale = 1.0
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    print(f"Processing scene: {scene_name}")
    print(f"  GT start frame: {mark_time}, Max frames: {video_max_time}")
    print(f"  GT total frames: {gt_total_frames}, Pred total frames: {pred_total_frames}")
    print(f"  Output: {output_path}")

    # 设置GT视频的起始位置
    gt_cap.set(cv2.CAP_PROP_POS_FRAMES, mark_time)

    frame_index = 0
    while frame_index < total_frames:
        # 读取GT和预测视频帧
        gt_ret, gt_frame = gt_cap.read()
        pred_ret, pred_frame = pred_cap.read()

        if not gt_ret or not pred_ret:
            break

        # 调整帧大小到640x720
        gt_resized = cv2.resize(gt_frame, frame_size)
        pred_resized = cv2.resize(pred_frame, frame_size)

        # 添加文本标签
        cv2.putText(gt_resized, "Ground Truth", (10, 40),
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(pred_resized, "Prediction", (10, 40),
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # 创建左右拼接视频
        combined_video = np.hstack((gt_resized, pred_resized))

        # 创建指标图
        combined_plot = create_metrics_plot(
            metrics_data, frame_index, video_max_time, plot_size
        )

        # 上下拼接最终帧
        final_frame = np.vstack((combined_video, combined_plot))

        out.write(final_frame)

        frame_index += 1
        print(f"  Processing frame: {frame_index}/{total_frames}\r", end="")

    # 释放资源
    gt_cap.release()
    pred_cap.release()
    out.release()

    print(f"\nCompleted: {output_path}")


def load_json_results(json_file):
    """
    加载JSON结果文件

    Returns:
        tuple: (video_max_time, data_list)
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 新的JSON格式: {"video_max_time": 100, "data": [...]}
        if isinstance(data, dict):
            video_max_time = data.get('video_max_time', 100)
            data_list = data.get('data', [])
            return video_max_time, data_list
        # 兼容旧格式
        elif isinstance(data, list):
            return 100, data
        else:
            print(f"Error: Unknown JSON format")
            return 100, []
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        return 100, []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return 100, []


def visualize_results(json_file, gt_root, test_root, output_dir=None):
    """
    主函数：读取JSON结果并生成可视化视频

    Args:
        json_file: JSON结果文件路径
        gt_root: Ground truth数据根目录
        test_root: 测试数据根目录
        output_dir: 输出目录（可选，默认为当前目录）
    """
    # 加载JSON结果
    video_max_time, results = load_json_results(json_file)

    if not results:
        print("No results to process.")
        return

    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(json_file), "visualization_output")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loaded {len(results)} results from {json_file}")
    print(f"Video max time: {video_max_time} frames")
    print(f"Output directory: {output_dir}\n")

    # 处理每个结果
    success_count = 0
    for result in results:
        if result.get('error') is not None:
            print(f"Skipping {result['path']} due to error: {result['error']}")
            continue

        # 提取信息
        data_path = result['path']
        perspective = result['perspective']
        test_type = result['test_type']
        mark_time = result.get('mark_time', 0)

        # 构造视频路径（按照process.py的方式）
        gt_video_path = os.path.join(gt_root, perspective, 'test', test_type, data_path, 'video.mp4')
        pred_video_path = os.path.join(test_root, perspective, test_type, data_path, 'video.mp4')

        # 检查视频是否存在
        if not os.path.exists(gt_video_path):
            print(f"GT video not found: {gt_video_path}")
            continue
        if not os.path.exists(pred_video_path):
            print(f"Prediction video not found: {pred_video_path}")
            continue

        # 准备指标数据
        metrics_data = {}

        # 添加lcm指标（mse）
        if 'lcm' in result and 'mse' in result['lcm']:
            metrics_data['mse'] = result['lcm']['mse']

        # 添加dino指标（dino_mse）
        if 'dino' in result and 'dino_mse' in result['dino']:
            metrics_data['dino_mse'] = result['dino']['dino_mse']

        if not metrics_data:
            print(f"No metrics data found for {data_path}")
            continue

        # 构造输出路径
        output_filename = f"{data_path}_visualized.mp4"
        output_path = os.path.join(output_dir, output_filename)

        # 生成可视化视频
        try:
            combine_videos_with_metrics(
                gt_video_path,
                pred_video_path,
                output_path,
                metrics_data,
                data_path,
                mark_time,
                video_max_time
            )
            success_count += 1
        except Exception as e:
            print(f"\nError processing {data_path}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Visualization complete!")
    print(f"Successfully processed: {success_count}/{len(results)} videos")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize video metrics with dynamic plots')
    parser.add_argument('--json_file', type=str,
                       default=None,
                       help='Path to JSON result file')
    parser.add_argument('--gt_root', type=str, default='../MIND-Data',
                       help='Ground truth data root directory')
    parser.add_argument('--test_root', type=str,
                       default='/media/wjp/gingerBackup/mind/structured_baselines/i2v',
                       help='Test data root directory')
    parser.add_argument('--output_dir', type=str, default='./',
                       help='Output directory for visualized videos')

    args = parser.parse_args()

    visualize_results(
        json_file=args.json_file,
        gt_root=args.gt_root,
        test_root=args.test_root,
        output_dir=args.output_dir
    )
