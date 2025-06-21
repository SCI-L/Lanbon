import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from Trajectory import Trajectory, geo_info 
from SrMReader import SrMReader 
import os
import classix 
import pandas as pd 
from sklearn.decomposition import PCA 
import random
from matplotlib.lines import Line2D

# 确保matplotlib能够显示中文
plt.rcParams['font.family'] = 'Hiragino Sans GB'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题


class TrajectoryFeatureExtractor:
    """轨迹特征提取器，用于从轨迹中提取特征向量"""
    # ... (此类代码未作修改，保持原样) ...
    def __init__(self, feature_types=None):
        self.available_features = [
            'spatial', 'sog_stats', 'cog_stats', 'length', 'duration',
            'turning_points', 'speed_change', 'acceleration'
        ]
        if feature_types is None:
            self.feature_types = self.available_features
        else:
            self.feature_types = [f for f in feature_types if f in self.available_features]

    def extract_features(self, trajectory):
        features = []
        if 'spatial' in self.feature_types:
            start_lon, start_lat = trajectory.lon[0], trajectory.lat[0]
            end_lon, end_lat = trajectory.lon[-1], trajectory.lat[-1]
            features.extend([start_lon, start_lat, end_lon, end_lat])
        if 'sog_stats' in self.feature_types:
            if len(trajectory.sog) > 0:
                mean_sog, max_sog, min_sog, std_sog = np.mean(trajectory.sog), np.max(trajectory.sog), np.min(trajectory.sog), np.std(trajectory.sog)
            else:
                mean_sog, max_sog, min_sog, std_sog = 0, 0, 0, 0
            features.extend([mean_sog, max_sog, min_sog, std_sog])
        if 'cog_stats' in self.feature_types:
            if len(trajectory.cog) > 0:
                sin_sum = np.sum(np.sin(np.radians(trajectory.cog)))
                cos_sum = np.sum(np.cos(np.radians(trajectory.cog)))
                mean_cog = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                max_cog, min_cog = np.max(trajectory.cog), np.min(trajectory.cog)
            else:
                mean_cog, max_cog, min_cog = 0, 0, 0
            features.extend([mean_cog, max_cog, min_cog])
        if 'direction' in self.feature_types:
            if len(trajectory.cog) > 1:
                direction_changes = [abs(trajectory.cog[i] - trajectory.cog[i - 1]) if abs(trajectory.cog[i] - trajectory.cog[i - 1]) <= 180 else 360 - abs(trajectory.cog[i] - trajectory.cog[i - 1]) for i in range(1, len(trajectory.cog))]
                mean_direction_change = np.mean(direction_changes) if direction_changes else 0
                max_direction_change = np.max(direction_changes) if direction_changes else 0
            else:
                mean_direction_change, max_direction_change = 0, 0
            features.extend([mean_direction_change, max_direction_change])
        if 'length' in self.feature_types:
            features.append(trajectory.get_length() if hasattr(trajectory, 'get_length') else 0)
        if 'duration' in self.feature_types:
            features.append((trajectory.ts[-1] - trajectory.ts[0]) if len(trajectory.ts) > 1 else 0)
        if 'turning_points' in self.feature_types:
            compressed_traj = trajectory.tdkc() if hasattr(trajectory, 'tdkc') else trajectory
            features.append(compressed_traj.point_num if hasattr(compressed_traj, 'point_num') else 0)
        if 'speed_change' in self.feature_types:
            if len(trajectory.sog) > 1:
                speed_changes = np.diff(trajectory.sog)
                mean_speed_change, max_speed_change = np.mean(np.abs(speed_changes)), np.max(np.abs(speed_changes))
            else:
                mean_speed_change, max_speed_change = 0, 0
            features.extend([mean_speed_change, max_speed_change])
        if 'acceleration' in self.feature_types:
            if len(trajectory.sog) > 1 and len(trajectory.ts) > 1:
                speed_diffs = np.diff(trajectory.sog)
                time_diffs = np.diff(trajectory.ts)
                accelerations = [sd / td for sd, td in zip(speed_diffs, time_diffs) if td > 0]
                if accelerations:
                    mean_acceleration, max_acceleration = np.mean(np.abs(accelerations)), np.max(np.abs(accelerations))
                else:
                    mean_acceleration, max_acceleration = 0, 0
            else:
                mean_acceleration, max_acceleration = 0, 0
            features.extend([mean_acceleration, max_acceleration])
        return np.array(features)


def visualize_clusters(trajectories_to_plot, labels, num_clusters, output_path="trajectory_clusters.png", params_text=""):
    # ... (此函数未作修改，保持原样) ...
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    color_count = len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1
    colors = plt.cm.rainbow(np.linspace(0, 1, color_count if color_count > 0 else 1))
    label_to_color = {}
    color_idx = 0
    for label in sorted(unique_labels):
        if label == -1:
            label_to_color[label] = 'k'
        else:
            label_to_color[label] = colors[color_idx]
            color_idx += 1
    for i, traj in enumerate(trajectories_to_plot):
        label = labels[i]
        color = label_to_color[label]
        plt.plot(traj.lon, traj.lat, color=color, alpha=0.5, linewidth=1)
    legend_handles = []
    for label in sorted(unique_labels):
        display_name = f'簇 {label}' if label != -1 else '噪声'
        if np.sum(labels == label) > 0:
            legend_handles.append(plt.Line2D([0], [0], color=label_to_color[label], lw=2, label=display_name))
    if legend_handles:
        plt.legend(handles=legend_handles, loc='best')
    plt.title(f'轨迹聚类结果 (聚类数: {num_clusters})')
    plt.xlabel('经度 (度)')
    plt.ylabel('纬度 (度)')
    plt.grid(True, linestyle='--', alpha=0.7)
    if params_text:
        plt.figtext(0.5, 0.01, params_text, ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"轨迹聚类可视化已保存到 {output_path}")

def explain_connection_path(classix_model, data_df, mmsi1, mmsi2):
    """
    解释两个MMSI为何被聚在同一簇，并直接显示2D和3D特征空间的可视化路径图。
    """
    # ... (此函数未作修改，保持原样) ...
    print(f"\n--- 正在分析 MMSI {mmsi1} 和 {mmsi2} 之间的连接路径 ---")
    try:
        idx1 = np.where(data_df.index == mmsi1)[0][0]
        idx2 = np.where(data_df.index == mmsi2)[0][0]
        print("\nCLASSIX 官方文本解释与2D可视化:")
        classix_model.explain(idx1, idx2, plot=True, alpha=0.2)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [lbl.replace(f'data point {idx1}', f'MMSI {mmsi1}').replace(f'data point {idx2}', f'MMSI {mmsi2}') for lbl in labels]
        ax.legend(handles, new_labels, loc='best')
        plt.tight_layout()
        print("正在显示 2D 可解释性图...")
        plt.show()
        print("\n正在准备 3D 可解释性图...")
        path_indices = classix_model.getPath(idx1, idx2)
    except IndexError:
        print(f"错误：无法在数据中定位到 MMSI {mmsi1} 或 {mmsi2}。")
        return
    except Exception as e:
        print(f"在解释过程中发生错误: {e}")
        return

    pca_3d = PCA(n_components=3)
    data_3d = pca_3d.fit_transform(data_df.values)
    num_points = data_3d.shape[0]
    sample_indices = random.sample(range(num_points), min(num_points, 50000))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_3d[sample_indices, 0], data_3d[sample_indices, 1], data_3d[sample_indices, 2],
                c=classix_model.labels_[sample_indices], cmap='viridis', s=1, alpha=0.2)
    legend_label_path = '连接路径'
    if len(path_indices) > 2:
        print("\n在3D图中：找到了连接两个群组的中间路径。")
        path_data_3d = data_3d[path_indices]
    else:
        print("\n在3D图中：两个轨迹在同一个起始群组中，用直线表示其直接连接。")
        path_data_3d = data_3d[[idx1, idx2]]
        legend_label_path = '连接路径 (同组直连)'
    ax.plot(path_data_3d[:, 0], path_data_3d[:, 1], path_data_3d[:, 2], c='r', alpha=0.8, linewidth=3)
    ax.scatter(data_3d[idx1, 0], data_3d[idx1, 1], data_3d[idx1, 2], marker='*', s=200, c='C0')
    ax.scatter(data_3d[idx2, 0], data_3d[idx2, 1], data_3d[idx2, 2], marker='*', s=200, c='C1')
    legend_elements = [
        Line2D([0], [0], color='r', lw=2, label=legend_label_path),
        Line2D([0], [0], marker='*', color='C0', markersize=10, label=f'MMSI {mmsi1}', linestyle='None'),
        Line2D([0], [0], marker='*', color='C1', markersize=10, label=f'MMSI {mmsi2}', linestyle='None')
    ]
    ax.legend(handles=legend_elements)
    ax.set_title(f'MMSI {mmsi1} 与 {mmsi2} 在3D特征空间的连接路径')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    print("正在显示 3D 可解释性图...")
    plt.show()


def main():
    """主函数，使用轨迹聚类算法"""
    # 1. 初始化和数据读取
    print(f"当前 CLASSIX 版本: {classix.__version__}")
    reader = SrMReader()
    data_path = '/Users/peng/Desktop/AIS数据/美国丹佛/message_type_1_2_3_20241111.txt'
    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}"); return
    print(f"正在读取AIS数据: {data_path}")
    trajectories_dict = reader.read_multiple_track(data_path, '1_2_3')
    original_trajectories_list = list(trajectories_dict.values())
    print(f"共读取 {len(original_trajectories_list)} 条轨迹")

    # 2. 轨迹过滤
    min_points_for_traj = 10
    filtered_trajectories = [t for t in original_trajectories_list if t.point_num >= min_points_for_traj]
    print(f"过滤后剩余 {len(filtered_trajectories)} 条轨迹 (点数 >= {min_points_for_traj})")
    if not filtered_trajectories: print("没有符合条件的轨迹进行聚类。"); return
    
    # 3. 轨迹压缩
    print("正在压缩轨迹...")
    compressed_trajectories = []
    mmsi_to_original_traj = {t.get_id(): t for t in filtered_trajectories}
    for traj in filtered_trajectories:
        try:
            compressed = traj.tdkc()
            if compressed.point_num >= 2: compressed_trajectories.append(compressed)
        except Exception as e: print(f"压缩轨迹 {traj.get_id()} 时出错: {e}")
    print(f"压缩后剩余 {len(compressed_trajectories)} 条轨迹")
    if not compressed_trajectories: print("压缩后没有符合条件的轨迹进行聚类。"); return
    
    # 4. 特征提取
    print("正在提取轨迹特征...")
    feature_types = ['spatial', 'sog_stats', 'cog_stats', 'acceleration']
    feature_extractor = TrajectoryFeatureExtractor(feature_types=feature_types)
    features_list, valid_trajectories, valid_mmsi_ids = [], [], []
    for traj in compressed_trajectories:
        try:
            feature_vector = feature_extractor.extract_features(traj)
            if np.all(np.isfinite(feature_vector)):
                features_list.append(feature_vector)
                valid_trajectories.append(traj)
                valid_mmsi_ids.append(traj.get_id())
            else: print(f"轨迹 {traj.get_id()} 的特征包含无效值，已跳过")
        except Exception as e: print(f"提取轨迹 {traj.get_id()} 特征时出错: {e}")
    if not features_list: print("没有有效的特征向量进行聚类。"); return
    features_array = np.array(features_list)
    print(f"提取了 {len(features_array)} 条轨迹的特征，每条特征维度: {features_array.shape[1]}")
    
    # 5. 特征标准化
    print("正在标准化特征...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)
    feature_column_names = [f'feature_{i}' for i in range(scaled_features.shape[1])]
    data_df = pd.DataFrame(scaled_features, index=valid_mmsi_ids, columns=feature_column_names)
    print(f"特征数据已转换为 DataFrame，包含 {data_df.shape[0]} 行，索引为 MMSI。")

    # 6. 使用CLASSIX进行聚类和超参数调优
    print("正在使用CLASSIX进行聚类...")
    radius_values = [0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.275, 0.3]
    minPts = 7
    best_silhouette = -1
    best_labels, best_radius, best_num_clusters, classix_model = None, None, 0, None
    for radius in radius_values:
        clx_instance = classix.CLASSIX(radius=radius, minPts=minPts, sorting='pca', group_merging='distance')
        clx_instance.fit(data_df)
        labels = clx_instance.labels_
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        if num_clusters <= 1: continue
        non_noise_indices = labels != -1
        if np.sum(non_noise_indices) > 1:
            silhouette_avg = silhouette_score(scaled_features[non_noise_indices], labels[non_noise_indices])
            print(f"半径 {radius} 产生了 {num_clusters} 个聚类，轮廓系数: {silhouette_avg:.4f}")
            if silhouette_avg > best_silhouette:
                best_silhouette, best_labels, best_radius, classix_model, best_num_clusters = \
                    silhouette_avg, labels, radius, clx_instance, num_clusters
    
    # 7. 处理聚类结果
    if best_labels is None:
        print("未能找到有效的聚类结果，使用默认参数")
        classix_model = classix.CLASSIX(radius=0.2, minPts=10, sorting='pca', group_merging='distance')
        classix_model.fit(data_df)
        best_labels, best_radius = classix_model.labels_, 0.2
        best_num_clusters = len(np.unique(best_labels)) - (1 if -1 in np.unique(best_labels) else 0)
    
    unique_labels = np.unique(best_labels)
    noise_count = np.sum(best_labels == -1)
    print(f"\n噪声点数量: {noise_count} ({noise_count/len(best_labels)*100:.2f}%)")
    silhouette_str_display = f"{best_silhouette:.4f}" if best_silhouette > -1 else "N/A"
    print(f"\n最佳聚类结果: 半径={best_radius}, 最小点数={minPts}, 簇数={best_num_clusters}, 轮廓系数={silhouette_str_display}")
    
    # 8. 保存MMSI列表
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    mmsi_list_file_path = os.path.join("/Users/peng/Desktop/results", "mmsi_cluster_grouped_list.txt")
    clusters_mmsis = {int(label): [] for label in unique_labels}
    for mmsi, label in zip(data_df.index, best_labels):
        clusters_mmsis[int(label)].append(mmsi)

    with open(mmsi_list_file_path, 'w', encoding='utf-8') as f:
        for label in sorted(clusters_mmsis.keys()):
            label_name = f"簇 {label}" if label != -1 else "噪声"
            unique_mmsis_in_cluster = list(dict.fromkeys(clusters_mmsis.get(label, [])))
            f.write(f"\n--- {label_name} (共 {len(unique_mmsis_in_cluster)} 条轨迹) ---\n")
            if unique_mmsis_in_cluster:
                for mmsi_item in unique_mmsis_in_cluster:
                    f.write(f"MMSI: {mmsi_item}\n")
            else:
                f.write("无轨迹\n")
                print("无轨迹")
    print(f"\nMMSI 码清单（按簇分组）已保存到 {mmsi_list_file_path}")
    
    # 9. 可视化聚类结果
    output_path_2d = os.path.join(output_dir, f"trajectory_clusters_r{best_radius}_m{minPts}_2D.png")
    params_text = f"参数: radius={best_radius}, minPts={minPts}, 特征={feature_types}, 轮廓系数={silhouette_str_display}"
    visualize_clusters(valid_trajectories, best_labels, best_num_clusters, output_path_2d, params_text)

    # 10. 交互式查询与解释
    print("\n--- MMSI 码聚类查询与解释 ---")
    print("你可以输入一个或两个 MMSI 码 (用逗号分隔) 进行详细解释，或输入 'q' 退出。")
    print("  - 输入单个MMSI (如: 'MMSI1')，将获得基本解释。")
    print("  - 输入两个MMSI (如: 'MMSI1, MMSI2')，将获得详细的路径分析和可视化图。")
    
    while True:
        mmsi_input_str = input("\n请输入MMSI码(s): ").strip()
        if mmsi_input_str.lower() == 'q':
            break

        mmsis_to_explain = [m.strip() for m in mmsi_input_str.split(',') if m.strip()]
        
        valid_mmsis = []
        for mmsi in mmsis_to_explain:
            if mmsi in data_df.index:
                valid_mmsis.append(mmsi)
            else:
                print(f"警告：MMSI码 {mmsi} 未在聚类数据中找到，可能已被过滤。")
                if mmsi in mmsi_to_original_traj:
                    print(f"  (注意：该MMSI的原始轨迹点数为 {mmsi_to_original_traj[mmsi].point_num})")
        
        if not valid_mmsis:
            print("没有有效的MMSI码可供查询。")
            continue
            
        try:
            if len(valid_mmsis) == 1:
                mmsi = valid_mmsis[0]
                idx_loc = np.where(data_df.index == mmsi)[0][0]
                print(f"\n正在对 MMSI {mmsi} (索引 {idx_loc}) 进行基本解释...")
                classix_model.explain(idx_loc, plot=True, alpha=0.1)
                plt.show() 

            elif len(valid_mmsis) >= 2:
                if len(valid_mmsis) > 2:
                    print(f"输入了 {len(valid_mmsis)} 个MMSI，将仅分析前两个: {valid_mmsis[0]} 和 {valid_mmsis[1]} 的路径。")
                mmsi1, mmsi2 = valid_mmsis[0], valid_mmsis[1]
                explain_connection_path(classix_model, data_df, mmsi1, mmsi2)

        except Exception as e:
            print(f"解释MMSI时发生错误: {e}")

if __name__ == "__main__":
    main()
