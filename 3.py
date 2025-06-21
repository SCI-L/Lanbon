import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from Trajectory import Trajectory, geo_info 
from SrMReader import SrMReader 
import os
from classix import CLASSIX
import pandas as pd # 导入 pandas 库
# from sklearn.decomposition import PCA # 删除：不再需要 PCA
# from random import sample # 删除：不再需要 sample 用于 3D 绘图中的采样

# 确保matplotlib能够显示中文
plt.rcParams['font.family'] = 'Hiragino Sans GB'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']


class TrajectoryFeatureExtractor:
    """轨迹特征提取器，用于从轨迹中提取特征向量"""

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
        # 提取空间特征（起点、终点）
        if 'spatial' in self.feature_types:
            start_lon, start_lat = trajectory.lon[0], trajectory.lat[0]
            end_lon, end_lat = trajectory.lon[-1], trajectory.lat[-1]
            features.extend([start_lon, start_lat, end_lon, end_lat])

        # 提取航速特征 (SOG - Speed Over Ground) 的统计量：平均值、最大值、最小值、标准差
        if 'sog_stats' in self.feature_types:
            if len(trajectory.sog) > 0:
                mean_sog = np.mean(trajectory.sog)
                max_sog = np.max(trajectory.sog)
                min_sog = np.min(trajectory.sog)
                std_sog = np.std(trajectory.sog)
            else:
                mean_sog, max_sog, min_sog, std_sog = 0, 0, 0, 0
            features.extend([mean_sog, max_sog, min_sog, std_sog])

        # 提取航向特征 (COG - Course Over Ground) 的统计量：平均值、最大值、最小值
        if 'cog_stats' in self.feature_types:
            if len(trajectory.cog) > 0:
                sin_sum = np.sum(np.sin(np.radians(trajectory.cog)))
                cos_sum = np.sum(np.cos(np.radians(trajectory.cog)))
                mean_cog = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                max_cog = np.max(trajectory.cog)
                min_cog = np.min(trajectory.cog)
            else:
                mean_cog, max_cog, min_cog = 0, 0, 0
            features.extend([mean_cog, max_cog, min_cog])

        # 提取方向变化特征
        if 'direction' in self.feature_types:
            if len(trajectory.cog) > 1:
                direction_changes = []
                for i in range(1, len(trajectory.cog)):
                    change = abs(trajectory.cog[i] - trajectory.cog[i - 1])
                    if change > 180:
                        change = 360 - change
                    direction_changes.append(change)
                mean_direction_change = np.mean(direction_changes) if direction_changes else 0
                max_direction_change = np.max(direction_changes) if direction_changes else 0
            else:
                mean_direction_change, max_direction_change = 0, 0
            features.extend([mean_direction_change, max_direction_change])

        # 提取轨迹长度
        if 'length' in self.feature_types:
            length = trajectory.get_length() if hasattr(trajectory, 'get_length') else 0
            features.append(length)

        # 提取轨迹持续时间
        if 'duration' in self.feature_types:
            duration = (trajectory.ts[-1] - trajectory.ts[0]) if len(trajectory.ts) > 1 else 0
            features.append(duration)

        # 提取转向点数量
        if 'turning_points' in self.feature_types:
            compressed_traj = trajectory.tdkc() if hasattr(trajectory, 'tdkc') else trajectory
            num_turning_points = compressed_traj.point_num if hasattr(compressed_traj, 'point_num') else 0
            features.append(num_turning_points)

        # 提取速度变化特征
        if 'speed_change' in self.feature_types:
            if len(trajectory.sog) > 1:
                speed_changes = np.diff(trajectory.sog)
                mean_speed_change = np.mean(np.abs(speed_changes))
                max_speed_change = np.max(np.abs(speed_changes))
            else:
                mean_speed_change, max_speed_change = 0, 0
            features.extend([mean_speed_change, max_speed_change])
        
        # 提取加速度特征
        if 'acceleration' in self.feature_types:
            if len(trajectory.sog) > 1 and len(trajectory.ts) > 1:
                speed_diffs = np.diff(trajectory.sog)
                time_diffs = np.diff(trajectory.ts)
                accelerations = [sd / td for sd, td in zip(speed_diffs, time_diffs) if td > 0]
                if accelerations:
                    mean_acceleration = np.mean(accelerations)
                    max_acceleration = np.max(accelerations)
                    min_acceleration = np.min(accelerations)
                else:
                    mean_acceleration, max_acceleration, min_acceleration = 0, 0, 0
            else:
                mean_acceleration, max_acceleration, min_acceleration = 0, 0, 0
            features.extend([mean_acceleration, max_acceleration, min_acceleration])
        return np.array(features)


def visualize_clusters(trajectories_to_plot, labels, num_clusters, output_path="trajectory_clusters.png",
                       params_text=""):
    """
    可视化聚类结果
    :param trajectories_to_plot: 用于绘图的轨迹对象列表
    :param labels: 每个轨迹的聚类标签
    :param num_clusters: 聚类的数量 (不包括噪声)
    :param output_path: 保存图片的路径
    :param params_text: 显示在图上的参数信息文本
    """
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    color_count = len(unique_labels) if -1 not in unique_labels else len(unique_labels) - 1
    colors = plt.cm.rainbow(np.linspace(0, 1, color_count if color_count > 0 else 1))
    label_to_color = {}
    color_idx = 0
    for label in sorted(unique_labels):
        if label == -1:
            label_to_color[label] = 'k'  # 噪声点用黑色表示
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

def main():
    """主函数，使用轨迹聚类算法"""
    # 1. 读取AIS数据
    reader = SrMReader()
    data_path = '/Users/peng/Desktop/AIS数据/美国丹佛/message_type_1_2_3_20241111.txt' # 请确保路径正确

    if not os.path.exists(data_path):
        print(f"文件不存在: {data_path}")
        return

    print(f"正在读取AIS数据: {data_path}")
    trajectories_dict = reader.read_multiple_track(data_path, '1_2_3')

    # 2. 转换为轨迹列表
    original_trajectories_list = list(trajectories_dict.values())
    print(f"共读取 {len(original_trajectories_list)} 条轨迹")

    # 3. 数据预处理：过滤掉点数太少的轨迹
    min_points_for_traj = 10
    filtered_trajectories = [traj for traj in original_trajectories_list if traj.point_num >= min_points_for_traj]
    print(f"过滤后剩余 {len(filtered_trajectories)} 条轨迹 (点数 >= {min_points_for_traj})")

    if not filtered_trajectories:
        print("没有符合条件的轨迹进行聚类。")
        return

    # 4. 轨迹压缩
    print("正在压缩轨迹...")
    compressed_trajectories = []
    mmsi_to_original_traj = {traj.get_id(): traj for traj in filtered_trajectories}

    for traj in filtered_trajectories:
        try:
            compressed_traj = traj.tdkc()
            if compressed_traj.point_num >= 2:
                compressed_trajectories.append(compressed_traj)
        except Exception as e:
            print(f"压缩轨迹 {traj.get_id()} 时出错: {e}")
    
    print(f"压缩后剩余 {len(compressed_trajectories)} 条轨迹")

    if not compressed_trajectories:
        print("压缩后没有符合条件的轨迹进行聚类。")
        return

    # 5. 特征提取
    print("正在提取轨迹特征...")
    feature_types=['spatial', 'sog_stats', 'cog_stats', 'acceleration'] 
    feature_extractor = TrajectoryFeatureExtractor(feature_types=feature_types)
    features_list = []
    valid_trajectories = [] # 存储用于聚类的有效轨迹对象
    valid_mmsi_ids = []

    for i, traj in enumerate(compressed_trajectories):
        try:
            feature_vector = feature_extractor.extract_features(traj)
            if np.all(np.isfinite(feature_vector)): 
                features_list.append(feature_vector)
                valid_trajectories.append(traj)
                valid_mmsi_ids.append(traj.get_id()) 
            else:
                print(f"轨迹 {traj.get_id()} 的特征包含无效值，已跳过")
        except Exception as e:
            print(f"提取轨迹 {traj.get_id()} 特征时出错: {e}")

    if not features_list:
        print("没有有效的特征向量进行聚类。")
        return

    features_array = np.array(features_list)
    print(f"提取了 {len(features_array)} 条轨迹的特征，每条特征维度: {features_array.shape[1]}")

    # 6. 特征标准化
    print("正在标准化特征...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_array)

    # 将标准化后的特征转换为 DataFrame，并以 MMSI 为索引
    # 为列命名，例如 'feature_0', 'feature_1', ...
    feature_column_names = [f'feature_{i}' for i in range(scaled_features.shape[1])]
    data_df = pd.DataFrame(scaled_features, index=valid_mmsi_ids, columns=feature_column_names)
    print(f"特征数据已转换为 DataFrame，包含 {data_df.shape[0]} 行，索引为 MMSI。")

    # 7. 使用CLASSIX进行聚类
    print("正在使用CLASSIX进行聚类...")
    radius_values = [0.1, 0.125, 0.15, 0.175, 0.2 , 0.225 , 0.275 , 0.3]
    minPts = 7
    best_silhouette = -1
    best_labels = None
    best_radius = None
    best_num_clusters = 0
    classix_model = None 

    for radius in radius_values:
        classix = CLASSIX(radius=radius, minPts=minPts, sorting='pca', group_merging='distance')
        # 传入 DataFrame 而不是 NumPy 数组
        classix.fit(data_df)
        labels = classix.predict(data_df)
        
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if num_clusters <= 1:
            print(f"半径 {radius} 产生了 {num_clusters} 个聚类，跳过评估")
            continue
        
        non_noise_indices = labels != -1
        if np.sum(non_noise_indices) > 1:
            # 计算轮廓系数时仍使用原始的 scaled_features 数组，因为它不依赖索引
            silhouette_avg = silhouette_score(scaled_features[non_noise_indices], 
                                             labels[non_noise_indices])
            print(f"半径 {radius} 产生了 {num_clusters} 个聚类，轮廓系数: {silhouette_avg:.4f}")
            
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_labels = labels
                best_radius = radius
                classix_model = classix 
                best_num_clusters = num_clusters

    if best_labels is None:
        print("未能找到有效的聚类结果，尝试使用默认参数")
        classix_model = CLASSIX(radius=0.2, minPts=10, sorting='pca', group_merging='distance')
        # 传入 DataFrame 而不是 NumPy 数组
        classix_model.fit(data_df)
        best_labels = classix_model.predict(data_df)
        best_radius = 0.5
        unique_labels = np.unique(best_labels)
        best_num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # 8. 输出聚类结果
    unique_labels = np.unique(best_labels)
    noise_count = np.sum(best_labels == -1)
    print(f"噪声点数量: {noise_count} ({noise_count/len(best_labels)*100:.2f}%)")
    print(f"\n最佳聚类结果:")
    print(f"- 半径: {best_radius}")
    print(f"- 中心点数量: {minPts}")
    print(f"- 聚类数量: {best_num_clusters}")
    print(f"- 轮廓系数: {best_silhouette:.4f}" if best_silhouette > -1 else "- 轮廓系数: 未计算")
    
    # 输出聚类解释 (整体，plot=True可能会生成图表)
    if classix_model is not None and hasattr(classix_model, 'explain'):
        print("\n整体聚类解释 (可能会弹出图表):")
        # alpha=1 表示不透明度，对于少量数据点，设为1可以看得更清楚
        classix_model.explain(plot=True, alpha=1) 

    # 统计每个聚类的轨迹数量
    print("\n各聚类的轨迹数量:")
    for label in unique_labels:
        count = np.sum(best_labels == label)
        label_name = f"簇 {label}" if label != -1 else "噪声"
        print(f"- {label_name}: {count} 条轨迹")

    # 输出 MMSI 码所属簇的文本清单 (按簇分组)
    print("\n--- MMSI 码及其所属簇的完整清单 (按簇分组) ---")
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    # 使用用户提供的硬编码路径
    mmsi_list_file_path = os.path.join("/Users/peng/Desktop/results", "mmsi_cluster_grouped_list.txt")
    
    clusters_mmsis = {int(label): [] for label in unique_labels}
    
    # ==================== 代码修复之处 ====================
    # 使用 zip 函数可以安全、高效地将 MMSI 与其对应的聚类标签配对。
    # 这种方法避免了在可能重复的索引上使用 .get_loc()，从而规避了之前的错误。
    for mmsi, label in zip(data_df.index, best_labels):
        cluster_label = int(label)  # 直接从 best_labels 中获取的 label 是一个标量，可安全转换
        clusters_mmsis[cluster_label].append(mmsi)
    # ======================================================

    with open(mmsi_list_file_path, 'w') as f:
        # 在写入文件时，最好也处理一下 clusters_mmsis 的键不存在的情况
        for label in sorted(unique_labels):
            label_name = f"簇 {label}" if label != -1 else "噪声"
            # 使用 .get(label, []) 来安全地获取列表，以防万一
            mmsis_in_cluster = clusters_mmsis.get(label, [])
            
            f.write(f"\n--- {label_name} (共 {len(mmsis_in_cluster)} 条轨迹) ---\n")
            print(f"\n--- {label_name} (共 {len(mmsis_in_cluster)} 条轨迹) ---")
            
            if mmsis_in_cluster:
                for mmsi_item in mmsis_in_cluster:
                    line = f"MMSI: {mmsi_item}"
                    f.write(line + '\n')
            else:
                f.write("无轨迹\n")
                print("无轨迹")

    print(f"\nMMSI 码清单（按簇分组）已保存到 {mmsi_list_file_path}")

    # 9. 可视化聚类结果 (2D)
    output_path_2d = os.path.join(output_dir, f"trajectory_clusters_r{best_radius}_m{minPts}_2D.png")
    silhouette_text = f"{best_silhouette:.4f}" if best_silhouette > -1 else "N/A"
    params_text = f"参数: radius={best_radius}, minPts={minPts}, 特征={feature_types}, 轮廓系数={silhouette_text}"
    visualize_clusters(valid_trajectories, best_labels, best_num_clusters, output_path_2d, params_text)


    # 10. 查询MMSI码所属簇及解释多条轨迹
    print("\n--- MMSI 码聚类查询与解释 ---")
    print("你可以输入一个或两个 MMSI 码 (用逗号分隔) 进行详细解释，或输入 'q' 退出。")
    print("例如：输入 'MMSI1, MMSI2' 来解释两艘船舶的聚类。")
    while True:
        mmsi_input_str = input("请输入MMSI码(s): ").strip()
        if mmsi_input_str.lower() == 'q':
            break

        mmsis_to_explain = [m.strip() for m in mmsi_input_str.split(',') if m.strip()]
        
        valid_input_mmsis = [] # 存储用户输入且在数据集中存在的MMSI
        for mmsi in mmsis_to_explain:
            if mmsi in data_df.index: # 直接检查 MMSI 是否在 DataFrame 的索引中
                valid_input_mmsis.append(mmsi)
                # 根据 MMSI 直接从 DataFrame 获取数据
                # 由于可能存在重复MMSI，我们只解释第一个匹配项
                mmsi_location = data_df.index.get_loc(mmsi)
                # 如果是重复索引，get_loc返回掩码，我们找到第一个True的位置
                if isinstance(mmsi_location, np.ndarray) and mmsi_location.dtype == bool:
                    label_index = np.where(mmsi_location)[0][0]
                elif isinstance(mmsi_location, slice):
                     label_index = mmsi_location.start
                else: # 是单个整数位置
                     label_index = mmsi_location

                cluster_label = int(best_labels[label_index])

                label_name = f"簇 {cluster_label}" if cluster_label != -1 else "噪声"
                print(f"MMSI码 {mmsi} 属于 {label_name}.")
                # 使用 .loc[] 通过索引获取行，对于重复索引，它会返回所有匹配行
                print(f"该轨迹的标准化特征向量(们): \n{data_df.loc[mmsi].values}")
            else:
                print(f"MMSI码 {mmsi} 未找到或在数据预处理阶段被过滤。")
                if mmsi in mmsi_to_original_traj:
                    print(f"注意：MMSI {mmsi} 的原始轨迹 ({mmsi_to_original_traj[mmsi].point_num}点) 可能因点数过少或压缩后点数不足而被过滤。")
        
        if valid_input_mmsis:
            try:
                if classix_model is not None and hasattr(classix_model, 'explain'):
                    print(f"\n正在尝试解释所选MMSI轨迹 ({valid_input_mmsis}) 的聚类...")
                    # 直接传入 MMSI 字符串列表
                    classix_model.explain(*valid_input_mmsis, plot=True,alpha=1) 
                    print("解释图表已生成（如果支持）。请查看弹出的 Matplotlib 窗口。")

            except Exception as e:
                print(f"解释所选MMSI聚类时发生错误: {e}")
                print("请检查 CLASSIX 库的文档以获取 explain 方法的正确用法。")
        else:
            print("没有有效的MMSI码可以进行解释。")


if __name__ == "__main__":
    main()
