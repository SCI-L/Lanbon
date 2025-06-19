import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from Trajectory import Trajectory, geo_info # 假设 Trajectory 模块已提供
from SrMReader import SrMReader # 假设 SrMReader 模块已提供
import os
from classix import CLASSIX

# 确保matplotlib能够显示中文
plt.rcParams['font.family'] = 'Hiragino Sans GB'
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']


class TrajectoryFeatureExtractor:
    """轨迹特征提取器，用于从轨迹中提取特征向量"""

    def __init__(self, feature_types=None):
        """
        初始化特征提取器
        :param feature_types: 要提取的特征类型列表，默认为None表示提取所有特征
        """
        # 增加了 'cog_stats', 'sog_stats', 'acceleration'
        self.available_features = [
            'spatial', 'sog_stats', 'cog_stats', 'length', 'duration',
            'turning_points', 'speed_change', 'acceleration'
        ]

        if feature_types is None:
            self.feature_types = self.available_features
        else:
            self.feature_types = [f for f in feature_types if f in self.available_features]

    def extract_features(self, trajectory):
        """
        从轨迹中提取特征
        :param trajectory: Trajectory对象
        :return: 特征向量
        """
        features = []

        # 提取空间特征（起点、终点）
        if 'spatial' in self.feature_types:
            start_lon, start_lat = trajectory.lon[0], trajectory.lat[0]
            end_lon, end_lat = trajectory.lon[-1], trajectory.lat[-1]
            features.extend([start_lon, start_lat, end_lon, end_lat])

        # 提取航速特征 (SOG - Speed Over Ground) 的统计量：平均值、最大值、最小值、标准差
        # 根据你的要求，这里明确包含了平均值、最大值、最小值
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
                # 航向的平均值通常需要考虑角度的特殊性（0和360是相同的）
                # 这里使用之前已有的循环平均方法，并添加最大值和最小值
                sin_sum = np.sum(np.sin(np.radians(trajectory.cog)))
                cos_sum = np.sum(np.cos(np.radians(trajectory.cog)))
                mean_cog = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                max_cog = np.max(trajectory.cog)
                min_cog = np.min(trajectory.cog)
            else:
                mean_cog, max_cog, min_cog = 0, 0, 0
            features.extend([mean_cog, max_cog, min_cog])


        # 提取方向变化特征（与cog_stats分开，保持原有逻辑，因为你没有明确要求删除）
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
                mean_direction_change, max_direction_change = 0, 0 # 注意这里mean_direction被移除，因为cog_stats已经包含
            features.extend([mean_direction_change, max_direction_change])

        # 提取轨迹长度
        if 'length' in self.feature_types:
            length = trajectory.get_length() if hasattr(trajectory, 'get_length') else 0
            features.append(length)

        # 提取轨迹持续时间
        if 'duration' in self.feature_types:
            duration = (trajectory.ts[-1] - trajectory.ts[0]) if len(trajectory.ts) > 1 else 0
            features.append(duration)

        # 提取转向点数量（使用DP算法的关键点）
        if 'turning_points' in self.feature_types:
            # 使用tdkc()方法获取压缩后的轨迹
            compressed_traj = trajectory.tdkc() if hasattr(trajectory, 'tdkc') else trajectory
            num_turning_points = compressed_traj.point_num if hasattr(compressed_traj, 'point_num') else 0
            features.append(num_turning_points)

        # 提取速度变化特征 (与sog_stats分开，保持原有逻辑)
        if 'speed_change' in self.feature_types:
            if len(trajectory.sog) > 1:
                speed_changes = np.diff(trajectory.sog)
                mean_speed_change = np.mean(np.abs(speed_changes))
                max_speed_change = np.max(np.abs(speed_changes))
            else:
                mean_speed_change, max_speed_change = 0, 0
            features.extend([mean_speed_change, max_speed_change])
        
        # 提取加速度特征：平均值、最大值、最小值
        if 'acceleration' in self.feature_types:
            if len(trajectory.sog) > 1 and len(trajectory.ts) > 1:
                # 计算速度差
                speed_diffs = np.diff(trajectory.sog)
                # 计算时间差（单位为秒，假设trajectory.ts是Unix时间戳）
                time_diffs = np.diff(trajectory.ts)
                
                # 避免除以零
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

    # 为每个聚类分配一个颜色
    # +1 for noise if present
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

    # 绘制每条轨迹
    for i, traj in enumerate(trajectories_to_plot):
        label = labels[i]
        color = label_to_color[label]
        plt.plot(traj.lon, traj.lat, color=color, alpha=0.5, linewidth=1)

    # 创建图例
    legend_handles = []
    for label in sorted(unique_labels):
        display_name = f'簇 {label}' if label != -1 else '噪声'
        # 检查此标签是否实际有点，然后再添加到图例中
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
    data_path = '/Users/peng/Desktop/AIS数据/美国丹佛/message_type_1_2_3_20241111.txt'  # 请确保路径正确

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
    for traj in filtered_trajectories:
        try:
            # 使用TDKC算法进行轨迹压缩
            compressed_traj = traj.tdkc()
            if compressed_traj.point_num >= 2:  # 确保压缩后至少有2个点
                compressed_trajectories.append(compressed_traj)
        except Exception as e:
            print(f"压缩轨迹 {traj.get_id()} 时出错: {e}")
    
    print(f"压缩后剩余 {len(compressed_trajectories)} 条轨迹")

    if not compressed_trajectories:
        print("压缩后没有符合条件的轨迹进行聚类。")
        return

    # 5. 特征提取
    print("正在提取轨迹特征...")
    # 修改这里的 feature_types 以包含新的特征
    feature_types=['spatial', 'sog_stats', 'cog_stats', 'acceleration'] 
    feature_extractor = TrajectoryFeatureExtractor(feature_types=feature_types)
    features_list = []
    valid_trajectories = []

    for traj in compressed_trajectories:
        try:
            feature_vector = feature_extractor.extract_features(traj)
            if np.all(np.isfinite(feature_vector)):  # 确保特征值都是有效的
                features_list.append(feature_vector)
                valid_trajectories.append(traj)
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

    # 7. 使用CLASSIX进行聚类
    print("正在使用CLASSIX进行聚类...")
    # 尝试不同的参数
    radius_values = [0.1, 0.125, 0.15, 0.175, 0.2 , 0.225 , 0.275 , 0.3]
    minPts = 7 # 定义minPts变量
    best_silhouette = -1  # 初始化为-1，因为轮廓系数范围是[-1,1]，越大越好
    best_labels = None
    best_radius = None
    best_num_clusters = 0

    for radius in radius_values:
        # 创建CLASSIX聚类器
        classix = CLASSIX(radius=radius, minPts=minPts, sorting='pca', group_merging='distance')
        classix.fit(scaled_features)
        labels = classix.predict(scaled_features)
        
        # 计算聚类数量（不包括噪声点）
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        # 如果只有一个聚类或全是噪声点，则跳过评估
        if num_clusters <= 1:
            print(f"半径 {radius} 产生了 {num_clusters} 个聚类，跳过评估")
            continue
        
        # 计算轮廓系数（只考虑非噪声点）
        non_noise_indices = labels != -1
        if np.sum(non_noise_indices) > 1:
            silhouette_avg = silhouette_score(scaled_features[non_noise_indices], 
                                             labels[non_noise_indices])
            print(f"半径 {radius} 产生了 {num_clusters} 个聚类，轮廓系数: {silhouette_avg:.4f}")
            
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_labels = labels
                best_radius = radius
                best_num_clusters = num_clusters

    if best_labels is None:
        print("未能找到有效的聚类结果，尝试使用默认参数")
        classix = CLASSIX(radius=0.2, minPts=10, sorting='pca', group_merging='distance')
        classix.fit(scaled_features)  # 先fit
        best_labels = classix.predict(scaled_features)  # 再predict
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
    # 输出聚类解释
    if hasattr(classix, 'explain'):
        print("\n聚类解释:")
        # 基本解释，不带图形
        # classix.explain()

        # 如果需要可视化聚类结果，取消下面的注释
        classix.explain(plot=True)

        # 如果需要解释特定数据点的聚类情况，取消下面的注释并提供有效的索引
        # 例如，查看第0个和第1个数据点
        # classix.explain(0, 1)

        # 或者查看两条特定的轨迹
        # 假设您想查看MMSI为123456789和987654321的两条轨迹
        # traj1_idx = valid_trajectories.index(next(t for t in valid_trajectories if t.get_id() == "538006748"))
        # traj2_idx = valid_trajectories.index(next(t for t in valid_trajectories if t.get_id() == "367776660"))
        # classix.explain(traj1_idx, traj2_idx)

    
    # 统计每个聚类的轨迹数量
    for label in unique_labels:
        count = np.sum(best_labels == label)
        label_name = f"簇 {label}" if label != -1 else "噪声"
        print(f"- {label_name}: {count} 条轨迹")

    # 9. 可视化聚类结果
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"trajectory_clusters_r{best_radius}_m{minPts}.png")
    
    # 修复格式化字符串错误
    silhouette_text = f"{best_silhouette:.4f}" if best_silhouette > -1 else "N/A"
    params_text = f"参数: radius={best_radius}, minPts={minPts}, 特征={feature_types}, 轮廓系数={silhouette_text}"
    visualize_clusters(valid_trajectories, best_labels, best_num_clusters, output_path, params_text)


if __name__ == "__main__":
    main()
