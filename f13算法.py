import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist
import random
from collections import defaultdict, deque
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class GardenPathAnalyzer:
    """江南古典园林路径分析器

    实现功能:
    - 加载园林数据
    - 提取道路元素坐标
    - 识别园林出入口
    - 构建路径网络图模型
    - 计算路径特征指标
    - 使用强化学习寻找最优游线
    """

    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        """初始化分析器

        Args:
            data_dir: 数据文件夹路径
        """
        self.data_dir = data_dir
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }
        self.element_types = {
            0: '半开放建筑', 1: '实体建筑', 2: '道路',
            3: '假山', 4: '水体', 5: '植物'
        }

        # 设置计算参数
        self.path_segment_threshold = 1500  # 路径段构建阈值(mm)
        self.turn_angle_threshold = np.pi/6  # 转弯角度阈值 (30度)
        self.entry_exit_clustering_eps = 5000  # 入口聚类参数(mm)
        self.step_size = 1000  # 路径采样步长(mm)
        self.visual_range = 10000  # 视野范围(mm)

        # 趣味性评分权重
        self.w_curv = 2.0   # 曲折度权重
        self.w_view = 3.0   # 异景程度权重
        self.w_exp = 1.5    # 探索性权重
        self.w_len = 0.1    # 路径长度权重（惩罚项）
        self.score_C = 1000  # 防止除零常数

        # RL参数
        self.gamma = 0.95  # 折扣因子
        self.alpha = 0.1   # 学习率
        self.epsilon = 0.2  # 探索率
        self.episodes = 1000  # 训练轮数

    def load_garden_data(self, garden_id):
        """加载单个园林的坐标数据

        Args:
            garden_id (int): 园林ID (1-10)

        Returns:
            dict: 包含各景观元素坐标的字典
        """
        garden_name = self.gardens[garden_id]
        data_path = f"{self.data_dir}/{garden_id}. {garden_name}/4-{garden_name}数据坐标.xlsx"

        garden_data = {
            'name': garden_name,
            'id': garden_id,
            'elements': {}
        }

        try:
            excel_file = pd.ExcelFile(data_path)

            # 读取6种景观元素数据
            for i, sheet_name in enumerate(excel_file.sheet_names):
                if i < 6:  # 只读取前6个工作表
                    element_name = self.element_types[i]
                    df = pd.read_excel(data_path, sheet_name=sheet_name)
                    garden_data['elements'][element_name] = df

            print(f"✓ 成功加载 {garden_name} 的数据")
            return garden_data

        except Exception as e:
            print(f"✗ 加载 {garden_name} 数据时出错: {e}")
            return None

    def extract_road_coordinates(self, garden_data):
        """从园林数据中提取道路坐标

        Args:
            garden_data (dict): 园林数据字典

        Returns:
            list: 道路坐标点列表 [(x1, y1), (x2, y2), ...]
        """
        road_coords = []

        if '道路' in garden_data['elements']:
            road_df = garden_data['elements']['道路']

            if len(road_df.columns) >= 2:
                # 使用第二列（不区分线段的点位坐标）
                coord_col = road_df.columns[1]

                for _, row in road_df.iterrows():
                    coord_str = str(row[coord_col])
                    if '{' in coord_str and '}' in coord_str:
                        try:
                            # 解析坐标字符串 {x,y,z}
                            coord_str = coord_str.strip('{}')
                            coords = [float(x.strip()) for x in coord_str.split(',')]
                            if len(coords) >= 2:
                                road_coords.append((coords[0], coords[1]))
                        except ValueError:
                            continue

        print(f"提取了 {len(road_coords)} 个道路坐标点")
        return road_coords

    def extract_building_coordinates(self, garden_data):
        """从园林数据中提取建筑物坐标

        Args:
            garden_data (dict): 园林数据字典

        Returns:
            tuple: (实体建筑坐标, 半开放建筑坐标)
        """
        solid_building_coords = []
        semi_open_building_coords = []

        if '实体建筑' in garden_data['elements']:
            solid_df = garden_data['elements']['实体建筑']
            if len(solid_df.columns) >= 2:
                coord_col = solid_df.columns[1]
                for _, row in solid_df.iterrows():
                    coord_str = str(row[coord_col])
                    if '{' in coord_str and '}' in coord_str:
                        try:
                            coord_str = coord_str.strip('{}')
                            coords = [float(x.strip()) for x in coord_str.split(',')]
                            if len(coords) >= 2:
                                solid_building_coords.append((coords[0], coords[1]))
                        except ValueError:
                            continue

        if '半开放建筑' in garden_data['elements']:
            semi_df = garden_data['elements']['半开放建筑']
            if len(semi_df.columns) >= 2:
                coord_col = semi_df.columns[1]
                for _, row in semi_df.iterrows():
                    coord_str = str(row[coord_col])
                    if '{' in coord_str and '}' in coord_str:
                        try:
                            coord_str = coord_str.strip('{}')
                            coords = [float(x.strip()) for x in coord_str.split(',')]
                            if len(coords) >= 2:
                                semi_open_building_coords.append((coords[0], coords[1]))
                        except ValueError:
                            continue

        print(f"提取了 {len(solid_building_coords)} 个实体建筑坐标点和 {len(semi_open_building_coords)} 个半开放建筑坐标点")
        return solid_building_coords, semi_open_building_coords

    def extract_landscape_elements(self, garden_data):
        """提取园林中的景观元素坐标

        Args:
            garden_data (dict): 园林数据字典

        Returns:
            dict: 各类景观元素的坐标点
        """
        elements = {}

        # 提取假山、水体和植物坐标
        for element_name in ['假山', '水体', '植物']:
            if element_name in garden_data['elements']:
                coords = []
                df = garden_data['elements'][element_name]

                if element_name == '植物':
                    # 植物数据包含坐标和半径
                    if len(df.columns) >= 2:
                        coord_col = df.columns[0]
                        radius_col = df.columns[1]
                        for _, row in df.iterrows():
                            coord_str = str(row[coord_col])
                            if '{' in coord_str and '}' in coord_str:
                                try:
                                    coord_str = coord_str.strip('{}')
                                    coord_parts = [float(x.strip()) for x in coord_str.split(',')]
                                    radius = float(row[radius_col])
                                    if len(coord_parts) >= 2:
                                        coords.append((coord_parts[0], coord_parts[1], radius))
                                except ValueError:
                                    continue
                else:
                    # 假山和水体只需要坐标
                    if len(df.columns) >= 2:
                        coord_col = df.columns[1]  # 使用不区分线段的坐标列
                        for _, row in df.iterrows():
                            coord_str = str(row[coord_col])
                            if '{' in coord_str and '}' in coord_str:
                                try:
                                    coord_str = coord_str.strip('{}')
                                    coord_parts = [float(x.strip()) for x in coord_str.split(',')]
                                    if len(coord_parts) >= 2:
                                        coords.append((coord_parts[0], coord_parts[1]))
                                except ValueError:
                                    continue

                elements[element_name] = coords
                print(f"提取了 {len(coords)} 个{element_name}坐标点")

        return elements

    def reconstruct_path_segments(self, road_coords):
        """重构路径段

        将离散的道路坐标点重构为连续的路径段

        Args:
            road_coords (list): 道路坐标点列表 [(x1, y1), (x2, y2), ...]

        Returns:
            list: 路径段列表，每个路径段是有序的坐标点序列
        """
        # 转换为numpy数组，便于计算
        road_points = np.array(road_coords)

        # 初始化
        path_segments = []
        remaining_points = set(range(len(road_points)))

        # 当还有未处理的点时，继续构建路径段
        while remaining_points:
            # 从剩余点中选择一个作为起点
            start_idx = next(iter(remaining_points))
            remaining_points.remove(start_idx)

            # 初始化新的路径段
            current_segment = [road_points[start_idx]]
            current_point = road_points[start_idx]

            # 向前扩展路径段
            while True:
                if not remaining_points:
                    break

                # 计算当前点到所有剩余点的距离
                distances = []
                for idx in remaining_points:
                    dist = np.linalg.norm(road_points[idx] - current_point)
                    distances.append((dist, idx))

                # 找到最近的点
                min_dist, nearest_idx = min(distances)

                # 如果最近点在阈值范围内，则添加到路径段
                if min_dist <= self.path_segment_threshold:
                    current_point = road_points[nearest_idx]
                    current_segment.append(current_point)
                    remaining_points.remove(nearest_idx)
                else:
                    break

            # 如果路径段至少包含2个点，则添加到结果中
            if len(current_segment) >= 2:
                path_segments.append(current_segment)

        print(f"重构了 {len(path_segments)} 条路径段")
        return path_segments

    def identify_key_vertices(self, path_segments, building_coords):
        """识别关键顶点

        包括端点、交叉点和建筑出入口

        Args:
            path_segments (list): 路径段列表
            building_coords (tuple): (实体建筑坐标, 半开放建筑坐标)

        Returns:
            tuple: (顶点列表, 顶点类型字典)
        """
        vertices = []
        vertex_types = {}  # 记录每个顶点的类型

        # 提取端点
        for i, segment in enumerate(path_segments):
            start_point = tuple(segment[0])
            end_point = tuple(segment[-1])

            vertices.append(start_point)
            vertices.append(end_point)

            # 标记端点类型
            if start_point not in vertex_types:
                vertex_types[start_point] = {'endpoint': True}
            else:
                vertex_types[start_point]['endpoint'] = True

            if end_point not in vertex_types:
                vertex_types[end_point] = {'endpoint': True}
            else:
                vertex_types[end_point]['endpoint'] = True

        # 识别交叉点 (为简化计算，将相近的点视为同一个交叉点)
        vertices_array = np.array(vertices)
        clustering = DBSCAN(eps=1000, min_samples=2).fit(vertices_array)

        # 处理聚类结果
        cluster_points = {}
        for i, label in enumerate(clustering.labels_):
            if label >= 0:  # 不是噪声点
                point = tuple(vertices_array[i])
                if label not in cluster_points:
                    cluster_points[label] = []
                cluster_points[label].append(point)

        # 标记交叉点
        intersection_vertices = []
        for label, points in cluster_points.items():
            if len(points) >= 2:  # 至少有2个路径相交
                # 使用平均坐标作为交叉点
                avg_point = tuple(np.mean(np.array(points), axis=0))
                intersection_vertices.append(avg_point)

                # 标记交叉点类型
                if avg_point not in vertex_types:
                    vertex_types[avg_point] = {'intersection': True}
                else:
                    vertex_types[avg_point]['intersection'] = True

        # 识别建筑出入口 (简化为建筑坐标与路径端点的近邻点)
        solid_coords, semi_open_coords = building_coords
        building_points = np.array(solid_coords + semi_open_coords)

        if len(building_points) > 0 and len(vertices_array) > 0:
            # 计算每个建筑点到路径端点的距离
            distances = cdist(building_points, vertices_array)

            # 找到每个建筑点最近的路径端点
            for i, building_point in enumerate(building_points):
                if distances[i].min() <= 3000:  # 设置距离阈值为3米
                    closest_idx = distances[i].argmin()
                    closest_vertex = tuple(vertices_array[closest_idx])

                    # 标记建筑入口
                    if closest_vertex not in vertex_types:
                        vertex_types[closest_vertex] = {'building_entrance': True}
                    else:
                        vertex_types[closest_vertex]['building_entrance'] = True

        # 合并所有顶点
        all_vertices = list(set(vertices + intersection_vertices))

        print(f"识别了 {len(all_vertices)} 个关键顶点")
        return all_vertices, vertex_types

    def identify_entry_exit(self, G, vertex_types):
        """识别园林出入口

        Args:
            G (networkx.Graph): 路径网络图
            vertex_types (dict): 顶点类型字典

        Returns:
            tuple: (入口顶点, 出口顶点)
        """
        # 获取所有端点
        endpoints = [v for v, attr in vertex_types.items() if 'endpoint' in attr and attr['endpoint']]

        if not endpoints:
            # 如果没有找到端点，使用度为1的顶点
            endpoints = [v for v, d in G.degree() if d == 1]

        if not endpoints:
            raise ValueError("无法识别园林入口和出口")

        # 转换为numpy数组进行聚类
        endpoints_array = np.array(endpoints)

        # 使用DBSCAN聚类，找出相对孤立的端点（可能是出入口）
        clustering = DBSCAN(eps=self.entry_exit_clustering_eps, min_samples=1).fit(endpoints_array)

        # 找出最外围的两个聚类中心作为出入口
        cluster_centers = []
        for label in set(clustering.labels_):
            if label >= 0:  # 不是噪声点
                cluster_points = endpoints_array[clustering.labels_ == label]
                center = np.mean(cluster_points, axis=0)
                cluster_centers.append((label, center))

        if len(cluster_centers) < 2:
            # 如果聚类数量不足，直接使用最远的两个端点
            if len(endpoints) >= 2:
                # 计算端点之间的距离矩阵
                dist_matrix = cdist(endpoints_array, endpoints_array)
                i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                entry = tuple(endpoints_array[i])
                exit_point = tuple(endpoints_array[j])
            else:
                # 如果只有一个端点，将它同时作为入口和出口
                entry = exit_point = tuple(endpoints_array[0])
        else:
            # 计算园林的边界轮廓
            try:
                all_nodes = np.array(list(G.nodes()))
                hull = ConvexHull(all_nodes)
                hull_center = np.mean(all_nodes[hull.vertices], axis=0)

                # 计算每个聚类中心到边界的距离
                boundary_dists = []
                for label, center in cluster_centers:
                    dist_to_boundary = np.linalg.norm(center - hull_center)
                    boundary_dists.append((label, center, dist_to_boundary))

                # 按照到边界的距离排序
                boundary_dists.sort(key=lambda x: x[2], reverse=True)

                # 选择距离最远的两个聚类中心作为入口和出口
                entry_label, entry_center, _ = boundary_dists[0]
                exit_label, exit_center, _ = boundary_dists[1] if len(boundary_dists) > 1 else boundary_dists[0]

                # 从各聚类中选择一个点作为实际的入口和出口
                entry_points = endpoints_array[clustering.labels_ == entry_label]
                exit_points = endpoints_array[clustering.labels_ == exit_label]

                entry = tuple(entry_points[0])
                exit_point = tuple(exit_points[0])
            except:
                # 如果凸包计算失败，退回到最远距离的方法
                dist_matrix = cdist(endpoints_array, endpoints_array)
                i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                entry = tuple(endpoints_array[i])
                exit_point = tuple(endpoints_array[j])

        # 标记入口和出口类型
        if entry in vertex_types:
            vertex_types[entry]['entry'] = True
        else:
            vertex_types[entry] = {'entry': True}

        if exit_point in vertex_types:
            vertex_types[exit_point]['exit'] = True
        else:
            vertex_types[exit_point] = {'exit': True}

        print(f"识别出入口: 入口坐标 {entry}, 出口坐标 {exit_point}")
        return entry, exit_point

    def build_path_graph(self, path_segments, vertices, vertex_types):
        """构建路径网络图

        Args:
            path_segments (list): 路径段列表
            vertices (list): 顶点列表
            vertex_types (dict): 顶点类型字典

        Returns:
            networkx.Graph: 路径网络图
        """
        # 创建无向图
        G = nx.Graph()

        # 添加所有顶点
        for v in vertices:
            G.add_node(v, pos=v)

        # 处理每个路径段
        for segment in path_segments:
            # 查找路径段上的顶点
            segment_points = np.array(segment)
            segment_vertices = []

            for v in vertices:
                # 计算顶点与路径段上各点的距离
                v_point = np.array(v)
                distances = np.linalg.norm(segment_points - v_point.reshape(1, 2), axis=1)

                # 如果顶点非常接近路径段上的某点，将其添加到段顶点列表
                if np.min(distances) < 1000:  # 1米阈值
                    segment_vertices.append(v)

            # 如果路径段上至少有两个顶点，添加相应的边
            if len(segment_vertices) >= 2:
                # 按照在路径段上的顺序排序顶点
                # 这里使用投影到路径方向上的距离作为排序依据
                start_point = np.array(segment[0])
                end_point = np.array(segment[-1])
                path_vector = end_point - start_point

                # 计算每个顶点在路径方向上的投影距离
                vertex_projections = []
                for v in segment_vertices:
                    v_point = np.array(v)
                    v_vector = v_point - start_point
                    projection = np.dot(v_vector, path_vector) / np.linalg.norm(path_vector)
                    vertex_projections.append((v, projection))

                # 按投影距离排序
                vertex_projections.sort(key=lambda x: x[1])
                ordered_vertices = [v for v, _ in vertex_projections]

                # 添加边
                for i in range(len(ordered_vertices) - 1):
                    v1 = ordered_vertices[i]
                    v2 = ordered_vertices[i+1]

                    # 计算边的几何属性
                    edge_length = np.linalg.norm(np.array(v1) - np.array(v2))

                    # 计算沿路径段的转折点
                    # 在v1和v2之间的路径段子序列
                    idx1 = np.argmin(np.linalg.norm(segment_points - np.array(v1).reshape(1, 2), axis=1))
                    idx2 = np.argmin(np.linalg.norm(segment_points - np.array(v2).reshape(1, 2), axis=1))

                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1

                    sub_segment = segment[idx1:idx2+1]
                    turn_count = self.count_turns(sub_segment)

                    # 添加边及其属性
                    G.add_edge(v1, v2,
                               length=edge_length,
                               geometry=sub_segment,
                               turns=turn_count)

        # 打印图的基本信息
        print(f"构建了路径网络图: {len(G.nodes())} 个顶点, {len(G.edges())} 条边")
        return G

    def count_turns(self, path_points):
        """计算路径上的转折点数量

        Args:
            path_points (list): 路径点列表

        Returns:
            int: 转折点数量
        """
        if len(path_points) < 3:
            return 0

        turn_count = 0
        points = np.array(path_points)

        for i in range(1, len(points) - 1):
            # 计算前后向量
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]

            # 归一化向量
            v1 = v1 / np.linalg.norm(v1)
            v2 = v2 / np.linalg.norm(v2)

            # 计算向量夹角
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # 如果夹角大于阈值，认为是一个转折点
            if angle > self.turn_angle_threshold:
                turn_count += 1

        return turn_count

    def calculate_path_features(self, G, path, landscape_elements):
        """计算路径的特征指标

        Args:
            G (networkx.Graph): 路径网络图
            path (list): 路径顶点序列
            landscape_elements (dict): 景观元素坐标

        Returns:
            dict: 路径特征指标
        """
        if len(path) < 2:
            return {
                'length': 0,
                'curvature': 0,
                'view_change': 0,
                'exploration': 0
            }

        # 计算路径长度
        path_length = 0
        path_geometry = []
        total_turns = 0

        for i in range(len(path) - 1):
            v1, v2 = path[i], path[i+1]
            edge_data = G.get_edge_data(v1, v2)
            path_length += edge_data['length']
            path_geometry.extend(edge_data['geometry'])
            total_turns += edge_data['turns']

        # 计算探索性 (顶点度数之和)
        exploration_score = 0
        for v in path[1:-1]:  # 排除起点和终点
            exploration_score += G.degree(v)

        # 计算异景程度 (视野变化)
        view_change_score = self.calculate_view_changes(path_geometry, landscape_elements)

        return {
            'length': path_length,
            'curvature': total_turns,
            'view_change': view_change_score,
            'exploration': exploration_score
        }

    def calculate_view_changes(self, path_geometry, landscape_elements):
        """计算路径上的视野变化

        Args:
            path_geometry (list): 路径几何坐标序列
            landscape_elements (dict): 景观元素坐标

        Returns:
            float: 视野变化得分
        """
        if len(path_geometry) < 2:
            return 0

        # 路径等距采样
        sampled_points = []
        path_array = np.array(path_geometry)

        # 计算路径总长度
        total_length = 0
        for i in range(len(path_array) - 1):
            segment_length = np.linalg.norm(path_array[i+1] - path_array[i])
            total_length += segment_length

        # 确定采样点数量 (每step_size取一个点)
        num_samples = max(2, int(total_length / self.step_size))

        # 等距采样
        accumulated_length = 0
        current_sample_idx = 0
        target_length = total_length / num_samples

        sampled_points.append(path_array[0])

        for i in range(len(path_array) - 1):
            segment_length = np.linalg.norm(path_array[i+1] - path_array[i])
            segment_direction = (path_array[i+1] - path_array[i]) / segment_length

            while accumulated_length + segment_length >= target_length * (current_sample_idx + 1):
                # 计算新采样点位置
                remaining = target_length * (current_sample_idx + 1) - accumulated_length
                sample_point = path_array[i] + segment_direction * remaining
                sampled_points.append(sample_point)
                current_sample_idx += 1

                if current_sample_idx >= num_samples - 1:
                    break

            accumulated_length += segment_length

            if current_sample_idx >= num_samples - 1:
                break

        # 添加路径终点
        if len(sampled_points) < num_samples:
            sampled_points.append(path_array[-1])

        # 计算每个采样点的视野内景观元素
        viewsheds = []

        for point in sampled_points:
            viewshed = set()

            # 检查每种景观元素
            for element_type, elements in landscape_elements.items():
                for element in elements:
                    if element_type == '植物':
                        # 植物有半径信息
                        x, y, radius = element
                        distance = np.linalg.norm(np.array([x, y]) - point)
                        if distance <= self.visual_range:
                            viewshed.add((element_type, x, y))
                    else:
                        # 其他元素只有坐标
                        x, y = element
                        distance = np.linalg.norm(np.array([x, y]) - point)
                        if distance <= self.visual_range:
                            viewshed.add((element_type, x, y))

            viewsheds.append(viewshed)

        # 计算视野变化总量
        view_changes = 0
        for i in range(1, len(viewsheds)):
            # 对称差集表示视野变化
            view_diff = len(viewsheds[i] ^ viewsheds[i-1])
            view_changes += view_diff

        return view_changes

    def calculate_interest_score(self, path_features):
        """计算路径的趣味性得分

        Args:
            path_features (dict): 路径特征指标

        Returns:
            float: 趣味性得分
        """
        # 提取各特征值
        L_len = path_features['length']
        L_curv = path_features['curvature']
        L_view = path_features['view_change']
        L_exp = path_features['exploration']

        # 使用模型中定义的公式计算得分
        numerator = (self.w_curv * L_curv + self.w_view * L_view + self.w_exp * L_exp)
        denominator = (self.w_len * L_len + self.score_C)

        interest_score = numerator / denominator
        return interest_score

    def find_optimal_path_rl(self, G, entry, exit_point, landscape_elements):
        """使用强化学习寻找最优游览路径

        使用Q-learning算法优化路径选择

        Args:
            G (networkx.Graph): 路径网络图
            entry (tuple): 入口坐标
            exit_point (tuple): 出口坐标
            landscape_elements (dict): 景观元素坐标

        Returns:
            tuple: (最优路径, 趣味性得分)
        """
        # 将顶点转换为索引，便于Q-table存储
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}

        n_states = len(nodes)

        # 初始化Q-table
        Q = np.zeros((n_states, n_states))

        # 访问计数，用于鼓励探索新路径
        visit_count = defaultdict(int)

        # 转换入口和出口为索引
        entry_idx = node_to_idx[entry]
        exit_idx = node_to_idx[exit_point]

        best_path = None
        best_score = -float('inf')

        print("开始强化学习训练...")

        # Q-learning 算法
        for episode in tqdm(range(self.episodes)):
            # 从入口开始
            current_state = entry_idx
            path = [idx_to_node[current_state]]
            path_edges = []  # 记录边以避免重复

            # 当前路径是否达到出口
            reached_exit = False

            # 每条路径最大步数限制，防止无限循环
            max_steps = min(100, len(G.nodes()) * 2)

            for step in range(max_steps):
                # 获取当前节点的邻居
                neighbors = list(G.neighbors(idx_to_node[current_state]))
                if not neighbors:
                    break

                neighbor_indices = [node_to_idx[n] for n in neighbors]

                # ε-greedy策略选择动作
                if np.random.rand() < self.epsilon:
                    # 随机探索，但尽量选择不在当前路径上的节点
                    valid_neighbors = [n_idx for n_idx in neighbor_indices
                                      if (idx_to_node[current_state], idx_to_node[n_idx]) not in path_edges]

                    if valid_neighbors:
                        next_state = np.random.choice(valid_neighbors)
                    else:
                        # 如果所有邻居都已访问，就从所有邻居中随机选择
                        next_state = np.random.choice(neighbor_indices)
                else:
                    # 利用当前Q值，选择最优动作
                    q_values = [Q[current_state, n_idx] - 0.1 * visit_count[(current_state, n_idx)]
                               for n_idx in neighbor_indices]

                    # 选择具有最高Q值的邻居
                    best_idx = np.argmax(q_values)
                    next_state = neighbor_indices[best_idx]

                # 记录这条边已访问
                path_edges.append((idx_to_node[current_state], idx_to_node[next_state]))

                # 更新访问计数
                visit_count[(current_state, next_state)] += 1

                # 添加下一个状态到路径
                path.append(idx_to_node[next_state])
                current_state = next_state

                # 如果到达出口，结束本次episode
                if current_state == exit_idx:
                    reached_exit = True
                    break

                # 如果路径中出现环路，也结束本次episode
                if len(set(path)) < len(path) - 1:
                    break

            # 只处理成功到达出口的路径
            if reached_exit:
                # 计算路径特征
                path_features = self.calculate_path_features(G, path, landscape_elements)

                # 计算趣味性得分作为奖励
                reward = self.calculate_interest_score(path_features)

                # 更新最佳路径
                if reward > best_score:
                    best_path = path
                    best_score = reward

                # 更新Q-table
                for i in range(len(path) - 1):
                    s = node_to_idx[path[i]]
                    s_next = node_to_idx[path[i+1]]

                    # 考虑到达终点的特殊奖励
                    if s_next == exit_idx:
                        Q[s, s_next] = (1 - self.alpha) * Q[s, s_next] + self.alpha * reward
                    else:
                        # 找出下一个状态的最大Q值
                        next_neighbors = [node_to_idx[n] for n in G.neighbors(path[i+1])]
                        max_q_next = np.max(Q[s_next, next_neighbors]) if next_neighbors else 0

                        # Q-learning更新公式
                        Q[s, s_next] = (1 - self.alpha) * Q[s, s_next] + \
                                      self.alpha * (reward / len(path) + self.gamma * max_q_next)

        # 如果没有找到有效路径，尝试使用最短路径
        if best_path is None:
            try:
                best_path = nx.shortest_path(G, entry, exit_point, weight='length')
                path_features = self.calculate_path_features(G, best_path, landscape_elements)
                best_score = self.calculate_interest_score(path_features)
            except:
                print("无法找到从入口到出口的路径")
                return [], [], 0

        print(f"找到最优路径，趣味性得分: {best_score:.4f}")

        # 计算最优路径的完整几何形态
        optimal_geometry = []
        for i in range(len(best_path) - 1):
            v1, v2 = best_path[i], best_path[i+1]
            edge_data = G.get_edge_data(v1, v2)
            optimal_geometry.extend(edge_data['geometry'])

        return best_path, optimal_geometry, best_score

    def visualize_garden_path(self, garden_data, G, optimal_path, optimal_geometry, entry, exit_point, garden_id):
        """可视化园林路径网络和最优游线

        Args:
            garden_data (dict): 园林数据
            G (networkx.Graph): 路径网络图
            optimal_path (list): 最优路径顶点序列
            optimal_geometry (list): 最优路径几何形态
            entry (tuple): 入口坐标
            exit_point (tuple): 出口坐标
            garden_id (int): 园林ID

        Returns:
            None
        """
        plt.figure(figsize=(14, 10))
        garden_name = self.gardens[garden_id]
        plt.title(f"{garden_name} 路径网络与最优游线", fontsize=18)

        # 绘制所有道路坐标点
        road_coords = self.extract_road_coordinates(garden_data)
        road_points = np.array(road_coords)
        plt.scatter(road_points[:, 0], road_points[:, 1], c='lightgray', s=5, alpha=0.5, label='道路点')

        # 绘制建筑物
        solid_coords, semi_open_coords = self.extract_building_coordinates(garden_data)
        if solid_coords:
            solid_points = np.array(solid_coords)
            plt.scatter(solid_points[:, 0], solid_points[:, 1], c='brown', s=10, alpha=0.7, label='实体建筑')

        if semi_open_coords:
            semi_points = np.array(semi_open_coords)
            plt.scatter(semi_points[:, 0], semi_points[:, 1], c='orange', s=10, alpha=0.7, label='半开放建筑')

        # 绘制图的边
        edge_x = []
        edge_y = []
        for (u, v) in G.edges():
            edge_data = G.get_edge_data(u, v)
            geometry = edge_data['geometry']
            for point in geometry:
                edge_x.append(point[0])
                edge_y.append(point[1])

            # 添加None分隔不同的边
            edge_x.append(None)
            edge_y.append(None)

        plt.plot(edge_x, edge_y, 'b-', linewidth=1, alpha=0.5, label='路径网络')

        # 绘制图的顶点
        node_x = []
        node_y = []
        for node in G.nodes():
            node_x.append(node[0])
            node_y.append(node[1])

        plt.scatter(node_x, node_y, c='blue', s=30, alpha=0.8, label='路径节点')

        # 绘制入口和出口
        plt.scatter([entry[0]], [entry[1]], c='green', s=200, marker='*', label='入口')
        plt.scatter([exit_point[0]], [exit_point[1]], c='red', s=200, marker='*', label='出口')

        # 绘制最优路径
        if optimal_geometry:
            opt_points = np.array(optimal_geometry)
            plt.plot(opt_points[:, 0], opt_points[:, 1], 'r-', linewidth=2.5, label='最优游线')

        # 标注路径上的关键点
        for i, node in enumerate(optimal_path):
            if i == 0 or i == len(optimal_path) - 1 or i % 5 == 0:  # 只标注部分点，避免拥挤
                plt.text(node[0], node[1], f"{i}", fontsize=12, ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.legend(loc='upper right', fontsize=12)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        # 保存图像
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{garden_name}_最优游线.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存 {garden_name} 的路径可视化图像")

    def process_garden(self, garden_id):
        """处理单个园林

        Args:
            garden_id (int): 园林ID

        Returns:
            tuple: (路径网络图, 最优路径, 趣味性得分)
        """
        print(f"\n===== 开始处理 {self.gardens[garden_id]} =====")

        # 加载园林数据
        garden_data = self.load_garden_data(garden_id)
        if not garden_data:
            print(f"无法加载 {self.gardens[garden_id]} 的数据，跳过")
            return None, None, 0

        # 提取道路坐标
        road_coords = self.extract_road_coordinates(garden_data)
        if not road_coords:
            print(f"{self.gardens[garden_id]} 没有道路坐标数据，跳过")
            return None, None, 0

        # 提取建筑坐标
        building_coords = self.extract_building_coordinates(garden_data)

        # 提取景观元素
        landscape_elements = self.extract_landscape_elements(garden_data)

        # 重构路径段
        path_segments = self.reconstruct_path_segments(road_coords)
        if not path_segments:
            print(f"{self.gardens[garden_id]} 无法重构路径段，跳过")
            return None, None, 0

        # 识别关键顶点
        vertices, vertex_types = self.identify_key_vertices(path_segments, building_coords)

        # 构建路径网络图
        G = self.build_path_graph(path_segments, vertices, vertex_types)

        # 识别出入口
        entry, exit_point = self.identify_entry_exit(G, vertex_types)

        # 寻找最优游线
        optimal_path, optimal_geometry, interest_score = self.find_optimal_path_rl(
            G, entry, exit_point, landscape_elements)

        # 可视化结果
        if optimal_path:
            self.visualize_garden_path(
                garden_data, G, optimal_path, optimal_geometry,
                entry, exit_point, garden_id)

        return G, optimal_path, interest_score

def main():
    """主函数"""
    analyzer = GardenPathAnalyzer()
    results = {}

    # 处理每个园林
    for garden_id in range(1, 11):
        G, optimal_path, interest_score = analyzer.process_garden(garden_id)

        if G and optimal_path:
            results[garden_id] = {
                'name': analyzer.gardens[garden_id],
                'interest_score': interest_score,
                'path_length': len(optimal_path)
            }

    # 输出结果汇总
    if results:
        print("\n===== 园林趣味性评分汇总 =====")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['interest_score'], reverse=True)

        for i, (garden_id, data) in enumerate(sorted_results):
            print(f"{i+1}. {data['name']}: {data['interest_score']:.4f}")

        # 可视化趣味性评分比较
        plt.figure(figsize=(12, 8))
        garden_names = [data['name'] for _, data in sorted_results]
        scores = [data['interest_score'] for _, data in sorted_results]

        plt.barh(garden_names, scores, color='skyblue')
        plt.xlabel('趣味性评分')
        plt.title('十大江南园林移步异景趣味性评分比较', fontsize=16)
        plt.grid(axis='x', alpha=0.3)

        # 在条形上标注具体分值
        for i, score in enumerate(scores):
            plt.text(score + 0.01, i, f'{score:.4f}', va='center')

        plt.tight_layout()
        plt.savefig("results/园林趣味性评分比较.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("已保存园林趣味性评分比较图表")
    else:
        print("没有成功处理任何园林数据")

if __name__ == "__main__":
    main()
