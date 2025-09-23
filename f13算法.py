import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, ConvexHull, cKDTree
from scipy.spatial.distance import cdist
import random
from collections import defaultdict, deque
from tqdm import tqdm
import warnings
import re
import time
from datetime import datetime
import json
import pickle
import matplotlib
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString
from shapely.ops import unary_union
import math
from scipy.spatial import distance_matrix

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class SmartGardenPathOptimizer:
    """
    智能园林路径优化系统 - 修正版

    修正问题：
    1. 智能识别真正的园林入口和出口 (基于建筑围墙间隙)
    2. 在地图上清楚标记最优路径 (路径与景观图合并)
    3. 优化景观元素大小，避免重叠
    """

    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        self.data_dir = data_dir
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }

        # 景观元素配置 (减小了点的大小)
        self.element_config = {
            '道路': {'color': '#FFD700', 'size': 5, 'marker': 'o', 'alpha': 0.7},  # 黄色道路, size从10->5
            '实体建筑': {'color': '#8B4513', 'size': 15, 'marker': 's', 'alpha': 0.9},
            '半开放建筑': {'color': '#FFA500', 'size': 12, 'marker': '^', 'alpha': 0.8},
            '假山': {'color': '#696969', 'size': 6, 'marker': 'o', 'alpha': 0.7}, # size从8->6
            '水体': {'color': '#4169E1', 'size': 6, 'marker': 'o', 'alpha': 0.8},
            '植物': {'color': '#228B22', 'size': 4, 'marker': 'o', 'alpha': 0.6}
        }

        # 智能识别参数
        self.smart_detection_params = {
            'entrance_detection_buffer': 5000,
            'boundary_margin': 2000,
            'entrance_road_threshold': 3000,
            'exit_similarity_threshold': 0.7,
            'building_cluster_eps': 10000, # 10米, 用于DBSCAN聚类建筑
            'building_cluster_min_samples': 3,
            'gap_threshold_factor': 1.5 # 间隙阈值因子
        }

        # 物理约束参数
        self.physical_params = {
            'road_connection_threshold': 3000,
            'building_access_threshold': 2500,
            'wall_buffer': 1000,
            'exploration_radius': 8000
        }

        # 路径优化权重
        self.optimization_weights = {
            'coverage_weight': 10.0,
            'novelty_weight': 5.0,
            'diversity_weight': 3.0,
            'repetition_penalty': 8.0,
            'length_penalty': 0.05,
        }

        # RL参数
        self.rl_config = {
            'episodes': 1000,
            'alpha': 0.12,
            'gamma': 0.95,
            'epsilon_start': 0.9,
            'epsilon_end': 0.1,
            'decay_rate': 0.995
        }

        self.create_output_directories()

    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            'results/smart_maps',
            'results/smart_paths',
            'results/smart_analysis',
            'results/entrance_detection'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def parse_coordinate_string(self, coord_str):
        """解析坐标字符串"""
        if pd.isna(coord_str):
            return None

        coord_str = str(coord_str).strip()
        patterns = [
            r'\{([^}]+)\}', r'\(([^)]+)\)', r'\[([^\]]+)\]',
            r'([0-9.-]+[,\s]+[0-9.-]+[,\s]*[0-9.-]*)'
        ]

        for pattern in patterns:
            match = re.search(pattern, coord_str)
            if match:
                try:
                    coord_part = match.group(1)
                    for sep in [',', ';', ' ', '\t']:
                        if sep in coord_part:
                            coords = [float(x.strip()) for x in coord_part.split(sep) if x.strip()]
                            if len(coords) >= 2:
                                return (float(coords[0]), float(coords[1]))
                except ValueError:
                    continue

        try:
            numbers = re.findall(r'-?\d+\.?\d*', coord_str)
            if len(numbers) >= 2:
                return (float(numbers[0]), float(numbers[1]))
        except:
            pass

        return None

    def load_garden_data(self, garden_id):
        """加载园林数据"""
        garden_name = self.gardens[garden_id]
        data_path = f"{self.data_dir}/{garden_id}. {garden_name}/4-{garden_name}数据坐标.xlsx"

        garden_data = {
            'id': garden_id,
            'name': garden_name,
            'elements': {}
        }

        try:
            excel_file = pd.ExcelFile(data_path)
            print(f"📖 加载 {garden_name} 数据...")

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(data_path, sheet_name=sheet_name)
                element_type = self.infer_element_type(sheet_name, df)
                if element_type:
                    coords = self.extract_coordinates_from_dataframe(df)
                    if coords:
                        garden_data['elements'][element_type] = coords
                        print(f"  ✓ {element_type}: {len(coords)} 个元素")

            return garden_data

        except Exception as e:
            print(f"❌ 加载 {garden_name} 数据失败: {e}")
            return None

    def infer_element_type(self, sheet_name, df):
        """推断元素类型"""
        sheet_lower = sheet_name.lower()
        type_mapping = {
            '道路': ['道路', 'road', 'path', '路'],
            '实体建筑': ['实体建筑', 'solid', 'building'],
            '半开放建筑': ['半开放建筑', 'semi', 'pavilion', '亭'],
            '假山': ['假山', 'mountain', 'rock', '山'],
            '水体': ['水体', 'water', '水', '池'],
            '植物': ['植物', 'plant', 'tree', '树', '花']
        }

        for element_type, keywords in type_mapping.items():
            if any(keyword in sheet_name or keyword in sheet_lower for keyword in keywords):
                return element_type
        return '道路'

    def extract_coordinates_from_dataframe(self, df):
        """从DataFrame中提取坐标"""
        coords = []
        for col in df.columns:
            for _, row in df.iterrows():
                coord_str = str(row[col])
                parsed_coord = self.parse_coordinate_string(coord_str)
                if parsed_coord:
                    coords.append(parsed_coord)
        return list(set(coords))

    def calculate_garden_boundaries(self, garden_elements):
        """计算园林边界"""
        all_coords = []
        for element_type, coords in garden_elements.items():
            all_coords.extend(coords)

        if not all_coords:
            return None

        coords_array = np.array(all_coords)

        boundaries = {
            'min_x': np.min(coords_array[:, 0]),
            'max_x': np.max(coords_array[:, 0]),
            'min_y': np.min(coords_array[:, 1]),
            'max_y': np.max(coords_array[:, 1]),
            'center_x': np.mean(coords_array[:, 0]),
            'center_y': np.mean(coords_array[:, 1])
        }

        return boundaries

    def smart_detect_entrance_exit(self, garden_elements, boundaries):
        """
        智能检测园林入口和出口 (新版逻辑)
        基于建筑围墙的间断点进行识别。
        """
        print("🔍 智能检测园林入口和出口 (基于建筑围墙)...")

        building_coords = garden_elements.get('实体建筑', [])
        road_coords = garden_elements.get('道路', [])

        if len(building_coords) < 5 or not road_coords:
            print("⚠️ 建筑或道路数据不足，回退到基于边界的旧版检测方法。")
            return self.smart_detect_entrance_exit_fallback(garden_elements, boundaries)

        building_array = np.array(building_coords)
        road_tree = cKDTree(road_coords)

        # 1. 使用DBSCAN对建筑点进行聚类
        db = DBSCAN(eps=self.smart_detection_params['building_cluster_eps'],
                    min_samples=self.smart_detection_params['building_cluster_min_samples']).fit(building_array)
        labels = db.labels_
        unique_labels = set(labels)

        # 2. 为每个建筑簇创建凸包(Shapely多边形)
        building_polygons = []
        for k in unique_labels:
            if k == -1: continue # 忽略噪声点
            class_member_mask = (labels == k)
            cluster_points = building_array[class_member_mask]
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                building_polygons.append(ShapelyPolygon(cluster_points[hull.vertices]))

        if not building_polygons:
            print("⚠️ 无法形成建筑簇，回退到基于边界的旧版检测方法。")
            return self.smart_detect_entrance_exit_fallback(garden_elements, boundaries)

        # 3. 合并所有建筑多边形，并提取其外部边界
        all_buildings_shape = unary_union(building_polygons)
        if hasattr(all_buildings_shape, 'exterior'):
            boundary_line = all_buildings_shape.exterior
        else: # 处理多个不相交的多边形集合
            print("⚠️ 建筑群不连续，使用其凸包作为边界。")
            all_building_points = np.vstack([list(poly.exterior.coords) for poly in building_polygons])
            hull = ConvexHull(all_building_points)
            boundary_line = LineString(all_building_points[hull.vertices])

        # 4. 识别边界线上的间隙
        boundary_points = list(boundary_line.coords)
        distances = [Point(boundary_points[i]).distance(Point(boundary_points[i+1])) for i in range(len(boundary_points)-1)]
        avg_dist = np.mean(distances)
        gap_threshold = avg_dist * self.smart_detection_params['gap_threshold_factor']

        gap_candidates = []
        for i in range(len(distances)):
            if distances[i] > gap_threshold:
                p1 = boundary_points[i]
                p2 = boundary_points[i+1]
                mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                gap_candidates.append({'mid_point': mid_point, 'gap_size': distances[i]})

        if not gap_candidates:
            print("⚠️ 未在建筑围墙上找到明显间隙，回退到旧版检测方法。")
            return self.smart_detect_entrance_exit_fallback(garden_elements, boundaries)

        # 5. 找到靠近间隙的道路点作为出入口候选
        entrance_candidates = []
        road_threshold = self.smart_detection_params['entrance_road_threshold']
        for gap in gap_candidates:
            nearby_roads_idx = road_tree.query_ball_point(gap['mid_point'], r=road_threshold)
            if nearby_roads_idx:
                # 选择最近的那个道路点
                nearest_road_dist = float('inf')
                nearest_road_point = None
                for road_idx in nearby_roads_idx:
                    dist = Point(gap['mid_point']).distance(Point(road_coords[road_idx]))
                    if dist < nearest_road_dist:
                        nearest_road_dist = dist
                        nearest_road_point = road_coords[road_idx]

                if nearest_road_point:
                    entrance_candidates.append({'position': nearest_road_point, 'gap_size': gap['gap_size']})

        if len(entrance_candidates) < 2:
            print("⚠️ 找到的入口候选不足两个，回退到旧版检测方法。")
            return self.smart_detect_entrance_exit_fallback(garden_elements, boundaries)

        # 6. 选择入口和出口
        # 按间隙大小排序，最大的作为入口
        entrance_candidates.sort(key=lambda x: x['gap_size'], reverse=True)
        entrance = entrance_candidates[0]['position']

        # 选择距离入口最远的候选点作为出口
        exit_point = None
        max_dist = -1
        for cand in entrance_candidates[1:]:
            dist = Point(entrance).distance(Point(cand['position']))
            if dist > max_dist:
                max_dist = dist
                exit_point = cand['position']

        if exit_point is None: # 如果只有一个候选
             return self.smart_detect_entrance_exit_fallback(garden_elements, boundaries)

        print(f"✅ 智能检测结果 (基于建筑):")
        print(f"   🚪 入口: {entrance}")
        print(f"   🏁 出口: {exit_point}")
        print(f"   📏 入口出口距离: {np.linalg.norm(np.array(entrance) - np.array(exit_point)):.0f}mm")

        return entrance, exit_point

    def smart_detect_entrance_exit_fallback(self, garden_elements, boundaries):
        """
        智能检测园林入口和出口的备用方法 (原始逻辑)
        基于边界附近的道路点进行识别。
        """
        road_coords = garden_elements.get('道路', [])
        if not road_coords or not boundaries:
            return None, None

        road_array = np.array(road_coords)

        # 寻找距离最远的两个道路点作为备选
        distances = cdist(road_array, road_array)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        entrance = tuple(road_array[i])
        exit_point = tuple(road_array[j])

        print(f"✅ 智能检测结果 (备用方法):")
        print(f"   🚪 入口: {entrance}")
        print(f"   🏁 出口: {exit_point}")

        return entrance, exit_point

    def find_building_access_points(self, building_coords, road_coords):
        """找到建筑的可进入点"""
        access_points = []
        threshold = self.physical_params['building_access_threshold']

        if not building_coords or not road_coords:
            return access_points

        road_tree = cKDTree(road_coords)

        for building_point in building_coords:
            distances, indices = road_tree.query(building_point, k=1)

            if distances < threshold:
                nearest_road = road_coords[indices]
                access_point = (
                    (building_point[0] + nearest_road[0]) / 2,
                    (building_point[1] + nearest_road[1]) / 2
                )
                access_points.append({
                    'position': access_point,
                    'building': building_point,
                    'road': nearest_road
                })

        return access_points

    def create_smart_movement_graph(self, garden_elements, entrance, exit_point):
        """创建智能移动图 - 包含真实入口出口"""
        G = nx.Graph()

        road_coords = garden_elements.get('道路', [])
        solid_buildings = garden_elements.get('实体建筑', [])
        semi_buildings = garden_elements.get('半开放建筑', [])

        all_buildings = solid_buildings + semi_buildings

        print(f"🏗️ 构建智能移动图...")
        print(f"   道路点: {len(road_coords)}")
        print(f"   建筑点: {len(all_buildings)}")
        print(f"   智能入口: {entrance}")
        print(f"   智能出口: {exit_point}")

        # 1. 添加道路节点
        for i, coord in enumerate(road_coords):
            node_type = 'road'
            if coord == entrance:
                node_type = 'entrance'
            elif coord == exit_point:
                node_type = 'exit'
            G.add_node(coord, type=node_type, id=f'road_{i}')

        # 2. 确保入口和出口在图中
        if entrance not in G.nodes():
            G.add_node(entrance, type='entrance', id='entrance_main')
        if exit_point not in G.nodes():
            G.add_node(exit_point, type='exit', id='exit_main')

        # 3. 找到建筑进入点
        access_points = self.find_building_access_points(all_buildings, road_coords)
        for i, access_info in enumerate(access_points):
            access_pos = access_info['position']
            G.add_node(access_pos, type='access', id=f'access_{i}',
                      building=access_info['building'])

        # 4. 连接道路节点
        road_threshold = self.physical_params['road_connection_threshold']
        all_movable_nodes = [n for n in G.nodes() if G.nodes[n]['type'] in ['road', 'entrance', 'exit', 'access']]

        if len(all_movable_nodes) > 1:
            coords_array = np.array(all_movable_nodes)
            tree = cKDTree(coords_array)

            for i, coord in enumerate(all_movable_nodes):
                indices = tree.query_ball_point(coord, road_threshold)

                for j in indices:
                    if i != j:
                        neighbor_coord = all_movable_nodes[j]
                        if not G.has_edge(coord, neighbor_coord):
                            distance = np.linalg.norm(np.array(coord) - np.array(neighbor_coord))
                            G.add_edge(coord, neighbor_coord,
                                     length=distance, type='movement')

        # 5. 连接建筑进入点到道路
        for access_info in access_points:
            access_pos = access_info['position']
            road_point = access_info['road']

            # 找到最近的道路节点
            road_nodes = [n for n in G.nodes() if G.nodes[n]['type'] in ['road', 'entrance', 'exit']]
            if road_nodes:
                distances = [np.linalg.norm(np.array(access_pos) - np.array(rn)) for rn in road_nodes]
                nearest_road_node = road_nodes[np.argmin(distances)]

                distance = np.linalg.norm(np.array(access_pos) - np.array(nearest_road_node))
                if distance < self.physical_params['building_access_threshold']:
                    G.add_edge(access_pos, nearest_road_node,
                              length=distance, type='access_to_road')

        print(f"✅ 智能移动图构建完成: {len(G.nodes())} 节点, {len(G.edges())} 边")
        return G

    def calculate_path_coverage(self, path, garden_elements):
        """计算路径覆盖率"""
        if len(path) < 2:
            return 0

        coverage_radius = self.physical_params['exploration_radius']
        covered_elements = set()
        total_elements = 0

        for element_type, coords in garden_elements.items():
            if not coords:
                continue

            total_elements += len(coords)
            coords_array = np.array(coords)

            for path_point in path:
                path_array = np.array(path_point)
                distances = np.linalg.norm(coords_array - path_array, axis=1)

                covered_indices = np.where(distances <= coverage_radius)[0]
                for idx in covered_indices:
                    element_id = f"{element_type}_{idx}"
                    covered_elements.add(element_id)

        coverage_rate = len(covered_elements) / max(total_elements, 1)
        return coverage_rate

    def calculate_path_novelty(self, path):
        """计算路径新奇性"""
        if len(path) < 3:
            return 0

        direction_changes = 0

        for i in range(1, len(path) - 1):
            v1 = np.array(path[i]) - np.array(path[i-1])
            v2 = np.array(path[i+1]) - np.array(path[i])

            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))

                if angle > np.pi / 4:
                    direction_changes += 1

        return direction_changes

    def calculate_path_repetition_penalty(self, path):
        """计算路径重复惩罚"""
        if len(path) < 2:
            return 0

        path_array = np.array(path)
        grid_size = 2000

        grid_coords = np.floor(path_array / grid_size).astype(int)
        unique_grids = len(set(tuple(coord) for coord in grid_coords))
        total_grids = len(grid_coords)

        repetition_rate = 1 - (unique_grids / max(total_grids, 1))
        return repetition_rate

    def calculate_smart_path_score(self, path, graph, garden_elements):
        """计算智能路径评分"""
        if len(path) < 2:
            return 0, {}

        coverage = self.calculate_path_coverage(path, garden_elements)
        novelty = self.calculate_path_novelty(path)
        repetition = self.calculate_path_repetition_penalty(path)

        # 路径长度
        path_length = 0
        for i in range(len(path) - 1):
            distance = np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
            path_length += distance

        # 路径多样性
        node_types = set()
        for node in path:
            if node in graph.nodes():
                node_type = graph.nodes[node].get('type', 'unknown')
                node_types.add(node_type)
        diversity = len(node_types)

        metrics = {
            'coverage': coverage,
            'novelty': novelty,
            'repetition': repetition,
            'diversity': diversity,
            'length': path_length
        }

        # 综合评分
        final_score = (
            self.optimization_weights['coverage_weight'] * coverage +
            self.optimization_weights['novelty_weight'] * novelty +
            self.optimization_weights['diversity_weight'] * diversity -
            self.optimization_weights['repetition_penalty'] * repetition -
            self.optimization_weights['length_penalty'] * path_length / 1000
        )

        return max(final_score, 0.01), metrics

    def smart_path_optimization(self, graph, garden_elements, entrance, exit_point):
        """智能路径优化"""
        print(f"🧠 开始智能路径优化...")
        print(f"   🚪 智能入口: {entrance}")
        print(f"   🏁 智能出口: {exit_point}")

        if not graph.has_node(entrance) or not graph.has_node(exit_point):
            print("❌ 智能入口或出口不在移动图中")
            return [], 0, {}

        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}

        n_states = len(nodes)
        start_idx = node_to_idx[entrance]
        end_idx = node_to_idx[exit_point]

        Q = np.zeros((n_states, n_states))
        visit_counts = np.zeros(n_states)

        best_path = []
        best_score = -float('inf')
        best_metrics = {}

        training_history = {
            'scores': [],
            'coverage': [],
            'lengths': [],
            'best_scores': []
        }

        print(f"🎯 开始强化学习: {self.rl_config['episodes']} 轮")

        for episode in tqdm(range(self.rl_config['episodes']), desc="智能路径优化"):
            current_state = start_idx
            path = [entrance]
            final_score = 0 # 初始化

            # 动态epsilon
            progress = episode / self.rl_config['episodes']
            epsilon = (self.rl_config['epsilon_start'] * (1 - progress) +
                      self.rl_config['epsilon_end'] * progress)

            max_steps = min(150, n_states * 4)

            for step in range(max_steps):
                current_node = idx_to_node[current_state]
                neighbors = list(graph.neighbors(current_node))

                if not neighbors:
                    break

                neighbor_indices = [node_to_idx[n] for n in neighbors if n in node_to_idx]
                if not neighbor_indices:
                    break

                # 智能探索策略
                if np.random.rand() < epsilon:
                    # 偏向少访问的节点
                    neighbor_visits = [visit_counts[idx] for idx in neighbor_indices]
                    min_visits = min(neighbor_visits)
                    least_visited = [idx for idx, visits in zip(neighbor_indices, neighbor_visits)
                                   if visits == min_visits]
                    next_state = np.random.choice(least_visited)
                else:
                    # Q值最大的动作
                    q_values = [Q[current_state, idx] for idx in neighbor_indices]
                    next_state = neighbor_indices[np.argmax(q_values)]

                next_node = idx_to_node[next_state]
                path.append(next_node)
                visit_counts[next_state] += 1

                # 判断是否结束 - 到达出口或路径足够长
                if (next_state == end_idx or
                    step >= max_steps - 1 or
                    len(path) > len(nodes) * 0.9):

                    # 计算得分
                    final_score, metrics = self.calculate_smart_path_score(
                        path, graph, garden_elements)

                    if final_score > best_score:
                        best_score = final_score
                        best_path = path.copy()
                        best_metrics = metrics.copy()

                        if episode % 100 == 0:
                            print(f"  🎯 Episode {episode}: 新最佳 {final_score:.3f}")

                    # Q值更新
                    for i in range(len(path) - 1):
                        s = node_to_idx[path[i]]
                        s_next = node_to_idx[path[i + 1]]

                        exploration_bonus = 1.0 / (visit_counts[s_next] + 1)
                        adjusted_reward = final_score + exploration_bonus

                        if i == len(path) - 2:
                            Q[s, s_next] = ((1 - self.rl_config['alpha']) * Q[s, s_next] +
                                           self.rl_config['alpha'] * adjusted_reward)
                        else:
                            future_neighbors = list(graph.neighbors(path[i + 1]))
                            if future_neighbors:
                                next_q_max = max([Q[s_next, node_to_idx[n]]
                                                for n in future_neighbors
                                                if n in node_to_idx] or [0])
                            else:
                                next_q_max = 0

                            Q[s, s_next] = ((1 - self.rl_config['alpha']) * Q[s, s_next] +
                                           self.rl_config['alpha'] *
                                           (adjusted_reward + self.rl_config['gamma'] * next_q_max))

                    break

                current_state = next_state

            # 记录历史
            training_history['scores'].append(final_score)
            training_history['coverage'].append(best_metrics.get('coverage', 0))
            training_history['lengths'].append(len(path))
            training_history['best_scores'].append(best_score)

        print(f"✅ 智能路径优化完成!")
        print(f"   🏆 最佳得分: {best_score:.4f}")
        print(f"   📏 路径长度: {len(best_path)} 节点")
        print(f"   📊 覆盖率: {best_metrics.get('coverage', 0):.3f}")

        return best_path, best_score, training_history

    def determine_legend_position(self, boundaries):
        """智能确定图例位置 - 避免挡住园林"""
        if not boundaries:
            return 'upper right'

        # 计算园林的主要分布区域
        width = boundaries['max_x'] - boundaries['min_x']
        height = boundaries['max_y'] - boundaries['min_y']
        center_x = boundaries['center_x']
        center_y = boundaries['center_y']

        # 根据园林形状和位置选择最佳图例位置
        if width > height:  # 园林比较宽
            if center_y > (boundaries['min_y'] + boundaries['max_y']) / 2:
                return 'lower right'  # 园林在上半部分，图例放下方
            else:
                return 'upper right'  # 园林在下半部分，图例放上方
        else:  # 园林比较高
            if center_x > (boundaries['min_x'] + boundaries['max_x']) / 2:
                return 'upper left'   # 园林在右半部分，图例放左边
            else:
                return 'upper right'  # 园林在左半部分，图例放右边

    def generate_smart_garden_map(self, garden_data, boundaries):
        """生成智能园林地图 - 图例不挡住园林"""
        garden_name = garden_data['name']

        print(f"🎨 生成 {garden_name} 智能地图...")

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(f"{garden_name} - 智能景观分布图", fontsize=16, fontweight='bold', pad=20)

        legend_elements = []

        # 绘制各类景观元素
        for element_type, coords in garden_data['elements'].items():
            if not coords:
                continue

            config = self.element_config.get(element_type,
                {'color': '#000000', 'size': 5, 'marker': 'o', 'alpha': 0.7})

            coords_array = np.array(coords)
            scatter = ax.scatter(coords_array[:, 0], coords_array[:, 1],
                               c=config['color'], s=config['size'],
                               marker=config['marker'], alpha=config['alpha'],
                               label=f"{element_type} ({len(coords)})")
            legend_elements.append(scatter)

        ax.set_xlabel('X (毫米)', fontsize=12)
        ax.set_ylabel('Y (毫米)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 智能图例定位
        legend_position = self.determine_legend_position(boundaries)
        ax.legend(handles=legend_elements, loc=legend_position, fontsize=9,
                 framealpha=0.95, fancybox=True, shadow=True)

        plt.tight_layout()

        map_filename = f"results/smart_maps/{garden_name}_智能地图.png"
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 智能地图已保存: {map_filename}")
        return map_filename

    def visualize_smart_optimal_path(self, garden_data, best_path, training_history,
                                   movement_graph, entrance, exit_point, boundaries):
        """
        可视化智能最优路径 - (新版) 将路径绘制在景观图上
        """
        garden_name = garden_data['name']

        print(f"🎯 生成 {garden_name} 智能最优路径图...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12), gridspec_kw={'width_ratios': [3, 2]})
        fig.suptitle(f'{garden_name} - 智能路径分析', fontsize=18, fontweight='bold')

        # --- 左图：景观 + 智能最优路径 ---
        ax1.set_title("智能最优游览路径与景观分布", fontsize=14, fontweight='bold')

        legend_elements_left = []

        # 绘制景观元素
        for element_type, coords in garden_data['elements'].items():
            if not coords: continue

            config = self.element_config.get(element_type)
            coords_array = np.array(coords)
            scatter = ax1.scatter(coords_array[:, 0], coords_array[:, 1],
                               c=config['color'], s=config['size'],
                               marker=config['marker'], alpha=config['alpha'],
                               label=f"{element_type}")
            legend_elements_left.append(scatter)

        # 绘制移动网络（可选，淡灰色背景）
        # for edge in movement_graph.edges():
        #     start, end = edge
        #     ax1.plot([start[0], end[0]], [start[1], end[1]],
        #            color='lightgray', linewidth=0.5, alpha=0.3, zorder=1)

        # 重点：清楚标记最优路径
        if len(best_path) > 1:
            path_array = np.array(best_path)

            # 主路径线 - 粗红线
            line = ax1.plot(path_array[:, 0], path_array[:, 1],
                          color='red', linewidth=3.5, alpha=0.8,
                          label=f'最优路径 ({len(best_path)}节点)', zorder=8)
            legend_elements_left.append(line[0])

            # 路径节点标记 - 小红点
            ax1.scatter(path_array[:, 0], path_array[:, 1],
                        c='darkred', s=15, alpha=0.7, zorder=9)

            # 智能入口标记 - 大绿星
            entrance_marker = ax1.scatter(entrance[0], entrance[1],
                                        c='lime', s=300, marker='*',
                                        edgecolors='darkgreen', linewidth=2,
                                        label='智能入口', zorder=12)
            legend_elements_left.append(entrance_marker)

            # 智能出口标记 - 大蓝星
            exit_marker = ax1.scatter(exit_point[0], exit_point[1],
                                    c='blue', s=300, marker='*',
                                    edgecolors='darkblue', linewidth=2,
                                    label='智能出口', zorder=12)
            legend_elements_left.append(exit_marker)

            # 路径方向箭头
            arrow_interval = max(1, len(best_path) // 10)
            for i in range(arrow_interval, len(best_path), arrow_interval):
                start_pos = best_path[i-1]
                end_pos = best_path[i]
                ax1.annotate('', xy=end_pos, xytext=start_pos,
                           arrowprops=dict(arrowstyle='->', color='darkred',
                                         lw=1.5, alpha=0.7), zorder=10)

        ax1.set_xlabel('X (毫米)', fontsize=12)
        ax1.set_ylabel('Y (毫米)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.set_aspect('equal')

        legend_position_left = self.determine_legend_position(boundaries)
        ax1.legend(handles=legend_elements_left, loc=legend_position_left,
                  fontsize=9, framealpha=0.9, fancybox=True, shadow=True)

        # --- 右图：训练历史 ---
        ax2.set_title("智能优化训练历史", fontsize=14, fontweight='bold')

        episodes = range(len(training_history['scores']))

        ax2_2 = ax2.twinx() # 双Y轴

        # 得分曲线
        line1, = ax2.plot(episodes, training_history['best_scores'],
                        color='red', linewidth=2.5, label='最佳得分', zorder=5)
        # 覆盖率曲线
        line2, = ax2_2.plot(episodes, training_history['coverage'],
                          color='green', alpha=0.7, linewidth=2, linestyle='--', label='覆盖率')

        # 标注最优路径对应的训练轮数
        if training_history['best_scores']:
            best_score_val = max(training_history['best_scores'])
            best_episode = training_history['best_scores'].index(best_score_val)
            ax2.axvline(x=best_episode, color='grey', linestyle=':', linewidth=1)
            ax2.annotate(f'最优解 (轮次 {best_episode})',
                        xy=(best_episode, best_score_val),
                        xytext=(best_episode, best_score_val * 0.9),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                        ha='center', va='top',
                        bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.7))

        ax2.set_xlabel('优化轮数', fontsize=12)
        ax2.set_ylabel('得分', color='red', fontsize=12)
        ax2_2.set_ylabel('覆盖率', color='green', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        ax2_2.tick_params(axis='y', labelcolor='green')
        ax2.grid(True, alpha=0.3)

        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='lower right', fontsize=9, framealpha=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应总标题

        path_filename = f"results/smart_paths/{garden_name}_智能最优路径.png"
        plt.savefig(path_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 智能路径图已保存: {path_filename}")
        return path_filename

    def process_single_garden_smart(self, garden_id):
        """处理单个园林 - 智能版本"""
        print(f"\n{'='*60}")
        print(f"🏛️  智能处理园林: {self.gardens[garden_id]} (ID: {garden_id})")
        print(f"🎯 智能特性: 真实入口出口 + 清晰路径标记 + 智能图例")
        print(f"{'='*60}")

        start_time = time.time()

        # 加载数据
        garden_data = self.load_garden_data(garden_id)
        if not garden_data or not garden_data['elements']:
            print(f"❌ {self.gardens[garden_id]} 数据加载失败")
            return None

        # 计算边界
        boundaries = self.calculate_garden_boundaries(garden_data['elements'])

        # 智能检测入口出口
        entrance, exit_point = self.smart_detect_entrance_exit(garden_data['elements'], boundaries)

        if not entrance or not exit_point:
            print(f"❌ {self.gardens[garden_id]} 智能入口出口检测失败")
            return None

        # 生成智能基础地图 (可选，如果只想看最终带路径的图，可以注释掉)
        # map_filename = self.generate_smart_garden_map(garden_data, boundaries)
        map_filename = "N/A (已合并到路径图)"

        # 创建智能移动图
        movement_graph = self.create_smart_movement_graph(garden_data['elements'], entrance, exit_point)

        if len(movement_graph.nodes()) < 2:
            print(f"❌ {self.gardens[garden_id]} 智能移动图节点不足")
            return None

        # 智能路径优化
        best_path, best_score, training_history = self.smart_path_optimization(
            movement_graph, garden_data['elements'], entrance, exit_point)

        if not best_path:
            print(f"❌ {self.gardens[garden_id]} 未找到智能最优路径")
            return None

        # 生成智能路径可视化 (新版，包含景观图)
        path_filename = self.visualize_smart_optimal_path(
            garden_data, best_path, training_history, movement_graph,
            entrance, exit_point, boundaries)

        # 计算最终指标
        final_score, final_metrics = self.calculate_smart_path_score(
            best_path, movement_graph, garden_data['elements'])

        processing_time = time.time() - start_time

        # 保存入口出口检测结果
        detection_result = {
            'garden_id': garden_id,
            'garden_name': self.gardens[garden_id],
            'boundaries': boundaries,
            'smart_entrance': entrance,
            'smart_exit': exit_point,
            'entrance_exit_distance': float(np.linalg.norm(np.array(entrance) - np.array(exit_point)))
        }

        with open(f'results/entrance_detection/{self.gardens[garden_id]}_入口检测.json',
                 'w', encoding='utf-8') as f:
            json.dump(detection_result, f, ensure_ascii=False, indent=2)

        result = {
            'garden_id': garden_id,
            'garden_name': self.gardens[garden_id],
            'map_filename': map_filename,
            'path_filename': path_filename,
            'smart_entrance': entrance,
            'smart_exit': exit_point,
            'best_score': best_score,
            'final_metrics': final_metrics,
            'path_length': len(best_path),
            'graph_nodes': len(movement_graph.nodes()),
            'graph_edges': len(movement_graph.edges()),
            'processing_time': processing_time,
            'boundaries': boundaries
        }

        print(f"✅ {self.gardens[garden_id]} 智能处理完成:")
        print(f"   🎯 路径图: {path_filename}")
        print(f"   🚪 智能入口: {entrance}")
        print(f"   🏁 智能出口: {exit_point}")
        print(f"   🏆 得分: {best_score:.4f}")
        print(f"   📊 覆盖率: {final_metrics['coverage']:.3f}")
        print(f"   ⏱️ 时间: {processing_time:.2f}秒")

        return result

    def batch_process_all_gardens_smart(self):
        """批量处理所有园林 - 智能版本"""
        print("🚀 智能园林路径优化系统启动!")
        print("🎯 智能修正:")
        print("   🧠 智能检测真实入口出口（基于建筑围墙间隙）")
        print("   🔴 最优路径与景观图合并显示")
        print("   🤏 优化景观元素点大小")
        print("=" * 80)

        start_time = time.time()
        results = []

        for garden_id in range(1, 11):
            try:
                result = self.process_single_garden_smart(garden_id)
                if result:
                    results.append(result)

            except Exception as e:
                print(f"❌ 处理园林 {garden_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        total_time = time.time() - start_time

        # 生成智能分析报告
        if results:
            self.generate_smart_analysis_report(results, total_time)
        else:
            print("🚫 未能成功处理任何园林，无法生成报告。")

        return results

    def generate_smart_analysis_report(self, results, total_time):
        """生成智能分析报告"""
        print(f"\n{'='*25} 智能分析报告 {'='*25}")

        if not results:
            print("❌ 没有成功处理的园林数据")
            return

        # 按得分排序
        sorted_results = sorted(results, key=lambda x: x['best_score'], reverse=True)

        # 创建智能分析图表
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle('江南古典园林智能路径优化分析报告', fontsize=18, fontweight='bold')

        # 创建复杂子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

        names = [r['garden_name'] for r in sorted_results]

        # 1. 综合得分排名
        ax1 = fig.add_subplot(gs[0, 0])
        scores = [r['best_score'] for r in sorted_results]
        bars1 = ax1.barh(names, scores, color='lightcoral', alpha=0.8)
        ax1.set_xlabel('综合得分')
        ax1.set_title('智能路径综合得分排名', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()

        for i, (bar, score) in enumerate(zip(bars1, scores)):
            ax1.text(score, i, f' {score:.2f}', va='center', ha='left', fontsize=8)

        # 2. 智能入口出口距离分析
        ax2 = fig.add_subplot(gs[0, 1])
        entrance_exit_distances = [np.linalg.norm(np.array(r['smart_entrance']) - np.array(r['smart_exit'])) for r in sorted_results]
        bars2 = ax2.barh(names, entrance_exit_distances, color='lightgreen', alpha=0.8)
        ax2.set_xlabel('入口出口距离 (mm)')
        ax2.set_title('智能入口出口距离', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()

        # 3. 覆盖率对比
        ax3 = fig.add_subplot(gs[0, 2])
        coverage_scores = [r['final_metrics']['coverage'] for r in sorted_results]
        bars3 = ax3.barh(names, coverage_scores, color='lightblue', alpha=0.8)
        ax3.set_xlabel('覆盖率')
        ax3.set_title('园林覆盖率对比', fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        ax3.invert_yaxis()

        # 4. 路径长度vs覆盖率散点图
        ax4 = fig.add_subplot(gs[1, 0])
        path_lengths = [r['path_length'] for r in sorted_results]
        ax4.scatter(path_lengths, coverage_scores, c='purple', alpha=0.6, s=100)
        ax4.set_xlabel('路径长度(节点数)')
        ax4.set_ylabel('覆盖率')
        ax4.set_title('路径长度 vs 覆盖率', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        for i, result in enumerate(sorted_results):
            ax4.annotate(result['garden_name'][0], # 仅用首字标注避免重叠
                        (path_lengths[i], coverage_scores[i]),
                        fontsize=8, alpha=0.8, ha='center', va='center')

        # 5. 图节点数量分析
        ax5 = fig.add_subplot(gs[1, 1])
        graph_nodes = [r['graph_nodes'] for r in sorted_results]
        graph_edges = [r['graph_edges'] for r in sorted_results]
        x_pos = np.arange(len(names))
        width = 0.35
        ax5.bar(x_pos - width/2, graph_nodes, width, label='节点数', color='lightsteelblue')
        ax5.bar(x_pos + width/2, graph_edges, width, label='边数', color='lightpink')
        ax5.set_ylabel('数量')
        ax5.set_title('移动图规模分析', fontweight='bold')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(names, rotation=45, ha='right')
        ax5.legend()

        # 6. 处理时间效率
        ax6 = fig.add_subplot(gs[1, 2])
        processing_times = [r['processing_time'] for r in sorted_results]
        bars6 = ax6.barh(names, processing_times, color='gold', alpha=0.8)
        ax6.set_xlabel('处理时间 (秒)')
        ax6.set_title('智能处理效率', fontweight='bold')
        ax6.grid(axis='x', alpha=0.3)
        ax6.invert_yaxis()

        # 7. 综合指标雷达图
        ax7 = fig.add_subplot(gs[2, :], polar=True)
        ax7.set_title('前五名园林综合指标对比 (雷达图)', fontweight='bold', pad=20)
        top5_results = sorted_results[:5]
        metrics_names = ['得分', '覆盖率', '新奇性', '多样性', '低重复']
        labels = [r['garden_name'] for r in top5_results]
        angles = np.linspace(0, 2*np.pi, len(metrics_names), endpoint=False).tolist()
        angles += angles[:1]

        for result in top5_results:
            metrics = result['final_metrics']
            values = [
                result['best_score'] / max(scores),
                metrics['coverage'],
                metrics['novelty'] / max([r['final_metrics']['novelty'] for r in results if r['final_metrics']['novelty']>0]),
                metrics['diversity'] / max([r['final_metrics']['diversity'] for r in results]),
                1 - metrics['repetition']
            ]
            values += values[:1]
            ax7.plot(angles, values, 'o-', linewidth=2, label=result['garden_name'])
            ax7.fill(angles, values, alpha=0.25)

        ax7.set_thetagrids(np.degrees(angles[:-1]), metrics_names)
        ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        analysis_filename = "results/smart_analysis/智能分析报告.png"
        plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # 打印详细统计
        print(f"📊 智能处理统计:")
        print(f"   成功处理: {len(results)}/10 个园林")
        print(f"   总用时: {total_time:.2f} 秒")
        if results: print(f"   平均用时: {total_time/len(results):.2f} 秒/园林")

        print(f"\n🧠 智能入口出口检测结果:")
        for result in sorted_results:
            dist = np.linalg.norm(np.array(result['smart_entrance']) - np.array(result['smart_exit']))
            print(f"   {result['garden_name']:<8}: 入口({result['smart_entrance'][0]:.0f}, {result['smart_entrance'][1]:.0f}) -> 出口({result['smart_exit'][0]:.0f}, {result['smart_exit'][1]:.0f}) | 距离: {dist:.0f}mm")

        print(f"\n🏆 智能优化排名:")
        for i, result in enumerate(sorted_results):
            metrics = result['final_metrics']
            print(f"   {i+1:2d}. {result['garden_name']:<8} 得分: {result['best_score']:6.2f} | 覆盖: {metrics['coverage']:.3f} | 重复: {metrics['repetition']:.3f} | 节点: {result['path_length']:3d}")

        final_results = {'results': sorted_results, 'analysis_filename': analysis_filename}
        with open('results/smart_analysis/完整智能结果.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)

        print(f"\n💾 智能分析文件已保存:")
        print(f"   📈 分析图表: {analysis_filename}")
        print(f"   📝 完整结果: results/smart_analysis/完整智能结果.json")

def main():
    """主函数 - 智能修正版本"""
    print("🏛️  江南古典园林智能路径优化系统 - 修正版")
    print("=" * 80)

    optimizer = SmartGardenPathOptimizer()

    results = optimizer.batch_process_all_gardens_smart()

    if results:
        print(f"\n🎉 智能修正系统运行完成！")
        print(f"✅ 成功处理 {len(results)}/10 个园林")
        print(f"🧠 智能入口出口：已基于建筑围墙间隙进行检测。")
        print(f"🔴 清晰路径标记：最优路径已直接绘制在景观图上。")
        print(f"🤏 元素大小优化：减小了图中元素的点大小，显示更清晰。")
        print(f"📁 详细结果保存在 'results/' 目录中。")
    else:
        print("❌ 智能修正系统运行失败或未处理任何园林。")

if __name__ == "__main__":
    main()
