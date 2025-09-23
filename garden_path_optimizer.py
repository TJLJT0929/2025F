import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay, ConvexHull, cKDTree
from scipy.spatial.distance import cdist
import json
import time
import warnings
from collections import defaultdict, deque
from tqdm import tqdm
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString
from shapely.ops import unary_union
import math
import itertools

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class GardenPathOptimizer:
    """
    基于图论与强化学习的园林路径优化器 (V2.2)

    针对用户反馈进行优化：
    1. 解决局部最优问题，提升路径覆盖率。
    2. 智能识别并选择最优出入口对。
    3. 强制路径覆盖园林核心景点。
    4. 平衡路径长度与游览趣味性。
    """

    def __init__(self, data_dir="results/garden_data"):
        self.data_dir = data_dir

        # 理论参数 - 基于论文定义
        self.graph_params = {
            'distance_threshold_epsilon': 1500,  # ε = 1.5m = 1500mm (路径段重构)
            'intersection_tolerance': 1000,      # 交叉点识别容差
            'boundary_access_threshold': 3000,   # 边界出入口识别阈值
            'poi_buffer': 3000,                  # 兴趣点缓冲区
            'turn_angle_threshold': np.pi/6      # θ_turn = π/6 (更敏感的转角)
        }

        # 游线特征量化参数 - 趣味性评分
        self.tour_params = {
            'sampling_interval': 500,            # 路径采样间隔
            'viewshed_radius': 5000,             # 视域半径
            'curvature_weight': 1.0,             # w_curv (曲折度)
            'view_change_weight': 2.0,           # w_view (异景度)
            'exploration_weight': 1.5,           # w_exp (探索性)
            'poi_coverage_weight': 5.0,          # w_poi (新: 景点覆盖奖励)
            'length_reward_weight': 0.001,       # w_len_reward (新: 路径长度奖励)
            'revisit_penalty': -0.5,             # (新: 重复访问惩罚)
        }

        # 强化学习参数
        self.rl_params = {
            'episodes': 2000,                    # 增加训练轮数
            'alpha': 0.1,                        # 学习率
            'gamma': 0.9,                        # 折扣因子
            'epsilon_start': 1.0,                # 从完全探索开始
            'epsilon_end': 0.05,
            'decay_rate': 0.998                  # 调整衰减率
        }

        self.create_output_directories()

    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            'results/path_optimization',
            'results/graph_models',
            'results/tour_analysis'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def load_garden_data(self, garden_name):
        """加载园林数据"""
        data_file = f"{self.data_dir}/{garden_name}_数据.json"
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                garden_data = json.load(f)

            for element_type in garden_data['elements']:
                garden_data['elements'][element_type] = [
                    tuple(coord) for coord in garden_data['elements'][element_type]
                ]

            return garden_data
        except Exception as e:
            print(f"❌ 加载 {garden_name} 数据失败: {e}")
            return None

    def extract_path_segments(self, road_coords):
        """路径段提取算法"""
        print("🔧 执行路径段提取算法...")
        if not road_coords: return []
        road_points = list(set(road_coords))
        segments, remaining_points = [], set(road_points)

        while remaining_points:
            start_point = next(iter(remaining_points))
            current_segment = [start_point]
            remaining_points.remove(start_point)

            # 正向扩展
            current_point = start_point
            while True:
                if not remaining_points: break
                distances = cdist([current_point], list(remaining_points))
                min_idx, min_distance = np.argmin(distances), np.min(distances)
                if min_distance <= self.graph_params['distance_threshold_epsilon']:
                    nearest_point = list(remaining_points)[min_idx]
                    current_segment.append(nearest_point)
                    remaining_points.remove(nearest_point)
                    current_point = nearest_point
                else: break

            # 反向扩展
            current_point = start_point
            while True:
                if not remaining_points: break
                distances = cdist([current_point], list(remaining_points))
                min_idx, min_distance = np.argmin(distances), np.min(distances)
                if min_distance <= self.graph_params['distance_threshold_epsilon']:
                    nearest_point = list(remaining_points)[min_idx]
                    current_segment.insert(0, nearest_point)
                    remaining_points.remove(nearest_point)
                    current_point = nearest_point
                else: break

            if len(current_segment) >= 2: segments.append(current_segment)
        print(f"✅ 提取到 {len(segments)} 个路径段")
        return segments

    def find_intersections(self, segments):
        """找到路径段交叉点"""
        intersections = []
        tolerance = self.graph_params['intersection_tolerance']
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i >= j: continue
                for p1 in seg1:
                    for p2 in seg2:
                        if np.linalg.norm(np.array(p1) - np.array(p2)) < tolerance:
                            intersections.append(tuple((np.array(p1) + np.array(p2)) / 2))
        return list(set(intersections))

    def identify_points_of_interest(self, garden_elements):
        """识别核心兴趣点 (POI) - V2.0 改进"""
        poi = []
        # 重点: 实体建筑、半开放建筑、水体都视为核心POI
        poi_types = ['实体建筑', '半开放建筑', '水体']
        for poi_type in poi_types:
            poi.extend(garden_elements.get(poi_type, []))
        return list(set(poi))

    def identify_access_points(self, road_coords, boundaries):
        """识别所有可能的出入口 - V2.0 新增"""
        access_points = []
        if not road_coords: return []

        road_array = np.array(road_coords)
        threshold = self.graph_params['boundary_access_threshold']

        # 检查靠近四条边界的道路点
        is_near_min_x = road_array[:, 0] - boundaries['min_x'] < threshold
        is_near_max_x = boundaries['max_x'] - road_array[:, 0] < threshold
        is_near_min_y = road_array[:, 1] - boundaries['min_y'] < threshold
        is_near_max_y = boundaries['max_y'] - road_array[:, 1] < threshold

        near_boundary_indices = np.where(is_near_min_x | is_near_max_x | is_near_min_y | is_near_max_y)[0]

        if len(near_boundary_indices) > 0:
            access_points = [tuple(p) for p in road_array[near_boundary_indices]]

        # 如果边界上找不到点，用最靠近角落的点作为备选
        if len(access_points) < 2:
            print("⚠️ 边界附近无足够道路点，采用角点逼近法")
            all_points = np.array(road_coords)
            corners = np.array([
                [boundaries['min_x'], boundaries['min_y']],
                [boundaries['max_x'], boundaries['min_y']],
                [boundaries['min_x'], boundaries['max_y']],
                [boundaries['max_x'], boundaries['max_y']]
            ])
            for corner in corners:
                distances = cdist([corner], all_points)
                closest_point_idx = np.argmin(distances)
                access_points.append(tuple(all_points[closest_point_idx]))

        return list(set(access_points))

    def select_optimal_entrance_exit(self, access_points):
        """从候选中选择距离最远的出入口对 - V2.0 新增"""
        if len(access_points) < 2:
            return None, None

        max_dist = -1
        best_pair = (None, None)

        for p1, p2 in itertools.combinations(access_points, 2):
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist > max_dist:
                max_dist = dist
                best_pair = (p1, p2)

        print(f"✅ 已选择最优出入口对，相距 {max_dist:.0f} mm")
        return best_pair[0], best_pair[1]

    def calculate_edge_weights(self, point_sequence):
        """计算边权重: 长度、几何序列、转折点数量"""
        if len(point_sequence) < 2:
            return {'length': 0, 'geometry': point_sequence, 'turns': 0}

        length = np.sum(np.linalg.norm(np.diff(point_sequence, axis=0), axis=1))

        turns = 0
        theta_turn = self.graph_params['turn_angle_threshold']
        for i in range(1, len(point_sequence) - 1):
            u = np.array(point_sequence[i]) - np.array(point_sequence[i-1])
            v = np.array(point_sequence[i+1]) - np.array(point_sequence[i])
            u_norm, v_norm = np.linalg.norm(u), np.linalg.norm(v)
            if u_norm > 0 and v_norm > 0:
                cos_angle = np.clip(np.dot(u, v) / (u_norm * v_norm), -1, 1)
                if np.arccos(cos_angle) > theta_turn:
                    turns += 1
        return {'length': length, 'geometry': point_sequence, 'turns': turns}

    def build_graph_model(self, garden_elements, boundaries):
        """构建园林路径网络图模型 - V2.0 改进"""
        print("🏗️ 构建园林路径网络图模型 (V2.0)...")
        road_coords = garden_elements.get('道路', [])
        segments = self.extract_path_segments(road_coords)
        if not segments: return None, None, None

        G = nx.Graph()
        endpoints = [s[0] for s in segments] + [s[-1] for s in segments]
        intersections = self.find_intersections(segments)
        poi = self.identify_points_of_interest(garden_elements)
        access_points = self.identify_access_points(road_coords, boundaries)

        entrance, exit_point = self.select_optimal_entrance_exit(access_points)
        if not entrance or not exit_point:
            print("❌ 无法确定出入口")
            return None, None, None

        all_vertices = list(set(endpoints + intersections + poi + access_points))

        # 将所有节点（包括出入口、POI）投影到最近的路径点上
        all_path_points_list = [p for seg in segments for p in seg]
        if not all_path_points_list:
             print("❌ 路径中没有点，无法构建KDTree")
             return None, None, None
        all_path_points = np.array(all_path_points_list)
        kdtree = cKDTree(all_path_points)

        vertex_map = {}
        for v in all_vertices:
            dist, idx = kdtree.query(v)
            projected_v = tuple(all_path_points[idx])
            vertex_map[v] = projected_v

            node_type = 'road'
            if v in poi: node_type = 'poi'
            if v in intersections: node_type = 'intersection'
            if v in endpoints: node_type = 'endpoint'
            if v == entrance: node_type = 'entrance'
            if v == exit_point: node_type = 'exit'

            G.add_node(projected_v, type=node_type, original_pos=v)

        entrance_proj = vertex_map.get(entrance)
        exit_proj = vertex_map.get(exit_point)
        if not entrance_proj or not exit_proj:
            print("❌ 投影后的出入口为空")
            return None, None, None

        for segment in segments:
            segment_vertices = sorted([v for v in G.nodes() if v in segment], key=segment.index)
            for i in range(len(segment_vertices) - 1):
                v1, v2 = segment_vertices[i], segment_vertices[i+1]
                start_idx, end_idx = segment.index(v1), segment.index(v2)
                if start_idx > end_idx: start_idx, end_idx = end_idx, start_idx
                edge_sequence = segment[start_idx:end_idx+1]
                edge_weights = self.calculate_edge_weights(edge_sequence)
                G.add_edge(v1, v2, **edge_weights)

        print(f"✅ 图模型构建完成: {len(G.nodes())} 顶点, {len(G.edges())} 边")
        return G, entrance_proj, exit_proj

    def calculate_tour_features(self, tour, graph):
        """计算游线的多维度特征"""
        if len(tour) < 2: return { 'length': 0, 'curvature': 0, 'exploration': 0, 'poi_coverage': 0 }

        length, curvature, exploration = 0, 0, 0
        visited_edges = set()

        for i in range(len(tour) - 1):
            u, v = tour[i], tour[i+1]
            if graph.has_edge(u, v):
                edge = tuple(sorted((u, v)))
                if edge not in visited_edges:
                    edge_data = graph[u][v]
                    length += edge_data.get('length', 0)
                    curvature += edge_data.get('turns', 0)
                    visited_edges.add(edge)

        # 探索性：访问的不同交叉口和端点的度数之和
        for node in set(tour):
            node_type = graph.nodes[node].get('type')
            if node_type in ['intersection', 'endpoint']:
                exploration += graph.degree(node)

        # 景点覆盖率
        poi_nodes_in_tour = sum(1 for node in set(tour) if graph.nodes[node].get('type') == 'poi')

        return {
            'length': length,
            'curvature': curvature,
            'exploration': exploration,
            'poi_coverage': poi_nodes_in_tour
        }

    def reinforcement_learning_optimization(self, graph, garden_elements, entrance, exit_node):
        """强化学习路径优化 - V2.2 修正"""
        print("🧠 开始强化学习路径优化 (V2.2)...")
        if not graph.has_node(entrance) or not graph.has_node(exit_node):
            print("❌ 入口或出口不在图中")
            return [], {}, {}

        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        n_states = len(nodes)
        start_idx, end_idx = node_to_idx[entrance], node_to_idx[exit_node]

        Q = defaultdict(lambda: np.zeros(n_states))

        best_tour, best_score, best_metrics = [], -float('inf'), {}
        history = {'episodes': [], 'scores': [], 'best_scores': [], 'path_lengths': []}

        epsilon = self.rl_params['epsilon_start']

        print(f"🎯 开始 {self.rl_params['episodes']} 轮训练...")
        for episode in tqdm(range(self.rl_params['episodes']), desc="路径优化"):
            current_state = start_idx
            tour = [entrance]
            visited_in_episode = {current_state}

            max_steps = n_states * 2 # 允许更长的路径
            for step in range(max_steps):
                current_node = idx_to_node[current_state]
                neighbors = list(graph.neighbors(current_node))
                if not neighbors: break

                if np.random.rand() < epsilon:
                    # --- V2.2 修正 ---
                    # 使用索引选择，避免numpy的维度错误
                    random_index = np.random.randint(0, len(neighbors))
                    next_node = neighbors[random_index]
                    next_state = node_to_idx[next_node]
                else:
                    q_values = Q[current_state]
                    # 优先选择未访问过的邻居
                    unvisited_neighbors = [n for n in neighbors if node_to_idx[n] not in visited_in_episode]
                    if unvisited_neighbors:
                        neighbor_indices = [node_to_idx[n] for n in unvisited_neighbors]
                        best_q_idx = np.argmax([q_values[i] for i in neighbor_indices])
                        next_state = neighbor_indices[best_q_idx]
                    else: # 如果都访问过，则正常选择
                        neighbor_indices = [node_to_idx[n] for n in neighbors]
                        best_q_idx = np.argmax([q_values[i] for i in neighbor_indices])
                        next_state = neighbor_indices[best_q_idx]

                # --- V2.0 奖励函数 ---
                reward = 0
                next_node = idx_to_node[next_state]
                node_type = graph.nodes[next_node]['type']

                # 1. 景点奖励
                if node_type == 'poi' and next_state not in visited_in_episode:
                    reward += self.tour_params['poi_coverage_weight']

                # 2. 重访惩罚
                if next_state in visited_in_episode:
                    reward += self.tour_params['revisit_penalty']

                # 3. 到达终点的大奖励
                if next_state == end_idx:
                    reward += 20 # 巨大奖励以鼓励到达终点

                # --- Q-Learning 更新 ---
                old_q_value = Q[current_state][next_state]
                future_max_q = np.max(Q[next_state]) if next_state in Q else 0

                new_q_value = (1 - self.rl_params['alpha']) * old_q_value + \
                              self.rl_params['alpha'] * (reward + self.rl_params['gamma'] * future_max_q)
                Q[current_state][next_state] = new_q_value

                tour.append(next_node)
                visited_in_episode.add(next_state)
                current_state = next_state

                if current_state == end_idx: break

            # 计算整条路径的综合评分
            features = self.calculate_tour_features(tour, graph)
            score = (self.tour_params['curvature_weight'] * features['curvature'] +
                     self.tour_params['exploration_weight'] * features['exploration'] +
                     self.tour_params['poi_coverage_weight'] * features['poi_coverage'] * 10 + # 放大POI覆盖的影响
                     self.tour_params['length_reward_weight'] * features['length'])

            if score > best_score:
                best_score = score
                best_tour = tour
                best_metrics = features

            history['episodes'].append(episode)
            history['scores'].append(score)
            history['best_scores'].append(best_score)
            history['path_lengths'].append(len(tour))

            epsilon = max(self.rl_params['epsilon_end'], epsilon * self.rl_params['decay_rate'])

        print(f"✅ 强化学习优化完成!")
        print(f"   🏆 最佳综合得分: {best_score:.4f}")
        print(f"   📏 最优路径: {len(best_tour)} 节点, 长度 {best_metrics.get('length', 0):.0f} mm")
        print(f"   🏞️ 覆盖景点数: {best_metrics.get('poi_coverage', 0)}")

        return best_tour, best_metrics, history

    def visualize_optimal_path(self, garden_data, graph, optimal_tour, training_history, tour_metrics):
        """在景观分布图上绘制最优路径 - V2.0 改进"""
        garden_name = garden_data['name']
        print(f"🎨 生成 {garden_name} 最优路径可视化...")

        fig = plt.figure(figsize=(24, 15))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

        ax_main = fig.add_subplot(gs[0, :])
        ax_main.set_title(f"{garden_name} - 最优游览路径分析 (V2.2)", fontsize=16, fontweight='bold')

        element_config = {
            '道路': {'color': '#d3d3d3', 'size': 5, 'marker': '.', 'alpha': 0.5},
            '实体建筑': {'color': '#8B4513', 'size': 80, 'marker': 's', 'alpha': 0.9},
            '半开放建筑': {'color': '#FFA500', 'size': 60, 'marker': '^', 'alpha': 0.8},
            '假山': {'color': '#696969', 'size': 40, 'marker': 'o', 'alpha': 0.7},
            '水体': {'color': '#4169E1', 'size': 50, 'marker': 'p', 'alpha': 0.8},
            '植物': {'color': '#228B22', 'size': 20, 'marker': 'o', 'alpha': 0.6}
        }

        for element_type, coords in garden_data['elements'].items():
            if not coords: continue
            config = element_config.get(element_type)
            coords_array = np.array(coords)
            ax_main.scatter(coords_array[:, 0], coords_array[:, 1], c=config['color'], s=config['size'],
                           marker=config['marker'], alpha=config['alpha'], label=element_type)

        # 绘制图的节点和边
        node_positions = {node: node for node in graph.nodes()}
        node_colors = []
        for node in graph.nodes():
            node_type = graph.nodes[node]['type']
            if node_type == 'entrance': node_colors.append('lime')
            elif node_type == 'exit': node_colors.append('blue')
            elif node_type == 'poi': node_colors.append('magenta')
            else: node_colors.append('gray')

        nx.draw_networkx_edges(graph, node_positions, ax=ax_main, edge_color='gray', alpha=0.4)

        # 绘制最优路径
        if len(optimal_tour) > 1:
            path_edges = list(zip(optimal_tour, optimal_tour[1:]))
            nx.draw_networkx_nodes(graph, node_positions, nodelist=optimal_tour, node_color='red', node_size=50, ax=ax_main)
            nx.draw_networkx_edges(graph, node_positions, edgelist=path_edges, edge_color='red', width=3.0, alpha=0.8, ax=ax_main)

            # 标记出入口
            entrance_node = optimal_tour[0]
            exit_node = optimal_tour[-1]
            ax_main.scatter(entrance_node[0], entrance_node[1], c='lime', s=500, marker='*', edgecolors='black', linewidth=1.5, label='入口', zorder=20)
            ax_main.scatter(exit_node[0], exit_node[1], c='blue', s=500, marker='*', edgecolors='black', linewidth=1.5, label='出口', zorder=20)

        ax_main.set_xlabel('X坐标 (毫米)', fontsize=12)
        ax_main.set_ylabel('Y坐标 (毫米)', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal')
        ax_main.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))

        # 训练历史图
        ax_train = fig.add_subplot(gs[1, 0])
        ax_train.set_title("RL训练过程", fontsize=12)
        ax_train.plot(training_history['episodes'], training_history['best_scores'], color='red', label='最佳得分')
        ax_train.set_xlabel('训练轮数')
        ax_train.set_ylabel('综合得分')
        ax_train.grid(True, alpha=0.3)

        # 路径特征分析
        ax_metrics = fig.add_subplot(gs[1, 1:])
        ax_metrics.set_title("最优路径特征分析 (V2.2)", fontsize=12)
        features = ['路径长度 (m)', '曲折度 (转折)', '探索性', '覆盖景点数']
        values = [tour_metrics.get('length', 0) / 1000, tour_metrics.get('curvature', 0),
                  tour_metrics.get('exploration', 0), tour_metrics.get('poi_coverage', 0)]
        bars = ax_metrics.bar(features, values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
        for bar in bars:
            yval = bar.get_height()
            ax_metrics.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}', va='bottom')
        ax_metrics.grid(True, alpha=0.3, axis='y')

        plt.tight_layout(pad=3.0)
        output_filename = f"results/path_optimization/{garden_name}_最优路径分析_V2_2.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 最优路径分析图已保存: {output_filename}")
        return output_filename

    def process_garden(self, garden_name):
        """处理单个园林的路径优化 (V2.2)"""
        print(f"\n{'='*60}\n🏛️ 路径优化 (V2.2): {garden_name}\n{'='*60}")
        start_time = time.time()

        garden_data = self.load_garden_data(garden_name)
        if not garden_data: return

        graph, entrance, exit_point = self.build_graph_model(
            garden_data['elements'], garden_data['boundaries'])
        if not graph:
            print(f"❌ {garden_name} 图模型构建失败")
            return

        optimal_tour, final_metrics, training_history = self.reinforcement_learning_optimization(
            graph, garden_data['elements'], entrance, exit_point)
        if not optimal_tour:
            print(f"❌ {garden_name} 未找到最优路径")
            return

        self.visualize_optimal_path(
            garden_data, graph, optimal_tour, training_history, final_metrics)

        print(f"\n✅ {garden_name} 优化完成，总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == '__main__':
    optimizer = GardenPathOptimizer()

    # 更新为包含所有10个园林的列表
    garden_list = ['拙政园', '留园', '寄畅园', '瞻园', '豫园',
                   '秋霞圃', '沈园', '怡园', '耦园', '绮园']

    for garden in garden_list:
        optimizer.process_garden(garden)
