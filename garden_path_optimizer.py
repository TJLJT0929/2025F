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

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class GardenPathOptimizer:
    """
    基于图论与强化学习的园林路径优化器
    
    基于论文 1.1-1.2.tex 的理论框架：
    1. 园林路径网络的图模型构建 (Section 1.1)
    2. 游线特征量化与"趣味性"定义 (Section 1.2)
    
    功能：
    1. 验证代码是否满足且正确使用了理论定义
    2. 用强化学习算法求解最优路径
    3. 在景观分布图上绘制最优路径
    """
    
    def __init__(self, data_dir="results/garden_data"):
        self.data_dir = data_dir
        
        # 理论参数 - 基于论文定义
        self.graph_params = {
            'distance_threshold_epsilon': 1500,  # ε = 1.5m = 1500mm (论文算法2.1)
            'intersection_tolerance': 1000,      # 交叉点识别容差
            'building_access_threshold': 2500,   # 建筑进入点阈值
            'poi_buffer': 3000,                  # 兴趣点缓冲区
            'turn_angle_threshold': np.pi/4      # θ_turn = π/4 (论文定义2.4)
        }
        
        # 游线特征量化参数 - 基于论文1.2节
        self.tour_params = {
            'sampling_interval': 500,            # 路径采样间隔 (论文定义2.7)
            'viewshed_radius': 5000,            # 视域半径
            'curvature_weight': 1.0,            # w_curv
            'view_change_weight': 2.0,          # w_view  
            'exploration_weight': 1.5,          # w_exp
            'length_penalty_weight': 0.1,      # w_len
            'penalty_constant': 1000.0          # C (防止分母为零)
        }
        
        # 强化学习参数
        self.rl_params = {
            'episodes': 1500,
            'alpha': 0.1,                       # 学习率
            'gamma': 0.95,                      # 折扣因子
            'epsilon_start': 0.9,
            'epsilon_end': 0.1,
            'decay_rate': 0.995
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
            
            # 转换坐标为numpy数组
            for element_type in garden_data['elements']:
                garden_data['elements'][element_type] = [
                    tuple(coord) for coord in garden_data['elements'][element_type]
                ]
            
            return garden_data
        except Exception as e:
            print(f"❌ 加载 {garden_name} 数据失败: {e}")
            return None
    
    def extract_path_segments(self, road_coords):
        """
        路径段提取算法 - 实现论文算法2.1
        将无序点集重构为有序路径段
        """
        print("🔧 执行路径段提取算法 (论文算法2.1)...")
        
        if not road_coords:
            return []
        
        road_points = list(set(road_coords))  # 去重
        segments = []  # S集合
        remaining_points = set(road_points)   # P_temp
        
        while remaining_points:
            # 选择起始点
            start_point = next(iter(remaining_points))
            current_segment = [start_point]
            remaining_points.remove(start_point)
            
            # 向前扩展
            current_point = start_point
            while True:
                if not remaining_points:
                    break
                
                # 找最近的点
                distances = [np.linalg.norm(np.array(current_point) - np.array(p)) 
                           for p in remaining_points]
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                nearest_point = list(remaining_points)[min_idx]
                
                # 检查距离阈值ε
                if min_distance <= self.graph_params['distance_threshold_epsilon']:
                    current_segment.append(nearest_point)
                    remaining_points.remove(nearest_point)
                    current_point = nearest_point
                else:
                    break
            
            # 向后扩展
            current_point = start_point
            while True:
                if not remaining_points:
                    break
                
                distances = [np.linalg.norm(np.array(current_point) - np.array(p)) 
                           for p in remaining_points]
                if not distances:
                    break
                    
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                nearest_point = list(remaining_points)[min_idx]
                
                if min_distance <= self.graph_params['distance_threshold_epsilon']:
                    current_segment.insert(0, nearest_point)
                    remaining_points.remove(nearest_point)
                    current_point = nearest_point
                else:
                    break
            
            if len(current_segment) >= 2:
                segments.append(current_segment)
        
        print(f"✅ 提取到 {len(segments)} 个路径段")
        return segments
    
    def find_intersections(self, segments):
        """找到路径段交叉点 - 实现论文定义2.2中的V_int"""
        intersections = []
        tolerance = self.graph_params['intersection_tolerance']
        
        for i, seg1 in enumerate(segments):
            for j, seg2 in enumerate(segments):
                if i >= j:
                    continue
                
                # 简化的交叉检测：找两条路径段中距离很近的点对
                for p1 in seg1:
                    for p2 in seg2:
                        distance = np.linalg.norm(np.array(p1) - np.array(p2))
                        if distance < tolerance:
                            intersection = ((np.array(p1) + np.array(p2)) / 2).tolist()
                            intersections.append(tuple(intersection))
        
        return list(set(intersections))  # 去重
    
    def identify_points_of_interest(self, garden_elements, boundaries):
        """识别兴趣点 - 实现论文定义2.2中的V_poi"""
        poi = []
        
        # 智能识别入口和出口
        road_coords = garden_elements.get('道路', [])
        if not road_coords:
            return []
        
        # 找边界附近的道路点作为入口出口候选
        road_array = np.array(road_coords)
        
        # 找距离边界最近的点作为入口
        boundary_points = [
            (boundaries['min_x'], boundaries['min_y']),
            (boundaries['max_x'], boundaries['min_y']),
            (boundaries['min_x'], boundaries['max_y']),
            (boundaries['max_x'], boundaries['max_y'])
        ]
        
        entrance_distances = []
        for road_point in road_coords:
            min_dist = min([np.linalg.norm(np.array(road_point) - np.array(bp)) 
                           for bp in boundary_points])
            entrance_distances.append((min_dist, road_point))
        
        entrance_distances.sort()
        
        # 入口：最靠近边界的点
        entrance = entrance_distances[0][1]
        
        # 出口：距离入口最远的道路点
        exit_distances = [np.linalg.norm(np.array(entrance) - np.array(rp)) 
                         for rp in road_coords]
        exit_idx = np.argmax(exit_distances)
        exit_point = road_coords[exit_idx]
        
        poi.extend([entrance, exit_point])
        
        # 添加重要建筑作为兴趣点
        buildings = garden_elements.get('实体建筑', [])
        if buildings:
            # 选择一些建筑作为兴趣点（如中心建筑）
            center_x, center_y = boundaries['center_x'], boundaries['center_y']
            building_distances = [np.linalg.norm(np.array(b) - np.array([center_x, center_y])) 
                                for b in buildings]
            central_buildings = sorted(zip(building_distances, buildings))[:3]  # 选择3个中心建筑
            poi.extend([b[1] for b in central_buildings])
        
        return poi
    
    def calculate_edge_weights(self, point_sequence):
        """
        计算边权重 - 实现论文定义2.4
        返回多维权重：长度、几何序列、转折点数量
        """
        if len(point_sequence) < 2:
            return {'length': 0, 'geometry': point_sequence, 'turns': 0}
        
        # 长度计算 - W_len(e)
        length = 0
        for i in range(len(point_sequence) - 1):
            length += np.linalg.norm(np.array(point_sequence[i+1]) - np.array(point_sequence[i]))
        
        # 转折点计算 - W_turns(e)
        turns = 0
        theta_turn = self.graph_params['turn_angle_threshold']
        
        for i in range(1, len(point_sequence) - 1):
            u = np.array(point_sequence[i]) - np.array(point_sequence[i-1])
            v = np.array(point_sequence[i+1]) - np.array(point_sequence[i])
            
            u_norm, v_norm = np.linalg.norm(u), np.linalg.norm(v)
            if u_norm > 0 and v_norm > 0:
                cos_angle = np.dot(u, v) / (u_norm * v_norm)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                if angle > theta_turn:
                    turns += 1
        
        return {
            'length': length,
            'geometry': point_sequence,
            'turns': turns
        }
    
    def build_graph_model(self, garden_elements, boundaries):
        """
        构建园林路径网络图模型 - 实现论文1.1节
        返回带权无向图G=(V, E, W)
        """
        print("🏗️ 构建园林路径网络图模型 (论文1.1节)...")
        
        # 1. 路径段重构
        road_coords = garden_elements.get('道路', [])
        segments = self.extract_path_segments(road_coords)
        
        if not segments:
            print("❌ 无法提取路径段")
            return None
        
        # 2. 构建图的顶点集V
        G = nx.Graph()
        
        # V_end: 端点
        endpoints = []
        for segment in segments:
            endpoints.extend([segment[0], segment[-1]])
        
        # V_int: 交叉点
        intersections = self.find_intersections(segments)
        
        # V_poi: 兴趣点  
        poi = self.identify_points_of_interest(garden_elements, boundaries)
        
        # 合并所有顶点
        all_vertices = list(set(endpoints + intersections + poi))
        
        # 添加顶点到图中，标记类型
        for vertex in all_vertices:
            node_type = 'road'
            if vertex in poi:
                if vertex == poi[0]:
                    node_type = 'entrance'
                elif vertex == poi[1]:
                    node_type = 'exit'
                else:
                    node_type = 'poi'
            elif vertex in intersections:
                node_type = 'intersection'
            elif vertex in endpoints:
                node_type = 'endpoint'
            
            G.add_node(vertex, type=node_type)
        
        # 3. 构建图的边集E和权重W
        for segment in segments:
            # 找出这个路径段上的所有顶点
            segment_vertices = [v for v in all_vertices if v in segment]
            segment_vertices.sort(key=lambda x: segment.index(x))
            
            # 创建边
            for i in range(len(segment_vertices) - 1):
                v1, v2 = segment_vertices[i], segment_vertices[i+1]
                
                # 提取v1到v2之间的点序列
                start_idx = segment.index(v1)
                end_idx = segment.index(v2)
                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx
                
                edge_sequence = segment[start_idx:end_idx+1]
                edge_weights = self.calculate_edge_weights(edge_sequence)
                
                G.add_edge(v1, v2, **edge_weights)
        
        print(f"✅ 图模型构建完成: {len(G.nodes())} 顶点, {len(G.edges())} 边")
        
        # 保存图模型
        graph_data = {
            'nodes': [(node, data) for node, data in G.nodes(data=True)],
            'edges': [(u, v, data) for u, v, data in G.edges(data=True)]
        }
        
        return G, poi[0], poi[1]  # 返回图和入口出口
    
    def calculate_path_length(self, tour, graph):
        """计算路径长度 L_len(L) - 论文公式"""
        if len(tour) < 2:
            return 0
        
        total_length = 0
        for i in range(len(tour) - 1):
            if graph.has_edge(tour[i], tour[i+1]):
                edge_data = graph[tour[i]][tour[i+1]]
                total_length += edge_data.get('length', 0)
        
        return total_length
    
    def calculate_path_curvature(self, tour, graph):
        """计算路径曲折度 L_curv(L) - 论文公式"""
        if len(tour) < 2:
            return 0
        
        total_turns = 0
        for i in range(len(tour) - 1):
            if graph.has_edge(tour[i], tour[i+1]):
                edge_data = graph[tour[i]][tour[i+1]]
                total_turns += edge_data.get('turns', 0)
        
        return total_turns
    
    def calculate_viewshed_changes(self, tour, garden_elements):
        """
        计算异景程度 L_view(L) - 实现论文定义2.6-2.7
        基于视域变化的量化
        """
        if len(tour) < 2:
            return 0
        
        # 获取所有景观元素作为观察对象
        landscape_objects = []
        for element_type, coords in garden_elements.items():
            if element_type != '道路':  # 道路不作为观察对象
                landscape_objects.extend(coords)
        
        if not landscape_objects:
            return 0
        
        # 路径采样
        sampled_points = []
        sampling_interval = self.tour_params['sampling_interval']
        
        for i in range(len(tour) - 1):
            p1, p2 = np.array(tour[i]), np.array(tour[i+1])
            distance = np.linalg.norm(p2 - p1)
            
            if distance > 0:
                num_samples = max(2, int(distance / sampling_interval))
                for j in range(num_samples):
                    t = j / (num_samples - 1)
                    sample_point = p1 + t * (p2 - p1)
                    sampled_points.append(tuple(sample_point))
        
        if len(sampled_points) < 2:
            return 0
        
        # 计算每个采样点的视域
        viewshed_radius = self.tour_params['viewshed_radius']
        total_view_changes = 0
        
        prev_viewshed = set()
        for point in sampled_points:
            current_viewshed = set()
            
            # 简化的视域计算：距离内的所有景观元素
            for obj in landscape_objects:
                distance = np.linalg.norm(np.array(point) - np.array(obj))
                if distance <= viewshed_radius:
                    current_viewshed.add(obj)
            
            # 计算视域变化
            if prev_viewshed:
                symmetric_diff = len(current_viewshed.symmetric_difference(prev_viewshed))
                total_view_changes += symmetric_diff
            
            prev_viewshed = current_viewshed
        
        return total_view_changes
    
    def calculate_exploration_score(self, tour, graph):
        """计算探索性 L_exp(L) - 论文公式"""
        if len(tour) < 2:
            return 0
        
        exploration_score = 0
        # 排除起点和终点的内部顶点
        for i in range(1, len(tour) - 1):
            node_degree = graph.degree(tour[i])
            exploration_score += node_degree
        
        return exploration_score
    
    def calculate_interest_score(self, tour, graph, garden_elements):
        """
        计算游线趣味性评分 F(L) - 实现论文公式
        F(L) = (w_curv * L_curv(L) + w_view * L_view(L) + w_exp * L_exp(L)) / (w_len * L_len(L) + C)
        """
        # 计算各项特征
        length = self.calculate_path_length(tour, graph)
        curvature = self.calculate_path_curvature(tour, graph)
        view_changes = self.calculate_viewshed_changes(tour, garden_elements)
        exploration = self.calculate_exploration_score(tour, graph)
        
        # 权重参数
        w_curv = self.tour_params['curvature_weight']
        w_view = self.tour_params['view_change_weight'] 
        w_exp = self.tour_params['exploration_weight']
        w_len = self.tour_params['length_penalty_weight']
        C = self.tour_params['penalty_constant']
        
        # 计算综合评分
        numerator = w_curv * curvature + w_view * view_changes + w_exp * exploration
        denominator = w_len * length + C
        
        interest_score = numerator / denominator if denominator > 0 else 0
        
        metrics = {
            'length': length,
            'curvature': curvature,
            'view_changes': view_changes,
            'exploration': exploration,
            'interest_score': interest_score
        }
        
        return interest_score, metrics
    
    def reinforcement_learning_optimization(self, graph, garden_elements, entrance, exit):
        """
        强化学习路径优化
        最大化趣味性评分F(L)
        """
        print("🧠 开始强化学习路径优化...")
        
        if not graph.has_node(entrance) or not graph.has_node(exit):
            print("❌ 入口或出口不在图中")
            return [], 0, {}
        
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        n_states = len(nodes)
        start_idx = node_to_idx[entrance]
        end_idx = node_to_idx[exit]
        
        # Q表
        Q = np.zeros((n_states, n_states))
        
        # 最优路径记录
        best_tour = []
        best_score = -float('inf')
        best_metrics = {}
        
        # 训练历史
        training_history = {
            'scores': [],
            'best_scores': [],
            'path_lengths': [],
            'episodes': []
        }
        
        print(f"🎯 开始 {self.rl_params['episodes']} 轮训练...")
        
        for episode in tqdm(range(self.rl_params['episodes']), desc="路径优化"):
            # 动态epsilon
            progress = episode / self.rl_params['episodes']
            epsilon = (self.rl_params['epsilon_start'] * (1 - progress) + 
                      self.rl_params['epsilon_end'] * progress)
            
            # 开始一轮游戏
            current_state = start_idx
            tour = [entrance]
            visited = set([current_state])
            
            max_steps = min(100, n_states * 2)
            
            for step in range(max_steps):
                current_node = idx_to_node[current_state]
                neighbors = list(graph.neighbors(current_node))
                
                if not neighbors:
                    break
                
                # 过滤已访问的邻居（避免简单循环）
                available_neighbors = [n for n in neighbors 
                                     if node_to_idx[n] not in visited or n == exit]
                
                if not available_neighbors:
                    available_neighbors = neighbors  # 如果无路可走，允许重访
                
                neighbor_indices = [node_to_idx[n] for n in available_neighbors]
                
                # epsilon-贪心策略
                if np.random.rand() < epsilon:
                    next_state = np.random.choice(neighbor_indices)
                else:
                    q_values = [Q[current_state, idx] for idx in neighbor_indices]
                    next_state = neighbor_indices[np.argmax(q_values)]
                
                next_node = idx_to_node[next_state]
                tour.append(next_node)
                visited.add(next_state)
                
                # 检查是否到达终点
                if next_state == end_idx:
                    break
                
                current_state = next_state
            
            # 计算这条路径的趣味性得分
            if len(tour) >= 2:
                score, metrics = self.calculate_interest_score(tour, graph, garden_elements)
                
                # 更新最优路径
                if score > best_score:
                    best_score = score
                    best_tour = tour.copy()
                    best_metrics = metrics.copy()
                
                # Q值更新
                for i in range(len(tour) - 1):
                    s = node_to_idx[tour[i]]
                    s_next = node_to_idx[tour[i+1]]
                    
                    # 延迟奖励：只在路径结束时给出完整奖励
                    if i == len(tour) - 2:  # 最后一步
                        Q[s, s_next] += self.rl_params['alpha'] * score
                    else:
                        # 中间步骤：当前奖励 + 未来期望
                        future_max = 0
                        if i + 1 < len(tour) - 1:
                            future_neighbors = list(graph.neighbors(tour[i+1]))
                            if future_neighbors:
                                future_q_values = [Q[s_next, node_to_idx[fn]] 
                                                 for fn in future_neighbors 
                                                 if fn in node_to_idx]
                                if future_q_values:
                                    future_max = max(future_q_values)
                        
                        target = score + self.rl_params['gamma'] * future_max
                        Q[s, s_next] += self.rl_params['alpha'] * (target - Q[s, s_next])
                
                # 记录训练历史
                training_history['scores'].append(score)
                training_history['best_scores'].append(best_score)
                training_history['path_lengths'].append(len(tour))
                training_history['episodes'].append(episode)
        
        print(f"✅ 强化学习优化完成!")
        print(f"   🏆 最佳趣味性得分: {best_score:.4f}")
        print(f"   📏 最优路径长度: {len(best_tour)} 节点")
        print(f"   📊 最优路径指标: {best_metrics}")
        
        return best_tour, best_score, training_history
    
    def visualize_optimal_path(self, garden_data, graph, optimal_tour, training_history, 
                             entrance, exit, tour_metrics):
        """在景观分布图上绘制最优路径"""
        garden_name = garden_data['name']
        
        print(f"🎯 生成 {garden_name} 最优路径可视化...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], width_ratios=[2, 1, 1])
        
        # 主图：景观 + 最优路径
        ax_main = fig.add_subplot(gs[0, :2])
        ax_main.set_title(f"{garden_name} - 基于图论与强化学习的最优游览路径", 
                         fontsize=14, fontweight='bold')
        
        # 景观元素配置
        element_config = {
            '道路': {'color': '#FFD700', 'size': 8, 'marker': 'o', 'alpha': 0.6},
            '实体建筑': {'color': '#8B4513', 'size': 20, 'marker': 's', 'alpha': 0.9},
            '半开放建筑': {'color': '#FFA500', 'size': 15, 'marker': '^', 'alpha': 0.8},
            '假山': {'color': '#696969', 'size': 10, 'marker': 'o', 'alpha': 0.7},
            '水体': {'color': '#4169E1', 'size': 12, 'marker': 'o', 'alpha': 0.8},
            '植物': {'color': '#228B22', 'size': 6, 'marker': 'o', 'alpha': 0.7}
        }
        
        # 绘制景观元素
        for element_type, coords in garden_data['elements'].items():
            if not coords:
                continue
            config = element_config.get(element_type, element_config['道路'])
            coords_array = np.array(coords)
            ax_main.scatter(coords_array[:, 0], coords_array[:, 1],
                           c=config['color'], s=config['size'], 
                           marker=config['marker'], alpha=config['alpha'],
                           label=f"{element_type}")
        
        # 绘制图的边（道路网络）
        for edge in graph.edges():
            p1, p2 = edge
            ax_main.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                        color='lightgray', linewidth=1, alpha=0.5, zorder=1)
        
        # 绘制最优路径
        if len(optimal_tour) > 1:
            tour_array = np.array(optimal_tour)
            
            # 主路径线
            line = ax_main.plot(tour_array[:, 0], tour_array[:, 1],
                              color='red', linewidth=4, alpha=0.9,
                              label=f'最优游览路径 (趣味性: {tour_metrics["interest_score"]:.3f})',
                              zorder=10)
            
            # 路径节点
            ax_main.scatter(tour_array[:, 0], tour_array[:, 1],
                           c='darkred', s=25, alpha=0.8, zorder=11)
            
            # 入口和出口标记
            ax_main.scatter(entrance[0], entrance[1], c='lime', s=400, 
                           marker='*', edgecolors='darkgreen', linewidth=3,
                           label='智能入口', zorder=15)
            ax_main.scatter(exit[0], exit[1], c='blue', s=400,
                           marker='*', edgecolors='darkblue', linewidth=3, 
                           label='智能出口', zorder=15)
            
            # 方向箭头
            arrow_interval = max(1, len(optimal_tour) // 8)
            for i in range(arrow_interval, len(optimal_tour), arrow_interval):
                start_pos = optimal_tour[i-1]
                end_pos = optimal_tour[i]
                ax_main.annotate('', xy=end_pos, xytext=start_pos,
                               arrowprops=dict(arrowstyle='->', color='darkred',
                                             lw=2, alpha=0.8), zorder=12)
        
        ax_main.set_xlabel('X坐标 (毫米)', fontsize=12)
        ax_main.set_ylabel('Y坐标 (毫米)', fontsize=12)
        ax_main.grid(True, alpha=0.3)
        ax_main.set_aspect('equal')
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 右上图：训练历史
        ax_train = fig.add_subplot(gs[0, 2])
        ax_train.set_title("强化学习训练过程", fontsize=12)
        episodes = training_history['episodes']
        ax_train.plot(episodes, training_history['best_scores'], 
                     color='red', linewidth=2, label='最佳得分')
        ax_train.set_xlabel('训练轮数')
        ax_train.set_ylabel('趣味性得分') 
        ax_train.grid(True, alpha=0.3)
        ax_train.legend()
        
        # 下方：路径特征分析
        ax_metrics = fig.add_subplot(gs[1, :])
        ax_metrics.set_title("最优路径特征分析 (基于论文1.2节理论)", fontsize=12)
        
        # 创建特征对比柱状图
        features = ['路径长度\n(mm)', '曲折度\n(转折点数)', '异景程度\n(视野变化)', '探索性\n(交叉点度数)']
        values = [tour_metrics['length'], tour_metrics['curvature'], 
                 tour_metrics['view_changes'], tour_metrics['exploration']]
        
        # 归一化显示
        normalized_values = []
        for i, (feature, value) in enumerate(zip(features, values)):
            if i == 0:  # 长度需要缩放
                normalized_values.append(value / 1000)  # 转换为米
            else:
                normalized_values.append(value)
        
        bars = ax_metrics.bar(features, normalized_values, 
                             color=['skyblue', 'lightcoral', 'lightgreen', 'orange'],
                             alpha=0.8)
        
        # 在柱子上显示数值
        for bar, value, original in zip(bars, normalized_values, values):
            height = bar.get_height()
            if features[bars.index(bar)] == '路径长度\n(mm)':
                text = f'{original:.0f}mm\n({value:.1f}m)'
            else:
                text = f'{original:.0f}'
            ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                           text, ha='center', va='bottom', fontsize=9)
        
        ax_metrics.set_ylabel('数值')
        ax_metrics.grid(True, alpha=0.3, axis='y')
        
        # 添加综合评分文本
        score_text = f"综合趣味性评分: {tour_metrics['interest_score']:.4f}\n"
        score_text += f"评分公式: F(L) = (w_curv·L_curv + w_view·L_view + w_exp·L_exp) / (w_len·L_len + C)"
        ax_metrics.text(0.02, 0.98, score_text, transform=ax_metrics.transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        output_filename = f"results/path_optimization/{garden_name}_最优路径分析.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 最优路径分析图已保存: {output_filename}")
        return output_filename
    
    def process_garden(self, garden_name):
        """处理单个园林的路径优化"""
        print(f"\n{'='*60}")
        print(f"🏛️ 路径优化: {garden_name}")
        print(f"📖 理论基础: 论文1.1-1.2节")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. 加载园林数据
        garden_data = self.load_garden_data(garden_name)
        if not garden_data:
            return None
        
        print(f"✅ 数据加载完成: {sum(len(coords) for coords in garden_data['elements'].values())} 个景观元素")
        
        # 2. 构建图模型
        try:
            graph, entrance, exit_point = self.build_graph_model(
                garden_data['elements'], garden_data['boundaries'])
            
            if not graph or len(graph.nodes()) < 3:
                print(f"❌ {garden_name} 图模型构建失败或节点不足")
                return None
                
        except Exception as e:
            print(f"❌ {garden_name} 图模型构建失败: {e}")
            return None
        
        # 3. 强化学习路径优化
        try:
            optimal_tour, best_score, training_history = self.reinforcement_learning_optimization(
                graph, garden_data['elements'], entrance, exit_point)
            
            if not optimal_tour:
                print(f"❌ {garden_name} 未找到最优路径")
                return None
                
        except Exception as e:
            print(f"❌ {garden_name} 路径优化失败: {e}")
            return None
        
        # 4. 计算最终指标
        final_score, final_metrics = self.calculate_interest_score(
            optimal_tour, graph, garden_data['elements'])
        
        # 5. 生成可视化
        try:
            viz_filename = self.visualize_optimal_path(
                garden_