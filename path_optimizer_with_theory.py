import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist
from collections import defaultdict, deque
from tqdm import tqdm
import warnings
import time
import json
import matplotlib
from shapely.geometry import Point, Polygon as ShapelyPolygon, LineString
from shapely.ops import unary_union
import math

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class TheoreticalPathOptimizer:
    """
    基于1.1-1.2.tex理论的路径优化器
    严格按照理论定义实现图模型构建和"趣味性"评价
    """
    
    def __init__(self):
        # 理论参数 - 基于1.1-1.2.tex定义
        self.theory_params = {
            # 路径段提取参数（算法2.1）
            'epsilon_threshold': 1500,  # 距离阈值ε，单位：mm
            
            # 图构建参数（定义2.2, 2.3）
            'intersection_tolerance': 800,  # 交叉点识别容差
            'poi_radius': 5000,  # 兴趣点识别半径
            
            # 视域计算参数（定义2.6, 2.7）
            'viewshed_radius': 8000,  # 视域半径
            'sampling_interval': 2000,  # 路径采样间隔
            
            # 转折点检测参数（定义2.4）
            'turn_angle_threshold': np.pi/6,  # 转折角度阈值（30度）
        }
        
        # 趣味性权重参数（公式中的w系数）
        self.interest_weights = {
            'w_curv': 2.0,    # 曲折度权重
            'w_view': 3.0,    # 异景程度权重  
            'w_exp': 1.5,     # 探索性权重
            'w_len': 0.01,    # 长度惩罚权重
            'C': 1000         # 常数C
        }
        
        # 强化学习参数
        self.rl_params = {
            'episodes': 800,
            'alpha': 0.15,    # 学习率
            'gamma': 0.9,     # 折扣因子
            'epsilon_start': 0.8,
            'epsilon_end': 0.05,
            'decay_rate': 0.995
        }
        
        self.create_output_directories()
    
    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            'results/theory_paths',
            'results/theory_analysis',
            'results/graph_models'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def extract_path_segments(self, road_coords):
        """
        算法2.1：路径段提取算法
        将离散的道路坐标点集处理成连续、有序的路径段
        """
        if not road_coords:
            return []
        
        print("🔨 执行算法2.1：路径段提取...")
        
        P_temp = road_coords.copy()  # 待处理点集
        segments = []  # 路径段集合 S
        epsilon = self.theory_params['epsilon_threshold']
        
        segment_id = 0
        while P_temp:
            # 选择起始点
            p_start = P_temp.pop(0)
            current_segment = [p_start]
            
            # 向前扩展
            while True:
                if not P_temp:
                    break
                
                p_curr = current_segment[-1]
                # 找最近点
                distances = [np.linalg.norm(np.array(p) - np.array(p_curr)) for p in P_temp]
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                if min_dist <= epsilon:
                    p_next = P_temp.pop(min_idx)
                    current_segment.append(p_next)
                else:
                    break
            
            # 向后扩展（处理起始点在路径中间的情况）
            P_temp_backup = P_temp.copy()
            while P_temp_backup:
                p_curr = current_segment[0]
                distances = [np.linalg.norm(np.array(p) - np.array(p_curr)) for p in P_temp_backup]
                if not distances:
                    break
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                if min_dist <= epsilon:
                    p_prev = P_temp_backup.pop(min_idx)
                    current_segment.insert(0, p_prev)
                    P_temp.remove(p_prev)
                else:
                    break
                P_temp_backup = [p for p in P_temp_backup if p in P_temp]
            
            if len(current_segment) >= 2:
                segments.append({
                    'id': f'S_{segment_id}',
                    'points': current_segment,
                    'length': self.calculate_segment_length(current_segment)
                })
                segment_id += 1
        
        print(f"✅ 提取到 {len(segments)} 条路径段")
        return segments
    
    def calculate_segment_length(self, points):
        """计算路径段长度（定义2.4中的W_len）"""
        if len(points) < 2:
            return 0
        
        total_length = 0
        for i in range(len(points) - 1):
            p1 = np.array(points[i])
            p2 = np.array(points[i + 1])
            total_length += np.linalg.norm(p2 - p1)
        
        return total_length
    
    def calculate_segment_turns(self, points):
        """计算路径段转折点数量（定义2.4中的W_turns）"""
        if len(points) < 3:
            return 0
        
        turn_count = 0
        theta_turn = self.theory_params['turn_angle_threshold']
        
        for i in range(1, len(points) - 1):
            p_prev = np.array(points[i - 1])
            p_curr = np.array(points[i])
            p_next = np.array(points[i + 1])
            
            # 计算向量
            u = p_curr - p_prev
            v = p_next - p_curr
            
            # 计算角度
            if np.linalg.norm(u) > 0 and np.linalg.norm(v) > 0:
                cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                
                if angle > theta_turn:
                    turn_count += 1
        
        return turn_count
    
    def build_graph_model(self, garden_data):
        """
        构建带权无向图G=(V,E,W)（定义2.2-2.4）
        """
        print("🏗️ 构建图模型G=(V,E,W)...")
        
        road_coords = garden_data['elements'].get('道路', [])
        building_coords = garden_data['elements'].get('实体建筑', []) + \
                         garden_data['elements'].get('半开放建筑', [])
        
        if not road_coords:
            return None
        
        # 1. 提取路径段
        segments = self.extract_path_segments(road_coords)
        
        # 2. 构建顶点集V
        vertices = self.build_vertices(segments, building_coords, road_coords)
        
        # 3. 构建边集E和权重W
        edges, edge_weights = self.build_edges_and_weights(segments, vertices)
        
        # 4. 创建NetworkX图
        G = nx.Graph()
        
        # 添加顶点
        for v_id, v_data in vertices.items():
            G.add_node(v_id, **v_data)
        
        # 添加边
        for edge_id, edge_data in edges.items():
            start_v, end_v = edge_data['vertices']
            weight_data = edge_weights[edge_id]
            G.add_edge(start_v, end_v, edge_id=edge_id, **weight_data)
        
        print(f"✅ 图模型构建完成: {len(G.nodes())} 个顶点, {len(G.edges())} 条边")
        
        return {
            'graph': G,
            'segments': segments,
            'vertices': vertices,
            'edges': edges,
            'edge_weights': edge_weights
        }
    
    def build_vertices(self, segments, building_coords, road_coords):
        """
        构建顶点集V（定义2.2）
        V = V_end ∪ V_int ∪ V_poi
        """
        vertices = {}
        vertex_id = 0
        tolerance = self.theory_params['intersection_tolerance']
        
        # V_end: 端点
        endpoints = []
        for segment in segments:
            points = segment['points']
            if len(points) >= 2:
                endpoints.extend([tuple(points[0]), tuple(points[-1])])
        
        # 去重端点
        unique_endpoints = list(set(endpoints))
        for ep in unique_endpoints:
            vertices[f'v_{vertex_id}'] = {
                'type': 'endpoint',
                'position': ep,
                'coords': ep
            }
            vertex_id += 1
        
        # V_int: 交叉点（简化实现，基于距离聚类）
        all_points = []
        for segment in segments:
            all_points.extend(segment['points'])
        
        if len(all_points) > 3:
            # 使用DBSCAN寻找密集区域作为交叉点
            points_array = np.array(all_points)
            clustering = DBSCAN(eps=tolerance, min_samples=3).fit(points_array)
            
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # 忽略噪声点
                    cluster_points = points_array[clustering.labels_ == cluster_id]
                    center = np.mean(cluster_points, axis=0)
                    center_tuple = tuple(center)
                    
                    # 检查是否已有近似顶点
                    is_duplicate = False
                    for v_data in vertices.values():
                        if np.linalg.norm(np.array(v_data['position']) - center) < tolerance:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        vertices[f'v_{vertex_id}'] = {
                            'type': 'intersection',
                            'position': center_tuple,
                            'coords': center_tuple
                        }
                        vertex_id += 1
        
        # V_poi: 兴趣点（入口出口，基于边界检测）
        if road_coords:
            road_array = np.array(road_coords)
            # 简单地选择最远的两个点作为入口出口
            distances = cdist(road_array, road_array)
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            
            entrance = tuple(road_array[i])
            exit_point = tuple(road_array[j])
            
            vertices[f'v_{vertex_id}'] = {
                'type': 'entrance',
                'position': entrance,
                'coords': entrance
            }
            vertex_id += 1
            
            vertices[f'v_{vertex_id}'] = {
                'type': 'exit', 
                'position': exit_point,
                'coords': exit_point
            }
            vertex_id += 1
        
        return vertices
    
    def build_edges_and_weights(self, segments, vertices):
        """
        构建边集E和权重W（定义2.3-2.4）
        """
        edges = {}
        edge_weights = {}
        edge_id = 0
        
        # 为每个路径段在顶点间创建边
        for segment in segments:
            points = segment['points']
            if len(points) < 2:
                continue
            
            # 找到该段起点和终点对应的顶点
            start_point = tuple(points[0])
            end_point = tuple(points[-1])
            
            start_vertex = None
            end_vertex = None
            tolerance = self.theory_params['intersection_tolerance']
            
            for v_id, v_data in vertices.items():
                v_pos = np.array(v_data['position'])
                if np.linalg.norm(v_pos - np.array(start_point)) < tolerance:
                    start_vertex = v_id
                if np.linalg.norm(v_pos - np.array(end_point)) < tolerance:
                    end_vertex = v_id
            
            if start_vertex and end_vertex and start_vertex != end_vertex:
                edge_key = f'e_{edge_id}'
                
                edges[edge_key] = {
                    'vertices': (start_vertex, end_vertex),
                    'segment_id': segment['id']
                }
                
                # 计算边的权重（定义2.4）
                edge_weights[edge_key] = {
                    'W_len': self.calculate_segment_length(points),
                    'W_geom': points,  # 几何序列
                    'W_turns': self.calculate_segment_turns(points)
                }
                
                edge_id += 1
        
        return edges, edge_weights
    
    def calculate_tour_path_features(self, tour_path, graph_model, garden_data):
        """
        计算游线特征（1.2.2节）
        """
        if not tour_path or len(tour_path) < 2:
            return {'L_len': 0, 'L_curv': 0, 'L_view': 0, 'L_exp': 0}
        
        G = graph_model['graph']
        
        # L_len: 路径长度
        L_len = 0
        for i in range(len(tour_path) - 1):
            if G.has_edge(tour_path[i], tour_path[i+1]):
                edge_data = G.edges[tour_path[i], tour_path[i+1]]
                L_len += edge_data.get('W_len', 0)
        
        # L_curv: 路径曲折度
        L_curv = 0
        for i in range(len(tour_path) - 1):
            if G.has_edge(tour_path[i], tour_path[i+1]):
                edge_data = G.edges[tour_path[i], tour_path[i+1]]
                L_curv += edge_data.get('W_turns', 0)
        
        # L_view: 异景程度（简化实现）
        L_view = self.calculate_view_change(tour_path, graph_model, garden_data)
        
        # L_exp: 探索性
        L_exp = 0
        for i in range(1, len(tour_path) - 1):  # 排除起点终点
            L_exp += G.degree(tour_path[i])
        
        return {
            'L_len': L_len,
            'L_curv': L_curv, 
            'L_view': L_view,
            'L_exp': L_exp
        }
    
    def calculate_view_change(self, tour_path, graph_model, garden_data):
        """
        计算异景程度L_view（定义2.6-2.7的简化实现）
        """
        if len(tour_path) < 2:
            return 0
        
        # 简化实现：基于路径经过的不同类型景观元素数量
        viewshed_radius = self.theory_params['viewshed_radius']
        vertices = graph_model['vertices']
        
        total_view_change = 0
        prev_visible_elements = set()
        
        for vertex_id in tour_path:
            if vertex_id not in vertices:
                continue
            
            position = np.array(vertices[vertex_id]['position'])
            current_visible_elements = set()
            
            # 检查各类景观元素是否在视域内
            for element_type, coords in garden_data['elements'].items():
                for i, coord in enumerate(coords):
                    distance = np.linalg.norm(position - np.array(coord))
                    if distance <= viewshed_radius:
                        current_visible_elements.add(f'{element_type}_{i}')
            
            # 计算与前一点的视域变化
            if prev_visible_elements:
                view_change = len(current_visible_elements.symmetric_difference(prev_visible_elements))
                total_view_change += view_change
            
            prev_visible_elements = current_visible_elements
        
        return total_view_change
    
    def calculate_interest_score(self, features):
        """
        计算趣味性评分F(L)（1.2.3节公式）
        """
        L_curv = features['L_curv']
        L_view = features['L_view'] 
        L_exp = features['L_exp']
        L_len = features['L_len']
        
        w = self.interest_weights
        
        numerator = w['w_curv'] * L_curv + w['w_view'] * L_view + w['w_exp'] * L_exp
        denominator = w['w_len'] * L_len + w['C']
        
        F_L = numerator / max(denominator, 1e-6)  # 防止除零
        
        return max(F_L, 0.001)  # 确保非负
    
    def find_entrance_exit(self, vertices):
        """找到入口和出口顶点"""
        entrance = None
        exit_point = None
        
        for v_id, v_data in vertices.items():
            if v_data['type'] == 'entrance':
                entrance = v_id
            elif v_data['type'] == 'exit':
                exit_point = v_id
        
        return entrance, exit_point
    
    def reinforcement_learning_optimization(self, graph_model, garden_data):
        """
        强化学习路径优化
        """
        print("🧠 启动强化学习路径优化...")
        
        G = graph_model['graph']
        vertices = graph_model['vertices']
        
        if len(G.nodes()) < 2:
            return [], 0, {}
        
        # 找入口出口
        entrance, exit_point = self.find_entrance_exit(vertices)
        if not entrance or not exit_point:
            nodes = list(G.nodes())
            entrance = nodes[0]
            exit_point = nodes[-1] if len(nodes) > 1 else nodes[0]
        
        print(f"🚪 入口: {entrance}, 🏁 出口: {exit_point}")
        
        # 节点映射
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        n_states = len(nodes)
        start_idx = node_to_idx[entrance]
        end_idx = node_to_idx[exit_point]
        
        # Q表
        Q = np.zeros((n_states, n_states))
        
        best_path = []
        best_score = -float('inf')
        training_history = {'scores': [], 'best_scores': []}
        
        print(f"🎯 开始训练: {self.rl_params['episodes']} 轮")
        
        for episode in tqdm(range(self.rl_params['episodes']), desc="强化学习训练"):
            current_state = start_idx
            path = [entrance]
            
            # 动态epsilon
            progress = episode / self.rl_params['episodes']
            epsilon = (self.rl_params['epsilon_start'] * (1 - progress) + 
                      self.rl_params['epsilon_end'] * progress)
            
            max_steps = min(100, len(nodes) * 2)
            
            for step in range(max_steps):
                current_node = idx_to_node[current_state]
                neighbors = list(G.neighbors(current_node))
                
                if not neighbors:
                    break
                
                neighbor_indices = [node_to_idx[n] for n in neighbors if n in node_to_idx]
                if not neighbor_indices:
                    break
                
                # ε-贪心策略
                if np.random.rand() < epsilon:
                    next_state = np.random.choice(neighbor_indices)
                else:
                    q_values = [Q[current_state, idx] for idx in neighbor_indices]
                    best_idx = np.argmax(q_values)
                    next_state = neighbor_indices[best_idx]
                
                next_node = idx_to_node[next_state]
                path.append(next_node)
                
                # 终止条件
                if next_state == end_idx or step >= max_steps - 1:
                    # 计算路径特征和趣味性评分
                    features = self.calculate_tour_path_features(path, graph_model, garden_data)
                    score = self.calculate_interest_score(features)
                    
                    # 更新最佳路径
                    if score > best_score:
                        best_score = score
                        best_path = path.copy()
                        if episode % 100 == 0:
                            print(f"  🎯 Episode {episode}: 新最佳评分 {score:.4f}")
                    
                    # Q值更新
                    for i in range(len(path) - 1):
                        s = node_to_idx[path[i]]
                        s_next = node_to_idx[path[i + 1]]
                        
                        if i == len(path) - 2:  # 最后一步
                            Q[s, s_next] = ((1 - self.rl_params['alpha']) * Q[s, s_next] + 
                                           self.rl_params['alpha'] * score)
                        else:
                            future_neighbors = list(G.neighbors(path[i + 1]))
                            if future_neighbors:
                                next_q_values = [Q[s_next, node_to_idx[n]] for n in future_neighbors 
                                               if n in node_to_idx]
                                next_q_max = max(next_q_values) if next_q_values else 0
                            else:
                                next_q_max = 0
                            
                            Q[s, s_next] = ((1 - self.rl_params['alpha']) * Q[s, s_next] + 
                                           self.rl_params['alpha'] * 
                                           (score + self.rl_params['gamma'] * next_q_max))
                    
                    training_history['scores'].append(score)
                    training_history['best_scores'].append(best_score)
                    break
                
                current_state = next_state
        
        print(f"✅ 强化学习完成！最佳评分: {best_score:.4f}")
        return best_path, best_score, training_history
    
    def visualize_optimal_path(self, garden_data, graph_model, best_path, best_score, training_history):
        """可视化最优路径"""
        garden_name = garden_data['name']
        print(f"🎨 生成 {garden_name} 理论路径分析图...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{garden_name} - 基于理论的路径优化分析', fontsize=16, fontweight='bold')
        
        # 1. 景观元素 + 图模型 + 最优路径
        ax1.set_title("图模型与最优路径", fontweight='bold')
        
        # 绘制景观元素
        element_configs = {
            '道路': {'color': '#FFD700', 'size': 8, 'alpha': 0.6},
            '实体建筑': {'color': '#8B4513', 'size': 25, 'alpha': 0.8},
            '半开放建筑': {'color': '#FFA500', 'size': 20, 'alpha': 0.8},
            '假山': {'color': '#696969', 'size': 12, 'alpha': 0.7},
            '水体': {'color': '#4169E1', 'size': 10, 'alpha': 0.8},
            '植物': {'color': '#228B22', 'size': 6, 'alpha': 0.6}
        }
        
        for element_type, coords in garden_data['elements'].items():
            if coords and element_type in element_configs:
                config = element_configs[element_type]
                coords_array = np.array(coords)
                ax1.scatter(coords_array[:, 0], coords_array[:, 1],
                           c=config['color'], s=config['size'], 
                           alpha=config['alpha'], label=element_type)
        
        # 绘制图的顶点
        G = graph_model['graph']
        vertices = graph_model['vertices']
        
        for v_id, v_data in vertices.items():
            pos = v_data['position']
            v_type = v_data['type']
            
            if v_type == 'entrance':
                ax1.scatter(pos[0], pos[1], c='lime', s=200, marker='*', 
                           edgecolors='black', linewidth=2, label='入口', zorder=10)
            elif v_type == 'exit':
                ax1.scatter(pos[0], pos[1], c='red', s=200, marker='*',
                           edgecolors='black', linewidth=2, label='出口', zorder=10)
            else:
                ax1.scatter(pos[0], pos[1], c='white', s=50, 
                           edgecolors='black', linewidth=1, alpha=0.8, zorder=5)
        
        # 绘制图的边
        for (u, v) in G.edges():
            pos_u = vertices[u]['position']
            pos_v = vertices[v]['position']
            ax1.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 
                    'gray', alpha=0.3, linewidth=1, zorder=1)
        
        # 绘制最优路径
        if len(best_path) > 1:
            path_coords = []
            for node_id in best_path:
                if node_id in vertices:
                    path_coords.append(vertices[node_id]['position'])
            
            if len(path_coords) > 1:
                path_array = np.array(path_coords)
                ax1.plot(path_array[:, 0], path_array[:, 1], 
                        'red', linewidth=4, alpha=0.8, label=f'最优路径', zorder=8)
                
                # 箭头指示方向
                for i in range(0, len(path_coords)-1, max(1, len(path_coords)//5)):
                    start = path_coords[i]
                    end = path_coords[i+1]
                    ax1.annotate('', xy=end, xytext=start,
                               arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
        
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. 训练历史
        ax2.set_title("强化学习训练历史", fontweight='bold')
        if training_history['scores']:
            episodes = range(len(training_history['scores']))
            ax2.plot(episodes, training_history['best_scores'], 'r-', linewidth=2, label='最佳评分')
            ax2.set_xlabel('训练轮数')
            ax2.set_ylabel('趣味性评分')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 路径特征分析
        ax3.set_title("路径特征分析", fontweight='bold')
        if best_path:
            features = self.calculate_tour_path_features(best_path, graph_model, garden_data)
            feature_names = ['长度\n(L_len)', '曲折度\n(L_curv)', '异景度\n(L_view)', '探索性\n(L_exp)']
            feature_values = [features['L_len']/1000, features['L_curv'], 
                            features['L_view'], features['L_exp']]
            
            bars = ax3.bar(feature_names, feature_values, color=['blue', 'green', 'orange', 'purple'], alpha=0.7)
            ax3.set_ylabel('特征值')
            
            # 添加数值标签
            for bar, value in zip(bars, feature_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(feature_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. 理论公式验证
        ax4.set_title("趣味性评分公式验证", fontweight='bold')
        if best_path:
            features = self.calculate_tour_path_features(best_path, graph_model, garden_data)
            w = self.interest_weights
            
            # 分子项
            numerator_terms = [
                w['w_curv'] * features['L_curv'],
                w['w_view'] * features['L_view'], 
                w['w_exp'] * features['L_exp']
            ]
            
            # 分母项
            denominator = w['w_len'] * features['L_len'] + w['C']
            
            labels = ['曲折度项\nw_curv×L_curv', '异景度项\nw_view×L_view', '探索性项\nw_exp×L_exp']
            bars = ax4.bar(labels, numerator_terms, color=['red', 'green', 'blue'], alpha=0.7)
            
            ax4.axhline(y=denominator/len(numerator_terms), color='orange', linestyle='--', 
                       linewidth=2, label=f'分母项均值: {denominator/len(numerator_terms):.1f}')
            
            ax4.set_ylabel('权重×特征值')
            ax4.legend()
            ax4.tick_params(axis='x', rotation=15)
            
            # 显示最终评分
            final_score = sum(numerator_terms) / denominator
            ax4.text(0.5, 0.95, f'最终趣味性评分 F(L) = {final_score:.4f}', 
                    transform=ax4.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7),
                    fontsize=12, fontweight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        path_filename = f"results/theory_paths/{garden_name}_理论路径分析.png"
        plt.savefig(path_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 理论路径分析图已保存: {path_filename}")
        return path_filename
    
    def process_garden_with_theory(self, garden_data):
        """基于理论处理单个园林"""
        garden_name = garden_data['name']
        print(f"\n{'='*60}")
        print(f"🏛️ 基于理论处理园林: {garden_name}")
        print(f"📖 严格按照1.1-1.2.tex理论实现")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 1. 构建图模型
        graph_model = self.build_graph_model(garden_data)
        if not graph_model:
            print(f"❌ {garden_name} 图模型构建失败")
            return None
        
        # 2. 强化学习优化
        best_path, best_score, training_history = self.reinforcement_learning_optimization(
            graph_model, garden_data)
        
        if not best_path:
            print(f"❌ {garden_name} 路径优化失败")
            return None
        
        # 3. 计算最终特征
        final_features = self.calculate_tour_path_features(best_path, graph_model, garden_data)
        
        # 4. 可视化
        path_filename = self.visualize_optimal_path(
            garden_data, graph_model, best_path, best_score, training_history)
        
        processing_time = time.time() - start_time
        
        # 5. 保存理论验证结果
        theory_result = {
            'garden_name': garden_name,
            'graph_info': {
                'vertices_count': len(graph_model['vertices']),
                'edges_count': len(graph_model['edges']),
                'segments_count': len(graph_model['segments'])
            },
            'path_features': final_features,
            'interest_score': best_score,
            'path_length': len(best_path),
            'processing_time': processing_time,
            'theory_weights': self.interest_weights
        }
        
        with open(f'results/theory_analysis/{garden_name}_理论验证.json', 'w', encoding='utf-8') as f:
            json.dump(theory_result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ {garden_name} 理论处理完成:")
        print(f"   📊 趣味性评分: {best_score:.4f}")
        print(f"   📏 路径长度: {final_features['L_len']:.0f}mm")
        print(f"   🔄 曲折度: {final_features['L_curv']}")
        print(f"   👁️ 异景度: {final_features['L_view']}")
        print(f"   🧭 探索性: {final_features['L_exp']}")
        print(f"   ⏱️ 处理时间: {processing_time:.2f}秒")
        
        return theory_result

def main():
    """主函数"""
    print("🏛️ 基于理论的江南古典园林路径优化系统")
    print("📖 严格按照1.1-1.2.tex理论框架实现")
    print("=" * 70)
    
    # 这里需要从第一部分获取园林数据
    # 示例用法（需要配合garden_data_loader.py使用）
    print("⚠️  请先运行 garden_data_loader.py 获取园林数据")
    print("然后将数据传递给本模块进行理论路径优化")
    
    optimizer = TheoreticalPathOptimizer()
    
    # 示例：处理单个园林（需要garden_data作为输入）
    # result = optimizer.process_garden_with_theory(garden_data)

if __name__ == "__main__":
    main()