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
warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class GardenVisualizationAndRLOptimizer:
    """
    园林可视化和强化学习路径优化系统

    功能模块：
    1. 园林景观元素可视化
    2. 强化学习路径优化训练
    3. 最优路径叠加显示
    4. 批量处理10个园林
    """

    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        self.data_dir = data_dir
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }

        # 景观元素配置
        self.element_config = {
            '道路': {'color': '#D3D3D3', 'size': 3, 'marker': '.', 'alpha': 0.6},
            '实体建筑': {'color': '#8B4513', 'size': 20, 'marker': 's', 'alpha': 0.8},
            '半开放建筑': {'color': '#FFA500', 'size': 15, 'marker': '^', 'alpha': 0.7},
            '假山': {'color': '#696969', 'size': 10, 'marker': 'o', 'alpha': 0.7},
            '水体': {'color': '#4169E1', 'size': 8, 'marker': 'o', 'alpha': 0.8},
            '植物': {'color': '#228B22', 'size': 5, 'marker': 'o', 'alpha': 0.6}
        }

        # RL训练参数
        self.rl_config = {
            'episodes': 500,
            'alpha': 0.1,
            'gamma': 0.95,
            'epsilon': 0.2,
            'decay_rate': 0.995
        }

        # 趣味性指标权重
        self.weights = {
            'interest': 3.0,      # 趣味性
            'diversity': 2.0,     # 多样性
            'novelty': 2.5,       # 新奇性
            'length_penalty': 0.1, # 长度惩罚
            'repetition_penalty': 1.0  # 重复性惩罚
        }

        # 创建输出目录
        self.create_output_directories()

    def create_output_directories(self):
        """创建输出目录结构"""
        directories = [
            'results/garden_maps',
            'results/optimal_paths',
            'results/rl_training',
            'results/analysis'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def parse_coordinate_string(self, coord_str):
        """解析坐标字符串"""
        if pd.isna(coord_str):
            return None

        coord_str = str(coord_str).strip()

        patterns = [
            r'\{([^}]+)\}',  # {x,y,z}
            r'\(([^)]+)\)',  # (x,y,z)
            r'\[([^\]]+)\]', # [x,y,z]
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

        return '道路'  # 默认类型

    def extract_coordinates_from_dataframe(self, df):
        """从DataFrame中提取坐标"""
        coords = []

        for col in df.columns:
            for _, row in df.iterrows():
                coord_str = str(row[col])
                parsed_coord = self.parse_coordinate_string(coord_str)
                if parsed_coord:
                    coords.append(parsed_coord)

        return list(set(coords))  # 去重

    def generate_garden_landscape_map(self, garden_data):
        """生成园林景观地图（第一步）"""
        garden_id = garden_data['id']
        garden_name = garden_data['name']

        print(f"🎨 生成 {garden_name} 景观地图...")

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_title(f"{garden_name} - 景观元素分布图", fontsize=16, fontweight='bold', pad=20)

        legend_elements = []
        total_elements = 0

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
                               label=f"{element_type} ({len(coords)}个)")

            legend_elements.append(scatter)
            total_elements += len(coords)

        # 设置图表属性
        ax.set_xlabel('X坐标 (毫米)', fontsize=12)
        ax.set_ylabel('Y坐标 (毫米)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 添加图例
        if legend_elements:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

        # 添加统计信息
        info_text = f"总景观元素: {total_elements}个\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # 保存景观地图
        map_filename = f"results/garden_maps/{garden_name}_景观地图.png"
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 景观地图已保存: {map_filename}")
        return map_filename, garden_data['elements']

    def build_path_network(self, road_coords, building_coords):
        """构建路径网络图"""
        if not road_coords:
            return nx.Graph()

        G = nx.Graph()
        road_array = np.array(road_coords)

        # 添加节点
        for i, coord in enumerate(road_coords):
            G.add_node(coord, pos=coord, id=i)

        # 构建边 - 使用KDTree优化
        if len(road_array) > 1:
            tree = cKDTree(road_array)
            connection_threshold = 5000  # 5米连接阈值

            for i, coord in enumerate(road_coords):
                # 找到附近的点
                distances, indices = tree.query(road_array[i], k=min(8, len(road_array)),
                                              distance_upper_bound=connection_threshold)

                for j, neighbor_idx in enumerate(indices):
                    if (neighbor_idx < len(road_coords) and
                        distances[j] < float('inf') and
                        neighbor_idx != i):

                        neighbor_coord = road_coords[neighbor_idx]
                        if not G.has_edge(coord, neighbor_coord):
                            G.add_edge(coord, neighbor_coord,
                                     length=distances[j],
                                     type='road')

        return G

    def calculate_path_metrics(self, path, graph, landscape_elements):
        """计算路径的多维度指标"""
        if len(path) < 2:
            return {
                'interest': 0, 'diversity': 0, 'novelty': 0,
                'length': 0, 'repetition': 0
            }

        # 1. 路径长度
        path_length = 0
        for i in range(len(path) - 1):
            if graph.has_edge(path[i], path[i+1]):
                edge_data = graph.get_edge_data(path[i], path[i+1])
                path_length += edge_data.get('length', 0)
            else:
                path_length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))

        # 2. 趣味性指标 - 基于路径曲折度
        interest_score = 0
        if len(path) > 2:
            angles = []
            for i in range(1, len(path) - 1):
                v1 = np.array(path[i]) - np.array(path[i-1])
                v2 = np.array(path[i+1]) - np.array(path[i])

                # 计算转弯角度
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)

            interest_score = np.mean(angles) if angles else 0

        # 3. 多样性指标 - 路径附近的景观元素种类
        diversity_score = 0
        view_radius = 10000  # 10米视野半径

        unique_elements = set()
        for point in path[::5]:  # 每5个点采样一次
            point_array = np.array(point)

            for element_type, coords in landscape_elements.items():
                if element_type == '道路':  # 跳过道路元素
                    continue

                if coords:
                    coords_array = np.array(coords)
                    distances = np.linalg.norm(coords_array - point_array, axis=1)

                    if np.any(distances <= view_radius):
                        unique_elements.add(element_type)

        diversity_score = len(unique_elements)

        # 4. 新奇性指标 - 基于路径节点的度数变化
        novelty_score = 0
        if graph and len(path) > 1:
            node_degrees = [graph.degree(node) for node in path if node in graph.nodes()]
            novelty_score = np.std(node_degrees) if node_degrees else 0

        # 5. 重复性指标 - 路径中重复访问的节点
        repetition_score = len(path) - len(set(path))

        return {
            'interest': interest_score,
            'diversity': diversity_score,
            'novelty': novelty_score,
            'length': path_length,
            'repetition': repetition_score
        }

    def calculate_reward(self, path, graph, landscape_elements):
        """计算强化学习奖励函数"""
        metrics = self.calculate_path_metrics(path, graph, landscape_elements)

        # 综合奖励计算
        reward = (self.weights['interest'] * metrics['interest'] +
                 self.weights['diversity'] * metrics['diversity'] +
                 self.weights['novelty'] * metrics['novelty'] -
                 self.weights['length_penalty'] * metrics['length'] / 1000 -  # 标准化长度
                 self.weights['repetition_penalty'] * metrics['repetition'])

        return max(reward, 0.01), metrics

    def train_rl_pathfinder(self, graph, landscape_elements, start_node, end_node):
        """训练强化学习路径查找器（第二步）"""
        print(f"🤖 开始强化学习训练...")
        print(f"   起点: {start_node}")
        print(f"   终点: {end_node}")

        if not graph.has_node(start_node) or not graph.has_node(end_node):
            print("❌ 起点或终点不在图中")
            return [], 0, {}

        # 节点映射
        nodes = list(graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for i, node in enumerate(nodes)}

        n_states = len(nodes)
        start_idx = node_to_idx[start_node]
        end_idx = node_to_idx[end_node]

        # Q-table初始化
        Q = np.zeros((n_states, n_states))
        visit_count = defaultdict(int)

        best_path = []
        best_reward = -float('inf')
        best_metrics = {}

        training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'best_rewards': []
        }

        print(f"📊 训练参数: {self.rl_config['episodes']} 轮训练")

        # Q-learning训练
        for episode in tqdm(range(self.rl_config['episodes']), desc="RL训练进度"):
            current_state = start_idx
            path = [start_node]
            episode_reward = 0

            # epsilon衰减
            epsilon = self.rl_config['epsilon'] * (self.rl_config['decay_rate'] ** episode)

            max_steps = min(50, n_states)

            for step in range(max_steps):
                # 获取邻居节点
                current_node = idx_to_node[current_state]
                neighbors = list(graph.neighbors(current_node))

                if not neighbors:
                    break

                neighbor_indices = [node_to_idx[n] for n in neighbors if n in node_to_idx]

                if not neighbor_indices:
                    break

                # epsilon-greedy策略
                if np.random.rand() < epsilon:
                    # 探索
                    next_state = np.random.choice(neighbor_indices)
                else:
                    # 利用
                    q_values = [Q[current_state, n] for n in neighbor_indices]
                    best_idx = np.argmax(q_values)
                    next_state = neighbor_indices[best_idx]

                next_node = idx_to_node[next_state]
                path.append(next_node)
                visit_count[(current_state, next_state)] += 1

                # 到达终点
                if next_state == end_idx:
                    # 计算最终奖励
                    final_reward, metrics = self.calculate_reward(path, graph, landscape_elements)
                    episode_reward = final_reward

                    # 更新最佳路径
                    if final_reward > best_reward:
                        best_reward = final_reward
                        best_path = path.copy()
                        best_metrics = metrics.copy()

                    # 反向传播奖励
                    for i in range(len(path) - 1):
                        s = node_to_idx[path[i]]
                        s_next = node_to_idx[path[i + 1]]

                        # 更新Q值
                        if i == len(path) - 2:  # 最后一步
                            Q[s, s_next] = ((1 - self.rl_config['alpha']) * Q[s, s_next] +
                                           self.rl_config['alpha'] * final_reward)
                        else:
                            # 中间步骤
                            next_q_max = np.max([Q[s_next, node_to_idx[n]]
                                               for n in graph.neighbors(path[i + 1])
                                               if n in node_to_idx])

                            Q[s, s_next] = ((1 - self.rl_config['alpha']) * Q[s, s_next] +
                                           self.rl_config['alpha'] *
                                           (final_reward / len(path) + self.rl_config['gamma'] * next_q_max))

                    break

                current_state = next_state

            # 记录训练历史
            training_history['episode_rewards'].append(episode_reward)
            training_history['episode_lengths'].append(len(path))
            training_history['best_rewards'].append(best_reward)

        print(f"✅ RL训练完成!")
        print(f"   最佳奖励: {best_reward:.4f}")
        print(f"   最佳路径长度: {len(best_path)} 个节点")
        print(f"   路径指标: {best_metrics}")

        return best_path, best_reward, training_history

    def find_start_end_nodes(self, road_coords):
        """寻找起点和终点"""
        if not road_coords:
            return None, None

        if len(road_coords) < 2:
            return road_coords[0], road_coords[0]

        # 找到距离最远的两个点作为起终点
        coords_array = np.array(road_coords)
        distances = cdist(coords_array, coords_array)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)

        return road_coords[i], road_coords[j]

    def visualize_optimal_path_overlay(self, garden_data, best_path, training_history):
        """在景观地图上叠加最优路径（第三步）"""
        garden_id = garden_data['id']
        garden_name = garden_data['name']

        print(f"🎯 生成 {garden_name} 最优路径叠加图...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # 左图：景观地图 + 最优路径
        ax1.set_title(f"{garden_name} - 最优游览路径", fontsize=14, fontweight='bold')

        # 绘制景观元素
        for element_type, coords in garden_data['elements'].items():
            if not coords:
                continue

            config = self.element_config.get(element_type,
                {'color': '#000000', 'size': 5, 'marker': 'o', 'alpha': 0.7})

            coords_array = np.array(coords)
            ax1.scatter(coords_array[:, 0], coords_array[:, 1],
                       c=config['color'], s=config['size'],
                       marker=config['marker'], alpha=config['alpha'],
                       label=f"{element_type}")

        # 绘制最优路径
        if len(best_path) > 1:
            path_array = np.array(best_path)
            ax1.plot(path_array[:, 0], path_array[:, 1],
                    color='red', linewidth=3, alpha=0.8,
                    label=f'最优路径 ({len(best_path)}个节点)')

            # 标记起点和终点
            ax1.scatter(best_path[0][0], best_path[0][1],
                       c='green', s=100, marker='*',
                       label='起点', zorder=10)
            ax1.scatter(best_path[-1][0], best_path[-1][1],
                       c='red', s=100, marker='*',
                       label='终点', zorder=10)

        ax1.set_xlabel('X坐标 (毫米)')
        ax1.set_ylabel('Y坐标 (毫米)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.set_aspect('equal')

        # 右图：训练历史
        ax2.set_title(f"{garden_name} - RL训练历史", fontsize=14, fontweight='bold')

        episodes = range(len(training_history['episode_rewards']))
        ax2_twin = ax2.twinx()

        # 奖励曲线
        line1 = ax2.plot(episodes, training_history['episode_rewards'],
                        color='blue', alpha=0.6, label='每轮奖励')
        line2 = ax2.plot(episodes, training_history['best_rewards'],
                        color='red', linewidth=2, label='最佳奖励')

        # 路径长度曲线
        line3 = ax2_twin.plot(episodes, training_history['episode_lengths'],
                             color='green', alpha=0.6, label='路径长度')

        ax2.set_xlabel('训练轮数')
        ax2.set_ylabel('奖励值', color='blue')
        ax2_twin.set_ylabel('路径长度', color='green')
        ax2.grid(True, alpha=0.3)

        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')

        plt.tight_layout()

        # 保存最优路径图
        path_filename = f"results/optimal_paths/{garden_name}_最优路径.png"
        plt.savefig(path_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 最优路径图已保存: {path_filename}")
        return path_filename

    def process_single_garden(self, garden_id):
        """处理单个园林的完整流程"""
        print(f"\n{'='*50}")
        print(f"🏛️  处理园林: {self.gardens[garden_id]} (ID: {garden_id})")
        print(f"{'='*50}")

        start_time = time.time()

        # 第一步：加载数据并生成景观地图
        garden_data = self.load_garden_data(garden_id)
        if not garden_data or not garden_data['elements']:
            print(f"❌ {self.gardens[garden_id]} 数据加载失败")
            return None

        map_filename, landscape_elements = self.generate_garden_landscape_map(garden_data)

        # 第二步：构建路径网络并训练RL
        road_coords = landscape_elements.get('道路', [])
        if not road_coords:
            print(f"⚠️ {self.gardens[garden_id]} 没有道路数据，创建模拟数据")
            # 创建简单的模拟道路
            road_coords = [(i*1000, j*1000) for i in range(10) for j in range(10)]

        building_coords = (landscape_elements.get('实体建筑', []) +
                          landscape_elements.get('半开放建筑', []))

        # 构建路径网络
        path_graph = self.build_path_network(road_coords, building_coords)

        if len(path_graph.nodes()) < 2:
            print(f"⚠️ {self.gardens[garden_id]} 路径网络节点不足")
            return None

        # 寻找起终点
        start_node, end_node = self.find_start_end_nodes(list(path_graph.nodes()))

        if not start_node or not end_node:
            print(f"⚠️ {self.gardens[garden_id]} 无法确定起终点")
            return None

        # 训练RL并找到最优路径
        best_path, best_reward, training_history = self.train_rl_pathfinder(
            path_graph, landscape_elements, start_node, end_node)

        if not best_path:
            print(f"❌ {self.gardens[garden_id]} 未找到有效路径")
            return None

        # 第三步：生成最优路径叠加图
        path_filename = self.visualize_optimal_path_overlay(garden_data, best_path, training_history)

        processing_time = time.time() - start_time

        result = {
            'garden_id': garden_id,
            'garden_name': self.gardens[garden_id],
            'map_filename': map_filename,
            'path_filename': path_filename,
            'best_reward': best_reward,
            'path_length': len(best_path),
            'processing_time': processing_time,
            'training_history': training_history
        }

        print(f"✅ {self.gardens[garden_id]} 处理完成:")
        print(f"   📸 景观地图: {map_filename}")
        print(f"   🎯 最优路径: {path_filename}")
        print(f"   🏆 最佳奖励: {best_reward:.4f}")
        print(f"   ⏱️ 处理时间: {processing_time:.2f} 秒")

        return result

    def batch_process_all_gardens(self):
        """批量处理所有10个园林"""
        print("🚀 开始批量处理所有园林...")
        print("=" * 80)

        start_time = time.time()
        results = []

        for garden_id in range(1, 11):
            try:
                result = self.process_single_garden(garden_id)
                if result:
                    results.append(result)

                    # 保存中间结果
                    with open(f'results/analysis/garden_{garden_id}_result.json', 'w', encoding='utf-8') as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"❌ 处理园林 {garden_id} 时出错: {e}")
                continue

        total_time = time.time() - start_time

        # 生成汇总分析
        self.generate_comprehensive_analysis(results, total_time)

        return results

    def generate_comprehensive_analysis(self, results, total_time):
        """生成综合分析报告"""
        print(f"\n{'='*20} 综合分析报告 {'='*20}")

        if not results:
            print("❌ 没有成功处理的园林数据")
            return

        # 按奖励排序
        sorted_results = sorted(results, key=lambda x: x['best_reward'], reverse=True)

        # 创建对比图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 奖励对比柱状图
        names = [r['garden_name'] for r in sorted_results]
        rewards = [r['best_reward'] for r in sorted_results]

        bars1 = ax1.barh(names, rewards, color='lightcoral', alpha=0.8)
        ax1.set_xlabel('最优路径奖励值')
        ax1.set_title('园林最优路径奖励排名', fontsize=12, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)

        for i, (bar, reward) in enumerate(zip(bars1, rewards)):
            ax1.text(reward + 0.01, i, f'{reward:.3f}', va='center', fontsize=9)

        # 2. 路径长度对比
        path_lengths = [r['path_length'] for r in sorted_results]

        bars2 = ax2.barh(names, path_lengths, color='lightblue', alpha=0.8)
        ax2.set_xlabel('最优路径节点数')
        ax2.set_title('园林最优路径长度对比', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)

        for i, (bar, length) in enumerate(zip(bars2, path_lengths)):
            ax2.text(length + 0.5, i, f'{length}', va='center', fontsize=9)

        # 3. 处理时间对比
        times = [r['processing_time'] for r in sorted_results]

        bars3 = ax3.barh(names, times, color='lightgreen', alpha=0.8)
        ax3.set_xlabel('处理时间 (秒)')
        ax3.set_title('园林处理效率对比', fontsize=12, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        for i, (bar, t) in enumerate(zip(bars3, times)):
            ax3.text(t + 0.1, i, f'{t:.1f}s', va='center', fontsize=9)

        # 4. 奖励与路径长度散点图
        ax4.scatter(path_lengths, rewards, c='purple', alpha=0.6, s=60)
        ax4.set_xlabel('路径长度 (节点数)')
        ax4.set_ylabel('奖励值')
        ax4.set_title('路径长度 vs 奖励关系', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 添加标签
        for i, r in enumerate(sorted_results):
            ax4.annotate(r['garden_name'],
                        (r['path_length'], r['best_reward']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)

        plt.tight_layout()

        # 保存综合分析图
        analysis_filename = "results/analysis/综合分析报告.png"
        plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
        plt.show()

        # 打印统计信息
        print(f"📊 处理统计:")
        print(f"   成功处理园林: {len(results)}/10")
        print(f"   总处理时间: {total_time:.2f} 秒")
        print(f"   平均处理时间: {total_time/len(results):.2f} 秒/园林")

        print(f"\n🏆 排名结果:")
        for i, result in enumerate(sorted_results):
            print(f"   {i+1:2d}. {result['garden_name']:<8} "
                  f"奖励: {result['best_reward']:6.3f} "
                  f"路径: {result['path_length']:3d}节点 "
                  f"时间: {result['processing_time']:5.1f}秒")

        # 保存详细结果
        with open('results/analysis/complete_results.json', 'w', encoding='utf-8') as f:
            json.dump(sorted_results, f, ensure_ascii=False, indent=2)

        print(f"\n💾 文件已保存:")
        print(f"   📈 综合分析图: {analysis_filename}")
        print(f"   📝 详细结果: results/analysis/complete_results.json")

def main():
    """主函数 - 执行完整的园林可视化和RL优化流程"""
    print("🏛️  江南古典园林可视化与强化学习路径优化系统")
    print("📋 执行流程:")
    print("   1️⃣  生成10个园林的景观分布图")
    print("   2️⃣  训练强化学习算法优化路径")
    print("   3️⃣  在景观图上叠加最优路径")
    print("   4️⃣  生成综合分析报告")
    print("=" * 80)

    # 创建系统实例
    system = GardenVisualizationAndRLOptimizer()

    # 执行批量处理
    results = system.batch_process_all_gardens()

    if results:
        print(f"\n🎉 系统运行完成！")
        print(f"✅ 成功处理 {len(results)}/10 个园林")
        print(f"📁 所有结果已保存在 results/ 目录下")
        print(f"🎯 查看 results/analysis/ 获取详细分析")
    else:
        print("❌ 系统运行失败，请检查数据和配置")

if __name__ == "__main__":
    main()
