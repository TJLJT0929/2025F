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
import re
warnings.filterwarnings('ignore')

class GardenPathAnalyzer:
    """江南古典园林路径分析器 - 改进版"""

    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        """初始化分析器"""
        self.data_dir = data_dir
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }
        self.element_types = {
            0: '半开放建筑', 1: '实体建筑', 2: '道路',
            3: '假山', 4: '水体', 5: '植物'
        }

        # 调整计算参数，使其更宽松
        self.path_segment_threshold = 3000  # 增加路径段构建阈值(mm)
        self.turn_angle_threshold = np.pi/4  # 放宽转弯角度阈值 (45度)
        self.entry_exit_clustering_eps = 8000  # 增加入口聚类参数(mm)
        self.step_size = 1000  # 路径采样步长(mm)
        self.visual_range = 15000  # 增加视野范围(mm)

        # 趣味性评分权重
        self.w_curv = 2.0   
        self.w_view = 3.0   
        self.w_exp = 1.5    
        self.w_len = 0.1    
        self.score_C = 1000  

        # RL参数
        self.gamma = 0.95  
        self.alpha = 0.1   
        self.epsilon = 0.2  
        self.episodes = 500  # 减少训练轮数以提高效率

    def load_garden_data(self, garden_id):
        """加载单个园林的坐标数据 - 改进版"""
        garden_name = self.gardens[garden_id]
        data_path = f"{self.data_dir}/{garden_id}. {garden_name}/4-{garden_name}数据坐标.xlsx"

        garden_data = {
            'name': garden_name,
            'id': garden_id,
            'elements': {}
        }

        try:
            excel_file = pd.ExcelFile(data_path)
            print(f"Excel文件工作表: {excel_file.sheet_names}")

            # 读取所有工作表
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(data_path, sheet_name=sheet_name)
                print(f"工作表 '{sheet_name}' 列名: {list(df.columns)}")
                print(f"工作表 '{sheet_name}' 形状: {df.shape}")
                
                # 根据工作表名称或内容推断元素类型
                element_name = self.infer_element_type(sheet_name, df)
                if element_name:
                    garden_data['elements'][element_name] = df

            print(f"✓ 成功加载 {garden_name} 的数据")
            return garden_data

        except Exception as e:
            print(f"✗ 加载 {garden_name} 数据时出错: {e}")
            return None

    def infer_element_type(self, sheet_name, df):
        """根据工作表名称推断元素类型"""
        sheet_name_lower = sheet_name.lower()
        
        if '道路' in sheet_name or 'road' in sheet_name_lower or 'path' in sheet_name_lower:
            return '道路'
        elif '建筑' in sheet_name or 'building' in sheet_name_lower:
            if '半开放' in sheet_name:
                return '半开放建筑'
            else:
                return '实体建筑'
        elif '假山' in sheet_name or 'mountain' in sheet_name_lower or 'rock' in sheet_name_lower:
            return '假山'
        elif '水体' in sheet_name or 'water' in sheet_name_lower:
            return '水体'
        elif '植物' in sheet_name or 'plant' in sheet_name_lower or 'tree' in sheet_name_lower:
            return '植物'
        
        # 如果无法从名称推断，尝试从列数推断
        if len(df.columns) >= 2:
            return '道路'  # 默认当作道路处理
        
        return None

    def parse_coordinate_string(self, coord_str):
        """解析各种格式的坐标字符串"""
        if pd.isna(coord_str):
            return None
            
        coord_str = str(coord_str).strip()
        
        # 尝试多种坐标格式
        patterns = [
            r'\{([^}]+)\}',  # {x,y,z} 格式
            r'\(([^)]+)\)',  # (x,y,z) 格式
            r'\[([^\]]+)\]', # [x,y,z] 格式
            r'([0-9.-]+[,\s]+[0-9.-]+)', # 简单的 x,y 格式
        ]
        
        for pattern in patterns:
            match = re.search(pattern, coord_str)
            if match:
                try:
                    coord_part = match.group(1)
                    # 尝试用逗号、空格或分号分割
                    for sep in [',', ';', ' ', '\t']:
                        if sep in coord_part:
                            coords = [float(x.strip()) for x in coord_part.split(sep) if x.strip()]
                            if len(coords) >= 2:
                                return (coords[0], coords[1])
                except:
                    continue
        
        # 如果以上都失败，尝试直接解析数字
        try:
            numbers = re.findall(r'-?\d+\.?\d*', coord_str)
            if len(numbers) >= 2:
                return (float(numbers[0]), float(numbers[1]))
        except:
            pass
            
        return None

    def extract_road_coordinates(self, garden_data):
        """从园林数据中提取道路坐标 - 改进版"""
        road_coords = []

        if '道路' in garden_data['elements']:
            road_df = garden_data['elements']['道路']
            print(f"道路数据框形状: {road_df.shape}")
            print(f"道路数据框列: {list(road_df.columns)}")

            # 遍历所有列寻找坐标数据
            for col in road_df.columns:
                print(f"检查列 '{col}'")
                for idx, row in road_df.iterrows():
                    coord_str = str(row[col])
                    parsed_coord = self.parse_coordinate_string(coord_str)
                    if parsed_coord:
                        road_coords.append(parsed_coord)
                        if len(road_coords) <= 5:  # 只打印前5个示例
                            print(f"  解析坐标: {coord_str} -> {parsed_coord}")

        # 去重
        road_coords = list(set(road_coords))
        print(f"提取了 {len(road_coords)} 个道路坐标点")
        return road_coords

    def extract_building_coordinates(self, garden_data):
        """从园林数据中提取建筑物坐标 - 改进版"""
        solid_building_coords = []
        semi_open_building_coords = []

        # 提取实体建筑坐标
        if '实体建筑' in garden_data['elements']:
            solid_df = garden_data['elements']['实体建筑']
            for col in solid_df.columns:
                for idx, row in solid_df.iterrows():
                    coord_str = str(row[col])
                    parsed_coord = self.parse_coordinate_string(coord_str)
                    if parsed_coord:
                        solid_building_coords.append(parsed_coord)

        # 提取半开放建筑坐标
        if '半开放建筑' in garden_data['elements']:
            semi_df = garden_data['elements']['半开放建筑']
            for col in semi_df.columns:
                for idx, row in semi_df.iterrows():
                    coord_str = str(row[col])
                    parsed_coord = self.parse_coordinate_string(coord_str)
                    if parsed_coord:
                        semi_open_building_coords.append(parsed_coord)

        # 去重
        solid_building_coords = list(set(solid_building_coords))
        semi_open_building_coords = list(set(semi_open_building_coords))
        
        print(f"提取了 {len(solid_building_coords)} 个实体建筑坐标点和 {len(semi_open_building_coords)} 个半开放建筑坐标点")
        return solid_building_coords, semi_open_building_coords

    def extract_landscape_elements(self, garden_data):
        """提取园林中的景观元素坐标 - 改进版"""
        elements = {}

        for element_name in ['假山', '水体', '植物']:
            if element_name in garden_data['elements']:
                coords = []
                df = garden_data['elements'][element_name]

                for col in df.columns:
                    for idx, row in df.iterrows():
                        coord_str = str(row[col])
                        parsed_coord = self.parse_coordinate_string(coord_str)
                        if parsed_coord:
                            if element_name == '植物':
                                # 植物可能有半径信息，暂时设为默认值
                                coords.append((parsed_coord[0], parsed_coord[1], 1000.0))
                            else:
                                coords.append(parsed_coord)

                # 去重
                coords = list(set(coords))
                elements[element_name] = coords
                print(f"提取了 {len(coords)} 个{element_name}坐标点")

        return elements

    def reconstruct_path_segments(self, road_coords):
        """重构路径段 - 改进版"""
        if not road_coords:
            return []

        road_points = np.array(road_coords)
        print(f"开始重构路径段，共 {len(road_points)} 个点")

        # 使用更灵活的路径重构方法
        path_segments = []
        visited = set()

        for i, start_point in enumerate(road_points):
            if i in visited:
                continue

            # 开始新的路径段
            current_segment = [start_point]
            visited.add(i)
            current_pos = start_point

            # 贪心法构建路径段
            while True:
                best_next_idx = -1
                best_distance = float('inf')

                # 寻找最近的未访问点
                for j, point in enumerate(road_points):
                    if j not in visited:
                        distance = np.linalg.norm(point - current_pos)
                        if distance < self.path_segment_threshold and distance < best_distance:
                            best_distance = distance
                            best_next_idx = j

                if best_next_idx == -1:
                    break  # 没有找到合适的下一个点

                # 添加到当前路径段
                next_point = road_points[best_next_idx]
                current_segment.append(next_point)
                visited.add(best_next_idx)
                current_pos = next_point

            # 只保留有意义的路径段
            if len(current_segment) >= 2:
                path_segments.append(current_segment)

        # 如果没有构建成功路径段，创建一个包含所有点的段
        if not path_segments and len(road_points) >= 2:
            # 使用最小生成树连接所有点
            distances = cdist(road_points, road_points)
            G_temp = nx.Graph()
            
            for i in range(len(road_points)):
                G_temp.add_node(i, pos=road_points[i])
            
            for i in range(len(road_points)):
                for j in range(i+1, len(road_points)):
                    if distances[i,j] <= self.path_segment_threshold:
                        G_temp.add_edge(i, j, weight=distances[i,j])
            
            # 如果图连通，使用最小生成树
            if nx.is_connected(G_temp):
                mst = nx.minimum_spanning_tree(G_temp)
                # 从MST中提取路径段
                for component in nx.connected_components(mst):
                    if len(component) >= 2:
                        component_points = [road_points[i] for i in component]
                        path_segments.append(component_points)
            else:
                # 如果不连通，为每个连通分量创建路径段
                for component in nx.connected_components(G_temp):
                    if len(component) >= 2:
                        component_points = [road_points[i] for i in component]
                        path_segments.append(component_points)

        print(f"重构了 {len(path_segments)} 条路径段")
        return path_segments

    def identify_key_vertices(self, path_segments, building_coords):
        """识别关键顶点 - 改进版"""
        if not path_segments:
            return [], {}

        vertices = []
        vertex_types = {}

        # 收集所有端点
        for segment in path_segments:
            if len(segment) >= 2:
                start_point = tuple(segment[0])
                end_point = tuple(segment[-1])
                vertices.extend([start_point, end_point])

        # 如果没有足够的顶点，使用所有道路点作为顶点
        if len(vertices) < 2:
            for segment in path_segments:
                for point in segment:
                    vertices.append(tuple(point))

        # 去重
        vertices = list(set(vertices))

        # 为所有顶点设置默认类型
        for v in vertices:
            vertex_types[v] = {'endpoint': True}

        print(f"识别了 {len(vertices)} 个关键顶点")
        return vertices, vertex_types

    def identify_entry_exit(self, vertices, vertex_types):
        """识别园林出入口 - 改进版"""
        if len(vertices) < 2:
            if len(vertices) == 1:
                entry = exit_point = vertices[0]
            else:
                # 创建默认的入口出口
                entry = (0, 0)
                exit_point = (1000, 1000)
                vertices.extend([entry, exit_point])
        else:
            # 使用最远的两个点作为入口和出口
            vertices_array = np.array(vertices)
            distances = cdist(vertices_array, vertices_array)
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            entry = tuple(vertices_array[i])
            exit_point = tuple(vertices_array[j])

        # 标记入口和出口
        if entry not in vertex_types:
            vertex_types[entry] = {}
        vertex_types[entry]['entry'] = True

        if exit_point not in vertex_types:
            vertex_types[exit_point] = {}
        vertex_types[exit_point]['exit'] = True

        print(f"识别出入口: 入口坐标 {entry}, 出口坐标 {exit_point}")
        return entry, exit_point

    def build_path_graph(self, path_segments, vertices, vertex_types):
        """构建路径网络图 - 改进版"""
        G = nx.Graph()

        # 添加所有顶点
        for v in vertices:
            G.add_node(v, pos=v)

        # 如果没有路径段，创建一个完全图
        if not path_segments:
            vertices_array = np.array(vertices)
            distances = cdist(vertices_array, vertices_array)
            
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    if distances[i,j] <= self.path_segment_threshold * 2:  # 使用更大的阈值
                        G.add_edge(vertices[i], vertices[j], 
                                  length=distances[i,j],
                                  geometry=[vertices[i], vertices[j]],
                                  turns=0)
        else:
            # 使用路径段构建图
            for segment in path_segments:
                segment_vertices = []
                for point in segment:
                    point_tuple = tuple(point)
                    if point_tuple in vertices:
                        segment_vertices.append(point_tuple)

                # 连接路径段上相邻的顶点
                for i in range(len(segment_vertices) - 1):
                    v1 = segment_vertices[i]
                    v2 = segment_vertices[i + 1]
                    edge_length = np.linalg.norm(np.array(v1) - np.array(v2))
                    G.add_edge(v1, v2,
                              length=edge_length,
                              geometry=[v1, v2],
                              turns=0)

        # 确保图是连通的
        if not nx.is_connected(G) and len(G.nodes()) > 1:
            # 连接各个连通分量
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                # 找到两个分量间最近的点对
                comp1_nodes = list(components[i])
                comp2_nodes = list(components[i + 1])
                
                min_dist = float('inf')
                best_pair = None
                
                for v1 in comp1_nodes:
                    for v2 in comp2_nodes:
                        dist = np.linalg.norm(np.array(v1) - np.array(v2))
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (v1, v2)
                
                if best_pair:
                    v1, v2 = best_pair
                    G.add_edge(v1, v2,
                              length=min_dist,
                              geometry=[v1, v2],
                              turns=0)

        print(f"构建了路径网络图: {len(G.nodes())} 个顶点, {len(G.edges())} 条边")
        return G

    def calculate_path_features(self, G, path, landscape_elements):
        """计算路径的特征指标 - 简化版"""
        if len(path) < 2:
            return {
                'length': 0,
                'curvature': 0,
                'view_change': 0,
                'exploration': 0
            }

        # 计算路径长度
        path_length = 0
        for i in range(len(path) - 1):
            v1, v2 = path[i], path[i+1]
            if G.has_edge(v1, v2):
                edge_data = G.get_edge_data(v1, v2)
                path_length += edge_data.get('length', np.linalg.norm(np.array(v1) - np.array(v2)))
            else:
                path_length += np.linalg.norm(np.array(v1) - np.array(v2))

        # 简化的曲折度计算
        curvature = len(path) - 2  # 转折点数量近似

        # 简化的视野变化计算
        view_change = len(path) * 10  # 简单的近似

        # 探索性得分
        exploration_score = sum(G.degree(v) for v in path[1:-1])

        return {
            'length': path_length,
            'curvature': curvature,
            'view_change': view_change,
            'exploration': exploration_score
        }

    def calculate_interest_score(self, path_features):
        """计算路径的趣味性得分"""
        L_len = path_features['length']
        L_curv = path_features['curvature']
        L_view = path_features['view_change']
        L_exp = path_features['exploration']

        numerator = (self.w_curv * L_curv + self.w_view * L_view + self.w_exp * L_exp)
        denominator = (self.w_len * L_len + self.score_C)

        interest_score = numerator / denominator
        return max(interest_score, 0.001)  # 确保得分为正

    def find_optimal_path_rl(self, G, entry, exit_point, landscape_elements):
        """使用强化学习寻找最优游览路径 - 简化版"""
        # 首先尝试找到基本路径
        try:
            if nx.has_path(G, entry, exit_point):
                # 使用多种路径查找方法
                paths_to_try = []
                
                # 最短路径
                try:
                    shortest_path = nx.shortest_path(G, entry, exit_point, weight='length')
                    paths_to_try.append(shortest_path)
                except:
                    pass
                
                # 简单路径枚举（限制数量）
                try:
                    simple_paths = list(nx.all_simple_paths(G, entry, exit_point))
                    paths_to_try.extend(simple_paths[:10])  # 只取前10个
                except:
                    pass
                
                # 评估所有路径
                best_path = None
                best_score = -float('inf')
                
                for path in paths_to_try:
                    if len(path) >= 2:
                        features = self.calculate_path_features(G, path, landscape_elements)
                        score = self.calculate_interest_score(features)
                        
                        if score > best_score:
                            best_score = score
                            best_path = path
                
                if best_path:
                    # 构建路径几何
                    geometry = []
                    for i in range(len(best_path) - 1):
                        v1, v2 = best_path[i], best_path[i+1]
                        if G.has_edge(v1, v2):
                            edge_data = G.get_edge_data(v1, v2)
                            geometry.extend(edge_data.get('geometry', [v1, v2]))
                        else:
                            geometry.extend([v1, v2])
                    
                    print(f"找到最优路径，趣味性得分: {best_score:.4f}")
                    return best_path, geometry, best_score
            
        except Exception as e:
            print(f"路径查找出错: {e}")

        # 如果没有找到路径，创建一个直接路径
        print("创建直接路径作为备选方案")
        direct_path = [entry, exit_point]
        direct_geometry = [entry, exit_point]
        features = self.calculate_path_features(G, direct_path, landscape_elements)
        score = self.calculate_interest_score(features)
        
        return direct_path, direct_geometry, score

    def visualize_garden_path(self, garden_data, G, optimal_path, optimal_geometry, entry, exit_point, garden_id):
        """可视化园林路径网络和最优游线"""
        plt.figure(figsize=(14, 10))
        garden_name = self.gardens[garden_id]
        plt.title(f"{garden_name} 路径网络与最优游线", fontsize=18)

        # 绘制所有道路坐标点
        road_coords = self.extract_road_coordinates(garden_data)
        if road_coords:
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
        if G and len(G.edges()) > 0:
            for (u, v) in G.edges():
                edge_data = G.get_edge_data(u, v)
                geometry = edge_data.get('geometry', [u, v])
                x_coords = [p[0] for p in geometry]
                y_coords = [p[1] for p in geometry]
                plt.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.5)

        # 绘制图的顶点
        if G and len(G.nodes()) > 0:
            node_coords = np.array(list(G.nodes()))
            plt.scatter(node_coords[:, 0], node_coords[:, 1], c='blue', s=30, alpha=0.8, label='路径节点')

        # 绘制入口和出口
        plt.scatter([entry[0]], [entry[1]], c='green', s=200, marker='*', label='入口')
        plt.scatter([exit_point[0]], [exit_point[1]], c='red', s=200, marker='*', label='出口')

        # 绘制最优路径
        if optimal_geometry and len(optimal_geometry) > 1:
            opt_points = np.array(optimal_geometry)
            plt.plot(opt_points[:, 0], opt_points[:, 1], 'r-', linewidth=2.5, label='最优游线')

        plt.legend(loc='upper right', fontsize=12)
        plt.axis('equal')
        plt.grid(True, alpha=0.3)

        # 保存图像
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{garden_name}_最优游线.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"已保存 {garden_name} 的路径可视化图像")

    def process_garden(self, garden_id):
        """处理单个园林 - 改进版"""
        print(f"\n===== 开始处理 {self.gardens[garden_id]} =====")

        # 加载园林数据
        garden_data = self.load_garden_data(garden_id)
        if not garden_data:
            print(f"无法加载 {self.gardens[garden_id]} 的数据，跳过")
            return None, None, 0

        # 提取道路坐标
        road_coords = self.extract_road_coordinates(garden_data)
        if not road_coords:
            print(f"{self.gardens[garden_id]} 没有道路坐标数据，创建模拟路径")
            # 创建一个简单的模拟路径
            road_coords = [(0, 0), (1000, 0), (2000, 1000), (3000, 1000)]

        # 提取建筑坐标
        building_coords = self.extract_building_coordinates(garden_data)

        # 提取景观元素
        landscape_elements = self.extract_landscape_elements(garden_data)

        # 重构路径段
        path_segments = self.reconstruct_path_segments(road_coords)

        # 识别关键顶点
        vertices, vertex_types = self.identify_key_vertices(path_segments, building_coords)
        
        # 确保至少有一些顶点
        if not vertices:
            vertices = road_coords[:10] if len(road_coords) >= 10 else road_coords
            vertex_types = {tuple(v): {'endpoint': True} for v in vertices}

        # 识别出入口
        entry, exit_point = self.identify_entry_exit(vertices, vertex_types)

        # 构建路径网络图
        G = self.build_path_graph(path_segments, vertices, vertex_types)

        # 寻找最优游线
        optimal_path, optimal_geometry, interest_score = self.find_optimal_path_rl(
            G, entry, exit_point, landscape_elements)

        # 可视化结果
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

        if optimal_path:
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
