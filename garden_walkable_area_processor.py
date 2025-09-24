import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import json
import warnings
from scipy.spatial import ConvexHull, Voronoi
from scipy.spatial.distance import cdist
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
import cv2
from collections import defaultdict

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class GardenWalkableAreaProcessor:
    """
    园林可行区域处理器
    
    功能：
    1. 识别不可通行区域（实体建筑、假山、水体）
    2. 识别可通行区域（半开放建筑、道路）
    3. 构建可行区域网格图
    4. 生成路径规划基础数据
    """
    
    def __init__(self, data_dir="results/garden_data"):
        self.data_dir = data_dir
        self.output_dir = "results/walkable_areas"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 元素类型配置 - 优化后的分类
        self.element_types = {
            'solid_non_walkable': ['实体建筑'],           # 实体不可通行（需要检测间隙门洞）
            'strict_non_walkable': ['水体'],              # 严格不可通行（精确边界）
            'mountain_non_walkable': ['假山'],            # 山体不可通行（除非有道路）
            'traversable': ['半开放建筑'],               # 可穿越建筑
            'roads': ['道路'],                          # 道路网络
            'neutral': ['植物']                        # 中性（可能影响视线但可通行）
        }
        
        # 优化处理参数
        self.processing_params = {
            'solid_building_buffer': 1200,       # 实体建筑缓冲区（减少以保留间隙）
            'water_buffer': 800,                 # 水体缓冲区（精确边界）
            'mountain_buffer': 1500,             # 山体缓冲区
            'semi_open_buffer': 500,             # 半开放建筑缓冲区（小缓冲允许穿越）
            'road_buffer': 2500,                 # 道路缓冲区（保证连通性）
            'gap_detection_threshold': 3000,     # 间隙检测阈值
            'doorway_min_width': 2000,           # 门洞最小宽度
            'mountain_road_buffer': 3000         # 山体内道路缓冲区
        }
        
        # 可视化配置
        self.colors = {
            '实体建筑': '#8B4513',    # 棕色
            '假山': '#696969',        # 灰色
            '水体': '#4169E1',        # 蓝色
            '半开放建筑': '#FFA500',  # 橙色
            '道路': '#FFD700',        # 金色
            '植物': '#228B22',        # 绿色
            'walkable_area': '#90EE90',     # 浅绿色 - 可行区域
            'non_walkable_area': '#FFB6C1', # 浅粉色 - 不可行区域
            'boundary': '#FF0000'           # 红色 - 边界
        }
        
    def load_garden_data(self, garden_name):
        """加载园林数据"""
        data_path = f"{self.data_dir}/{garden_name}_数据.json"
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                garden_data = json.load(f)
            return garden_data
        except Exception as e:
            print(f"❌ 加载 {garden_name} 数据失败: {e}")
            return None
    
    def create_polygon_from_points(self, points, buffer_distance=1000):
        """从点集创建多边形区域"""
        if len(points) < 3:
            return None
        
        points_array = np.array(points)
        
        try:
            # 使用DBSCAN聚类处理密集点
            clustering = DBSCAN(eps=buffer_distance*2, min_samples=3).fit(points_array)
            
            polygons = []
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:  # 噪声点
                    continue
                
                cluster_points = points_array[clustering.labels_ == cluster_id]
                
                if len(cluster_points) >= 3:
                    try:
                        # 创建凸包
                        hull = ConvexHull(cluster_points)
                        hull_points = cluster_points[hull.vertices]
                        
                        # 创建多边形并添加缓冲区
                        polygon = Polygon(hull_points).buffer(buffer_distance)
                        if polygon.is_valid and not polygon.is_empty:
                            polygons.append(polygon)
                    except Exception:
                        continue
            
            # 合并所有多边形
            if polygons:
                return unary_union(polygons)
            else:
                # 如果聚类失败，使用所有点创建一个大的凸包
                hull = ConvexHull(points_array)
                hull_points = points_array[hull.vertices]
                return Polygon(hull_points).buffer(buffer_distance)
                
        except Exception:
            # 备选方案：直接使用凸包
            try:
                hull = ConvexHull(points_array)
                hull_points = points_array[hull.vertices]
                return Polygon(hull_points).buffer(buffer_distance)
            except:
                return None
    
    def create_walkable_areas_from_roads(self, road_points, buffer_distance=2000):
        """从道路点创建可行走区域 - 优化版本"""
        if not road_points:
            return None
        
        try:
            road_points_array = np.array(road_points)
            
            # 为每个道路点创建缓冲区，然后合并
            road_polygons = []
            
            for point in road_points_array:
                point_geom = Point(point).buffer(buffer_distance)
                road_polygons.append(point_geom)
            
            # 合并所有道路多边形
            if road_polygons:
                return unary_union(road_polygons)
            
        except Exception as e:
            print(f"创建道路区域失败: {e}")
            
        return None
    
    def detect_building_gaps(self, building_coords, gap_threshold=3000):
        """检测建筑间的间隙门洞"""
        if len(building_coords) < 4:  # 至少需要两个建筑物
            return []
        
        building_array = np.array(building_coords)
        gaps = []
        
        try:
            # 使用较大的聚类半径来分离不同的建筑
            # 调整聚类参数，使用更合理的距离
            clustering = DBSCAN(eps=gap_threshold*0.8, min_samples=2).fit(building_array)
            
            # 识别不同建筑群
            unique_labels = set(clustering.labels_)
            if -1 in unique_labels:
                unique_labels.remove(-1)  # 移除噪声标签
            
            print(f"   DBSCAN聚类结果: {len(unique_labels)} 个建筑群")
            
            if len(unique_labels) < 2:
                return gaps  # 需要至少两个建筑群才能有间隙
            
            building_clusters = []
            for label in unique_labels:
                cluster_points = building_array[clustering.labels_ == label]
                building_clusters.append(cluster_points)
                print(f"   建筑群 {label}: {len(cluster_points)} 个点")
            
            # 计算簇之间的间隙
            for i in range(len(building_clusters)):
                for j in range(i + 1, len(building_clusters)):
                    cluster1 = building_clusters[i]
                    cluster2 = building_clusters[j]
                    
                    # 找到两个簇的中心点
                    center1 = np.mean(cluster1, axis=0)
                    center2 = np.mean(cluster2, axis=0)
                    center_distance = np.linalg.norm(center1 - center2)
                    
                    # 计算簇之间的最近距离
                    min_dist = float('inf')
                    closest_points = None
                    
                    for p1 in cluster1:
                        for p2 in cluster2:
                            dist = np.linalg.norm(p1 - p2)
                            if dist < min_dist:
                                min_dist = dist
                                closest_points = (p1, p2)
                    
                    print(f"   簇 {i} 和簇 {j} 之间: 中心距离 {center_distance:.0f}mm, 最近距离 {min_dist:.0f}mm")
                    
                    # 如果距离在合理范围内，认为是门洞
                    if self.processing_params['doorway_min_width'] <= min_dist <= gap_threshold:
                        gap_center = ((closest_points[0] + closest_points[1]) / 2).tolist()
                        gaps.append({
                            'center': gap_center,
                            'width': min_dist,
                            'points': closest_points,
                            'clusters': (center1.tolist(), center2.tolist())
                        })
                        print(f"   ✅ 发现间隙: 宽度 {min_dist:.0f}mm")
            
            return gaps
            
        except Exception as e:
            print(f"间隙检测失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_mountain_walkable_areas(self, mountain_coords, road_coords, mountain_buffer=1500, road_buffer=3000):
        """创建山体区域的可行区域（仅限道路）"""
        if not mountain_coords:
            return None, None
        
        try:
            # 创建山体不可通行区域
            mountain_polygon = self.create_polygon_from_points(mountain_coords, buffer_distance=mountain_buffer)
            
            mountain_walkable_areas = []
            
            if road_coords and mountain_polygon:
                # 找到山体内的道路点
                road_array = np.array(road_coords)
                mountain_roads = []
                
                for road_point in road_array:
                    point_geom = Point(road_point)
                    if mountain_polygon.contains(point_geom) or mountain_polygon.intersects(point_geom.buffer(road_buffer)):
                        mountain_roads.append(road_point)
                
                # 为山体内的道路创建通行区域
                if mountain_roads:
                    road_walkable = self.create_walkable_areas_from_roads(mountain_roads, buffer_distance=road_buffer)
                    if road_walkable:
                        mountain_walkable_areas.append(road_walkable)
            
            mountain_walkable_union = unary_union(mountain_walkable_areas) if mountain_walkable_areas else None
            
            return mountain_polygon, mountain_walkable_union
            
        except Exception as e:
            print(f"山体区域处理失败: {e}")
            return None, None
    
    def create_optimized_building_areas(self, building_coords, buffer_distance=1200):
        """创建优化的建筑区域（保留间隙）"""
        if len(building_coords) < 3:
            return None
        
        try:
            # 检测建筑间隙
            gaps = self.detect_building_gaps(building_coords, self.processing_params['gap_detection_threshold'])
            
            # 创建建筑多边形，但使用较小的缓冲区以保留间隙
            building_polygon = self.create_polygon_from_points(building_coords, buffer_distance=buffer_distance)
            
            # 从建筑区域中挖除间隙门洞
            if building_polygon and gaps:
                for gap in gaps:
                    # 为每个间隙创建一个通道
                    gap_buffer = max(gap['width'] * 0.6, self.processing_params['doorway_min_width'])
                    gap_area = Point(gap['center']).buffer(gap_buffer)
                    building_polygon = building_polygon.difference(gap_area)
            
            return building_polygon
            
        except Exception as e:
            print(f"优化建筑区域创建失败: {e}")
            return self.create_polygon_from_points(building_coords, buffer_distance=buffer_distance)
    
    def process_garden_walkable_areas(self, garden_data):
        """处理园林可行区域 - 优化版本"""
        garden_name = garden_data['name']
        print(f"🏗️ 处理 {garden_name} 可行区域（优化版本）...")
        
        boundaries = garden_data['boundaries']
        elements = garden_data['elements']
        
        # 分类处理不同类型的元素
        solid_non_walkable_polygons = []    # 实体建筑（需要间隙检测）
        strict_non_walkable_polygons = []   # 严格不可通行（水体）
        mountain_areas = []                 # 山体区域  
        traversable_areas = []              # 可穿越区域（半开放建筑）
        road_areas = []                     # 道路网络
        
        road_coords = elements.get('道路', [])  # 道路坐标用于山体处理
        
        for element_type, coords in elements.items():
            if not coords:
                continue
            
            if element_type in self.element_types['solid_non_walkable']:
                # 实体建筑 - 使用优化的间隙检测
                print(f"  🏠 处理实体建筑 ({len(coords)} 个点)...")
                polygon = self.create_optimized_building_areas(
                    coords, 
                    buffer_distance=self.processing_params['solid_building_buffer']
                )
                if polygon:
                    solid_non_walkable_polygons.append(polygon)
                    
            elif element_type in self.element_types['strict_non_walkable']:
                # 水体 - 严格边界处理
                print(f"  💧 处理水体区域 ({len(coords)} 个点)...")
                polygon = self.create_polygon_from_points(
                    coords, 
                    buffer_distance=self.processing_params['water_buffer']
                )
                if polygon:
                    strict_non_walkable_polygons.append(polygon)
                    
            elif element_type in self.element_types['mountain_non_walkable']:
                # 山体 - 特殊处理（只有道路可通行）
                print(f"  ⛰️  处理山体区域 ({len(coords)} 个点)...")
                mountain_polygon, mountain_walkable = self.create_mountain_walkable_areas(
                    coords, road_coords,
                    mountain_buffer=self.processing_params['mountain_buffer'],
                    road_buffer=self.processing_params['mountain_road_buffer']
                )
                if mountain_polygon:
                    mountain_areas.append({
                        'non_walkable': mountain_polygon,
                        'walkable': mountain_walkable
                    })
                    
            elif element_type in self.element_types['traversable']:
                # 半开放建筑 - 可穿越，使用小缓冲区
                print(f"  🏛️ 处理半开放建筑 ({len(coords)} 个点)...")
                polygon = self.create_polygon_from_points(
                    coords, 
                    buffer_distance=self.processing_params['semi_open_buffer']
                )
                if polygon:
                    traversable_areas.append(polygon)
                    
            elif element_type in self.element_types['roads']:
                # 道路 - 确保连通性
                print(f"  🛤️  处理道路网络 ({len(coords)} 个点)...")
                polygon = self.create_walkable_areas_from_roads(
                    coords, 
                    buffer_distance=self.processing_params['road_buffer']
                )
                if polygon:
                    road_areas.append(polygon)
        
        # 合并各类区域
        solid_non_walkable_area = unary_union(solid_non_walkable_polygons) if solid_non_walkable_polygons else None
        strict_non_walkable_area = unary_union(strict_non_walkable_polygons) if strict_non_walkable_polygons else None
        traversable_area = unary_union(traversable_areas) if traversable_areas else None
        road_area = unary_union(road_areas) if road_areas else None
        
        # 处理山体区域
        mountain_non_walkable_areas = []
        mountain_walkable_areas = []
        
        for mountain_data in mountain_areas:
            if mountain_data['non_walkable']:
                mountain_non_walkable_areas.append(mountain_data['non_walkable'])
            if mountain_data['walkable']:
                mountain_walkable_areas.append(mountain_data['walkable'])
        
        mountain_non_walkable_area = unary_union(mountain_non_walkable_areas) if mountain_non_walkable_areas else None
        mountain_walkable_area = unary_union(mountain_walkable_areas) if mountain_walkable_areas else None
        
        # 创建园林边界
        garden_boundary = Polygon([
            (boundaries['min_x'], boundaries['min_y']),
            (boundaries['max_x'], boundaries['min_y']),
            (boundaries['max_x'], boundaries['max_y']),
            (boundaries['min_x'], boundaries['max_y'])
        ])
        
        # 计算最终可行区域
        print("  🧮 计算最终可行区域...")
        
        # 从园林边界开始
        final_walkable_area = garden_boundary
        
        # 减去所有不可通行区域
        non_walkable_areas = []
        
        if solid_non_walkable_area:
            non_walkable_areas.append(solid_non_walkable_area)
        if strict_non_walkable_area:
            non_walkable_areas.append(strict_non_walkable_area)
        if mountain_non_walkable_area:
            non_walkable_areas.append(mountain_non_walkable_area)
        
        # 合并所有不可通行区域
        total_non_walkable_area = unary_union(non_walkable_areas) if non_walkable_areas else None
        
        if total_non_walkable_area:
            final_walkable_area = final_walkable_area.difference(total_non_walkable_area)
        
        # 添加明确的可通行区域
        explicit_walkable_areas = []
        
        if traversable_area:
            explicit_walkable_areas.append(traversable_area)
        if road_area:
            explicit_walkable_areas.append(road_area)
        if mountain_walkable_area:
            explicit_walkable_areas.append(mountain_walkable_area)
        
        explicit_walkable_union = unary_union(explicit_walkable_areas) if explicit_walkable_areas else None
        
        if explicit_walkable_union:
            final_walkable_area = final_walkable_area.union(explicit_walkable_union)
        
        result = {
            'garden_name': garden_name,
            'boundaries': boundaries,
            'solid_non_walkable_area': solid_non_walkable_area,
            'strict_non_walkable_area': strict_non_walkable_area,
            'mountain_non_walkable_area': mountain_non_walkable_area,
            'mountain_walkable_area': mountain_walkable_area,
            'traversable_area': traversable_area,
            'road_area': road_area,
            'total_non_walkable_area': total_non_walkable_area,
            'explicit_walkable_area': explicit_walkable_union,
            'final_walkable_area': final_walkable_area,
            'garden_boundary': garden_boundary,
            'elements': elements
        }
        
        print(f"  ✅ {garden_name} 可行区域处理完成")
        return result
    
    def plot_polygon(self, ax, polygon, color, alpha=0.6, label=None):
        """绘制多边形"""
        if polygon is None or polygon.is_empty:
            return
        
        if hasattr(polygon, 'geoms'):  # MultiPolygon
            for geom in polygon.geoms:
                self.plot_single_polygon(ax, geom, color, alpha, label)
                label = None  # 只显示一次标签
        else:  # Single Polygon
            self.plot_single_polygon(ax, polygon, color, alpha, label)
    
    def plot_single_polygon(self, ax, polygon, color, alpha, label):
        """绘制单个多边形"""
        if polygon.is_empty:
            return
        
        x, y = polygon.exterior.xy
        ax.fill(x, y, color=color, alpha=alpha, label=label)
        ax.plot(x, y, color=color, alpha=0.8, linewidth=1)
        
        # 绘制内部孔洞
        for interior in polygon.interiors:
            x, y = interior.xy
            ax.fill(x, y, color='white', alpha=1.0)
            ax.plot(x, y, color=color, alpha=0.8, linewidth=1)
    
    def visualize_walkable_areas(self, result):
        """可视化可行区域 - 优化版本"""
        garden_name = result['garden_name']
        
        fig = plt.figure(figsize=(24, 18))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)
        
        fig.suptitle(f"{garden_name} - 优化可行区域分析", fontsize=16, fontweight='bold')
        
        # 1. 原始元素分布图 (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("原始景观元素分布", fontsize=14)
        for element_type, coords in result['elements'].items():
            if coords:
                coords_array = np.array(coords)
                color = self.colors.get(element_type, '#000000')
                ax1.scatter(coords_array[:, 0], coords_array[:, 1], 
                           c=color, alpha=0.7, s=20, label=f"{element_type} ({len(coords)})")
        
        ax1.set_xlabel('X坐标 (mm)')
        ax1.set_ylabel('Y坐标 (mm)')
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. 实体建筑处理 (中上)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("实体建筑间隙处理", fontsize=14)
        
        self.plot_polygon(ax2, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        self.plot_polygon(ax2, result['solid_non_walkable_area'], '#FFB6C1', 
                         alpha=0.7, label='实体建筑（保留间隙）')
        
        # 标注实体建筑点
        building_coords = result['elements'].get('实体建筑', [])
        if building_coords:
            building_array = np.array(building_coords)
            ax2.scatter(building_array[:, 0], building_array[:, 1], 
                       c='#8B4513', alpha=0.8, s=15, label='建筑点')
        
        ax2.set_xlabel('X坐标 (mm)')
        ax2.set_ylabel('Y坐标 (mm)')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 3. 水体严格处理 (右上)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title("水体严格边界", fontsize=14)
        
        self.plot_polygon(ax3, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        self.plot_polygon(ax3, result['strict_non_walkable_area'], '#4169E1', 
                         alpha=0.8, label='水体严格边界')
        
        water_coords = result['elements'].get('水体', [])
        if water_coords:
            water_array = np.array(water_coords)
            ax3.scatter(water_array[:, 0], water_array[:, 1], 
                       c='#4169E1', alpha=0.8, s=15, label='水体点')
        
        ax3.set_xlabel('X坐标 (mm)')
        ax3.set_ylabel('Y坐标 (mm)')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # 4. 山体道路处理 (左中)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title("山体区域道路通行", fontsize=14)
        
        self.plot_polygon(ax4, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        self.plot_polygon(ax4, result['mountain_non_walkable_area'], '#696969', 
                         alpha=0.6, label='山体不可通行')
        self.plot_polygon(ax4, result['mountain_walkable_area'], '#90EE90', 
                         alpha=0.8, label='山体内道路')
        
        mountain_coords = result['elements'].get('假山', [])
        road_coords = result['elements'].get('道路', [])
        if mountain_coords:
            mountain_array = np.array(mountain_coords)
            ax4.scatter(mountain_array[:, 0], mountain_array[:, 1], 
                       c='#696969', alpha=0.8, s=10, label='山体点')
        if road_coords:
            road_array = np.array(road_coords)
            ax4.scatter(road_array[:, 0], road_array[:, 1], 
                       c='#FFD700', alpha=0.8, s=8, label='道路点')
        
        ax4.set_xlabel('X坐标 (mm)')
        ax4.set_ylabel('Y坐标 (mm)')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        # 5. 半开放建筑可穿越 (中中)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title("半开放建筑可穿越", fontsize=14)
        
        self.plot_polygon(ax5, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        self.plot_polygon(ax5, result['traversable_area'], '#FFA500', 
                         alpha=0.6, label='半开放建筑（可穿越）')
        
        semi_coords = result['elements'].get('半开放建筑', [])
        if semi_coords:
            semi_array = np.array(semi_coords)
            ax5.scatter(semi_array[:, 0], semi_array[:, 1], 
                       c='#FFA500', alpha=0.8, s=15, label='半开放建筑点')
        
        ax5.set_xlabel('X坐标 (mm)')
        ax5.set_ylabel('Y坐标 (mm)')  
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # 6. 道路网络连通性 (右中)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title("道路网络连通性", fontsize=14)
        
        self.plot_polygon(ax6, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        self.plot_polygon(ax6, result['road_area'], '#FFD700', 
                         alpha=0.7, label='道路网络')
        
        if road_coords:
            road_array = np.array(road_coords)
            ax6.scatter(road_array[:, 0], road_array[:, 1], 
                       c='#FFD700', alpha=0.8, s=10, label='道路点')
        
        ax6.set_xlabel('X坐标 (mm)')
        ax6.set_ylabel('Y坐标 (mm)')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        ax6.set_aspect('equal')
        
        # 7. 总体不可通行区域 (左下)
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.set_title("总体不可通行区域", fontsize=14)
        
        self.plot_polygon(ax7, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        self.plot_polygon(ax7, result['total_non_walkable_area'], '#FFB6C1', 
                         alpha=0.7, label='总体不可通行')
        
        ax7.set_xlabel('X坐标 (mm)')
        ax7.set_ylabel('Y坐标 (mm)')
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
        ax7.set_aspect('equal')
        
        # 8. 明确可通行区域 (中下)
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.set_title("明确可通行区域", fontsize=14)
        
        self.plot_polygon(ax8, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        self.plot_polygon(ax8, result['explicit_walkable_area'], '#90EE90', 
                         alpha=0.7, label='明确可通行')
        
        ax8.set_xlabel('X坐标 (mm)')
        ax8.set_ylabel('Y坐标 (mm)')
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
        ax8.set_aspect('equal')
        
        # 9. 最终可行区域（路径规划基础） (右下)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.set_title("最终可行区域（路径规划基础）", fontsize=14, fontweight='bold')
        
        # 绘制园林边界
        self.plot_polygon(ax9, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        
        # 绘制不可通行区域
        self.plot_polygon(ax9, result['total_non_walkable_area'], '#FFB6C1', 
                         alpha=0.5, label='不可通行区域')
        
        # 绘制最终可行区域
        self.plot_polygon(ax9, result['final_walkable_area'], '#90EE90', 
                         alpha=0.6, label='最终可行区域')
        
        # 添加植物位置（影响视线但不影响通行）
        plant_coords = result['elements'].get('植物', [])
        if plant_coords:
            plant_array = np.array(plant_coords)
            ax9.scatter(plant_array[:, 0], plant_array[:, 1], 
                       c=self.colors['植物'], alpha=0.6, s=8, label='植物')
        
        ax9.set_xlabel('X坐标 (mm)')
        ax9.set_ylabel('Y坐标 (mm)')
        ax9.legend(fontsize=10)
        ax9.grid(True, alpha=0.3)
        ax9.set_aspect('equal')
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"{self.output_dir}/{garden_name}_优化可行区域分析.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 优化可行区域分析图已保存: {filename}")
        return filename
    
    def calculate_walkable_statistics(self, result):
        """计算可行区域统计信息 - 优化版本"""
        boundaries = result['boundaries']
        total_area = boundaries['width'] * boundaries['height']
        
        # 计算各区域面积
        garden_boundary_area = result['garden_boundary'].area if result['garden_boundary'] else 0
        
        # 分类统计不可通行区域
        solid_non_walkable_area = result['solid_non_walkable_area'].area if result['solid_non_walkable_area'] else 0
        strict_non_walkable_area = result['strict_non_walkable_area'].area if result['strict_non_walkable_area'] else 0
        mountain_non_walkable_area = result['mountain_non_walkable_area'].area if result['mountain_non_walkable_area'] else 0
        total_non_walkable_area = result['total_non_walkable_area'].area if result['total_non_walkable_area'] else 0
        
        # 分类统计可通行区域
        traversable_area = result['traversable_area'].area if result['traversable_area'] else 0
        road_area = result['road_area'].area if result['road_area'] else 0
        mountain_walkable_area = result['mountain_walkable_area'].area if result['mountain_walkable_area'] else 0
        explicit_walkable_area = result['explicit_walkable_area'].area if result['explicit_walkable_area'] else 0
        
        final_walkable_area = result['final_walkable_area'].area if result['final_walkable_area'] else 0
        
        statistics = {
            'garden_name': result['garden_name'],
            'total_bounding_area': total_area,
            'garden_boundary_area': garden_boundary_area,
            
            # 分类不可通行区域统计
            'solid_non_walkable_area': solid_non_walkable_area,
            'strict_non_walkable_area': strict_non_walkable_area, 
            'mountain_non_walkable_area': mountain_non_walkable_area,
            'total_non_walkable_area': total_non_walkable_area,
            
            # 分类可通行区域统计
            'traversable_area': traversable_area,
            'road_area': road_area,
            'mountain_walkable_area': mountain_walkable_area,
            'explicit_walkable_area': explicit_walkable_area,
            'final_walkable_area': final_walkable_area,
            
            # 比例统计
            'walkable_ratio': final_walkable_area / garden_boundary_area if garden_boundary_area > 0 else 0,
            'non_walkable_ratio': total_non_walkable_area / garden_boundary_area if garden_boundary_area > 0 else 0,
            'solid_building_ratio': solid_non_walkable_area / garden_boundary_area if garden_boundary_area > 0 else 0,
            'water_ratio': strict_non_walkable_area / garden_boundary_area if garden_boundary_area > 0 else 0,
            'mountain_ratio': mountain_non_walkable_area / garden_boundary_area if garden_boundary_area > 0 else 0,
            'road_ratio': road_area / garden_boundary_area if garden_boundary_area > 0 else 0,
            
            # 元素数量统计
            'elements_count': {k: len(v) for k, v in result['elements'].items()}
        }
        
        return statistics
    
    def save_walkable_data(self, result, statistics):
        """保存可行区域数据"""
        garden_name = result['garden_name']
        
        # 准备可序列化的数据
        walkable_data = {
            'garden_name': garden_name,
            'boundaries': result['boundaries'],
            'statistics': statistics,
            'elements_count': {k: len(v) for k, v in result['elements'].items()},
            # 注意：Shapely几何对象不能直接序列化，这里只保存统计信息
            'areas': {
                'total_area': statistics['total_bounding_area'],
                'walkable_area': statistics['final_walkable_area'],
                'non_walkable_area': statistics['non_walkable_area']
            }
        }
        
        # 保存数据
        filename = f"{self.output_dir}/{garden_name}_可行区域数据.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(walkable_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 可行区域数据已保存: {filename}")
        return filename
    
    def process_single_garden(self, garden_name):
        """处理单个园林的可行区域"""
        print(f"\n{'='*60}")
        print(f"🏛️ 处理园林可行区域: {garden_name}")
        print(f"{'='*60}")
        
        # 加载园林数据
        garden_data = self.load_garden_data(garden_name)
        if not garden_data:
            return None
        
        # 处理可行区域
        result = self.process_garden_walkable_areas(garden_data)
        
        # 生成可视化图像
        image_filename = self.visualize_walkable_areas(result)
        
        # 计算统计信息
        statistics = self.calculate_walkable_statistics(result)
        
        # 保存数据
        data_filename = self.save_walkable_data(result, statistics)
        
        # 打印统计信息
        print(f"📊 {garden_name} 优化可行区域统计:")
        print(f"   🏛️ 园林边界面积: {statistics['garden_boundary_area']/1000000:.1f} 平方米")
        print(f"   🏠 实体建筑面积: {statistics['solid_non_walkable_area']/1000000:.1f} 平方米 "
              f"({statistics['solid_building_ratio']*100:.1f}%)")
        print(f"   💧 水体面积: {statistics['strict_non_walkable_area']/1000000:.1f} 平方米 "
              f"({statistics['water_ratio']*100:.1f}%)")
        print(f"   ⛰️  山体面积: {statistics['mountain_non_walkable_area']/1000000:.1f} 平方米 "
              f"({statistics['mountain_ratio']*100:.1f}%)")
        print(f"   🛤️  道路面积: {statistics['road_area']/1000000:.1f} 平方米 "
              f"({statistics['road_ratio']*100:.1f}%)")
        print(f"   🚫 总不可行面积: {statistics['total_non_walkable_area']/1000000:.1f} 平方米 "
              f"({statistics['non_walkable_ratio']*100:.1f}%)")
        print(f"   ✅ 最终可行面积: {statistics['final_walkable_area']/1000000:.1f} 平方米 "
              f"({statistics['walkable_ratio']*100:.1f}%)")
        
        return {
            'garden_name': garden_name,
            'statistics': statistics,
            'image_filename': image_filename,
            'data_filename': data_filename
        }
    
    def batch_process_all_gardens(self):
        """批量处理所有园林"""
        print("🚀 园林可行区域处理器启动!")
        print("📋 任务: 识别和构建所有园林的可行区域")
        print("=" * 80)
        
        # 获取所有园林数据文件
        garden_files = [f for f in os.listdir(self.data_dir) if f.endswith('_数据.json')]
        gardens = [f.replace('_数据.json', '') for f in garden_files]
        
        results = []
        
        for garden_name in gardens:
            try:
                result = self.process_single_garden(garden_name)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ 处理 {garden_name} 时出错: {e}")
                continue
        
        # 生成总结报告
        if results:
            self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results):
        """生成总结报告"""
        print(f"\n{'='*30} 可行区域处理总结 {'='*30}")
        
        if not results:
            print("❌ 没有成功处理的园林数据")
            return
        
        # 创建对比图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('十园林可行区域对比分析', fontsize=16, fontweight='bold')
        
        garden_names = [r['garden_name'] for r in results]
        walkable_areas = [r['statistics']['final_walkable_area']/1000000 for r in results]
        non_walkable_areas = [r['statistics']['non_walkable_area']/1000000 for r in results]
        walkable_ratios = [r['statistics']['walkable_ratio']*100 for r in results]
        total_areas = [r['statistics']['garden_boundary_area']/1000000 for r in results]
        
        # 1. 可行区域面积对比
        bars1 = ax1.bar(garden_names, walkable_areas, color='lightgreen', alpha=0.7)
        ax1.set_title('各园林可行区域面积对比')
        ax1.set_ylabel('可行区域面积 (平方米)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        for bar, area in zip(bars1, walkable_areas):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{area:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 可行比例对比
        bars2 = ax2.bar(garden_names, walkable_ratios, color='skyblue', alpha=0.7)
        ax2.set_title('各园林可行区域比例对比')
        ax2.set_ylabel('可行区域比例 (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, ratio in zip(bars2, walkable_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. 总面积 vs 可行面积散点图
        scatter = ax3.scatter(total_areas, walkable_areas, c='purple', alpha=0.6, s=100)
        ax3.set_title('园林总面积 vs 可行面积关系')
        ax3.set_xlabel('园林总面积 (平方米)')
        ax3.set_ylabel('可行面积 (平方米)')
        ax3.grid(True, alpha=0.3)
        
        for i, name in enumerate(garden_names):
            ax3.annotate(name, (total_areas[i], walkable_areas[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')
        
        # 4. 堆叠条形图：可行 vs 不可行
        width = 0.6
        x_pos = np.arange(len(garden_names))
        
        bars_walkable = ax4.bar(x_pos, walkable_areas, width, label='可行区域', 
                               color='lightgreen', alpha=0.7)
        bars_non_walkable = ax4.bar(x_pos, non_walkable_areas, width, 
                                   bottom=walkable_areas, label='不可行区域',
                                   color='lightcoral', alpha=0.7)
        
        ax4.set_title('各园林区域构成对比')
        ax4.set_ylabel('面积 (平方米)')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(garden_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存对比图
        summary_filename = f"{self.output_dir}/园林可行区域对比分析.png"
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印排名信息
        print(f"\n🏆 可行区域面积排名:")
        sorted_by_area = sorted(results, key=lambda x: x['statistics']['final_walkable_area'], reverse=True)
        for i, result in enumerate(sorted_by_area):
            area = result['statistics']['final_walkable_area'] / 1000000
            ratio = result['statistics']['walkable_ratio'] * 100
            print(f"   {i+1:2d}. {result['garden_name']:<8}: {area:8.0f} 平方米 ({ratio:5.1f}%)")
        
        print(f"\n📊 可行比例排名:")
        sorted_by_ratio = sorted(results, key=lambda x: x['statistics']['walkable_ratio'], reverse=True)
        for i, result in enumerate(sorted_by_ratio):
            ratio = result['statistics']['walkable_ratio'] * 100
            area = result['statistics']['final_walkable_area'] / 1000000
            print(f"   {i+1:2d}. {result['garden_name']:<8}: {ratio:5.1f}% ({area:.0f} 平方米)")
        
        # 保存汇总数据
        summary_data = {
            'processing_summary': {
                'total_gardens': len(results),
                'successful_gardens': len(results)
            },
            'results': results,
            'summary_chart': summary_filename
        }
        
        with open(f'{self.output_dir}/可行区域处理总结.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 总结报告已保存:")
        print(f"   📊 对比图表: {summary_filename}")
        print(f"   📋 详细数据: {self.output_dir}/可行区域处理总结.json")


def main():
    """主函数"""
    print("🏛️ 江南古典园林可行区域处理器 - 优化版本")
    print("=" * 70)
    print("📋 优化功能说明:")
    print("   - 🏠 实体建筑: 智能间隙检测，保留门洞通道")
    print("   - 💧 水体区域: 严格边界处理，完全不可通行") 
    print("   - ⛰️  山体区域: 只允许道路通行，其余严格禁止")
    print("   - 🏛️ 半开放建筑: 视线通透，允许穿越通行")
    print("   - 🛤️  道路网络: 优化连通性，保证通行宽度")
    print("   - 🔍 间隙识别: 自动检测建筑间门洞，避免过度合并")
    print("=" * 70)
    
    processor = GardenWalkableAreaProcessor()
    results = processor.batch_process_all_gardens()
    
    if results:
        print(f"\n🎉 可行区域处理完成！")
        print(f"✅ 成功处理 {len(results)} 个园林")
        print(f"📁 结果保存在 '{processor.output_dir}/' 目录中")
        print(f"📋 下一步: 基于可行区域进行路径规划和趣味性建模")
    else:
        print("❌ 可行区域处理失败或未处理任何园林。")


if __name__ == "__main__":
    main()
