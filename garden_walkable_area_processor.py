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
        
        # 元素类型配置
        self.element_types = {
            'non_walkable': ['实体建筑', '假山', '水体'],  # 不可通行
            'walkable': ['半开放建筑', '道路'],           # 可通行
            'neutral': ['植物']                        # 中性（可能影响视线但可通行）
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
        """从道路点创建可行走区域"""
        if not road_points:
            return None
        
        try:
            road_points_array = np.array(road_points)
            
            # 创建道路缓冲区
            road_polygons = []
            
            # 使用较小的聚类参数来保持道路的连续性
            clustering = DBSCAN(eps=buffer_distance*1.5, min_samples=2).fit(road_points_array)
            
            for cluster_id in set(clustering.labels_):
                if cluster_id == -1:
                    continue
                
                cluster_points = road_points_array[clustering.labels_ == cluster_id]
                
                if len(cluster_points) >= 2:
                    # 为道路点创建缓冲区
                    for point in cluster_points:
                        point_geom = Point(point).buffer(buffer_distance)
                        road_polygons.append(point_geom)
            
            if road_polygons:
                return unary_union(road_polygons)
            
        except Exception as e:
            print(f"创建道路区域失败: {e}")
            
        return None
    
    def process_garden_walkable_areas(self, garden_data):
        """处理园林可行区域"""
        garden_name = garden_data['name']
        print(f"🏗️ 处理 {garden_name} 可行区域...")
        
        boundaries = garden_data['boundaries']
        elements = garden_data['elements']
        
        # 创建不可通行区域
        non_walkable_polygons = []
        walkable_polygons = []
        
        for element_type, coords in elements.items():
            if not coords:
                continue
            
            if element_type in self.element_types['non_walkable']:
                # 不可通行区域（实体建筑、假山、水体）
                polygon = self.create_polygon_from_points(coords, buffer_distance=1500)
                if polygon:
                    non_walkable_polygons.append(polygon)
                    
            elif element_type in self.element_types['walkable']:
                # 可通行区域（半开放建筑、道路）
                if element_type == '道路':
                    # 道路使用特殊处理
                    polygon = self.create_walkable_areas_from_roads(coords, buffer_distance=2000)
                else:
                    # 半开放建筑
                    polygon = self.create_polygon_from_points(coords, buffer_distance=1000)
                
                if polygon:
                    walkable_polygons.append(polygon)
        
        # 合并区域
        non_walkable_area = unary_union(non_walkable_polygons) if non_walkable_polygons else None
        walkable_area = unary_union(walkable_polygons) if walkable_polygons else None
        
        # 创建园林边界
        garden_boundary = Polygon([
            (boundaries['min_x'], boundaries['min_y']),
            (boundaries['max_x'], boundaries['min_y']),
            (boundaries['max_x'], boundaries['max_y']),
            (boundaries['min_x'], boundaries['max_y'])
        ])
        
        # 计算最终可行区域：园林边界内 - 不可通行区域 + 明确的可通行区域
        final_walkable_area = garden_boundary
        
        if non_walkable_area:
            final_walkable_area = final_walkable_area.difference(non_walkable_area)
        
        if walkable_area:
            final_walkable_area = final_walkable_area.union(walkable_area)
        
        result = {
            'garden_name': garden_name,
            'boundaries': boundaries,
            'non_walkable_area': non_walkable_area,
            'walkable_area': walkable_area,
            'final_walkable_area': final_walkable_area,
            'garden_boundary': garden_boundary,
            'elements': elements
        }
        
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
        """可视化可行区域"""
        garden_name = result['garden_name']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f"{garden_name} - 可行区域分析", fontsize=16, fontweight='bold')
        
        # 1. 原始元素分布图
        ax1.set_title("原始景观元素分布", fontsize=14)
        for element_type, coords in result['elements'].items():
            if coords:
                coords_array = np.array(coords)
                color = self.colors.get(element_type, '#000000')
                ax1.scatter(coords_array[:, 0], coords_array[:, 1], 
                           c=color, alpha=0.7, s=20, label=f"{element_type} ({len(coords)})")
        
        ax1.set_xlabel('X坐标 (mm)')
        ax1.set_ylabel('Y坐标 (mm)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 2. 不可通行区域
        ax2.set_title("不可通行区域识别", fontsize=14)
        
        # 绘制园林边界
        self.plot_polygon(ax2, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        
        # 绘制不可通行区域
        self.plot_polygon(ax2, result['non_walkable_area'], self.colors['non_walkable_area'], 
                         alpha=0.7, label='不可通行区域')
        
        # 标注原始元素点
        for element_type in self.element_types['non_walkable']:
            coords = result['elements'].get(element_type, [])
            if coords:
                coords_array = np.array(coords)
                ax2.scatter(coords_array[:, 0], coords_array[:, 1], 
                           c=self.colors[element_type], alpha=0.8, s=15, 
                           label=f"{element_type}点")
        
        ax2.set_xlabel('X坐标 (mm)')
        ax2.set_ylabel('Y坐标 (mm)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # 3. 明确可通行区域
        ax3.set_title("明确可通行区域", fontsize=14)
        
        # 绘制园林边界
        self.plot_polygon(ax3, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        
        # 绘制可通行区域
        self.plot_polygon(ax3, result['walkable_area'], self.colors['walkable_area'], 
                         alpha=0.7, label='明确可通行区域')
        
        # 标注原始元素点
        for element_type in self.element_types['walkable']:
            coords = result['elements'].get(element_type, [])
            if coords:
                coords_array = np.array(coords)
                ax3.scatter(coords_array[:, 0], coords_array[:, 1], 
                           c=self.colors[element_type], alpha=0.8, s=15, 
                           label=f"{element_type}点")
        
        ax3.set_xlabel('X坐标 (mm)')
        ax3.set_ylabel('Y坐标 (mm)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_aspect('equal')
        
        # 4. 最终可行区域
        ax4.set_title("最终可行区域（路径规划基础）", fontsize=14)
        
        # 绘制园林边界
        self.plot_polygon(ax4, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')
        
        # 绘制不可通行区域
        self.plot_polygon(ax4, result['non_walkable_area'], self.colors['non_walkable_area'], 
                         alpha=0.5, label='不可通行区域')
        
        # 绘制最终可行区域
        self.plot_polygon(ax4, result['final_walkable_area'], self.colors['walkable_area'], 
                         alpha=0.6, label='最终可行区域')
        
        # 添加植物位置（影响视线但不影响通行）
        plant_coords = result['elements'].get('植物', [])
        if plant_coords:
            plant_array = np.array(plant_coords)
            ax4.scatter(plant_array[:, 0], plant_array[:, 1], 
                       c=self.colors['植物'], alpha=0.6, s=10, label='植物')
        
        ax4.set_xlabel('X坐标 (mm)')
        ax4.set_ylabel('Y坐标 (mm)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        
        # 保存图像
        filename = f"{self.output_dir}/{garden_name}_可行区域分析.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 可行区域分析图已保存: {filename}")
        return filename
    
    def calculate_walkable_statistics(self, result):
        """计算可行区域统计信息"""
        boundaries = result['boundaries']
        total_area = boundaries['width'] * boundaries['height']
        
        # 计算各区域面积
        garden_boundary_area = result['garden_boundary'].area if result['garden_boundary'] else 0
        non_walkable_area = result['non_walkable_area'].area if result['non_walkable_area'] else 0
        walkable_area = result['walkable_area'].area if result['walkable_area'] else 0
        final_walkable_area = result['final_walkable_area'].area if result['final_walkable_area'] else 0
        
        statistics = {
            'garden_name': result['garden_name'],
            'total_bounding_area': total_area,
            'garden_boundary_area': garden_boundary_area,
            'non_walkable_area': non_walkable_area,
            'explicit_walkable_area': walkable_area,
            'final_walkable_area': final_walkable_area,
            'walkable_ratio': final_walkable_area / garden_boundary_area if garden_boundary_area > 0 else 0,
            'non_walkable_ratio': non_walkable_area / garden_boundary_area if garden_boundary_area > 0 else 0
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
        print(f"📊 {garden_name} 可行区域统计:")
        print(f"   🏛️ 园林边界面积: {statistics['garden_boundary_area']/1000000:.1f} 平方米")
        print(f"   🚫 不可通行面积: {statistics['non_walkable_area']/1000000:.1f} 平方米 "
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
    print("🏛️ 江南古典园林可行区域处理器")
    print("=" * 60)
    print("📋 功能说明:")
    print("   - 识别不可通行区域：实体建筑、假山、水体")
    print("   - 识别可通行区域：半开放建筑、道路")
    print("   - 构建路径规划基础网格")
    print("   - 生成可视化分析图表")
    print("=" * 60)
    
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
