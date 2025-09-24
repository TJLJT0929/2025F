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
    园林可行区域处理器 - 简化版（只生成明确可通行区域图）
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
        """可视化可行区域 - 只生成明确可通行区域图"""
        garden_name = result['garden_name']

        # 创建单个图形，只显示明确可通行区域
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        fig.suptitle(f"{garden_name} - 明确可通行区域", fontsize=16, fontweight='bold')

        # 设置标题
        ax.set_title("明确可通行区域识别", fontsize=14)

        # 绘制园林边界
        self.plot_polygon(ax, result['garden_boundary'], 'lightgray', alpha=0.3, label='园林边界')

        # 绘制可通行区域
        self.plot_polygon(ax, result['walkable_area'], self.colors['walkable_area'],
                         alpha=0.7, label='明确可通行区域')

        # 标注原始元素点
        for element_type in self.element_types['walkable']:
            coords = result['elements'].get(element_type, [])
            if coords:
                coords_array = np.array(coords)
                ax.scatter(coords_array[:, 0], coords_array[:, 1],
                           c=self.colors[element_type], alpha=0.8, s=30,
                           label=f"{element_type}点")

        ax.set_xlabel('X坐标 (mm)')
        ax.set_ylabel('Y坐标 (mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        # 保存图像
        filename = f"{self.output_dir}/{garden_name}_明确可通行区域.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 明确可通行区域图已保存: {filename}")
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

        # 生成可视化图像（只生成明确可通行区域图）
        image_filename = self.visualize_walkable_areas(result)

        print(f"✅ {garden_name} 明确可通行区域图生成完成")

        return {
            'garden_name': garden_name,
            'image_filename': image_filename
        }

    def batch_process_all_gardens(self):
        """批量处理所有园林 - 简化版"""
        print("🚀 园林可行区域处理器启动!")
        print("📋 任务: 生成所有园林的明确可通行区域图")
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

        return results


def main():
    """主函数 - 简化版"""
    print("🏛️ 江南古典园林可行区域处理器 - 明确可通行区域图生成")
    print("=" * 60)
    print("📋 功能说明:")
    print("   - 只生成明确可通行区域图")
    print("   - 显示半开放建筑和道路区域")
    print("=" * 60)

    processor = GardenWalkableAreaProcessor()
    results = processor.batch_process_all_gardens()

    if results:
        print(f"\n🎉 明确可通行区域图生成完成！")
        print(f"✅ 成功处理 {len(results)} 个园林")
        print(f"📁 结果保存在 '{processor.output_dir}/' 目录中")
        print(f"📸 生成的文件命名格式: '园林名_明确可通行区域.png'")
    else:
        print("❌ 图像生成失败或未处理任何园林。")


if __name__ == "__main__":
    main()
