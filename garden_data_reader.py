import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
import warnings
import re
import json
import matplotlib
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.ops import unary_union

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class GardenDataReader:
    """
    园林数据读取与景观图生成器
    按照理论文档1.1节的要求，构建园林路径网络的图模型基础数据
    """

    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        self.data_dir = data_dir
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }

        # 景观元素配置 - 对应理论文档中的景观元素集O
        self.element_config = {
            '道路': {'color': '#FFD700', 'size': 5, 'marker': 'o', 'alpha': 0.7, 'type': 'path'},
            '实体建筑': {'color': '#8B4513', 'size': 15, 'marker': 's', 'alpha': 0.9, 'type': 'solid_building'},
            '半开放建筑': {'color': '#FFA500', 'size': 12, 'marker': '^', 'alpha': 0.8, 'type': 'semi_building'},
            '假山': {'color': '#696969', 'size': 6, 'marker': 'o', 'alpha': 0.7, 'type': 'mountain'},
            '水体': {'color': '#4169E1', 'size': 6, 'marker': 'o', 'alpha': 0.8, 'type': 'water'},
            '植物': {'color': '#228B22', 'size': 4, 'marker': 'o', 'alpha': 0.6, 'type': 'plant'}
        }

        self.create_output_directories()

    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            'results/garden_maps',
            'results/garden_data',
            'results/processed_data'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def parse_coordinate_string(self, coord_str):
        """
        解析坐标字符串 - 提取原始道路坐标测量数据
        对应理论文档中的P_road = {p_1, p_2, ..., p_N}，其中p_i = (x_i, y_i) ∈ R²
        """
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
        """
        加载园林数据 - 构建景观元素集合O
        返回包含所有景观元素坐标的结构化数据
        """
        garden_name = self.gardens[garden_id]
        data_path = f"{self.data_dir}/{garden_id}. {garden_name}/4-{garden_name}数据坐标.xlsx"

        garden_data = {
            'id': garden_id,
            'name': garden_name,
            'elements': {},
            'raw_path_points': []  # 对应理论文档中的P_road原始点集
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
                        # 如果是道路数据，保存为原始路径点集
                        if element_type == '道路':
                            garden_data['raw_path_points'].extend(coords)
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
        return '道路'  # 默认为道路

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

    def calculate_garden_boundaries(self, garden_elements):
        """
        计算园林边界 - 用于后续图构建的空间约束
        """
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
            'center_y': np.mean(coords_array[:, 1]),
            'width': np.max(coords_array[:, 0]) - np.min(coords_array[:, 0]),
            'height': np.max(coords_array[:, 1]) - np.min(coords_array[:, 1])
        }

        return boundaries

    def determine_legend_position(self, boundaries):
        """智能确定图例位置 - 避免遮挡园林主体"""
        if not boundaries:
            return 'upper right'

        # 根据园林形状和位置选择最佳图例位置
        if boundaries['width'] > boundaries['height']:  # 园林比较宽
            if boundaries['center_y'] > (boundaries['min_y'] + boundaries['max_y']) / 2:
                return 'lower right'
            else:
                return 'upper right'
        else:  # 园林比较高
            if boundaries['center_x'] > (boundaries['min_x'] + boundaries['max_x']) / 2:
                return 'upper left'
            else:
                return 'upper right'

    def generate_garden_landscape_map(self, garden_data, boundaries):
        """
        生成园林景观分布图
        可视化景观元素集合O = {O_1, O_2, ..., O_P}
        """
        garden_name = garden_data['name']

        print(f"🎨 生成 {garden_name} 景观分布图...")

        fig, ax = plt.subplots(figsize=(16, 12))
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
                               label=f"{element_type} ({len(coords)})")
            legend_elements.append(scatter)
            total_elements += len(coords)

        # 添加统计信息
        if boundaries:
            ax.text(0.02, 0.98,
                   f"园林规模: {boundaries['width']:.0f}×{boundaries['height']:.0f}mm\n"
                   f"元素总数: {total_elements}\n"
                   f"道路点数: {len(garden_data['raw_path_points'])}",
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8),
                   verticalalignment='top')

        ax.set_xlabel('X坐标 (毫米)', fontsize=12)
        ax.set_ylabel('Y坐标 (毫米)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 智能图例定位
        legend_position = self.determine_legend_position(boundaries)
        ax.legend(handles=legend_elements, loc=legend_position, fontsize=9,
                 framealpha=0.95, fancybox=True, shadow=True)

        plt.tight_layout()

        map_filename = f"results/garden_maps/{garden_name}_景观分布图.png"
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 景观分布图已保存: {map_filename}")
        return map_filename

    def save_garden_processed_data(self, garden_data, boundaries):
        """
        保存处理后的园林数据供第二部分使用
        包含理论文档中定义的所有基础数据结构
        """
        garden_name = garden_data['name']

        processed_data = {
            'garden_info': {
                'id': garden_data['id'],
                'name': garden_name,
                'boundaries': boundaries
            },
            'landscape_elements': garden_data['elements'],  # 景观元素集O
            'raw_path_points': garden_data['raw_path_points'],  # P_road原始道路点集
            'element_statistics': {
                element_type: len(coords)
                for element_type, coords in garden_data['elements'].items()
            },
            'spatial_info': {
                'total_elements': sum(len(coords) for coords in garden_data['elements'].values()),
                'garden_area': boundaries['width'] * boundaries['height'] if boundaries else 0,
                'density': sum(len(coords) for coords in garden_data['elements'].values()) /
                          (boundaries['width'] * boundaries['height'] / 1000000) if boundaries and boundaries['width'] * boundaries['height'] > 0 else 0
            }
        }

        # 保存为JSON格式
        data_filename = f"results/processed_data/{garden_name}_processed_data.json"
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

        print(f"💾 处理数据已保存: {data_filename}")
        return data_filename

    def process_single_garden(self, garden_id):
        """处理单个园林 - 数据读取与可视化"""
        print(f"\n{'='*50}")
        print(f"🏛️ 处理园林: {self.gardens[garden_id]} (ID: {garden_id})")
        print(f"{'='*50}")

        # 加载数据
        garden_data = self.load_garden_data(garden_id)
        if not garden_data or not garden_data['elements']:
            print(f"❌ {self.gardens[garden_id]} 数据加载失败")
            return None

        # 计算边界
        boundaries = self.calculate_garden_boundaries(garden_data['elements'])

        # 生成景观分布图
        map_filename = self.generate_garden_landscape_map(garden_data, boundaries)

        # 保存处理后的数据
        data_filename = self.save_garden_processed_data(garden_data, boundaries)

        result = {
            'garden_id': garden_id,
            'garden_name': self.gardens[garden_id],
            'map_filename': map_filename,
            'data_filename': data_filename,
            'boundaries': boundaries,
            'element_count': sum(len(coords) for coords in garden_data['elements'].values()),
            'path_points_count': len(garden_data['raw_path_points'])
        }

        print(f"✅ {self.gardens[garden_id]} 处理完成:")
        print(f"   📊 景观图: {map_filename}")
        print(f"   💾 数据文件: {data_filename}")
        print(f"   📈 元素总数: {result['element_count']}")
        print(f"   🛤️ 道路点数: {result['path_points_count']}")

        return result

    def batch_process_all_gardens(self):
        """批量处理所有园林"""
        print("🚀 园林数据读取与景观图生成系统启动!")
        print("📋 任务: 读取十个园林数据，生成景观元素分布图")
        print("=" * 60)

        results = []

        for garden_id in range(1, 11):
            try:
                result = self.process_single_garden(garden_id)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ 处理园林 {garden_id} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 生成总结报告
        if results:
            self.generate_summary_report(results)

        return results

    def generate_summary_report(self, results):
        """生成数据读取总结报告"""
        print(f"\n{'='*30} 数据读取报告 {'='*30}")

        if not results:
            print("❌ 没有成功处理的园林数据")
            return

        # 创建统计图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('十个园林景观元素统计分析', fontsize=16, fontweight='bold')

        names = [r['garden_name'] for r in results]
        element_counts = [r['element_count'] for r in results]
        path_counts = [r['path_points_count'] for r in results]

        # 1. 元素总数对比
        bars1 = ax1.bar(names, element_counts, color='lightblue', alpha=0.8)
        ax1.set_title('各园林景观元素总数', fontweight='bold')
        ax1.set_ylabel('元素数量')
        ax1.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars1, element_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom')

        # 2. 道路点数对比
        bars2 = ax2.bar(names, path_counts, color='lightgreen', alpha=0.8)
        ax2.set_title('各园林道路点数', fontweight='bold')
        ax2.set_ylabel('道路点数量')
        ax2.tick_params(axis='x', rotation=45)

        # 3. 园林规模对比
        areas = []
        for result in results:
            if result['boundaries']:
                area = result['boundaries']['width'] * result['boundaries']['height'] / 1000000  # 转换为平方米
                areas.append(area)
            else:
                areas.append(0)

        bars3 = ax3.bar(names, areas, color='lightcoral', alpha=0.8)
        ax3.set_title('各园林占地面积', fontweight='bold')
        ax3.set_ylabel('面积 (平方米)')
        ax3.tick_params(axis='x', rotation=45)

        # 4. 元素密度分析
        densities = [area / max(count, 1) for area, count in zip(areas, element_counts)]
        bars4 = ax4.bar(names, densities, color='gold', alpha=0.8)
        ax4.set_title('各园林元素密度', fontweight='bold')
        ax4.set_ylabel('平方米/元素')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        summary_filename = "results/garden_data/园林数据统计报告.png"
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # 打印统计信息
        print(f"📊 数据处理统计:")
        print(f"   成功处理: {len(results)}/10 个园林")
        print(f"   总元素数: {sum(element_counts)}")
        print(f"   总道路点数: {sum(path_counts)}")

        print(f"\n🏛️ 各园林详细信息:")
        for result in results:
            area_info = ""
            if result['boundaries']:
                area = result['boundaries']['width'] * result['boundaries']['height'] / 1000000
                area_info = f" | 面积: {area:.1f}㎡"
            print(f"   {result['garden_name']:<8}: 元素 {result['element_count']:>3} | 道路点 {result['path_points_count']:>4}{area_info}")

        # 保存完整结果
        summary_data = {
            'processing_summary': {
                'total_gardens': len(results),
                'total_elements': sum(element_counts),
                'total_path_points': sum(path_counts),
                'average_elements_per_garden': sum(element_counts) / len(results) if results else 0
            },
            'garden_results': results
        }

        with open('results/garden_data/园林数据处理结果.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 报告文件已保存:")
        print(f"   📈 统计图表: {summary_filename}")
        print(f"   📝 完整结果: results/garden_data/园林数据处理结果.json")

def main():
    """主函数 - 园林数据读取与景观图生成"""
    print("🏛️ 江南古典园林数据读取与景观图生成系统")
    print("=" * 60)

    reader = GardenDataReader()
    results = reader.batch_process_all_gardens()

    if results:
        print(f"\n🎉 数据读取系统运行完成！")
        print(f"✅ 成功处理 {len(results)}/10 个园林")
        print(f"📁 所有文件保存在 'results/' 目录中")
        print(f"🔗 处理后的数据可供第二部分路径优化使用")
    else:
        print("❌ 数据读取系统运行失败")

if __name__ == "__main__":
    main()