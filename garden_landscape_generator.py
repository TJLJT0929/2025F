import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import re
import json
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class GardenLandscapeGenerator:
    """
    园林景观元素分布图生成器

    功能：
    1. 读取十个园林数据表中各子表的坐标
    2. 生成园林景观元素的分布图
    3. 保存园林数据供后续路径优化使用
    """

    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        self.data_dir = data_dir
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }

        # 景观元素配置 - 优化后的显示参数
        self.element_config = {
            '道路': {'color': '#FFD700', 'size': 8, 'marker': 'o', 'alpha': 0.8, 'label': '道路'},
            '实体建筑': {'color': '#8B4513', 'size': 20, 'marker': 's', 'alpha': 0.9, 'label': '实体建筑'},
            '半开放建筑': {'color': '#FFA500', 'size': 15, 'marker': '^', 'alpha': 0.8, 'label': '半开放建筑'},
            '假山': {'color': '#696969', 'size': 10, 'marker': 'o', 'alpha': 0.7, 'label': '假山'},
            '水体': {'color': '#4169E1', 'size': 12, 'marker': 'o', 'alpha': 0.8, 'label': '水体'},
            '植物': {'color': '#228B22', 'size': 6, 'marker': 'o', 'alpha': 0.7, 'label': '植物'}
        }

        self.create_output_directories()

    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            'results/landscape_maps',
            'results/garden_data'
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

    def load_garden_data(self, garden_id):
        """加载园林数据"""
        garden_name = self.gardens[garden_id]
        data_path = f"{self.data_dir}/{garden_id}. {garden_name}/4-{garden_name}数据坐标.xlsx"

        garden_data = {
            'id': garden_id,
            'name': garden_name,
            'elements': {},
            'boundaries': None
        }

        try:
            excel_file = pd.ExcelFile(data_path)
            print(f"📖 加载 {garden_name} 数据...")

            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(data_path, sheet_name=sheet_name)
                element_type = self.infer_element_type(sheet_name, df)
                coords = self.extract_coordinates_from_dataframe(df)

                if coords:
                    garden_data['elements'][element_type] = coords
                    print(f"  ✓ {element_type}: {len(coords)} 个元素")

            # 计算园林边界
            garden_data['boundaries'] = self.calculate_garden_boundaries(garden_data['elements'])

            return garden_data

        except Exception as e:
            print(f"❌ 加载 {garden_name} 数据失败: {e}")
            return None

    def calculate_garden_boundaries(self, garden_elements):
        """计算园林边界"""
        all_coords = []
        for element_type, coords in garden_elements.items():
            all_coords.extend(coords)

        if not all_coords:
            return None

        coords_array = np.array(all_coords)

        boundaries = {
            'min_x': float(np.min(coords_array[:, 0])),
            'max_x': float(np.max(coords_array[:, 0])),
            'min_y': float(np.min(coords_array[:, 1])),
            'max_y': float(np.max(coords_array[:, 1])),
            'center_x': float(np.mean(coords_array[:, 0])),
            'center_y': float(np.mean(coords_array[:, 1])),
            'width': float(np.max(coords_array[:, 0]) - np.min(coords_array[:, 0])),
            'height': float(np.max(coords_array[:, 1]) - np.min(coords_array[:, 1]))
        }

        return boundaries

    def determine_legend_position(self, boundaries):
        """智能确定图例位置 - 避免挡住园林"""
        if not boundaries:
            return 'upper right'

        width = boundaries['width']
        height = boundaries['height']
        center_x = boundaries['center_x']
        center_y = boundaries['center_y']

        # 根据园林形状和位置选择最佳图例位置
        if width > height:  # 园林比较宽
            if center_y > (boundaries['min_y'] + boundaries['max_y']) / 2:
                return 'lower right'
            else:
                return 'upper right'
        else:  # 园林比较高
            if center_x > (boundaries['min_x'] + boundaries['max_x']) / 2:
                return 'upper left'
            else:
                return 'upper right'

    def generate_landscape_map(self, garden_data):
        """生成园林景观分布图"""
        garden_name = garden_data['name']
        boundaries = garden_data['boundaries']

        print(f"🎨 生成 {garden_name} 景观分布图...")

        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(f"{garden_name} - 景观元素分布图", fontsize=16, fontweight='bold', pad=20)

        legend_elements = []

        # 绘制各类景观元素
        for element_type, coords in garden_data['elements'].items():
            if not coords:
                continue

            config = self.element_config.get(element_type,
                {'color': '#000000', 'size': 5, 'marker': 'o', 'alpha': 0.7, 'label': element_type})

            coords_array = np.array(coords)
            scatter = ax.scatter(coords_array[:, 0], coords_array[:, 1],
                               c=config['color'], s=config['size'],
                               marker=config['marker'], alpha=config['alpha'],
                               label=f"{config['label']} ({len(coords)})")
            legend_elements.append(scatter)

        ax.set_xlabel('X坐标 (毫米)', fontsize=12)
        ax.set_ylabel('Y坐标 (毫米)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

        # 智能图例定位
        legend_position = self.determine_legend_position(boundaries)
        ax.legend(handles=legend_elements, loc=legend_position, fontsize=10,
                 framealpha=0.95, fancybox=True, shadow=True)

        # 添加园林基本信息
        info_text = f"园林规模: {boundaries['width']:.0f}×{boundaries['height']:.0f} mm"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                facecolor='lightblue', alpha=0.7))

        plt.tight_layout()

        # 保存图片
        map_filename = f"results/landscape_maps/{garden_name}_景观分布图.png"
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"💾 景观分布图已保存: {map_filename}")
        return map_filename

    def save_garden_data(self, garden_data):
        """保存园林数据为JSON文件，供后续路径优化使用"""
        garden_name = garden_data['name']

        # 转换numpy数组为列表，确保JSON序列化
        serializable_data = {
            'id': garden_data['id'],
            'name': garden_data['name'],
            'elements': {k: [list(coord) for coord in v] for k, v in garden_data['elements'].items()},
            'boundaries': garden_data['boundaries']
        }

        data_filename = f"results/garden_data/{garden_name}_数据.json"
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

        print(f"💾 园林数据已保存: {data_filename}")
        return data_filename

    def process_single_garden(self, garden_id):
        """处理单个园林"""
        print(f"\n{'='*50}")
        print(f"🏛️ 处理园林: {self.gardens[garden_id]} (ID: {garden_id})")
        print(f"{'='*50}")

        # 加载数据
        garden_data = self.load_garden_data(garden_id)
        if not garden_data or not garden_data['elements']:
            print(f"❌ {self.gardens[garden_id]} 数据加载失败")
            return None

        # 生成景观分布图
        map_filename = self.generate_landscape_map(garden_data)

        # 保存园林数据
        data_filename = self.save_garden_data(garden_data)

        result = {
            'garden_id': garden_id,
            'garden_name': garden_data['name'],
            'map_filename': map_filename,
            'data_filename': data_filename,
            'elements_count': {k: len(v) for k, v in garden_data['elements'].items()},
            'boundaries': garden_data['boundaries']
        }

        print(f"✅ {garden_data['name']} 处理完成:")
        print(f"   🎨 分布图: {map_filename}")
        print(f"   📊 数据文件: {data_filename}")
        print(f"   📈 元素统计: {result['elements_count']}")

        return result

    def batch_process_all_gardens(self):
        """批量处理所有园林"""
        print("🚀 园林景观元素分布图生成器启动!")
        print("📋 任务: 读取十个园林数据并生成景观分布图")
        print("=" * 80)

        results = []

        for garden_id in range(1, 11):
            try:
                result = self.process_single_garden(garden_id)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"❌ 处理园林 {garden_id} 时出错: {e}")
                continue

        # 生成总结报告
        if results:
            self.generate_summary_report(results)

        return results

    def generate_summary_report(self, results):
        """生成总结报告 - 生成四张独立的图表"""
        print(f"\n{'='*25} 处理总结报告 {'='*25}")

        if not results:
            print("❌ 没有成功处理的园林数据")
            return

        garden_names = [r['garden_name'] for r in results]

        # 1. 各园林元素总数对比
        fig1, ax1 = plt.subplots(figsize=(14, 8))
        
        element_counts = defaultdict(list)
        for result in results:
            for element_type in ['道路', '实体建筑', '半开放建筑', '假山', '水体', '植物']:
                count = result['elements_count'].get(element_type, 0)
                element_counts[element_type].append(count)

        x_pos = np.arange(len(garden_names))
        width = 0.12
        colors = ['#FFD700', '#8B4513', '#FFA500', '#696969', '#4169E1', '#228B22']

        for i, (element_type, counts) in enumerate(element_counts.items()):
            ax1.bar(x_pos + i*width, counts, width,
                   label=element_type, color=colors[i], alpha=0.8)

        ax1.set_xlabel('园林', fontsize=12)
        ax1.set_ylabel('元素数量', fontsize=12)
        ax1.set_xticks(x_pos + width*2.5)
        ax1.set_xticklabels(garden_names, rotation=45, ha='right', fontsize=11)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        chart1_filename = "results/landscape_maps/园林元素数量对比图.png"
        plt.savefig(chart1_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 园林规模对比
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        areas = [r['boundaries']['width'] * r['boundaries']['height'] / 1000000
                for r in results]  # 转换为平方米
        bars = ax2.bar(garden_names, areas, color='lightcoral', alpha=0.8)
        ax2.set_xlabel('园林', fontsize=12)
        ax2.set_ylabel('占地面积 (平方米)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45, labelsize=11)
        ax2.grid(True, alpha=0.3)

        # 在柱子上显示数值
        for bar, area in zip(bars, areas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{area:.0f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        chart2_filename = "results/landscape_maps/园林占地面积对比图.png"
        plt.savefig(chart2_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 元素密度分析 - 改进标注显示完整园林名称
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        
        densities = []
        for result in results:
            total_elements = sum(result['elements_count'].values())
            area = result['boundaries']['width'] * result['boundaries']['height'] / 1000000
            density = total_elements / area if area > 0 else 0
            densities.append(density)

        # 使用更大的散点
        scatter = ax3.scatter(areas, densities, c='purple', alpha=0.6, s=150)
        ax3.set_xlabel('占地面积 (平方米)', fontsize=12)
        ax3.set_ylabel('元素密度 (个/平方米)', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # 使用完整园林名称和更大字体进行标注
        for i, name in enumerate(garden_names):
            ax3.annotate(name, (areas[i], densities[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.8, edgecolor='gray'))

        plt.tight_layout()
        chart3_filename = "results/landscape_maps/园林规模与元素密度关系图.png"
        plt.savefig(chart3_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 各类元素占比饼图（以所有园林总和为基准）
        fig4, ax4 = plt.subplots(figsize=(10, 10))
        
        total_by_type = defaultdict(int)
        for result in results:
            for element_type, count in result['elements_count'].items():
                total_by_type[element_type] += count

        labels = list(total_by_type.keys())
        sizes = list(total_by_type.values())
        colors_pie = [self.element_config[label]['color'] for label in labels]

        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie,
                                          autopct='%1.1f%%', textprops={'fontsize': 11})

        # 为饼图块设置透明度
        for wedge in wedges:
            wedge.set_alpha(0.8)

        plt.tight_layout()
        chart4_filename = "results/landscape_maps/景观元素类型分布饼图.png"
        plt.savefig(chart4_filename, dpi=300, bbox_inches='tight')
        plt.close()

        # 打印统计信息
        print(f"📊 处理统计:")
        print(f"   成功处理: {len(results)}/10 个园林")
        print(f"   图表1: {chart1_filename}")
        print(f"   图表2: {chart2_filename}")
        print(f"   图表3: {chart3_filename}")
        print(f"   图表4: {chart4_filename}")

        print(f"\n🏛️ 园林规模排名:")
        sorted_by_area = sorted(results, key=lambda x: x['boundaries']['width'] * x['boundaries']['height'], reverse=True)
        for i, result in enumerate(sorted_by_area):
            area = result['boundaries']['width'] * result['boundaries']['height'] / 1000000
            print(f"   {i+1:2d}. {result['garden_name']:<8}: {area:8.0f} 平方米")

        print(f"\n🌿 元素丰富度排名:")
        sorted_by_elements = sorted(results, key=lambda x: sum(x['elements_count'].values()), reverse=True)
        for i, result in enumerate(sorted_by_elements):
            total = sum(result['elements_count'].values())
            print(f"   {i+1:2d}. {result['garden_name']:<8}: {total:4d} 个景观元素")

        # 保存详细结果
        summary_data = {
            'processing_summary': {
                'total_gardens': len(results),
                'successful_gardens': len(results),
                'failed_gardens': 10 - len(results)
            },
            'results': results,
            'chart_files': [chart1_filename, chart2_filename, chart3_filename, chart4_filename]
        }

        with open('results/garden_data/园林处理总结.json', 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)

        print(f"\n💾 详细数据已保存: results/garden_data/园林处理总结.json")


def main():
    """主函数"""
    print("🏛️ 江南古典园林景观元素分布图生成器")
    print("=" * 60)

    generator = GardenLandscapeGenerator()
    results = generator.batch_process_all_gardens()

    if results:
        print(f"\n🎉 景观分布图生成完成！")
        print(f"✅ 成功处理 {len(results)}/10 个园林")
        print(f"📁 结果保存在 'results/' 目录中")
        print(f"📋 下一步: 使用 garden_path_optimizer.py 进行路径优化")
    else:
        print("❌ 景观分布图生成失败或未处理任何园林。")


if __name__ == "__main__":
    main()
