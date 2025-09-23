import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import warnings
import matplotlib
from collections import defaultdict

warnings.filterwarnings('ignore')

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

class GardenDataLoader:
    """
    园林数据加载与可视化类
    根据1.1-1.2.tex理论，构建园林路径网络的基础数据结构
    """
    
    def __init__(self, data_dir="赛题F江南古典园林美学特征建模附件资料"):
        self.data_dir = data_dir
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }
        
        # 景观元素配置 - 对应理论中的景观元素集O
        self.element_config = {
            '道路': {'color': '#FFD700', 'size': 8, 'marker': 'o', 'alpha': 0.8},
            '实体建筑': {'color': '#8B4513', 'size': 20, 'marker': 's', 'alpha': 0.9},
            '半开放建筑': {'color': '#FFA500', 'size': 15, 'marker': '^', 'alpha': 0.8},
            '假山': {'color': '#696969', 'size': 10, 'marker': 'o', 'alpha': 0.7},
            '水体': {'color': '#4169E1', 'size': 8, 'marker': 'o', 'alpha': 0.8},
            '植物': {'color': '#228B22', 'size': 6, 'marker': 'o', 'alpha': 0.6}
        }
        
        self.create_output_directories()
    
    def create_output_directories(self):
        """创建输出目录"""
        directories = [
            'results/garden_maps',
            'results/data_analysis'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def parse_coordinate_string(self, coord_str):
        """
        解析坐标字符串
        对应理论中的二维点集 P_road = {p_1, p_2, ..., p_N}, 其中 p_i = (x_i, y_i) ∈ R^2
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
        加载园林数据
        构建理论中描述的景观元素集合 O = {O_1, O_2, ..., O_P}
        """
        garden_name = self.gardens[garden_id]
        data_path = f"{self.data_dir}/{garden_id}. {garden_name}/4-{garden_name}数据坐标.xlsx"
        
        garden_data = {
            'id': garden_id,
            'name': garden_name,
            'elements': {}  # 对应理论中的景观元素集合 O
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
        return '道路'
    
    def extract_coordinates_from_dataframe(self, df):
        """从DataFrame中提取坐标"""
        coords = []
        for col in df.columns:
            for _, row in df.iterrows():
                coord_str = str(row[col])
                parsed_coord = self.parse_coordinate_string(coord_str)
                if parsed_coord:
                    coords.append(parsed_coord)
        return list(set(coords))
    
    def calculate_garden_boundaries(self, garden_elements):
        """
        计算园林边界
        为后续的图构建和路径规划提供空间约束
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
            'center_y': np.mean(coords_array[:, 1])
        }
        
        return boundaries
    
    def generate_garden_map(self, garden_data, boundaries):
        """
        生成园林景观分布图
        可视化理论中的景观元素集合 O
        """
        garden_name = garden_data['name']
        print(f"🎨 生成 {garden_name} 景观分布图...")
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_title(f"{garden_name} - 景观元素分布图", fontsize=16, fontweight='bold', pad=20)
        
        legend_elements = []
        
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
        
        ax.set_xlabel('X (毫米)', fontsize=12)
        ax.set_ylabel('Y (毫米)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 智能图例定位
        ax.legend(handles=legend_elements, loc='best', fontsize=10,
                 framealpha=0.95, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        map_filename = f"results/garden_maps/{garden_name}_景观分布图.png"
        plt.savefig(map_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 景观分布图已保存: {map_filename}")
        return map_filename
    
    def analyze_garden_statistics(self, garden_data):
        """
        分析园林统计信息
        为理论模型提供基础数据支撑
        """
        stats = {
            'garden_name': garden_data['name'],
            'element_counts': {},
            'total_elements': 0,
            'element_densities': {}
        }
        
        for element_type, coords in garden_data['elements'].items():
            count = len(coords)
            stats['element_counts'][element_type] = count
            stats['total_elements'] += count
        
        # 计算密度（相对比例）
        for element_type, count in stats['element_counts'].items():
            stats['element_densities'][element_type] = count / max(stats['total_elements'], 1)
        
        return stats
    
    def process_all_gardens(self):
        """
        处理所有园林数据
        生成基础的景观分布图
        """
        print("🏛️ 开始加载所有园林数据...")
        print("=" * 60)
        
        all_results = []
        
        for garden_id in range(1, 11):
            try:
                # 加载数据
                garden_data = self.load_garden_data(garden_id)
                if not garden_data or not garden_data['elements']:
                    print(f"❌ {self.gardens[garden_id]} 数据加载失败")
                    continue
                
                # 计算边界
                boundaries = self.calculate_garden_boundaries(garden_data['elements'])
                
                # 生成地图
                map_filename = self.generate_garden_map(garden_data, boundaries)
                
                # 统计分析
                stats = self.analyze_garden_statistics(garden_data)
                
                result = {
                    'garden_data': garden_data,
                    'boundaries': boundaries,
                    'map_filename': map_filename,
                    'statistics': stats
                }
                
                all_results.append(result)
                print(f"✅ {self.gardens[garden_id]} 处理完成")
                
            except Exception as e:
                print(f"❌ 处理园林 {garden_id} 时出错: {e}")
                continue
        
        # 生成汇总分析
        if all_results:
            self.generate_summary_analysis(all_results)
        
        print(f"\n✅ 数据加载完成，共处理 {len(all_results)}/10 个园林")
        return all_results
    
    def generate_summary_analysis(self, results):
        """生成汇总分析图表"""
        print("📊 生成汇总分析...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('江南古典园林景观元素统计分析', fontsize=18, fontweight='bold')
        
        # 提取数据
        garden_names = [r['statistics']['garden_name'] for r in results]
        
        # 1. 总元素数量对比
        total_counts = [r['statistics']['total_elements'] for r in results]
        bars1 = ax1.bar(garden_names, total_counts, color='lightblue', alpha=0.8)
        ax1.set_title('各园林景观元素总数', fontweight='bold')
        ax1.set_ylabel('元素总数')
        ax1.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars1, total_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(count), ha='center', va='bottom', fontsize=9)
        
        # 2. 元素类型分布
        element_types = set()
        for r in results:
            element_types.update(r['statistics']['element_counts'].keys())
        element_types = list(element_types)
        
        element_data = []
        for element_type in element_types:
            type_counts = []
            for r in results:
                count = r['statistics']['element_counts'].get(element_type, 0)
                type_counts.append(count)
            element_data.append(type_counts)
        
        x_pos = np.arange(len(garden_names))
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']
        
        bottom = np.zeros(len(garden_names))
        for i, (element_type, counts) in enumerate(zip(element_types, element_data)):
            color = colors[i % len(colors)]
            ax2.bar(x_pos, counts, bottom=bottom, label=element_type, 
                   color=color, alpha=0.8)
            bottom += counts
        
        ax2.set_title('各园林元素类型分布（堆叠图）', fontweight='bold')
        ax2.set_ylabel('元素数量')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(garden_names, rotation=45)
        ax2.legend(loc='upper left', fontsize=8)
        
        # 3. 道路点密度分析
        road_counts = []
        for r in results:
            road_count = r['statistics']['element_counts'].get('道路', 0)
            road_counts.append(road_count)
        
        ax3.scatter(range(len(garden_names)), road_counts, 
                   c='red', s=100, alpha=0.7)
        ax3.set_title('各园林道路点数量', fontweight='bold')
        ax3.set_ylabel('道路点数量')
        ax3.set_xticks(range(len(garden_names)))
        ax3.set_xticklabels(garden_names, rotation=45)
        ax3.grid(True, alpha=0.3)
        
        for i, (name, count) in enumerate(zip(garden_names, road_counts)):
            ax3.annotate(f'{count}', (i, count), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)
        
        # 4. 园林规模对比（基于边界面积）
        areas = []
        for r in results:
            boundary = r['boundaries']
            if boundary:
                width = boundary['max_x'] - boundary['min_x']
                height = boundary['max_y'] - boundary['min_y']
                area = width * height / 1000000  # 转换为平方米
                areas.append(area)
            else:
                areas.append(0)
        
        bars4 = ax4.barh(garden_names, areas, color='lightgreen', alpha=0.8)
        ax4.set_title('各园林占地面积估算', fontweight='bold')
        ax4.set_xlabel('面积 (平方米)')
        ax4.invert_yaxis()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        summary_filename = "results/data_analysis/园林数据汇总分析.png"
        plt.savefig(summary_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 汇总分析图已保存: {summary_filename}")

def main():
    """主函数 - 数据加载与可视化"""
    print("🏛️ 江南古典园林数据加载与可视化系统")
    print("📖 基于1.1-1.2.tex理论框架")
    print("=" * 60)
    
    loader = GardenDataLoader()
    results = loader.process_all_gardens()
    
    if results:
        print(f"\n🎉 数据加载系统运行完成！")
        print(f"✅ 成功处理 {len(results)}/10 个园林")
        print(f"📁 结果保存在 'results/' 目录中")
        print(f"📊 景观分布图: results/garden_maps/")
        print(f"📈 统计分析: results/data_analysis/")
    else:
        print("❌ 数据加载系统运行失败")

if __name__ == "__main__":
    main()