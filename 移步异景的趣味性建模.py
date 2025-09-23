
问题1：移步异景的趣味性建模
江南古典园林美学特征建模分析

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体环境
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

class InterestAnalyzer:
    """移步异景趣味性分析器

    主要功能：
    - 数据加载与预处理
    - 路径特征计算（路径长度、转折点、复杂度）
    - 景观多样性分析（Shannon指数）
    - 趣味性综合评分
    - 可视化图表生成
    """

    def __init__(self):
        """初始化分析器"""
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }
        self.garden_data = {}
        self.element_types = {
            0: '半开放建筑', 1: '实体建筑', 2: '道路',
            3: '假山', 4: '水体', 5: '植物'
        }

    def load_garden_data(self, garden_id):
        """加载单个园林的坐标数据

        Args:
            garden_id (int): 园林ID (1-10)

        Returns:
            bool: 加载是否成功
        """
        garden_name = self.gardens[garden_id]
        data_path = f"赛题F江南古典园林美学特征建模附件资料/{garden_id}. {garden_name}/4-{garden_name}数据坐标.xlsx"

        try:
            excel_file = pd.ExcelFile(data_path)
            garden_info = {
                'name': garden_name,
                'id': garden_id,
                'elements': {}
            }

            # 读取6种景观元素数据
            for i, sheet_name in enumerate(excel_file.sheet_names):
                if i < 6:  # 只读取前6个工作表
                    df = pd.read_excel(data_path, sheet_name=sheet_name)
                    element_name = self.element_types[i]
                    garden_info['elements'][element_name] = df

            self.garden_data[garden_id] = garden_info
            print(f"✓ 成功加载 {garden_name} 的数据")
            return True

        except Exception as e:
            print(f"✗ 加载 {garden_name} 数据时出错: {e}")
            return False

    def load_all_gardens(self):
        """加载所有园林数据

        Returns:
            bool: 是否全部加载成功
        """
        print("开始加载所有园林数据...")
        success_count = 0
        for garden_id in range(1, 11):
            if self.load_garden_data(garden_id):
                success_count += 1

        print(f"数据加载完成: {success_count}/10 个园林")
        return success_count == 10

    def extract_coordinates(self, df):
        """从数据框中提取坐标点

        Args:
            df (DataFrame): 包含坐标数据的数据框

        Returns:
            list: 坐标点列表 [(x1, y1), (x2, y2), ...]
        """
        coordinates = []

        if len(df.columns) >= 2:
            # 使用第二列（不区分线段的点位坐标）
            coord_col = df.columns[1]
            for idx, row in df.iterrows():
                coord_str = str(row[coord_col])
                if '{' in coord_str and '}' in coord_str:
                    try:
                        # 解析坐标字符串 {x,y,z}
                        coord_str = coord_str.strip('{}')
                        coords = [float(x.strip()) for x in coord_str.split(',')]
                        if len(coords) >= 2:
                            coordinates.append((coords[0], coords[1]))
                    except ValueError:
                        continue  # 跳过无效坐标

        return coordinates
    

  
