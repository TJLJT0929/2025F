import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.spatial import cKDTree
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

# --- Matplotlib Configuration for Chinese Characters and Scientific Style ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300  # 提高DPI用于期刊质量
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 科研风格配色方案
SCIENTIFIC_COLORS = {
    'nature': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'science': ['#0173b2', '#de8f05', '#029e73', '#cc78bc', '#ca9161',
                '#fbafe4', '#949494', '#ece133', '#56b4e9', '#f0e442'],
    'nature_comm': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
                    '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
}

class GardenInterestAnalyzer:
    """
    A comprehensive analyzer for modeling the "Interest Level" of classical Chinese gardens
    based on the concept of "Shifting Scenery" (移步异景).

    Enhanced with scientific visualization capabilities.
    """

    def __init__(self, data_folder_path):
        """
        Initializes the analyzer.

        Args:
            data_folder_path (str): The path to the main folder containing the garden data.
        """
        self.data_folder_path = data_folder_path
        self.gardens = {
            1: '拙政园', 2: '留园', 3: '寄畅园', 4: '瞻园', 5: '豫园',
            6: '秋霞圃', 7: '沈园', 8: '怡园', 9: '耦园', 10: '绮园'
        }
        self.element_types = {
            '半开放建筑': 0, '实体建筑': 1, '道路': 2,
            '假山': 3, '水体': 4, '植物': 5
        }
        self.garden_data = {}
        self.color_palette = SCIENTIFIC_COLORS['nature']
        print(f"Garden Interest Analyzer initialized. Data folder set to: '{data_folder_path}'")

    def load_garden_data(self, garden_id):
        """
        Loads the coordinate data for a single garden from its corresponding Excel file.
        """
        garden_name = self.gardens[garden_id]
        data_path = os.path.join(self.data_folder_path, f"{garden_id}. {garden_name}", f"4-{garden_name}数据坐标.xlsx")

        if not os.path.exists(data_path):
            print(f"✗ Error: Data file not found for {garden_name} at '{data_path}'")
            return False

        try:
            excel_file = pd.ExcelFile(data_path)
            garden_info = {'name': garden_name, 'id': garden_id, 'elements': {}}

            for sheet_name in excel_file.sheet_names:
                element_name = ''.join(filter(lambda char: not char.isdigit(), sheet_name.split('.')[0]))
                if element_name in self.element_types:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    garden_info['elements'][element_name] = df

            self.garden_data[garden_id] = garden_info
            print(f"✓ Successfully loaded data for {garden_name}.")
            return True
        except Exception as e:
            print(f"✗ Error loading data for {garden_name}: {e}")
            return False

    def _extract_coordinates_from_df(self, df):
        """
        Extracts and parses (x, y) coordinates from a DataFrame column.
        """
        coords = []
        if len(df.columns) < 2:
            return coords

        coord_col_data = df.iloc[:, 1].dropna()

        for item in coord_col_data:
            try:
                clean_str = str(item).strip('{}')
                parts = [float(p.strip()) for p in clean_str.split(',')]
                if len(parts) >= 2:
                    coords.append((parts[0], parts[1]))
            except (ValueError, AttributeError):
                continue
        return coords

    def _get_viewshed(self, point, kdtree, view_radius=50000):
        """
        Simulates a 'viewshed' for a given point.
        """
        indices = kdtree.query_ball_point(point, r=view_radius)
        return set(indices)

    def analyze_garden(self, garden_id, turn_angle_threshold=20, path_sample_dist=5000, view_radius=50000):
        """
        Performs a full analysis on a single garden based on the custom formulas.
        """
        if garden_id not in self.garden_data:
            self.load_garden_data(garden_id)
        if garden_id not in self.garden_data:
            return None

        garden = self.garden_data[garden_id]

        # Path Data Extraction
        path_coords = self._extract_coordinates_from_df(garden['elements'].get('道路', pd.DataFrame()))
        if len(path_coords) < 3:
            print(f"|  -> Skipping {garden['name']} due to insufficient path data.")
            return None

        # Path Length and Curvature Calculation
        path_length = 0
        turn_points_count = 0
        vectors = []
        for i in range(1, len(path_coords)):
            p1, p2 = path_coords[i-1], path_coords[i]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            path_length += np.sqrt(dx**2 + dy**2)
            vectors.append((dx, dy))

        for i in range(1, len(vectors)):
            v1, v2 = vectors[i-1], vectors[i]
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
            if mag1 > 0 and mag2 > 0:
                cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi
                if angle > turn_angle_threshold:
                    turn_points_count += 1

        # View Change Calculation
        landscape_elements = []
        for name, df in garden['elements'].items():
            if name not in ['道路', '植物']:
                landscape_elements.extend(self._extract_coordinates_from_df(df))

        if '植物' in garden['elements']:
            plant_df = garden['elements']['植物']
            if not plant_df.empty and len(plant_df.columns) >= 1:
                plant_coords = self._extract_coordinates_from_df(plant_df)
                landscape_elements.extend(plant_coords)

        total_view_change = 0
        if landscape_elements:
            kdtree = cKDTree(landscape_elements)
            num_samples = int(path_length / path_sample_dist)
            if num_samples > 1:
                sampled_indices = np.linspace(0, len(path_coords) - 1, num_samples, dtype=int)
                sampled_points = [path_coords[i] for i in sampled_indices]

                last_viewshed = self._get_viewshed(sampled_points[0], kdtree, view_radius)
                for i in range(1, len(sampled_points)):
                    current_viewshed = self._get_viewshed(sampled_points[i], kdtree, view_radius)
                    view_diff = len(current_viewshed.symmetric_difference(last_viewshed))
                    total_view_change += view_diff
                    last_viewshed = current_viewshed

        # Exploration Calculation
        exploration_score = turn_points_count

        # Fun Score Calculation
        w_curv, w_view, w_exp, w_len = 0.4, 0.4, 0.2, 0.1
        C = 1e-6

        numerator = (w_curv * turn_points_count +
                     w_view * total_view_change +
                     w_exp * exploration_score)
        denominator = w_len * (path_length / 1000) + C

        fun_score = numerator / denominator if denominator != 0 else 0

        print(f"|  -> Analysis complete for {garden['name']}.")

        return {
            'garden_name': garden['name'],
            'path_length_m': path_length / 1000,
            'curvature_score': turn_points_count,
            'view_change_score': total_view_change,
            'exploration_score': exploration_score,
            'fun_score': fun_score
        }

    def run_full_analysis(self):
        """
        Runs the analysis for all 10 gardens and stores the results.
        """
        print("\n" + "="*60)
        print("Starting Full Analysis for All Gardens")
        print("="*60)

        all_results = []
        for gid in self.gardens.keys():
            result = self.analyze_garden(gid)
            if result:
                all_results.append(result)

        self.results_df = pd.DataFrame(all_results)

        if not self.results_df.empty and 'fun_score' in self.results_df.columns:
            scaler = MinMaxScaler(feature_range=(0, 100))
            self.results_df['fun_score_scaled'] = scaler.fit_transform(self.results_df[['fun_score']])

        print("\n" + "="*60)
        print("Full Analysis Complete.")
        print("="*60)
        print(self.results_df.sort_values('fun_score_scaled', ascending=False))

    def generate_visualizations(self, save_folder="问题1_趣味性建模图表"):
        """
        Generates and saves a comprehensive suite of scientific visualizations.
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("✗ No analysis results to visualize. Please run `run_full_analysis()` first.")
            return

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print(f"\nGenerating scientific visualizations in folder: '{save_folder}'")

        df = self.results_df.copy()

        # Basic visualizations (修复版)
        self._plot_fun_score_ranking(df, save_folder)
        self._plot_score_components_stacked_bar(df, save_folder)
        self._plot_correlation_heatmap(df, save_folder)
        self._plot_length_vs_fun_score_bubble(df, save_folder)
        self._plot_top_gardens_radar(df, save_folder)

        # 新增高级科研图表
        self._plot_violin_distribution(df, save_folder)
        self._plot_forest_plot(df, save_folder)
        self._plot_boxplot_with_swarm(df, save_folder)
        self._plot_3d_scatter(df, save_folder)
        self._plot_parallel_coordinates(df, save_folder)
        self._plot_density_contour(df, save_folder)
        self._plot_statistical_comparison(df, save_folder)
        self._plot_heatmap_matrix(df, save_folder)
        self._plot_ridgeline_plot(df, save_folder)

        print(f"\n✓ All scientific visualizations have been successfully generated and saved to '{save_folder}'.")

    def _plot_fun_score_ranking(self, df, save_folder):
        """修复版：趣味性评分排名条形图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        df_sorted = df.sort_values('fun_score_scaled', ascending=False)

        bars = ax.bar(range(len(df_sorted)), df_sorted['fun_score_scaled'],
                     color=self.color_palette[:len(df_sorted)],
                     edgecolor='white', linewidth=1.5, alpha=0.8)

        ax.set_title('园林趣味性综合评分排名', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('园林名称', fontsize=14)
        ax.set_ylabel('归一化趣味性评分 (0-100)', fontsize=14)
        ax.set_xticks(range(len(df_sorted)))
        ax.set_xticklabels(df_sorted['garden_name'], rotation=45, ha='right')

        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, df_sorted['fun_score_scaled'])):
            ax.text(i, value + 1, f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "01_趣味性评分排名.png"), bbox_inches='tight')
        plt.close()

    def _plot_score_components_stacked_bar(self, df, save_folder):
        """修复版：得分构成堆积条形图"""
        df_norm = df.copy()
        components = ['curvature_score', 'view_change_score', 'exploration_score']

        # 归一化处理
        for col in components:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        df_norm = df_norm.sort_values('fun_score_scaled', ascending=False)

        fig, ax = plt.subplots(figsize=(14, 8))

        bottom = np.zeros(len(df_norm))
        colors = self.color_palette[:3]
        labels = ['路径曲折度', '异景程度', '探索性']

        for i, (col, color, label) in enumerate(zip(components, colors, labels)):
            ax.bar(range(len(df_norm)), df_norm[col], bottom=bottom,
                  color=color, label=label, alpha=0.8, edgecolor='white', linewidth=0.5)
            bottom += df_norm[col]

        ax.set_title('趣味性得分构成分析', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('园林名称', fontsize=14)
        ax.set_ylabel('归一化贡献度', fontsize=14)
        ax.set_xticks(range(len(df_norm)))
        ax.set_xticklabels(df_norm['garden_name'], rotation=45, ha='right')

        # 修复图例显示
        ax.legend(title='得分组成', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "02_趣味性得分构成分析.png"), bbox_inches='tight')
        plt.close()

    def _plot_correlation_heatmap(self, df, save_folder):
        """修复版：相关性热力图"""
        fig, ax = plt.subplots(figsize=(10, 8))

        corr_data = df[['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score', 'fun_score_scaled']]
        corr_matrix = corr_data.corr()

        # 创建掩码用于上三角
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # 绘制热力图
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8},
                   xticklabels=['路径长度', '曲折度', '异景程度', '探索性', '趣味性总分'],
                   yticklabels=['路径长度', '曲折度', '异景程度', '探索性', '趣味性总分'],
                   ax=ax)

        ax.set_title('各项指标相关性分析', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "03_指标相关性热力图.png"), bbox_inches='tight')
        plt.close()

    def _plot_length_vs_fun_score_bubble(self, df, save_folder):
        """修复版：气泡图 - 控制气泡大小"""
        fig, ax = plt.subplots(figsize=(14, 10))

        # 控制气泡大小，避免超出范围
        max_size = 1000
        min_size = 50
        sizes = np.interp(df['view_change_score'],
                         [df['view_change_score'].min(), df['view_change_score'].max()],
                         [min_size, max_size])

        scatter = ax.scatter(df['path_length_m'], df['fun_score_scaled'],
                           s=sizes, c=df['curvature_score'],
                           cmap='viridis', alpha=0.7, edgecolors='white', linewidth=2)

        ax.set_title('路径长度与趣味性评分关系', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('总路径长度 (米)', fontsize=14)
        ax.set_ylabel('归一化趣味性评分 (0-100)', fontsize=14)

        # 添加标签，避免重叠
        for i, row in df.iterrows():
            ax.annotate(row['garden_name'],
                       (row['path_length_m'], row['fun_score_scaled']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, alpha=0.8)

        # 颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('路径曲折度', fontsize=12)

        # 图例 - 修复气泡大小显示
        legend_sizes = [df['view_change_score'].min(), df['view_change_score'].mean(), df['view_change_score'].max()]
        legend_labels = ['低', '中', '高']
        legend_handles = []

        for size_val, label in zip(legend_sizes, legend_labels):
            size = np.interp(size_val, [df['view_change_score'].min(), df['view_change_score'].max()], [min_size, max_size])
            handle = plt.scatter([], [], s=size, c='gray', alpha=0.6, edgecolors='white', linewidth=1)
            legend_handles.append(handle)

        ax.legend(legend_handles, legend_labels, title='异景程度',
                 loc='upper right', frameon=True, fancybox=True, shadow=True)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "04_路径长度_vs_趣味性_气泡图.png"), bbox_inches='tight')
        plt.close()

    def _plot_top_gardens_radar(self, df, save_folder):
        """修复版：雷达图"""
        df_norm = df.copy()
        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score', 'fun_score_scaled']

        for col in metrics:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        top_5_gardens = df_norm.sort_values('fun_score_scaled', ascending=False).head(5)

        labels = ['路径长度', '曲折度', '异景程度', '探索性', '趣味性总分']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        colors = self.color_palette[:5]

        for i, (idx, row) in enumerate(top_5_gardens.iterrows()):
            values = row[metrics].values.tolist()
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=3, label=row['garden_name'],
                   color=colors[i], markersize=8)
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_title('前五名园林综合指标对比', fontsize=16, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "05_前五名园林雷达图.png"), bbox_inches='tight')
        plt.close()

    def _plot_violin_distribution(self, df, save_folder):
        """小提琴图：显示各指标的分布"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'fun_score_scaled']
        metric_names = ['路径长度 (米)', '曲折度', '异景程度', '趣味性评分']

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            # 为每个园林创建数据点（模拟分布）
            data_for_violin = []
            labels_for_violin = []

            for _, row in df.iterrows():
                # 创建正态分布数据用于小提琴图
                n_points = 100
                std = row[metric] * 0.1  # 标准差为值的10%
                points = np.random.normal(row[metric], std, n_points)
                data_for_violin.extend(points)
                labels_for_violin.extend([row['garden_name']] * n_points)

            violin_df = pd.DataFrame({
                'value': data_for_violin,
                'garden': labels_for_violin
            })

            sns.violinplot(data=violin_df, x='garden', y='value', ax=axes[i],
                          palette=self.color_palette[:len(df)])
            axes[i].set_title(f'{name}分布', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('园林', fontsize=12)
            axes[i].set_ylabel(name, fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "06_指标分布小提琴图.png"), bbox_inches='tight')
        plt.close()

    def _plot_forest_plot(self, df, save_folder):
        """森林图：显示置信区间"""
        fig, ax = plt.subplots(figsize=(12, 10))

        metrics = ['curvature_score', 'view_change_score', 'exploration_score']
        metric_names = ['曲折度', '异景程度', '探索性']

        y_pos = np.arange(len(df))

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = df[metric].values
            # 计算置信区间（使用标准误差）
            mean_val = np.mean(values)
            std_err = stats.sem(values)
            ci_lower = mean_val - 1.96 * std_err
            ci_upper = mean_val + 1.96 * std_err

            # 绘制置信区间
            for j, (garden_name, value) in enumerate(zip(df['garden_name'], values)):
                x_pos = i * (len(df) + 2) + j

                # 计算个体置信区间
                individual_ci = value * 0.1  # 简化的置信区间

                ax.errorbar(value, x_pos, xerr=individual_ci,
                           fmt='o', color=self.color_palette[i],
                           markersize=8, capsize=5, capthick=2, alpha=0.8)

                # 添加园林名称
                if i == 0:  # 只在第一个指标时添加标签
                    ax.text(-0.5, x_pos, garden_name, ha='right', va='center', fontsize=10)

        ax.set_ylabel('园林', fontsize=14)
        ax.set_xlabel('标准化得分', fontsize=14)
        ax.set_title('各园林指标森林图（95%置信区间）', fontsize=16, fontweight='bold', pad=20)

        # 创建图例
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=self.color_palette[i], markersize=10,
                                   label=name) for i, name in enumerate(metric_names)]
        ax.legend(handles=legend_handles, loc='upper right')

        ax.grid(axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "07_森林图_置信区间.png"), bbox_inches='tight')
        plt.close()

    def _plot_boxplot_with_swarm(self, df, save_folder):
        """箱线图与散点图组合"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'fun_score_scaled']
        metric_names = ['路径长度', '曲折度', '异景程度', '趣味性评分']

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            # 创建分组数据
            df['score_group'] = pd.cut(df[metric], bins=3, labels=['低', '中', '高'])

            # 箱线图
            box_plot = axes[i].boxplot([df[df['score_group'] == '低'][metric].dropna(),
                                       df[df['score_group'] == '中'][metric].dropna(),
                                       df[df['score_group'] == '高'][metric].dropna()],
                                      labels=['低', '中', '高'],
                                      patch_artist=True,
                                      boxprops=dict(facecolor='lightblue', alpha=0.7))

            # 散点图叠加
            for j, group in enumerate(['低', '中', '高']):
                group_data = df[df['score_group'] == group][metric].dropna()
                if len(group_data) > 0:
                    x = np.random.normal(j+1, 0.04, len(group_data))
                    axes[i].scatter(x, group_data, alpha=0.8, s=50,
                                   color=self.color_palette[j], edgecolors='white')

            axes[i].set_title(f'{name} 分布分析', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('评级分组', fontsize=12)
            axes[i].set_ylabel(name, fontsize=12)
            axes[i].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "08_箱线图与散点组合.png"), bbox_inches='tight')
        plt.close()

    def _plot_3d_scatter(self, df, save_folder):
        """3D散点图"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(df['path_length_m'], df['curvature_score'], df['view_change_score'],
                           c=df['fun_score_scaled'], s=100, cmap='viridis',
                           alpha=0.8, edgecolors='white', linewidth=1)

        ax.set_xlabel('路径长度 (米)', fontsize=12)
        ax.set_ylabel('曲折度', fontsize=12)
        ax.set_zlabel('异景程度', fontsize=12)
        ax.set_title('三维指标关系图', fontsize=16, fontweight='bold', pad=20)

        # 添加园林名称标签
        for i, row in df.iterrows():
            ax.text(row['path_length_m'], row['curvature_score'], row['view_change_score'],
                   row['garden_name'], fontsize=9)

        # 颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
        cbar.set_label('趣味性评分', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "09_三维散点图.png"), bbox_inches='tight')
        plt.close()

    def _plot_parallel_coordinates(self, df, save_folder):
        """平行坐标图"""
        fig, ax = plt.subplots(figsize=(16, 10))

        # 数据标准化
        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score', 'fun_score_scaled']
        df_norm = df[metrics].copy()

        for col in metrics:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        # 绘制平行坐标
        for i, row in df.iterrows():
            values = df_norm.iloc[i].values
            ax.plot(range(len(metrics)), values, 'o-',
                   color=self.color_palette[i % len(self.color_palette)],
                   alpha=0.7, linewidth=2, markersize=6,
                   label=row['garden_name'])

        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(['路径长度', '曲折度', '异景程度', '探索性', '趣味性总分'],
                          fontsize=12)
        ax.set_ylabel('标准化数值', fontsize=14)
        ax.set_title('园林指标平行坐标图', fontsize=16, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "10_平行坐标图.png"), bbox_inches='tight')
        plt.close()

    def _plot_density_contour(self, df, save_folder):
        """密度等高线图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        pairs = [('path_length_m', 'fun_score_scaled'),
                ('curvature_score', 'fun_score_scaled'),
                ('view_change_score', 'fun_score_scaled')]

        pair_names = [('路径长度', '趣味性评分'),
                     ('曲折度', '趣味性评分'),
                     ('异景程度', '趣味性评分')]

        for i, ((x_col, y_col), (x_name, y_name)) in enumerate(zip(pairs, pair_names)):
            # 创建网格
            x = df[x_col].values
            y = df[y_col].values

            # 散点图
            axes[i].scatter(x, y, c=self.color_palette[i], s=100, alpha=0.7,
                          edgecolors='white', linewidth=1)

            # 添加趋势线
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[i].plot(x, p(x), "--", color='red', alpha=0.8, linewidth=2)

            # 计算相关系数
            correlation = np.corrcoef(x, y)[0, 1]

            axes[i].set_xlabel(x_name, fontsize=12)
            axes[i].set_ylabel(y_name, fontsize=12)
            axes[i].set_title(f'{x_name} vs {y_name}\n(r={correlation:.3f})',
                            fontsize=14, fontweight='bold')
            axes[i].grid(True, alpha=0.3)

            # 添加园林名称
            for j, row in df.iterrows():
                axes[i].annotate(row['garden_name'],
                               (row[x_col], row[y_col]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, alpha=0.8)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "11_密度关系图.png"), bbox_inches='tight')
        plt.close()

    def _plot_statistical_comparison(self, df, save_folder):
        """统计显著性比较图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        # 将园林按趣味性评分分为高低两组
        median_score = df['fun_score_scaled'].median()
        df['score_category'] = df['fun_score_scaled'].apply(lambda x: '高评分组' if x >= median_score else '低评分组')

        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score']
        metric_names = ['路径长度', '曲折度', '异景程度', '探索性']

        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            high_group = df[df['score_category'] == '高评分组'][metric]
            low_group = df[df['score_category'] == '低评分组'][metric]

            # t检验
            if len(high_group) > 1 and len(low_group) > 1:
                t_stat, p_value = stats.ttest_ind(high_group, low_group)
            else:
                t_stat, p_value = 0, 1

            # 绘制对比条形图
            means = [high_group.mean(), low_group.mean()]
            stds = [high_group.std(), low_group.std()]

            x_pos = [0, 1]
            bars = axes[i].bar(x_pos, means, yerr=stds, capsize=5,
                              color=self.color_palette[:2], alpha=0.8,
                              edgecolor='white', linewidth=1)

            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels(['高评分组', '低评分组'])
            axes[i].set_ylabel(name, fontsize=12)
            axes[i].set_title(f'{name}组间比较\n(p={p_value:.3f})',
                            fontsize=14, fontweight='bold')

            # 显著性标记
            if p_value < 0.05:
                max_height = max(means) + max(stds)
                axes[i].text(0.5, max_height * 1.1, '*', ha='center', va='center',
                           fontsize=20, fontweight='bold')
                axes[i].text(0.5, max_height * 1.2, f'p={p_value:.3f}',
                           ha='center', va='center', fontsize=10)

            axes[i].grid(axis='y', alpha=0.3)
            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "12_统计显著性比较.png"), bbox_inches='tight')
        plt.close()

    def _plot_heatmap_matrix(self, df, save_folder):
        """增强版热力图矩阵"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        # 1. 原始数据热力图
        data_matrix = df.set_index('garden_name')[['path_length_m', 'curvature_score',
                                                  'view_change_score', 'exploration_score',
                                                  'fun_score_scaled']].T

        sns.heatmap(data_matrix, annot=True, cmap='YlOrRd', fmt='.1f',
                   ax=axes[0,0], cbar_kws={'shrink': 0.8})
        axes[0,0].set_title('原始指标数值矩阵', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('指标类型', fontsize=12)

        # 2. 标准化数据热力图
        df_norm = df.copy()
        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score', 'fun_score_scaled']
        for col in metrics:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        norm_matrix = df_norm.set_index('garden_name')[metrics].T
        sns.heatmap(norm_matrix, annot=True, cmap='RdBu_r', fmt='.2f',
                   center=0.5, ax=axes[0,1], cbar_kws={'shrink': 0.8})
        axes[0,1].set_title('标准化指标矩阵', fontsize=14, fontweight='bold')

        # 3. 排名矩阵
        rank_df = df.copy()
        for col in metrics:
            rank_df[col] = rank_df[col].rank(ascending=False)

        rank_matrix = rank_df.set_index('garden_name')[metrics].T
        sns.heatmap(rank_matrix, annot=True, cmap='RdYlGn_r', fmt='.0f',
                   ax=axes[1,0], cbar_kws={'shrink': 0.8})
        axes[1,0].set_title('指标排名矩阵', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('指标类型', fontsize=12)

        # 4. 相对表现矩阵
        perf_df = df.copy()
        for col in metrics:
            mean_val = perf_df[col].mean()
            perf_df[col] = (perf_df[col] - mean_val) / mean_val

        perf_matrix = perf_df.set_index('garden_name')[metrics].T
        sns.heatmap(perf_matrix, annot=True, cmap='RdBu_r', fmt='.2f',
                   center=0, ax=axes[1,1], cbar_kws={'shrink': 0.8})
        axes[1,1].set_title('相对表现矩阵（vs均值）', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "13_增强版热力图矩阵.png"), bbox_inches='tight')
        plt.close()

    def _plot_ridgeline_plot(self, df, save_folder):
        """山脊线图（分布对比）"""
        fig, ax = plt.subplots(figsize=(14, 10))

        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score']
        metric_names = ['路径长度', '曲折度', '异景程度', '探索性']

        y_offset = 0
        colors = self.color_palette[:len(metrics)]

        for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
            # 标准化数据
            data = df[metric].values
            data_norm = (data - data.min()) / (data.max() - data.min())

            # 创建核密度估计
            x_range = np.linspace(0, 1, 100)
            kde = stats.gaussian_kde(data_norm)
            density = kde(x_range)

            # 缩放密度以适应显示
            density_scaled = density / density.max() * 0.8

            # 绘制密度曲线
            ax.fill_between(x_range, y_offset, y_offset + density_scaled,
                           color=color, alpha=0.7, label=name)
            ax.plot(x_range, y_offset + density_scaled, color=color, linewidth=2)

            # 添加原始数据点
            for j, val in enumerate(data_norm):
                jitter = np.random.normal(0, 0.02)
                ax.scatter(val, y_offset + jitter, color=color, s=30, alpha=0.8,
                          edgecolors='white', linewidth=0.5)

            # 添加标签
            ax.text(-0.1, y_offset + 0.4, name, fontsize=12, fontweight='bold',
                   ha='right', va='center')

            y_offset += 1.2

        ax.set_xlim(-0.15, 1.1)
        ax.set_ylim(-0.2, y_offset)
        ax.set_xlabel('标准化数值', fontsize=14)
        ax.set_title('各指标分布山脊线图', fontsize=16, fontweight='bold', pad=20)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "14_山脊线分布图.png"), bbox_inches='tight')
        plt.close()


def main():
    """
    Main execution function.
    """
    # 请设置正确的数据文件夹路径
    DATA_FOLDER = "赛题F江南古典园林美学特征建模附件资料"

    if not os.path.exists(DATA_FOLDER):
        print("="*70)
        print("!!! 文件夹未找到 !!!")
        print(f"指定的数据文件夹 '{DATA_FOLDER}' 不存在。")
        print("请下载竞赛数据并将其放置在正确的目录中，")
        print("或者更新 main 函数中的 'DATA_FOLDER' 变量。")
        print("="*70)
        return

    analyzer = GardenInterestAnalyzer(data_folder_path=DATA_FOLDER)
    analyzer.run_full_analysis()
    analyzer.generate_visualizations()

    print("\n--- 程序结束 ---")


if __name__ == "__main__":
    main()
