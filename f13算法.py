import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# --- Matplotlib Configuration for Chinese Characters ---
# This setup ensures that Chinese characters are displayed correctly in the plots.
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 120
plt.style.use('seaborn-v0_8-whitegrid')


class GardenInterestAnalyzer:
    """
    A comprehensive analyzer for modeling the "Interest Level" of classical Chinese gardens
    based on the concept of "Shifting Scenery" (移步异景).

    This class implements the custom definitions provided in the user's .tex file:
    1.  **Path Formalization**: Models garden paths to calculate length and curvature.
    2.  **View Change (异景程度)**: Quantifies how the scenery changes as one walks along a path
        by simulating a "viewshed" and calculating the symmetric difference between views at
        consecutive points.
    3.  **Exploration (探索性)**: Measures the "maze-like" quality of a path by considering the
        number of choices (degree) at intersections.
    4.  **Fun Score (趣味性评分)**: A composite score based on the weighted combination of
        View Change, Curvature, and Exploration, penalized by the path length.
    5.  **Enhanced Visualization**: Generates a suite of unique and informative charts to
        present the analysis results.
    """

    def __init__(self, data_folder_path):
        """
        Initializes the analyzer.

        Args:
            data_folder_path (str): The path to the main folder containing the garden data.
                                    Example: "赛题F江南古典园林美学特征建模附件资料"
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
        print(f"Garden Interest Analyzer initialized. Data folder set to: '{data_folder_path}'")

    def load_garden_data(self, garden_id):
        """
        Loads the coordinate data for a single garden from its corresponding Excel file.

        Args:
            garden_id (int): The ID of the garden (1-10).

        Returns:
            bool: True if data was loaded successfully, False otherwise.
        """
        garden_name = self.gardens[garden_id]
        # Construct the file path dynamically based on the provided data folder structure
        data_path = os.path.join(self.data_folder_path, f"{garden_id}. {garden_name}", f"4-{garden_name}数据坐标.xlsx")

        if not os.path.exists(data_path):
            print(f"✗ Error: Data file not found for {garden_name} at '{data_path}'")
            return False

        try:
            excel_file = pd.ExcelFile(data_path)
            garden_info = {'name': garden_name, 'id': garden_id, 'elements': {}}

            for sheet_name in excel_file.sheet_names:
                # Use element name from the sheet name for robustness
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
        This function handles the specific '{x, y, z}' string format.

        Args:
            df (DataFrame): The DataFrame containing coordinate data.

        Returns:
            list: A list of (x, y) tuples.
        """
        coords = []
        if len(df.columns) < 2:
            return coords

        # Use the second column which contains non-segmented coordinates
        coord_col_data = df.iloc[:, 1].dropna()

        for item in coord_col_data:
            try:
                # Clean and parse the string format like '{x, y, z}'
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
        It finds all landscape elements within a certain radius of the point.

        Args:
            point (tuple): The (x, y) coordinates of the viewpoint.
            kdtree (cKDTree): A k-d tree built from the coordinates of all landscape elements.
            view_radius (float): The radius (in mm) to consider for the viewshed.

        Returns:
            set: A set of indices representing the landscape elements visible from the point.
        """
        # query_ball_point returns indices of all points within the radius
        indices = kdtree.query_ball_point(point, r=view_radius)
        return set(indices)

    def analyze_garden(self, garden_id, turn_angle_threshold=20, path_sample_dist=5000, view_radius=50000):
        """
        Performs a full analysis on a single garden based on the custom formulas.

        Args:
            garden_id (int): The ID of the garden.
            turn_angle_threshold (float): The angle (in degrees) to define a significant turn.
            path_sample_dist (float): The distance (in mm) for sampling points along the path for view analysis.
            view_radius (float): The radius (in mm) for the simulated viewshed.

        Returns:
            dict: A dictionary containing all calculated metrics for the garden.
        """
        if garden_id not in self.garden_data:
            self.load_garden_data(garden_id)
        if garden_id not in self.garden_data:
            return None

        garden = self.garden_data[garden_id]

        # 1. Path Data Extraction
        path_coords = self._extract_coordinates_from_df(garden['elements'].get('道路', pd.DataFrame()))
        if len(path_coords) < 3:
            print(f"|  -> Skipping {garden['name']} due to insufficient path data.")
            return None

        # 2. Path Length (L_len) and Curvature (L_curv) Calculation
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

        # 3. View Change (L_view) Calculation
        landscape_elements = []
        for name, df in garden['elements'].items():
            if name not in ['道路', '植物']:
                landscape_elements.extend(self._extract_coordinates_from_df(df))

        # Add plant centers as landscape elements
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
                    # Symmetric difference: (A - B) U (B - A)
                    view_diff = len(current_viewshed.symmetric_difference(last_viewshed))
                    total_view_change += view_diff
                    last_viewshed = current_viewshed

        # 4. Exploration (L_exp) Calculation
        # Simplified: Count high-degree nodes. A turn point represents a node of at least degree 2.
        # We approximate exploration score as the number of turns, as each turn offers a new direction.
        exploration_score = turn_points_count

        # 5. Fun Score (F(L)) Calculation
        # F(L) = (w_curv * L_curv + w_view * L_view + w_exp * L_exp) / (w_len * L_len + C)
        w_curv, w_view, w_exp, w_len = 0.4, 0.4, 0.2, 0.1
        C = 1e-6 # Small constant to prevent division by zero

        numerator = (w_curv * turn_points_count +
                     w_view * total_view_change +
                     w_exp * exploration_score)
        denominator = w_len * (path_length / 1000) + C # Normalize length to meters

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

        # Normalize the fun score for better comparison (0-100 scale)
        if not self.results_df.empty and 'fun_score' in self.results_df.columns:
            scaler = MinMaxScaler(feature_range=(0, 100))
            self.results_df['fun_score_scaled'] = scaler.fit_transform(self.results_df[['fun_score']])

        print("\n" + "="*60)
        print("Full Analysis Complete.")
        print("="*60)
        print(self.results_df.sort_values('fun_score_scaled', ascending=False))

    def generate_visualizations(self, save_folder="问题1_新版趣味性建模图表"):
        """
        Generates and saves a suite of custom visualizations.
        """
        if not hasattr(self, 'results_df') or self.results_df.empty:
            print("✗ No analysis results to visualize. Please run `run_full_analysis()` first.")
            return

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        print(f"\nGenerating visualizations in folder: '{save_folder}'")

        df = self.results_df.copy()

        # Viz 1: Fun Score Ranking Bar Chart
        self._plot_fun_score_ranking(df, save_folder)
        # Viz 2: Fun Score Components Breakdown
        self._plot_score_components_stacked_bar(df, save_folder)
        # Viz 3: Correlation Heatmap of All Metrics
        self._plot_correlation_heatmap(df, save_folder)
        # Viz 4: Path Length vs. Fun Score (with component bubbles)
        self._plot_length_vs_fun_score_bubble(df, save_folder)
        # Viz 5: Radar Chart for Top Gardens
        self._plot_top_gardens_radar(df, save_folder)
        # Viz 6: Paired Plot for Detailed Metric Comparison
        self._plot_paired_comparison(df, save_folder)

        print(f"\n✓ All visualizations have been successfully generated and saved to '{save_folder}'.")

    def _plot_fun_score_ranking(self, df, save_folder):
        """Viz 1: Bar chart showing the final scaled 'Fun Score' for each garden."""
        plt.figure(figsize=(16, 9))
        df_sorted = df.sort_values('fun_score_scaled', ascending=False)

        palette = sns.color_palette("viridis", n_colors=len(df_sorted))
        bars = plt.bar(df_sorted['garden_name'], df_sorted['fun_score_scaled'], color=palette)

        plt.title('园林“趣味性”综合评分排名 (根据您的自定义公式)', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('园林名称', fontsize=14, labelpad=10)
        plt.ylabel('归一化趣味性评分 (0-100)', fontsize=14, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "1_趣味性评分排名.png"))
        plt.close()

    def _plot_score_components_stacked_bar(self, df, save_folder):
        """Viz 2: Stacked bar chart showing the contribution of each component to the 'Fun Score'."""
        df_norm = df.copy()
        # Normalize components to show their relative contribution
        for col in ['curvature_score', 'view_change_score', 'exploration_score']:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        df_norm = df_norm.sort_values('fun_score_scaled', ascending=False)

        plt.figure(figsize=(16, 10))
        df_norm.plot(x='garden_name', y=['curvature_score', 'view_change_score', 'exploration_score'],
                     kind='bar', stacked=True, figsize=(16,10),
                     colormap='plasma', width=0.8)

        plt.title('趣味性得分构成分析 (各组成部分贡献度)', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('园林名称', fontsize=14, labelpad=10)
        plt.ylabel('归一化贡献度', fontsize=14, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(['路径曲折度 (L_curv)', '异景程度 (L_view)', '探索性 (L_exp)'], title='得分来源', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "2_趣味性得分构成分析.png"))
        plt.close()

    def _plot_correlation_heatmap(self, df, save_folder):
        """Viz 3: Heatmap showing the correlation between all calculated metrics."""
        plt.figure(figsize=(12, 10))
        corr_df = df[['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score', 'fun_score_scaled']].corr()

        # Custom labels for the heatmap
        labels = ['路径长度', '曲折度', '异景程度', '探索性', '趣味性总分']

        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, xticklabels=labels, yticklabels=labels, annot_kws={"size": 12})

        plt.title('各项指标之间的相关性分析', fontsize=20, fontweight='bold', pad=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "3_指标相关性热力图.png"))
        plt.close()

    def _plot_length_vs_fun_score_bubble(self, df, save_folder):
        """Viz 4: Bubble chart showing Path Length vs. Fun Score, with bubble size representing View Change."""
        plt.figure(figsize=(16, 9))

        sizes = df['view_change_score'] * 2 # Scale bubble size

        scatter = plt.scatter(df['path_length_m'], df['fun_score_scaled'],
                              s=sizes, c=df['curvature_score'],
                              cmap='viridis', alpha=0.7, edgecolors="w", linewidth=0.5)

        plt.title('路径长度 vs 趣味性评分', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('总路径长度 (米)', fontsize=14, labelpad=10)
        plt.ylabel('归一化趣味性评分 (0-100)', fontsize=14, labelpad=10)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add labels for each point
        for i, row in df.iterrows():
            plt.text(row['path_length_m'], row['fun_score_scaled'] + 1, row['garden_name'],
                     ha='center', va='bottom', fontsize=10)

        # Create a colorbar and a legend for bubble size
        cbar = plt.colorbar(scatter)
        cbar.set_label('路径曲折度', fontsize=12)

        # Legend for bubble size
        for size_val in [df['view_change_score'].min(), df['view_change_score'].mean(), df['view_change_score'].max()]:
            plt.scatter([], [], s=size_val * 2, c='k', alpha=0.5, label=f'{int(size_val)}')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=2, title='异景程度 (气泡大小)', loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "4_路径长度_vs_趣味性_气泡图.png"))
        plt.close()

    def _plot_top_gardens_radar(self, df, save_folder):
        """Viz 5: Radar chart comparing the metrics for the top 5 gardens."""
        df_norm = df.copy()
        metrics = ['path_length_m', 'curvature_score', 'view_change_score', 'exploration_score', 'fun_score_scaled']
        for col in metrics:
            df_norm[col] = (df_norm[col] - df_norm[col].min()) / (df_norm[col].max() - df_norm[col].min())

        top_5_gardens = df_norm.sort_values('fun_score_scaled', ascending=False).head(5)

        labels = np.array(['路径长度', '曲折度', '异景程度', '探索性', '趣味性总分'])
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

        for i, row in top_5_gardens.iterrows():
            values = row[metrics].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=row['garden_name'])
            ax.fill(angles, values, alpha=0.1)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        plt.title('趣味性排名前五园林各项指标对比', fontsize=20, fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "5_前五名园林雷达图.png"))
        plt.close()

    def _plot_paired_comparison(self, df, save_folder):
        """Viz 6: Seaborn pairplot for detailed comparison between metrics."""
        plt.figure(figsize=(12, 12))

        plot_df = df[['garden_name', 'path_length_m', 'curvature_score', 'view_change_score', 'fun_score_scaled']].copy()
        plot_df.columns = ['园林', '路径长度', '曲折度', '异景程度', '趣味性总分']

        pair_plot = sns.pairplot(plot_df, hue='园林', palette='tab10', corner=True)
        pair_plot.fig.suptitle('各项关键指标配对关系图', y=1.02, fontsize=20, fontweight='bold')

        plt.savefig(os.path.join(save_folder, "6_指标配对关系图.png"))
        plt.close()


def main():
    """
    Main execution function.
    """
    # --- IMPORTANT ---
    # Please set the correct path to your data folder here.
    # The folder should be the one containing "1. 拙政园", "2. 留园", etc.
    DATA_FOLDER = "赛题F江南古典园林美学特征建模附件资料"

    if not os.path.exists(DATA_FOLDER):
        print("="*70)
        print("!!! FOLDER NOT FOUND !!!")
        print(f"The specified data folder '{DATA_FOLDER}' does not exist.")
        print("Please download the competition data and place it in the correct directory,")
        print("or update the 'DATA_FOLDER' variable in the `main` function.")
        print("="*70)
        return

    analyzer = GardenInterestAnalyzer(data_folder_path=DATA_FOLDER)
    analyzer.run_full_analysis()
    analyzer.generate_visualizations()

    print("\n--- End of Program ---")


if __name__ == "__main__":
    main()
