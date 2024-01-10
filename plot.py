import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os

class Plotter:
    def __init__(self):
        self.plot_folder = "plots"
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def draw_plots(self, file_path):
        try:
            data = pd.read_json(file_path)
            plot_paths = []
            columns = data.columns

            # Confusion matrix
            true_labels = data['gt_corners']
            predicted_labels = data['rb_corners']

            cm = confusion_matrix(true_labels, predicted_labels)

            normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(normalized_cm, annot=True, cmap='Blues', fmt='g', 
                        xticklabels=['4.0', '6.0', '8.0', '10.0'],
                        yticklabels=['4.0', '6.0', '8.0', '10.0'])
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Normalized confusion Matrix')
            plot_path = f"{self.plot_folder}/confusion_matrix.png"
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
            
            # Floor vs Ceiling mean
            sns.scatterplot(x="floor_mean", y="ceiling_mean",
                        data=data,
                        size="mean")
            plt.xlabel('Floor Mean')
            plt.ylabel('Ceiling Mean')
            plt.title('Floor Mean vs Ceiling Mean')
            plot_path = f"{self.plot_folder}/floor_ceiling_mean.png"
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
            
            # Floor vs Ceiling min
            sns.scatterplot(x="floor_min", y="ceiling_min",
                        data=data,
                        size="min")
            plt.xlabel('Floor Min')
            plt.ylabel('Ceiling Min')
            plt.title('Floor Min vs Ceiling Min')
            plot_path = f"{self.plot_folder}/floor_ceiling_min.png"
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
            
            # Floor vs Ceiling max
            sns.scatterplot(x="floor_max", y="ceiling_max",
                        data=data,
                        size="max")
            plt.xlabel('Floor Max')
            plt.ylabel('Ceiling Max')
            plt.title('Floor Max vs Ceiling Max')
            plot_path = f"{self.plot_folder}/floor_ceiling_max.png"
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
            
            # Consistency and Precision of ceiling
            data['ceiling_range'] = data['ceiling_max'] - data['ceiling_min']
            
            sns.scatterplot(x="ceiling_mean", y="ceiling_range",
                        data=data)
            plt.xlabel('Ceiling Mean')
            plt.ylabel('Ceiling Min-Max range')
            plt.title('Ceiling consistency and precision plot')
            plot_path = f"{self.plot_folder}/consistency_ceiling.png"
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
            
            # Consistency and Precision of floor
            data['floor_range'] = data['floor_max'] - data['floor_min']
            
            sns.scatterplot(x="floor_mean", y="floor_range",
                        data=data)
            plt.xlabel('Floor Mean')
            plt.ylabel('Floor Min-Max range')
            plt.title('Floor consistency and precision plot')
            plot_path = f"{self.plot_folder}/consistency_floor.png"
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()

            return plot_paths

        except Exception as e:
            print(f"Error: {e}")
            return None

def main():
    plotter = Plotter()
    json_file_path = "deviation.json"
    paths = plotter.draw_plots(json_file_path)
    if paths:
        print("Plots created at:")
        for path in paths:
            print(path)
    else:
        print("Plot creation failed.")
        
if __name__ == "__main__":
    main()