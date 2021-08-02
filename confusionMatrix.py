import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import plot_confusion_matrix,confusion_matrix

def printConfusionMatrix(y_pred, y_ground):
    displayLabels = ['0','1','2','3','4']
    matrix = confusion_matrix(y_ground, y_pred, normalize='true')
    df_cm = pd.DataFrame(matrix, displayLabels,
                         displayLabels)

    plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="f", cmap='Blues')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label (Ground Truth)")
    plt.title("Normalized confusion matrix")
    plt.show()