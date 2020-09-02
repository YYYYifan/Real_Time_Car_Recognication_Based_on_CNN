import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



data = np.asarray([0.9791666666666666, 0.9817708333333334, 0.9766839378238342, 0.9792207792207793])
name_list = ["Accuracy", "Precision", "Recall", "F1 Score"]
color_list = ["coral", "gold", "turquoise", "hotpink"]

plt.title("Model Evaluation")

plt.bar(range(len(data)), data, color=color_list, tick_label=name_list)
plt.ylim(0.972, 0.984)

def to_percent(temp, position):
    return '%1.1f' % (100*temp) + '%'

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(to_percent))

plt.savefig("./images/Model Evaluation.png", dpi=900,  bbox_inches='tight')
plt.show()