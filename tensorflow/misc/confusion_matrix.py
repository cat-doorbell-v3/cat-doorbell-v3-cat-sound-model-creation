import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Confusion matrix data
confusion_matrix = np.array([[22, 0, 0],
                             [2, 10, 6],
                             [0, 4, 54]])

# Labels for the classes
class_labels = ['Class 1', 'Class 2', 'Class 3']

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
