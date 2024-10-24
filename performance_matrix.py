import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import seaborn as sns

# Load the saved model
model = tf.keras.models.load_model('model/alzheimer_model.h5')

# Load the label encoder
class_indices = joblib.load('model/label_encoder.pkl')
class_labels = list(class_indices.keys())

# Define data parameters
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
data_dir = "C:/Users/shakt/Downloads/Data"

# ImageDataGenerator for loading validation data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=0.2)

validation_data = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(validation_data)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Predictions
predictions = model.predict(validation_data)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_data.classes

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Classification Report
class_report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)

# Calculate AUC for each class
y_true_binary = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
auc = roc_auc_score(y_true_binary, predictions, average="macro")
print(f"Model AUC: {auc:.4f}")

### Visualization Section ###

# 1. Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 2. Plot AUC Scores for Each Class
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(class_labels):
    class_auc = roc_auc_score(y_true_binary[:, i], predictions[:, i])
    plt.bar(class_label, class_auc)

plt.title('AUC Score for Each Class')
plt.xlabel('Class')
plt.ylabel('AUC Score')
plt.ylim([0, 1])
plt.show()

# 3. Plot Training Accuracy and Validation Accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Validation Accuracy'], [val_accuracy])
plt.ylim([0, 1])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.show()

# 4. Plot Training Loss and Validation Loss
plt.figure(figsize=(8, 6))
plt.bar(['Validation Loss'], [val_loss])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.show()

# 5. Plot Precision, Recall, and F1-Score for Each Class
precision = [class_report[label]['precision'] for label in class_labels]
recall = [class_report[label]['recall'] for label in class_labels]
f1_score = [class_report[label]['f1-score'] for label in class_labels]

x = np.arange(len(class_labels))  # Label locations
width = 0.2  # Bar width

fig, ax = plt.subplots(figsize=(10, 7))
rects1 = ax.bar(x - width, precision, width, label='Precision')
rects2 = ax.bar(x, recall, width, label='Recall')
rects3 = ax.bar(x + width, f1_score, width, label='F1-Score')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Classes')
ax.set_title('Precision, Recall, and F1-Score by Class')
ax.set_xticks(x)
ax.set_xticklabels(class_labels)
ax.legend()

# Attach text labels to bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.show()

# 6. Plot ROC Curve for Each Class (Corrected Code)
plt.figure(figsize=(8, 6))
for i, class_label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_true_binary[:, i], predictions[:, i])  # Corrected roc_curve usage
    plt.plot(fpr, tpr, label=f'{class_label} (AUC = {roc_auc_score(y_true_binary[:, i], predictions[:, i]):.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc="lower right")
plt.show()

