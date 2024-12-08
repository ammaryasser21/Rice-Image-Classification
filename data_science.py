import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from google.colab import drive
drive.mount('/content/drive')

# !ls "<your_path>/rice data"
# !unzip "<your_path>/rice data/archive (2).zip" -d "<your_path>/rice data/archive (2)"

"""## Define paths for dataset and split directories"""

dataset_path = "<your_path>/archive"
train_dir = "<your_path>/train"
test_dir = "<your_path>/test"
val_dir = "<your_path>/validation"


dataset_path ="https://www.kaggle.com/datasets/mbsoroush/rice-images-dataset"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

"""## Split images into train, test, and validation sets"""

rice_types = ["Arborio","Basmati","Ipsala", "Jasmine", "Karacadag"]

for rice_type in rice_types:
    rice_type_dir = os.path.join(dataset_path, rice_type)
    if os.path.isdir(rice_type_dir):
        images = os.listdir(rice_type_dir)

        train_images, temp_images = train_test_split(images, test_size=0.2, random_state=42)  # 80% for train
        val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)  # 10% for val, 10% for test

        os.makedirs(os.path.join(train_dir, rice_type), exist_ok=True)
        os.makedirs(os.path.join(val_dir, rice_type), exist_ok=True)
        os.makedirs(os.path.join(test_dir, rice_type), exist_ok=True)

        for img in train_images:
            src = os.path.join(rice_type_dir, img)
            dest = os.path.join(train_dir, rice_type, img)
            shutil.copy(src, dest)

        for img in val_images:
            src = os.path.join(rice_type_dir, img)
            dest = os.path.join(val_dir, rice_type, img)
            shutil.copy(src, dest)

        for img in test_images:
            src = os.path.join(rice_type_dir, img)
            dest = os.path.join(test_dir, rice_type, img)
            shutil.copy(src, dest)

"""# Data preprocessing"""

image_size = (224, 224)
batch_size = 32

data_gen_no_aug = ImageDataGenerator(rescale=1.0 / 255.0)

train_data = data_gen_no_aug.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = data_gen_no_aug.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = data_gen_no_aug.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

"""# Pretrained model: ResNet50"""

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=image_size + (3,))

for layer in base_model.layers:
    layer.trainable = False

global_pool = GlobalAveragePooling2D()(base_model.output)
fc1 = Dense(128, activation='relu')(global_pool)
dropout = Dropout(0.5)(fc1)
output = Dense(train_data.num_classes, activation='softmax')(dropout)

pretrained_model = Model(inputs=base_model.input, outputs=output)
pretrained_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
pretrained_model.summary()

history_pretrained = pretrained_model.fit(train_data, validation_data=val_data, epochs=2)

"""## Save the model"""

pretrained_model.save('<your_path>/archive/pretrained_model.h5')
pretrained_model=load_model('<your_path>/archive/pretrained_model.h5')

"""## Evaluate Pretrained Model"""

print("Evaluating Pretrained Model")
pretrained_eval = pretrained_model.evaluate(test_data)

print("Confusion Matrix for Pretrained Model")
predictions = pretrained_model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(test_data.class_indices.keys()),
            yticklabels=list(test_data.class_indices.keys()))
plt.title("Confusion Matrix - Pretrained Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("Classification Report for Pretrained Model")
print(classification_report(true_classes, predicted_classes, target_names=list(test_data.class_indices.keys())))

"""# Custom model"""

custom_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=image_size + (3,)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

custom_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
custom_model.summary()

history_custom = custom_model.fit(train_data, validation_data=val_data, epochs=2)

"""## Save the model"""

custom_model.save('<your_path>/archive/custom_model.h5')
custom_model =load_model('<your_path>/archive/custom_model.h5')

"""## Evaluate Custom Model"""

print("Evaluating Custom Model")
custom_eval = custom_model.evaluate(test_data)

print("Confusion Matrix for Custom Model")
predictions_custom = custom_model.predict(test_data)
predicted_classes_custom = np.argmax(predictions_custom, axis=1)
true_classes = test_data.classes

conf_matrix_custom = confusion_matrix(true_classes, predicted_classes_custom)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_custom, annot=True, fmt='d', cmap='Greens',
            xticklabels=list(test_data.class_indices.keys()),
            yticklabels=list(test_data.class_indices.keys()))
plt.title("Confusion Matrix - Custom Model")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("Classification Report for Custom Model")
print(classification_report(true_classes, predicted_classes_custom, target_names=list(test_data.class_indices.keys())))
