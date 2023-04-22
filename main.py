import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from text_processing import TextProcessing
from image_processing import ImageProcessing

# Load the dataset
text_data_file_path = './Dataset/LabeledText.xlsx'
data = pd.read_excel(text_data_file_path)

# Text Processing
text_processing = TextProcessing(text_data_file_path)
data['Caption'] = data['Caption'].apply(text_processing.preprocess_text)

# Split Text Data
X_train_preprocessed, X_test_preprocessed, y_train, y_test = text_processing.split_data(data['Caption'], data['LABEL'])

# Train Text Classifier
text_processing.train_model(X_train_preprocessed, y_train)
y_pred = text_processing.predict(X_test_preprocessed)

# Evaluate Text Classifier
print(classification_report(y_test, y_pred))

# Image Processing
image_data_files_path = './Dataset/image'
image_processing = ImageProcessing(image_data_files_path)

# Load images and labels
X_img, y_img = image_processing.load_all_images()

# Split Image Data
X_train_img, X_test_img, y_train_img, y_test_img = image_processing.split_data(X_img, y_img)
y_train_img_onehot = image_processing.label_to_onehot(y_train_img)
y_test_img_onehot = image_processing.label_to_onehot(y_test_img)

# Train Image Classifier
image_processing.build_model()
image_processing.compile_model()
image_processing.load_weights_if_exists()

if not os.path.exists(image_processing.weights_path):    
    image_processing.train_model(X_train_img, y_train_img_onehot, X_test_img, y_test_img_onehot, epochs=4)

# Evaluate Image Classifier
y_pred_img_onehot = image_processing.model.predict(X_test_img)
y_pred_img = image_processing.label_to_onehot(y_pred_img_onehot, inverse=True)
print(classification_report(y_test_img, y_pred_img))
