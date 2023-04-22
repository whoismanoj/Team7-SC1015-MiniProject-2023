import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from text_processing import TextProcessing
from image_processing import ImageProcessing
from sentiment_prediction import predict_sentiment

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'

    text_data_file_path = './Dataset/LabeledText.xlsx'
    image_dir = './Dataset/image'

    text_processing = TextProcessing(text_data_file_path)
    image_processing = ImageProcessing(image_dir)

    if mode == 'train':
        # Load the dataset
        data = pd.read_excel(text_data_file_path)

        # Text Processing
        data['Caption'] = data['Caption'].apply(text_processing.preprocess_text)

        # Split Text Data
        X_train_preprocessed, X_test_preprocessed, y_train, y_test = text_processing.split_data(data['Caption'], data['LABEL'])

        # Train Text Classifier
        text_processing.train_model(X_train_preprocessed, y_train)
        y_pred = text_processing.predict(X_test_preprocessed)

        # Evaluate Text Classifier
        print(classification_report(y_test, y_pred))

        # Image Processing
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

    elif mode == 'production':
        # Load the text model
        text_processing.load_model()

        # Load the image model
        image_processing.build_model()
        image_processing.compile_model()
        image_processing.load_weights_if_exists()

        # Predict the sentiment using input text and images
        input_text = "My cat is soooo cute!"
        image_files = ["sample_pictures/cat_1.jpg", "sample_pictures/cat_2.jpg"]  # Provide the list of image paths

        sentiment = predict_sentiment(input_text, image_files, text_processing, image_processing)
        print("Sentiment:", sentiment)

    else:
        print(f"Unknown mode '{mode}'. Please use either 'train' or 'production'.")

if __name__ == "__main__":
    main()
