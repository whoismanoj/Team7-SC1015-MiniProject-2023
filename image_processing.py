import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImageProcessing:

    def __init__(self, image_dir, img_size=(224, 224), weights_path='./model_weights/'):
        self.image_dir = image_dir
        self.img_size = img_size
        self.weights_path = weights_path
        self.model = None
        self.lb = None
        self.model_saved = None

    def model_exists(self):
        if self.model_saved is None:
            self.model_saved = os.path.exists(os.path.join(self.model_path, "model.json")) and os.path.exists(os.path.join(self.model_path, "model_weights.h5"))
        return self.model_saved
    
    @staticmethod
    def load_images(image_dir, label, img_size):
        img_list = []
        label_list = []
        
        for img_name in os.listdir(os.path.join(image_dir, label)):
            img_path = os.path.join(image_dir, label, img_name)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(img_size, Image.LANCZOS)
            img_list.append(np.array(img))
            label_list.append(label)
            
        return img_list, label_list

    def load_all_images(self):
        positive_images, positive_labels = self.load_images(self.image_dir, "Positive", self.img_size)
        neutral_images, neutral_labels = self.load_images(self.image_dir, "Neutral", self.img_size)
        negative_images, negative_labels = self.load_images(self.image_dir, "Negative", self.img_size)

        X_img = np.array(positive_images + neutral_images + negative_images)
        y_img = np.array(positive_labels + neutral_labels + negative_labels)

        return X_img, y_img

    @staticmethod
    def split_data(X_img, y_img):
        return train_test_split(X_img, y_img, test_size=0.2, random_state=42, stratify=y_img)

    def label_to_onehot(self, label, inverse=False):
        if self.lb is None:
            self.lb = LabelBinarizer()
        if inverse:
            return self.lb.inverse_transform(label)
        return self.lb.fit_transform(label)

    @staticmethod
    def configure_gpu():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def build_model(self):
        # Load EfficientNetB0 pre-trained model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(self.img_size[0], self.img_size[1], 3))

        # Add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # Add a fully-connected layer
        x = Dense(1024, activation='relu')(x)

        # Add a logistic layer with 3 classes (Negative, Neutral, Positive)
        predictions = Dense(3, activation='softmax')(x)

        # Create the final model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze the layers in the base model
        for layer in base_model.layers:
            layer.trainable = False
            
    def compile_model(self):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train_img, y_train_img_onehot, X_test_img, y_test_img_onehot, batch_size=16, epochs=5):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Data augmentation
        datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                     zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

        # Train the model with the augmented data
        self.model.fit(datagen.flow(X_train_img, y_train_img_onehot, batch_size=batch_size),
                       validation_data=(X_test_img, y_test_img_onehot), epochs=epochs, verbose=1)

        # Save the trained model weights
        self.model.save_weights(self.weights_path)
        print("Model weights saved to", self.weights_path)

    def load_weights_if_exists(self):
        if os.path.exists(self.weights_path):
            self.model.load_weights(self.weights_path)
            print("Loaded model weights from", self.weights_path)
        else:
            print("No saved model weights found.")

    def predict(self, X_test_img, prod=False):
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        y_pred_img_onehot = self.model.predict(X_test_img)
        
        if prod:
            return y_pred_img_onehot
        else:
            return self.label_to_onehot(y_pred_img_onehot, inverse=True)
