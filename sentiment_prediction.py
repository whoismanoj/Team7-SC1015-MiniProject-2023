import numpy as np
from PIL import Image

def predict_sentiment(input_text, image_files, text_processing, image_processing):
    # Prepare the input text
    input_text_processed = text_processing.preprocess_text(input_text)
    text_prediction = text_processing.predict([input_text_processed], True)[0]

    if not image_files:
        return text_processing.label_to_onehot(text_prediction, True)[0]

    # Prepare the input images
    input_images = []
    for image_file in image_files:
        img = Image.open(image_file).convert("RGB")
        img = img.resize(image_processing.img_size, Image.LANCZOS)
        input_images.append(np.array(img))

    input_images = np.array(input_images)
    image_prediction = image_processing.predict(input_images, True)

    # Combine the predictions
    combined_prediction = 0.7 * text_prediction + 0.3 * np.mean(image_prediction, axis=0)

    # Get the final sentiment label
    final_label = text_processing.label_to_onehot(combined_prediction, True)[0]
    return final_label
