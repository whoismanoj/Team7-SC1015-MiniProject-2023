# SC1015-MiniProject-2023

# Sentiment Analysis with Text and Image

This project combines text and image-based sentiment analysis to predict sentiment labels for a given text and set of images. It uses Natural Language Processing (NLP) techniques for text-based sentiment analysis and a Convolutional Neural Network (CNN) for image-based sentiment analysis. The project is designed with flexibility in mind, allowing users to run it in training, testing, or production modes.

## Installation

1. Clone the repository to your local machine:
    ```
    git clone https://github.com/YANG-U2120594G/SC1015-MiniProject-2023.git
    ```
2. Install the required packages using pip:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Place your labeled text dataset in the `./Dataset` directory as `LabeledText.xlsx`.
2. Place your image dataset in the `./Dataset/image` directory.
3. Run the `main.py` file and select the desired mode (train, test, or production, default: production):
    ```
    python main.py --mode=train
    ```
4. If running in production mode, you can use the provided GUI to input text and upload images for sentiment prediction.

<img width="302" alt="image" src="https://user-images.githubusercontent.com/131555765/233835573-fa014438-b406-447c-9b31-565fdaa8fce9.png">

## License

This project is not licensed.

