import tkinter as tk
from tkinter import filedialog
from sentiment_prediction import predict_sentiment

class App(tk.Tk):
    def __init__(self, text_processing, image_processing):
        super().__init__()
        self.title("Sentiment Prediction")
        self.geometry("400x300")
        self.text_processing = text_processing
        self.image_processing = image_processing
        self.image_files = []

        self.label = tk.Label(self, text="Enter the text:")
        self.label.pack(pady=10)

        self.input_text = tk.Entry(self)
        self.input_text.pack(pady=10)

        self.browse_button = tk.Button(self, text="Browse Images", command=self.browse_files)
        self.browse_button.pack(pady=10)

        self.predict_button = tk.Button(self, text="Predict Sentiment", command=self.predict)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(self, text="")
        self.result_label.pack(pady=10)

    def browse_files(self):
        self.image_files = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image files", "*.jpg;*.png")])
        if self.image_files:
            files_label_text = f"Selected {len(self.image_files)} image(s)"
        else:
            files_label_text = "No images selected"
        self.files_label = tk.Label(self, text=files_label_text)
        self.files_label.pack(pady=10)

    def predict(self):
        input_text = self.input_text.get()
        sentiment = predict_sentiment(input_text, self.image_files, self.text_processing, self.image_processing)
        self.result_label.config(text=f"Predicted sentiment: {sentiment}")

def run_gui(text_processing, image_processing):
    app = App(text_processing, image_processing)
    app.mainloop()
