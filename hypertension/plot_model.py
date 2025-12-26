import tensorflow as tf
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, models  # type: ignore
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model  # type: ignore
from hypertensive_model import build_cnn_model, build_cnn_lstm_model

def plot_and_save_model(model: keras.Model, model_name: str, file_path: str) -> None:
    """
    Plots and saves the architecture of a Keras model to a file.

    Args:
        model (keras.Model): The Keras model to be plotted.
        model_name (str): The name of the model (for title purposes).
        file_path (str): The file path where the plot image will be saved.
    """
    # Plot the model architecture
    plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)

    # Display the model architecture
    img = plt.imread(file_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'{model_name} Architecture', fontsize=16)
    plt.show()

if __name__ == "__main__":
    # Build and plot CNN model
    cnn_model = build_cnn_model(input_shape=(60, 5, 1))
    plot_and_save_model(cnn_model, "CNN Model", "cnn_model_architecture.png")

    # Build and plot CNN-LSTM model
    cnn_lstm_model = build_cnn_lstm_model(input_shape=(60, 5, 1))
    plot_and_save_model(cnn_lstm_model, "CNN-LSTM Model", "cnn_lstm_model_architecture.png")