import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import datetime
import zipfile
import os


# function to load and resize the image
def load_and_prep_image(filename, img_shape=224, scale=True):
    """
    Reads in an image from filename, turns it into a tensor and reshapes into
    (224, 224, 3).
    Parameters
    ----------
    filename (str): string filename of target image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """

    # read the image
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, [img_shape, img_shape])
    if scale:
        return img / 255.
    else:
        return img


# function to predict an image and plot it.
def pred_and_plot(model, filename, class_names):
    """
    Imports an image located at filename, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.
    """
    img = load_and_prep_image(filename)
    prediction = model.predict(tf.expand_dims(img, axis=0))

    if len(prediction[0]) > 1:
        pred_class = class_names[prediction.argmax()]
    else:
        pred_class = class_names[int(tf.round(prediction)[0][0])]

    plt.imshow(img)
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


# function to create TensorBoard Callback
def create_tensorboard_callback(dir_name, experiment_name):
    """
    Creates a TensorBoard callback instand to store log files.
    Stores log files with the filepath:
        "dir_name/experiment_name/current_datetime/"
    Args:
        dir_name: target directory to store TensorBoard log files
        experiment_name: name of experiment directory (e.g. efficientnet_model_1)
    """
    log_dir = dir_name + "/" + experiment_name + "/" + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir
    )
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# function to plot loss curves
def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    Args:
        history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # plotting loss
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # plotting accuracy
    plt.figure()
    plt.plot(epochs, train_accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


# function to compare history of two models
def compare_history(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.

    Args:
        original_history: History object from original model (before new_history)
        new_history: History object from continued model training (after original_history)
        initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """

    # getting original history metrics
    acc = original_history.history['accuracy']
    loss = original_history.history['loss']
    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']

    # combining the original with new
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']
    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy.')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.xlabel('epochs')
    plt.title('Training and Validation Loss.')
    plt.show()


# function to unzip files
def unzip_data(filename):
    """
    Unzips filename into the current working directory.

    Args:
        filename (str): a filepath to a target zip folder to be unzipped.
    """
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall()
    zip_ref.close()


# function to walkthrough a directory
def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its content.

    Args:
        dir_path (str): target directory.

    Returns:
        A print out of:
            number of subdirectories in the dir_path.
            number of images (files) in each subdirectory.
            name of each subdirectory.
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories in and {len(filenames)} images in '{dirpath}'."
        )


# function to evaluate: accuracy, precision, recall and f1-score
def calculate_results(y_true, y_preds):
    """
    Calculates the model accuracy, precision, recall and f1-score of a binary classification model.

    Args:
        y_true: true labels in the form of 1D array.
        y_preds: predicted labels in the form of 1d array.

    Returns:
        A dictionary of Accuracy, precision, recall and f1-score.
    """
    model_acc = accuracy_score(y_true, y_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_preds, average='weighted')

    model_results = {
        'accuracy': model_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return model_results


# function to compare two histories
def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compare two Model's History Objects.
    """

    acc = original_history.history['accuracy']
    loss = original_history.history['loss']
    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']

    # combining the histories:
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']
    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1],
        plt.ylim(),
        label='Start of Fine Tuning'
    )
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot(
        [initial_epochs - 1, initial_epochs - 1],
        plt.ylim(),
        label='Start of Fine Tuning'
    )
    plt.legend(loc='upper right')
    plt.xlabel('epochs')
    plt.title('Training and Validation Loss')
    plt.show()
