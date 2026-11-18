import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Parameters
IMG_SIZE = 128
MODEL_PATH = 'leaves_disease_detector_model_optimized.h5'
CLASS_LABELS_PATH = 'class_labels.npy'

# Load model and class labels
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(" Trained model not found. Please run 'train_and_save_model.py' first.")
if not os.path.exists(CLASS_LABELS_PATH):
    raise FileNotFoundError(" Class labels file not found. Please ensure 'train_and_save_model.py' ran successfully.")

model = tf.keras.models.load_model(MODEL_PATH)
class_labels = np.load(CLASS_LABELS_PATH, allow_pickle=True).item()


def predict_leaf_disease(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        max_confidence = np.max(predictions)
        print(max_confidence)
        class_idx = np.argmax(predictions)
        class_name = class_labels[class_idx]

        confidence_threshold = 0.80
        if max_confidence < confidence_threshold:
            return f" The Leaf image does NOT belong to the dataset classes. Try another image."
        elif "healthy" in class_name.lower():
            return f" The leaf is HEALTHY ({class_name}) with {max_confidence * 100:.2f}% confidence."
        else:
            return f" The leaf is DISEASED with: {class_name} (Confidence: {max_confidence * 100:.2f}%)"

    except Exception as e:
        return f" An error occurred: {str(e)}"


# Example usage
if __name__ == "__main__":
    test_image_path = r"C:\Users\Administrator\Downloads\Telegram Desktop\Plant_Disease_Detection\Out_of_Dataet\mulberry.jpg"
    result = predict_leaf_disease(test_image_path)
    print(result)
