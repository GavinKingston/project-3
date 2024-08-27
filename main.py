import pandas as pd
from sklearn.preprocessing import StandardScaler
import PIL.Image as Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gradio as gr
import PIL
from sklearn.preprocessing import OneHotEncoder
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import time


def preprocess_data(data):
    # Add columns for weapon type and weapon id based on the filename
    data[["weapon_type", "weapon_id"]] = data["labelfile"].str.replace(r"\..*$", "", regex=True).str.split("_", expand=True)
    data = data.loc[data['train_id'] == 1].copy()
    data.reset_index(drop=True, inplace=True)
    return data

def open_image(image):
    image_path = './Resources/weapon_detection/train/images/' + image
    try:
        # Open the image
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

def process_image(image):
    image = image.resize((128, 128))  # Resize image to 128x128
    image = image.convert('RGB')  # Convert to RGB
    return np.array(image)
    
def augment_images(images, labels, n_augmentations=10):

    # create an ImageDataGenerator object to augment the images
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # setup empty lists to store the augmented images and labels and a counter to keep track of how many augmented images we have created
    X_aug = []
    y_aug = []
    counter = 0

    # loop through the training set and augment the images
    for i in range(images.shape[0]):
        print(f"Augmenting image {i}")
        img = images[i].reshape((1, 128, 128, 3))
        label = labels[i]

        # loop through the augmented images and add them to the list
        for aug_img in datagen.flow(img, batch_size=1):
            counter += 1
            print(f"Augmenting image {i} - {counter}")
            X_aug.append(aug_img[0])
            y_aug.append(label)
            
            # break the loop if we have created the specified number of augmented images
            if counter == n_augmentations:
                counter = 0
                break

        # add the original image to the list
        X_aug.append(img[0])
        y_aug.append(label)

        # convert the augmented images and labels to numpy arrays
        augmented_images = np.array(X_aug)
        augmented_labels = np.array(y_aug)

    return augmented_images, augmented_labels

def process_frame(image, model, encoder):
    # Preprocess the image to match the input format of your model
    # This could include resizing, normalization, etc.
    image = process_image(image)
    image = image.reshape((1, 128, 128, 3))
    image = image / 255.0  # Normalize the image

    # Make a prediction using the model
    predictions = model.predict(image)

    # Process the prediction (e.g., return the class with the highest probability)
    #predicted_class = np.argmax(prediction, axis=1)[0]

    prediction = encoder.inverse_transform(predictions)[0][0]
    confidence_score = predictions[0][np.argmax(predictions)]
    print(f"Prediction: {prediction} with an accuracy of {confidence_score}")
    return f"{prediction} with an accuracy of {confidence_score}"

    return predicted_class

def train_model(X, y, encoder):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(encoder.categories_[0]), activation='sigmoid')
        #tf.keras.layers.Dense(len(encoder.classes_), activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save('./models/weapon_detection_model.keras')
    return model


def load_model():
    model = tf.keras.models.load_model('./models/weapon_detection_model.keras')
    return model

def gradio_input_fn(image, encoder, model):
    image = process_image(image)
    image = image.reshape((1, 128, 128, 3))
    image = image / 255.0
    predictions = model.predict(image)

    if type(encoder) == MultiLabelBinarizer:
        predictions_binary = (predictions > 0.5).astype(int)
        predictions_labels = encoder.inverse_transform(predictions_binary)

        print(f"Predicted Weapon Types: {predictions_labels}")
        return f"Predicted Weapon Types: {predictions_labels}"
    else:
        prediction = encoder.inverse_transform(predictions)[0][0]
        confidence_score = predictions[0][np.argmax(predictions)]
        print(f"Prediction: {prediction} with an accuracy of {confidence_score}")
        return f"{prediction} with an accuracy of {confidence_score}"

def load_images(image_paths):
    images = [process_image(open_image(image_path)) for image_path in image_paths]
    return np.array(images)

def build_encoder(y):
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    with open('./models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    return y_encoded, encoder

def multi_label_binarizer(y):
    y_list_of_lists = [[label] for label in y]
    encoder = MultiLabelBinarizer()
    y_encoded = encoder.fit_transform(y_list_of_lists)
    with open('./models/encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    return y_encoded, encoder

def load_encoder():
    if not os.path.exists('./models/encoder.pkl'):
        return None
    with open('./models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def opencv(model, encoder):
    cam = cv2.VideoCapture(0)
    last_recorded_time = time.time() # this keeps track of the last time a frame was processed
    while True:
        curr_time = time.time() # grab the current time

        _, img = cam.read()

        cv2.imshow('img', img)

        if curr_time - last_recorded_time >= 2.0:
            # it has been at least 2 seconds
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(process_frame(Image.fromarray(color_img), model, encoder))
            last_recorded_time = curr_time


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

if __name__ == '__main__':

    if not os.path.exists('./Resources/weapon_detection_model.keras') or not os.path.exists('./Resources/encoder.pkl'):
        print("Model or Encoder not found. Training model and encoder...")

        # Load the data
        data = pd.read_csv('./Resources/metadata.csv')
        
        # Preprocess the data to gather labels and process image data
        data = preprocess_data(data)
        
        images = load_images(data["imagefile"])
        y_encoded, encoder = build_encoder(data["weapon_type"])
        #y_encoded, encoder = multi_label_binarizer(data["weapon_type"])

        images, labels = augment_images(images, y_encoded, 100)

        # Train the model
        model = train_model(images, labels, encoder)
    else:
        model = load_model()
        encoder = load_encoder()

    opencv(model, encoder)
    gr.Interface(fn=lambda input_data: gradio_input_fn(input_data, encoder, model), inputs=gr.Image(label="Image", type="pil"), outputs=gr.Textbox(label="Weapon Type")).launch()

#    model = train_model(data)

#    print(data.head(10))

#    data = preprocess_data(data)```