import pandas as pd
from sklearn.preprocessing import StandardScaler
import PIL.Image as Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gradio as gr
import PIL

target_size = (128, 128)

def preprocess_data(data):

    # Add columns for weapon type and weapon id based on the filename
    data[["weapon_type", "weapon_id"]] = data["labelfile"].str.replace(r"\..*$", "", regex=True).str.split("_", expand=True)
    data = data.loc[data['train_id'] == 1].copy()
    data.reset_index(drop=True, inplace=True)
    
    # Get all of the images
    #images = [open_image(x) for x in data["imagefile"]]
    images = data["imagefile"].apply(open_image)
    
    # get the most common size of images and resize all images to that size
    global target_size
    target_size = get_most_common_size(images) 

    print(f"Target size: {target_size}")
    #data["image"] = [process_image(image, target_size) for image in images]
    data["image"] = images.apply(lambda x: process_image(x, target_size))

    return data

def open_image(image):
    image_path = './Resources/weapon_detection/train/images/' + image
    try:
        # Open the image
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")

def get_most_common_size(images):
    sizes = [image.size for image in images]
    size_counts = pd.Series(sizes).value_counts()
    target_size = size_counts.idxmax()
    return target_size

def process_image(image, target_size):

    # Convert the image to RGB if it is not
    if image.mode != 'RGB':
        image = image.convert('RGB')

    #resize the image
    resized_image = image.resize(target_size, resample = Image.LANCZOS)

    #convert the image to a numpy array
    resized_image = np.array(image).astype(np.float32)

    #normalize the image
    resized_image = resized_image / 255.0
    return resized_image

def augment_images(data):
    X = data["image"]
    y_train = data["target"]
    
    X_train_processed = [np.expand_dims(x, axis=-1) for x in X]

    print(f"X_train_processed: {X_train_processed}")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),         # Random rotation (20 degrees)
        tf.keras.layers.RandomTranslation(0.1, 0.1), # Random horizontal and vertical shift
        tf.keras.layers.RandomZoom(0.2),             # Random zoom
        tf.keras.layers.RandomFlip('horizontal')     # Random horizontal flip
    ])

    print(y_train.shape)
    print(y_train.head(10))

    X_train_aug = []
    y_train_aug = []

    # loop through the training set and augment the images
    for i in range(len(X_train_processed)):
        # augment 10 more images for each image in the training set
        print(f"Augmenting image {i}")
        for j in range(1):
            print(f"Augmenting image {i} - {j}")
            X_train_aug.append(data_augmentation(X_train_processed[i], training=True)[0].numpy())
            y_train_aug.append(y_train[i])
        
    return X_train_aug, y_train_aug

def train_model(data):

    X = data["image"]
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save('./Resources/weapon_detection_model')
    return model


def load_model():
    model = tf.keras.models.load_model('./Resources/weapon_detection_model')
    return model

def gradio_input_fn(data):
    global target_size
    data = process_image(data, target_size)
    model = load_model()
    prediction = model.predict(data)
    return prediction



if __name__ == '__main__':

    # Load the data
    data = pd.read_csv('./Resources/metadata.csv')

    # Preprocess the data to gather labels and process image data
    data = preprocess_data(data)

    # Augment more images from existing images
    X_train_aug, y_train_aug = augment_images(data)

    for x in X_train_aug:
        print(x)
    for y in y_train_aug:
        print(y)

    # Concatenate the original and augmented images
    #X = pd.concat([X_train_aug.flatten(), data["image"]], ignore_index=True)
    #y = pd.concat([y_train_aug.flatten(), data["target"]], ignore_index=True)

    # Train the model
    #model = train_model({"image": X, "target": y})

    #gr.Interface(fn=gradio_input_fn, inputs=gr.Image(label="Image", type="pil"), outputs=gr.Textbox(label="Weapon Type")).launch()

#    model = train_model(data)

#    print(data.head(10))

#    data = preprocess_data(data)```