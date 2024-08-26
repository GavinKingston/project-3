import pandas as pd
from sklearn.preprocessing import StandardScaler
import PIL.Image as Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import gradio as gr
import PIL
from sklearn.preprocessing import OneHotEncoder

target_size = (128, 128)

def preprocess_data(data):

    # Add columns for weapon type and weapon id based on the filename
    data[["weapon_type", "weapon_id"]] = data["labelfile"].str.replace(r"\..*$", "", regex=True).str.split("_", expand=True)
    data = data.loc[data['train_id'] == 1].copy()
    data.reset_index(drop=True, inplace=True)
    
    # Get all of the images
    images = [open_image(x) for x in data["imagefile"]]
    
    # get the most common size of images and resize all images to that size
    target_size = get_most_common_size(images) 

    data["image"] = [process_image(image, target_size) for image in images]
   # data["image"] = images.apply(lambda x: process_image(x, target_size))


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
    resized_image = np.array(resized_image).astype(np.float32)

    #normalize the image
    resized_image = resized_image / 255.0
    
    return resized_image

def augment_images(data):
    X = [open_image(i) for i in data["imagefile"]]
    y= data["target"]
    
    # Convert list of images to numpy array with the correct shape
    #X_train_processed = np.array([np.array(x) for x in X])
    #X_train_processed = [np.expand_dims(x, axis=-1) for x in X]

    #print(f"X_train_processed: {X_train_processed}")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),         # Random rotation (20 degrees)
        tf.keras.layers.RandomTranslation(0.1, 0.1), # Random horizontal and vertical shift
        tf.keras.layers.RandomZoom(0.2),             # Random zoom
        tf.keras.layers.RandomFlip('horizontal')     # Random horizontal flip
    ])

    X_aug = []
    y_aug = []

    # loop through the training set and augment the images
    for i in range(len(X)):
        # augment 10 more images for each image in the training set
        print(f"Augmenting image {i}")
        #i = np.expand_dims(i, axis=0)  
        for j in range(1):
            print(f"Augmenting image {i} - {j}")
            augmented_image = data_augmentation(X[i], training=True)[0].numpy()

            # Resize the augmented image due to zooming
            #global target_size
            #resized_image = tf.image.resize(augmented_image, target_size)
            print(f"augmented_image shape: {augmented_image.shape}")

            #X_train_aug.append(resized_image.numpy())
            X_train_aug.append(augmented_image)
            y_train_aug.append(y_train[i])

    # Convert lists to numpy arrays
    #X_train_aug = np.array(X_train_aug)
    #y_train_aug = np.array(y_train_aug)
        
    return X_train_aug, y_train_aug

def one_hot_encode(y):
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))
    columns = encoder.get_feature_names_out(["weapon_type"])
    y = pd.DataFrame(y_encoded, columns=columns)
    print(f"y shape: {y.shape}")
    return y, encoder


def train_model(X, y):

    # Convert list of images to numpy array with the correct shape
    X = np.array([np.array(x) for x in X])
    y = np.array(y.values)

    print(f"y shape training: {y.shape}")
    #for img in X.values:
    #    print(f"img shape: {img.shape}")
    #print(f"X type: {type(X)}")
    #X_arr = np.concatenate(X, axis=0)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"y_train shape: {y_train.shape}")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

#    print(f"X_train: {X_train[0]}")
#    print(f"y_train: {y_train[0]}")

    print(f"X_train length: {len(X_train)}")
    print(f"y_train length: {len(y_train)}")

   # print(f"X_test dtype: {X_train[0].dtype}")
    print(f"X_train full dtype: {X_train.dtype}")
    #print(f"y_test dtype: {y_train[0].dtype}")
    print(f"y_train full dtype: {y_train.dtype}")

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save('./Resources/weapon_detection_model.keras')
    return model


def load_model():
    model = tf.keras.models.load_model('./Resources/weapon_detection_model.keras')
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
    X, y = augment_images(data)
    y, encoder = one_hot_encode(data["weapon_type"])

#    print(f"X_train_aug: {X_train_aug}")
    # Concatenate the original and augmented images
    #print(f"x_train_aug: {X_train_aug.shape}")
    #print(f"X_train: {data['image'].shape}")
    #X = X_train_aug + data["image"]
    #y = y_train_aug + y

#    print(f"X: {X}")
#    print(f"y: {y}")

    # Train the model
    model = train_model(data["image"], y)

    #gr.Interface(fn=gradio_input_fn, inputs=gr.Image(label="Image", type="pil"), outputs=gr.Textbox(label="Weapon Type")).launch()

#    model = train_model(data)

#    print(data.head(10))

#    data = preprocess_data(data)```