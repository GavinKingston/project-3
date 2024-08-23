# The Digital Gangsters presents: Weapon Image Classifier

# Group 3 Project Members
Nicholas Major, Gavin Kingston, Philip Schrank, Joe Middleton

# Executive Summary
Our goal is to create a Convolutional Neural Network (CNN) specifically designed to classify images of weapons. This model is optimized for integration with OpenCV, enabling real-time analysis of videos and live security feeds. By leveraging this technology, we aim to enhance threat detection capabilities and proactively mitigate potential security risks.

# Dataset and Resources
[metadata.csv](https://github.com/GavinKingston/project-3/blob/Major/Resources/metadata.csv) 

Weapon Detection Dataset: https://dasci.es/transferencia/open-data/24705/

Weapon Detection for Security and Video Surveillance: https://sci2s.ugr.es/weapons-detection

[train.csv](https://github.com/GavinKingston/project-3/blob/Major/Resources/dataset/train.csv)

# Data Preparation 
Two datasets were imported: one containing metadata of weapon images and another with general images. The metadata files were processed to extract relevant labels and IDs.  All images were loaded from the specified directories, converted to RGB format, and cleaned to ensure only valid images were included.

# Image Preprocessing
Images were resized to a uniform dimension of 64x60 pixels. Each image was converted to a floating-point numpy array and normalized to have pixel values between 0 and 1, which standardizes the input for the model.

# Data Augmentation
An augmentation model was created using TensorFlow to randomly rotate, translate, zoom, and flip images. This increases the variability of the training data and helps the model generalize better. Each original training image was augmented to produce additional images, effectively expanding the dataset.

# Model Building
A CNN was defined using TensorFlow and Keras, consisting of multiple convolutional layers, max pooling layers, and dense layers. This architecture is designed to extract features from images and classify them effectively.

# Model Training
The model was compiled with the RMSProp optimizer and binary cross-entropy loss function, suitable for binary classification tasks. The training process involved fitting the model to the augmented training data, validating its performance with a separate test dataset, and iterating over multiple epochs to optimize accuracy.

# Model Saving
The trained model was serialized and saved as a pickle file (binary_model.pkl). This allows the model to be loaded and used for inference in future applications.






