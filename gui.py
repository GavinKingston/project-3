import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
import pickle

l = ['Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun', 'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword']

def predict(model, target_size, labels):
    """
    This is a helper function to launch a Gradio UI, which
    can then be used to enter an image to detect if there is
    a weapon in the image

    Args:
        model: the tensor flow CNN model
        target_size (int, int): the height and width to resize the image
        labels (list): the list of the target labels (weapon names)
    """
    def run_model(img):
        resized_img = img.resize(target_size, resample=(Image.LANCZOS))
        float_img = np.array(resized_img).astype(np.float32)
        normalized_img = float_img/255
        expanded_img = np.expand_dims(normalized_img, axis=0)
        prediction = model.predict(expanded_img)
        max_arg = tf.argmax(prediction, axis=1)
        max_index = max_arg.numpy()[0]
        weapon_name = labels[max_index]
        confidence = tf.gather(prediction[0], max_index).numpy()
        if confidence > 0.5:
            return f"The image contains a {weapon_name} with a confidence of {'{:,.2%}'.format(confidence)}"
        
        return "No weapon detected in the image"
    demo = gr.Interface(run_model, gr.Image(type="pil"), gr.Textbox())
    demo.launch()

if __name__ == '__main__':
    pkl_model = "./pew_pew_pew_model.pkl"
    with open(pkl_model, 'rb') as file:
        model = pickle.load(file)
        
    predict(model, (64, 60), l)