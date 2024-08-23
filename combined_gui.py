import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
import pickle

weapon_model_pkl = 'pew_pew_pew_model.pkl'
with open(weapon_model_pkl,'rb') as file:
    model_2 = pickle.load(file)  
    
theme = gr.themes.Soft(
    primary_hue="stone",
    secondary_hue="sky",
)

with gr.Blocks(theme=theme) as demo:
    ...
def predict(model, target_size):
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
        labels = [1,0]
        resized_img = img.resize(target_size, resample=(Image.LANCZOS))
        float_img = np.array(resized_img).astype(np.float32)
        normalized_img = float_img/255
        expanded_img = np.expand_dims(normalized_img, axis=0)
        prediction = model.predict(expanded_img)
        max_arg = tf.argmax(prediction, axis=1)
        max_index = max_arg.numpy()[0]
        weapon_name = labels[max_index]
        print(weapon_name)
        if weapon_name == 1:
            ## load model to use
            labels = ['Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun', 'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword']
            prediction = model_2.predict(expanded_img)
            pred_copy = prediction[0].copy()
            max_val = np.sort(pred_copy)[-1]
            second_max_val = np.sort(pred_copy)[-2]
            max_idx = np.where(pred_copy == max_val)[0][0]
            second_max_idx = np.where(pred_copy == second_max_val)[0][0]
            first_weapon = labels[max_idx]
            second_weapon = labels[second_max_idx]
            
            if max_val > 0.5:
                return f"The image contains a {first_weapon} with a confidence of {'{:,.2%}'.format(max_val)}."
            if second_max_val > 0.3 and second_max_val < max_val:
                return f"\nThe image could also be a {second_weapon} with a confidence of {'{:,.2%}'.format(second_max_val)}"
        if weapon_name == 0:
            return "No weapon detected in the image"
    demo = gr.Interface(run_model, gr.Image(type="pil"), gr.Textbox())
    demo.launch()