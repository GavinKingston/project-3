import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image
import pickle
import time
import cv2

# Load weapon classification model
weapon_model_pkl = './models/pew_pew_pew_model_2.pkl'
with open(weapon_model_pkl,'rb') as file:
    model_2 = pickle.load(file)  

# Set weapon_list binary
weapon_list = [1,0]

# Create Gradio classifier
theme = gr.themes.Soft(
    primary_hue="stone",
    secondary_hue="sky",
)

# initiate gradio with theme

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
        pred_copy = prediction[0].copy()
        max_val = np.sort(pred_copy)[-1]
        second_max_val = np.sort(pred_copy)[-2]
        max_idx = np.where(pred_copy == max_val)[0][0]
        second_max_idx = np.where(pred_copy == second_max_val)[0][0]
        first_weapon = labels[max_idx]
        second_weapon = labels[second_max_idx]

        # create classifications for if there is a weapon in the image or not
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
            
            if max_val > 0.6:
                return f"The image contains a {first_weapon} with a confidence of {'{:,.2%}'.format(max_val)}."
            if second_max_val > 0.3 and second_max_val < max_val:
                return f"The image could contain a {first_weapon} with a confidence of {'{:,.2%}'.format(max_val)}. The image could also be a {second_weapon} with a confidence of {'{:,.2%}'.format(second_max_val)}"
            else:
                return f'Image contains no weapon \n the image could contain a {first_weapon} with a confidence of {"{:,.2%}".format(max_val)}.'
        if weapon_name == 0:
            
            if max_val > 0.5:
                labels = ['Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun', 'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword']
                prediction = model_2.predict(expanded_img)
                pred_copy = prediction[0].copy()
                max_val = np.sort(pred_copy)[-1]
                second_max_val = np.sort(pred_copy)[-2]
                max_idx = np.where(pred_copy == max_val)[0][0]
                second_max_idx = np.where(pred_copy == second_max_val)[0][0]
                first_weapon = labels[max_idx]
                second_weapon = labels[second_max_idx]

                if max_val > 0.7:
                    return f"No weapon detected in the image\nThe image could contain a {first_weapon} with a confidence of {'{:,.2%}'.format(max_val)}."
                if second_max_val > 0.1:
                    return f"No weapon detected in the image\nThe image could contain a {first_weapon} with a confidence of {'{:,.2%}'.format(max_val)}. The image could also be a {second_weapon} with a confidence of {'{:,.2%}'.format(second_max_val)}"
            else:
                return "No weapon detected in the image"
        
    demo = gr.Interface(run_model,
                        gr.Image(type="pil"),
                        
                        gr.Textbox())
    demo.launch()
    
#with gr.Blocks(theme=theme) as demo:
#    ...
def vid_predict(model, target_size):

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
        
        # create classifications for if there is a weapon in the image or not
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
                return f"The image could contain a {first_weapon} with a confidence of {'{:,.2%}'.format(max_val)}. The image could also be a {second_weapon} with a confidence of {'{:,.2%}'.format(second_max_val)}"
        if weapon_name == 0:
            return "No weapon detected in the image"


    cam = cv2.VideoCapture(0)
    last_recorded_time = time.time() # this keeps track of the last time a frame was processed
    while True:
        curr_time = time.time() # grab the current time

        _, img = cam.read()

        #cv2.imshow('img', img)

        if curr_time - last_recorded_time >= 2.0:
            # it has been at least 2 seconds
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(run_model(Image.fromarray(color_img)))
            last_recorded_time = curr_time


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
#    demo_2 = gr.Interface(run_model,
#                        gr.Image(type="pil"),
#                        
#                        gr.Textbox())
#    demo_2.launch()