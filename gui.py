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
        pred_copy = prediction[0].copy()
        max_val = np.sort(pred_copy)[-1]
        second_max_val = np.sort(pred_copy)[-2]
        max_idx = np.where(pred_copy == max_val)[0][0]
        second_max_idx = np.where(pred_copy == second_max_val)[0][0]
        first_weapon = labels[max_idx]
        second_weapon = labels[second_max_idx]
        output_message = "No weapon detected in the image"
        if max_val > 0.5:
            output_message = f"The image contains a {first_weapon} with a confidence of {'{:,.2%}'.format(max_val)}."
        
        if second_max_val > 0.3 and second_max_val < max_val:
            output_message = output_message + f"\nThe image could also be a {second_weapon} with a confidence of {'{:,.2%}'.format(second_max_val)}"
        
        return output_message
    demo = gr.Interface(run_model, gr.Image(type="pil"), gr.Textbox())
    demo.launch()

if __name__ == '__main__':
    print(pickle.format_version)
    # load model to use
    #model_pkl = 'binary_model.pkl'
    model_pkl = 'pew_pew_pew_model.pkl'
    # to load in    
    #load model from pickle file
    with open(model_pkl, 'rb') as file:  
        model = pickle.load(file)  
        
    predict(model, (64, 60), l)