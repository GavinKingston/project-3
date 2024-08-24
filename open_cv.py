import cv2
import numpy as np
import time
import pickle
from PIL import Image

l = ['Automatic Rifle', 'Bazooka', 'Grenade Launcher', 'Handgun', 'Knife', 'Shotgun', 'SMG', 'Sniper', 'Sword']

def vid_predict(model, target_size, labels):

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

    cam = cv2.VideoCapture(0)
    last_recorded_time = time.time() # this keeps track of the last time a frame was processed
    while True:
        curr_time = time.time() # grab the current time

        _, img = cam.read()

        cv2.imshow('img', img)

        if curr_time - last_recorded_time >= 2.0:
            # it has been at least 2 seconds
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            print(run_model(Image.fromarray(color_img)))
            last_recorded_time = curr_time


        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

if __name__ == '__main__':
    print(pickle.format_version)
    # load model to use
    model_pkl = 'pew_pew_pew_model.pkl'
    # to load in    
    #load model from pickle file
    with open(model_pkl, 'rb') as file:  
        model = pickle.load(file)  
        
    vid_predict(model, (64, 60), l)