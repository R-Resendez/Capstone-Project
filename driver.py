# time used for FPS counter
# os used for navigating pi FS as well as using build-in linux tools for playing audio.
# numpy used for generating blank images and pixel modification
# threading used for keeping the video feed process separate from image processing in the hopes of improving fps
#tflite intterpreter used for evaluating images and getting results for image classification
# cv2 used for handling the video feed.

import time
import os
import numpy as np
from threading import Thread

# from PIL import Image
from tflite_runtime.interpreter import Interpreter
import cv2

#Video feed as a class allows for easier handling of threading for this specific process.
class CameraFeed:
#1280,720
    ########################################
    # INIT function sets up class variables used for handing video feed.
    # feed handles the primary video capture which is where individual frames are pulled.
    # cv2 default paramters for width and height of the capture feed are 3 and 4 in the set function.
    # success is a bool variables that is output along with reading the videofeed, img is the current frame
    # Stop cam is a bool designed for handing the primary videofeed loop
    ########################################
    def __init__(self,resolution=(640,480)):
        self.feed = cv2.VideoCapture(0)
        self.feed.set(3,resolution[0])
        self.feed.set(4,resolution[1])

        self.success, self.img = self.feed.read()

        self.stop_cam = False
    ########################################
    # START function sets up the parallel portion of the code.
    # The thread is cteayed and assigned the update function which handles pulling new frame data.
    # t.start will start the thread 
    ########################################
    def start(self):
        t = Thread(target=self.update,args=())
        t.start()
        return self

    ########################################
    # UPDATE function continuously runs and stores the next frame in the img variable
    # If the bool for stoppping the camera is true than update ends and video feed stops.
    # else it will continue to assign the next frame to the image variable.
    ########################################
    def update(self):
        while True:
            if self.stop_cam:
                self.feed.release()
                return
            self.success, self.img = self.feed.read()

    ########################################
    # GET_IMG function returns the current img from the video feed.
    ########################################
    def get_img(self):
        return self.img

    ########################################
    # STOP function sets the stop bool which will cause the update function to end.
    ########################################
    def stop(self):
        self.stop_cam = True


####################### END CLASS DEFINITION ###########################


    ########################################
    # LOAD_LABELS function opens the labels file from thge resources folder.
    ########################################
def load_labels(label_path):
    with open(label_path, 'r') as label_file:
        return {i: line.strip() for i, line in enumerate(label_file.readlines())}


    ########################################
    # SET_INPUT_TENSORS function sets the input paramters for the TF lite model to evaluate
    # tensor_index is assigned the index value from the interpreter.
    # input_tensor takes the index and assigns it the interpreters tensor
    # image is assigned the input tensor as raw data 
    ########################################
def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


    ########################################
    # CLASSIFY_IMAGE function classifies the input image with a probability and label
    # Input tensor is set and ready to be evaluated.
    # Invoking the interpreter tells the model to start the classification and identify the image
    # list of dictionarys is output from the interpreter and the results from the first entry are stored as the full output.
    # The index that was assigned from the output function is passed to the get_tensor function which will return a copy of the output tensor
    # np.squeeze then removes all columns of length 1 to normalize the result.
    ########################################
def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]


def main():

    # Variables for width and height for testing purposes mostly.
    width = 640
    height = 480
    # Counters for storing images confirming package drop-off/pirate-prevention
    # images are also a good way to continuously get new data for updating the model with additonal training.
    pirate_ct = 0
    carrier_ct = 0

    # Start up the Camerage feed and provide with with custom dimentions if needed.
    cameraFeed = CameraFeed(resolution=(width,height))
    cameraFeed.start()
    time.sleep(1)
    cv2.namedWindow('Porch Pirate Detection System', cv2.WINDOW_NORMAL)

    # Setup a bunch of pathways that will be necessary for playing the audio files as well as locating the model and labels files.
    model_path = 'resources/models/model.tflite'
    label_path = 'resources/models/labels.txt'
    carrier_mp3 = 'resources/audio/mail_carrier.mp3'
    pirate_mp3 = 'resources/audio/porch_pirate.mp3'

    # Interpreter is set to the model_path and the labels are then loaded.
    interpreter = Interpreter(model_path)
    labels = load_labels(label_path)

    # Allocte enough memory needed for models input.
    # Collect input and output details from the interpreter
    # Update width and height to match input details (Should always match anyway this is for testing purposes.)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # The Primary running loop. This is where all the image classification happens.
    while True:
        # Grab the time that the frame was first grabbed to start an accurate measure of FPS.
        # Also make a copy of the grabbed frame for performing manipulations while still having the original for comparison.
        frame_start = time.time()
        cur_img = cameraFeed.get_img()
        cp_img = cur_img.copy()

        # Convert the image to RGB, and resize the image to match our display.
        frame_rgb = cv2.cvtColor(cp_img, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))


        # input_data = np.expand_dims(frame_resized, axis=0)

        # Classify the image based on the current frame and grab the computed label id and probability.
        results = classify_image(interpreter, frame_resized)
        label_id, prob = results[0]

        # Check if the mail_carrier probabiliy is above the needed threshold and if so trigger the OS to play an audio file.
        # After that cv2 will write to the device the frame that had the identified actor.
        # This process is then repeated for the porch pirate.
        if labels[label_id] == "mail_carrier" and prob > .9:
            os.system("mpg123 " + carrier_mp3)
            cv2.imwrite('resources/newPhotoData/carriers/carrier_img' + (str(carrier_ct)).zfill(3) + '.jpg',cur_img)
            carrier_ct = carrier_ct+1
            time.sleep(10)
        
        if labels[label_id] == "porch_pirate" and prob > .9:
            os.system("mpg123 " + pirate_mp3)
            cv2.imwrite('resources/newPhotoData/pirates/pirate_img' + (str(pirate_ct)).zfill(3) + '.jpg',cur_img)
            pirate_ct=pirate_ct+1
            time.sleep(10)

        # Two Mostly debugging statements, they print to the console the label and probability of each frame.
        print("Current Actor is most likely:" + labels[label_id] + " Probability: " + str(prob))
        print("Current FPS is: " + str(1/(time.time() - frame_start)))

        # Now that classifiction is done the FPS count can be added to the cv2 window with color, font, size, and positon.
        cv2.putText(frame_resized, 'FPS: {0:.2f}'.format(1/(time.time() - frame_start)),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        cv2.imshow('Porch Pirate Detection System', frame_resized)

        # If the user pressed q on the keyboard than the loop will break allowing the program to gracefully exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Destroys all cv2 windows and clears the memory for them.
    # The stop boolean for the camera feed is set which will end the threaded process. 
    cv2.destroyAllWindows()
    cameraFeed.stop()

if __name__ == '__main__':
  main()
