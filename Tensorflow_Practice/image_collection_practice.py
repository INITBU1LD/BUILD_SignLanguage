import cv2
import os
import time
import uuid

#specifying our images path and saving it into this file path
IMAGES_PATH = "Tensorflow/workspace/images/collectedimages"

labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
number_imgs = 15

for label in labels:
    os.mkdir 
    {'Tensorflow\workspace\images\collectedimages\\'+label}
    cap = cv2.VideoCapture(1) #depends on your camera, you can play around with the values
    print('Collecting images for {}'.format(label))
    time.sleep(5) #sleep for 5 seconds, it will give us time to sleep between images in order to collect img
    for imgnum in range(number_imgs): #loop through the number of images
        ret, frame = cap.read() #set up capture
        imagename = os.path.join(IMAGES_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1()))) #image path
        cv2.imwrite(imagename, frame) #write it to directory
        cv2.imshow('frame', frame) #show on screen
        time.sleep(5) #sleep

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

