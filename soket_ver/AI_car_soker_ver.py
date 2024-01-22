import os
import argparse
import cv2
import numpy as np
import sys
import time
import importlib.util
from threading import Thread
import threading
import RPi.GPIO as GPIO
import YB_Pcb_Car
import socket
import json

car = YB_Pcb_Car.YB_Pcb_Car()

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

EchoPin = 18
TrigPin = 16
Left1 = 13
Left2 = 15
Right1 = 11
Right2 = 7

GPIO.setup(EchoPin,GPIO.IN)
GPIO.setup(TrigPin,GPIO.OUT)
GPIO.setup(Left1,GPIO.IN)
GPIO.setup(Left2,GPIO.IN)
GPIO.setup(Right1,GPIO.IN)
GPIO.setup(Right2,GPIO.IN)

object_name = ""
dist = 20
track = 4
speed = 50

def Distance():
    global dist
    GPIO.output(TrigPin,GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin,GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin,GPIO.LOW)
    
    t3 = time.time()
    
    while not GPIO.input(EchoPin):
        t4 = time.time()
        if (t4 - t3) > 0.03 :
            return -1
        
    t1 = time.time()
    
    while GPIO.input(EchoPin):
        t5 = time.time()
        if(t5 - t1) > 0.03 :
            return -1
        
    t2 = time.time()
    time.sleep(0.02)
    #print ("distance_1 is %d " % (((t2 - t1)* 340 / 2) * 100))
    return ((t2 - t1)* 340 / 2) * 100

def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
        distance = Distance()
        while int(distance) == -1 :
            distance = Distance()
        while (int(distance) >= 500 or int(distance) == 0) :
            distance = Distance()
        ultrasonic.append(distance)
        num = num + 1
        time.sleep(0.04)
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3])/3
    return distance


def avoid():
    global dist
    while True:
        dist = Distance_test()
        time.sleep(0.1)
def tracking():
    global track
    while True:
        #right 0
        Left1Value = GPIO.input(Left1);
        Left2Value = GPIO.input(Left2);
        Right1Value = GPIO.input(Right1);
        Right2Value = GPIO.input(Right2);
        track = Left1Value + Left2Value + Right1Value + Right2Value
        time.sleep(0.1)
        
        '''
        if (Tracking_Left1Value == False and Tracking_Left2Value == False and Tracking_Right1Value == False and Tracking_Right2Value ==False):
            car.Car_Run(30, 30)
        if stop == 4:
            speed = 0
            time.sleep(3)
        else:
            speed = 60
            time.sleep(0.2)
            #print (Left1Value)
            #print (Left2Value)
            #print (Right1Value)
            #print (Right2Value)
            #print ('---')
        print('tracking = ', speed)
        '''


class VideoStream:
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        (self.grabbed, self.frame) = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

def traffic():
    global object_name
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='320x320')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = '/home/pi/AI_CAR/custom_model_lite'
    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'label_map.pbtxt'
    min_conf_threshold = float(0.5)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu
    
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    if use_TPU:
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'edgetpu.tflite'       

    CWD_PATH = os.getcwd()

    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    if labels[0] == '???':
        del(labels[0])

    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname):
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)
    
    while True:
        t1 = cv2.getTickCount()
        frame1 = videostream.read()
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
        
        object_name = "None"
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                object_name = labels[int(classes[i])]
                #print(object_name)
                #print(scores[i])
                label = '%s: %d%%' % (object_name, int(scores[i]*100))
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, labelSize[1] + 10) 
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        if cv2.waitKey(1) == ord('q'):
            break
        
        
def car_controller():
    global dist
    global track
    global object_name
    global speed
    while True:
        speed = 40
        while object_name == "red":
            if (object_name == "red" and track == 0) or dist < 10:
                car.Car_Stop()
                time.sleep(0.2)
            elif object_name =="red":
                if speed > 30:
                    speed = speed - 1
                car.Car_Run(speed, speed)
                time.sleep(0.2)
        if dist < 10:
            car.Car_Stop()
            time.sleep(0.3)
        else:
            car.Car_Run(speed, speed)

def send_data(data):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('192.168.1.134', 9999)
    sock.connect(server_address)
    json_data = json.dumps(data)
    
    sock.sendall(json_data.encode())

    sock.close()


if __name__ == '__main__':
    thread1 = threading.Thread(target=avoid)
    thread1.Daemon = True
    thread1.start()
    thread2 = threading.Thread(target=tracking)
    thread2.Daemon = True
    thread2.start()
    thread3 = threading.Thread(target=traffic)
    thread3.Daemon = True
    thread3.start()
    thread4 = threading.Thread(target=car_controller)
    thread4.Daemon = True
    thread4.start()
    
    try:
        while True:
            #print("object_name:", object_name)
            #print("track:", track)
            #print("dist:", dist)
            #print("speed:", speed)
            data = {"object_name" : object_name, "track" : track, "dict" : int(dist), "speed" : speed}
            send_data(data)
            print(data)
            time.sleep(2)
                
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    videostream.stop()



