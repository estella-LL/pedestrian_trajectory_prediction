import os
import cv2
import time
import argparse
import torch
import sys
import numpy as np
from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from Ui_show import Ui_mainWindow
from PyQt5 import QtWidgets,QtMultimedia
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import QTimer
from predictor.model_lstm import LSTM
from collections import defaultdict

trace = {}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--display", default=1,action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")#
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()

class myWindow(QtWidgets.QWidget,Ui_mainWindow):
    def __init__(self, parent=None):
        super(myWindow, self).__init__()
        self.parent = parent
        self.setupUi(self)
        self.preWight() #initialize the control
        self.PreParameters() #initializing variables
        self.controler() #control corresponding slot function
        # self.use_cuda = 1       #need a graphics card
        self.use_cuda = 0
        self.count_frame = 0
        self.id_set = {}        # The dictionary is used to see if there are any related ids recently
        self.pre_history={}     #Store the coordinates of the first trajectory point predicted for each frame
        self.pre_history=defaultdict(list)
        self.loss={} #Store the loss value for each id

        self.initModel()

    def preWight(self):
        self.pushButton_1.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.pushButton_4.setEnabled(False)

    def PreParameters(self):
        self.player=QtMultimedia.QMediaPlayer() #Player

    def controler(self): #Signal slot settings
        self.pushButton_1.clicked.connect(self.OpenVideo)
        self.pushButton_2.clicked.connect(self.doPredict)
        self.pushButton_4.clicked.connect(self.CloseVideo)

    def load_file(self):
        directory, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", '', "all files(*.*)")
        self.path = directory
        if self.path == '' and self.path == ' ':
            return
        if self.path == '':
            self.lineEdit_2.setText("You didn't select a video! ")
            return
        else:
            # Path display
            videoStr = str(self.path)
            self.lineEdit_2.setText(videoStr)
            return directory


    def OpenVideo(self):
        self.filename=self.load_file()
        #Button enabled, the video has been loaded and you can click the start button
        self.pushButton_1.setEnabled(False)
        self.pushButton_2.setEnabled(True)

    def initModel(self):

        print("Loading model...")
        '''Load related parameters'''
        args = parse_args()
        cfg = get_config()
        cfg.merge_from_file(args.config_detection)
        cfg.merge_from_file(args.config_deepsort)
        # Detecting target
        self.detector = build_detector(cfg, use_cuda=self.use_cuda)
        # Tracking Target
        self.deepsort = build_tracker(cfg, use_cuda=self.use_cuda)
        # Prediction Model
        # self.model = LSTM(0,10)
        self.model = LSTM(0, 50)
        self.model.load_state_dict(torch.load('./predictor/lstm_pre.pth'))
        device = torch.device(device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # self.model.to(device)
        print("Loading completed! ")

    def predict_road(self,x_tmp:list):
        x_tmp = np.array(x_tmp)
        # x = np.zeros((10,2))
        x = np.zeros((50, 2))
        x[:, 0] = (x_tmp[:,0] + x_tmp[:,2]) / 2
        x[:, 1] = (x_tmp[:, 1] + x_tmp[:, 3]) / 2
        x_min , x_max = np.min(x[:,0]),np.max(x[:,0])
        y_min , y_max = np.min(x[:,1]),np.max(x[:,1])
        x[:, 0] = (x[:, 0] - x_min) / (x_max - x_min)
        x[:, 1] = (x[:, 1] - y_min) / (y_max - y_min)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        x = x.float()       # int -> float
        '''torch.FloatTensor torch.Size([1, 10, 2])'''
        y_pre = self.model(x)
        '''Attributes are scaled to a specified minimum and maximum value (usually 1-0). This can be achieved with the preprocessing.MinMaxScaler class.'''
        '''Commonly used minimum and maximum normalization methods(x-min(x))/(max(x)-min(x))'''
        y_pre[:, :, 0] = y_pre[:, :, 0] * (x_max - x_min) + x_min
        y_pre[:, :, 1] = y_pre[:, :, 1] * (y_max - y_min) + y_min
        y = y_pre.detach().numpy().squeeze()
        y = y.astype(int)
        y = y.tolist()
        # 2 dimensions, the last dimension represents x,y, the second to last dimension represents 10 numbers
        return y

    def show_pic(self):
        #Read a frame
        success,frame=self.cap.read()
        self.count_frame += 1
        print(self.count_frame)
        im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox_xywh, cls_conf, cls_ids = self.detector(im)
        mask = cls_ids == 0
        bbox_xywh = bbox_xywh[mask]
        bbox_xywh[:, 3:] *= 1.0
        cls_conf = cls_conf[mask]
        # tracking
        outputs = self.deepsort.update(bbox_xywh, cls_conf, im)
        print("Number of targets",len(outputs))
        self.lineEdit_3.setText(str(len(outputs)))
        outputs = np.array(outputs)         # outputs[Number of pedestrians in the current frame: 5 columns]
        # If there is an id in id_set but not in output, it means that the outdated s needs to be deleted.
        for s in self.id_set:
            if s not in list(outputs[:, -1]):
                self.id_set.pop(self,s)
        # If a new id appears in the output and is not in id_set, it means that it is new or appears after being blocked.
        for o in outputs:
            if o[-1] not in self.id_set:
                self.id_set[o[-1]] = [[o[0],o[1],o[2],o[3]]]
            # If the id that appears in output is also in set_id, it means that it needs to be updated
            if o[-1] in self.id_set:
                self.id_set[o[-1]].append([o[0],o[1],o[2],o[3]])
        tra_ans = {}

        if self.count_frame > 51:
            for s in self.id_set:
                if len(self.id_set[s]) >= 51:
                    temp = self.id_set[s][-50:]     # 2D
                    ans = self.predict_road(temp)       # 50x2 --> 50x2
                    tra_ans[s] = ans
                    # print(tra_ans[s][0])
                    self.pre_history[s].append(tra_ans[s][0])
                    # print(self.pre_history)

        '''Drawing'''
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            out = list(identities)
            for id in identities:
                index = out.index(id)
                if id not in trace.keys():
                    trace[id] = [[int((outputs[index][0] + outputs[index][2]) / 2),
                                  int((outputs[index][1] + outputs[index][3]) / 2)]]
                else:
                    trace[id].append([int((outputs[index][0] + outputs[index][2]) / 2),
                                      int((outputs[index][1] + outputs[index][3]) / 2)])
                # print("Historical trace:",trace[id])
                if len(self.pre_history[id]) > 10:
                    trace_tensor = torch.tensor(trace[id][-10:]).float()
                    pre_history_tensor=torch.tensor(self.pre_history[id][-11:-1]).float()
                    loss_func = torch.nn.MSELoss(reduction='mean')
                    loss = loss_func(trace_tensor, pre_history_tensor)
                    loss1 = torch.sqrt(loss)
                    print(loss1)
            frame = draw_boxes(frame, bbox_xyxy, identities, trace=trace, pre=tra_ans, pre_h=self.pre_history)
            for bb_xyxy in bbox_xyxy:
                bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

        if success:
            # convert RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.label.setPixmap(QPixmap.fromImage(img))
            self.label.setScaledContents(True)#Make the image adapt to the label size

    def doPredict(self):
        self.pushButton_4.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        # Set the timer
        self.v_timer = QTimer()  # self.
        # Read Video
        self.cap = cv2.VideoCapture(self.filename)
        if not self.cap:
            print("Failed to open the video")
            return
        #Get the FPS of the video
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # Set the timer period in milliseconds
        self.v_timer.start(int(1000 / self.fps))
        # Connect the slot function of the timer period overflow to display a frame of video
        self.v_timer.timeout.connect(self.show_pic)

    def CloseVideo(self):
        if QtWidgets.QMessageBox.information(self,str("End reminder"),str("Do you really want to end the prediction?")):
            app.quit()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = myWindow()
    myshow.show()
    sys.exit(app.exec_())
