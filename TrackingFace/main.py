import torch
from torch import cuda
from face_detector import YoloDetector
import numpy as np
from PIL import Image
import cv2



class TrackFace():
    """
    программа содержит pre-trained yolov5 и собственный трекер лиц
    трекер реализован координатным методом
    """
    
    def __init__(self, target_size = 480, min_face_size = 30, mode = 'cam', video_path = None):

        if cuda.is_available():
            cuda.empty_cache()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.target_size = target_size
        self.model = YoloDetector(target_size=self.target_size, device=self.device, min_face=min_face_size)
        self.mode = mode
        if self.mode == 'cam':
            self.cam = cv2.VideoCapture(0)
        else:
            self.cam = cv2.VideoCapture(video_path)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', fourcc, 5, (self.target_size, self.target_size))
        cv2.namedWindow('FaceTracking')

        # tracking var
        self.init_bboxes = dict()

    # основной метод класса для захвата фреймов
    def start(self):
        self.launch()
        while True:
            ret, frame = self.cam.read()
            if not ret:
                print("failed to grab frame")
                break
            frame = cv2.resize(frame, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
            frame = self.detectFace(frame)
            cv2.imshow('FaceTracking', frame)
            if self.mode != 'cam':
                self.out.write(frame)
            if cv2.waitKey(1) == ord('q'):
                break
        if self.mode != 'cam':
            self.out.release()
        self.cam.release()
        cv2.destroyAllWindows()


    # первый запуск
    def launch(self):
        ret, frame = self.cam.read()
        if not ret:
            print("failed to grab frame")
            raise RuntimeError
        frame = cv2.resize(frame, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        bboxes, _ = self.model.predict(frame)
        if len(bboxes[0]) > 0:
            for idx in range(len(bboxes[0])):
                if idx not in self.init_bboxes:
                    self.init_bboxes[idx] = bboxes[0][idx] # словарь из начальных {id: bbox}


    # функция отрисовки bbox + face_id
    def draw_info(self, frame, bboxes, assignment):
        for current_idx, prev_idx in assignment.items():
            bbox_itm = bboxes[current_idx]
            frame = cv2.rectangle(frame, 
                                (bbox_itm[0], bbox_itm[1]), 
                                (bbox_itm[2], bbox_itm[3]), 
                                (255, 0, 0), 
                                5)
            
            label = 'face id: ' + str(prev_idx)
            (w, h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
   
            frame = cv2.rectangle(frame, (bbox_itm[0], bbox_itm[1] - 20), (bbox_itm[0] + w, bbox_itm[1]), (255, 0, 0), -1)
            frame = cv2.putText(frame, label, (bbox_itm[0], bbox_itm[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        return frame         
    
    # предсказание сетки
    def detectFace(self, frame):
        bboxes, _ = self.model.predict(frame)
        if len(bboxes[0]) > 0:
            assignment = self.intersection(bboxes[0])  # return --> {id_for_max_iou: previous_id, ...}
            frame = self.draw_info(frame, bboxes[0], assignment)
            return frame
        else:
            return frame  

    # рассчет коэффициента перекрытия 
    def iou(self, prev_coords, curr_coords):
        x_0 = max(prev_coords[0], curr_coords[0])
        y_0 = max(prev_coords[1], curr_coords[1])
        x_1 = min(prev_coords[2], curr_coords[2])
        y_1 = min(prev_coords[3], curr_coords[3])
        inter_area = (x_1 - x_0)*(y_1 - y_0)
        a = (prev_coords[2] - prev_coords[0])*(prev_coords[3] - prev_coords[1])
        b = (curr_coords[2] - curr_coords[0])*(curr_coords[3] - curr_coords[1])
        union_area = a + b - inter_area
        return inter_area / (union_area + 0.000001)

    # расчет геометрических характеристик
    def intersection(self, bboxes):

        assignment = dict()

        if sum(self.init_bboxes) == 0 or len(self.init_bboxes) != len(bboxes):
            for key, value in enumerate(bboxes):
                self.init_bboxes[key] = value
                
        # для каждого исходного id находим геометрические характеристики
        for id_0, prev_coords in self.init_bboxes.items():
            x_prev_c = (prev_coords[0] + prev_coords[2]) / 2
            y_prev_c = (prev_coords[1] + prev_coords[3]) / 2
            w_prev = prev_coords[2] - prev_coords[0]
            h_prev = prev_coords[3] - prev_coords[1]
            iou_areas = [0, 0]
            iou_last = 0
            for idx_curr, curr_coords in enumerate(bboxes):

                # расчет координат центроид, ширины и высоты
                x_current_c = (curr_coords[0] + curr_coords[2]) / 2
                y_current_c = (curr_coords[1] + curr_coords[3]) / 2
                w_current = curr_coords[2] - curr_coords[0]
                h_current = curr_coords[3] - curr_coords[1]

                # расчет расстояний между центроидами
                dx = abs(x_current_c - x_prev_c)
                dy = abs(y_current_c - y_prev_c)

                if dx < (w_prev + w_current)/2 and dy < (h_prev + h_current)/2:
                    # то есть bbox на двух фреймах (i, i + 1) пересекаются ==> рассчитаем IOU
                    iou_value = self.iou(prev_coords, curr_coords)
                    if iou_value > iou_last:
                        iou_areas[0] = idx_curr
                        iou_areas[1] = iou_value
                        iou_last = iou_value

            if iou_areas[0] not in assignment:
                assignment[iou_areas[0]] = id_0
                
        self.init_bboxes = dict()
        for old_id, new_id in assignment.items():
            self.init_bboxes[new_id] = bboxes[old_id]
        return assignment
    

if __name__ == '__main__':
    track_model = TrackFace(target_size = 512, 
                            min_face_size = 10, 
                            mode = 'cam',
                            video_path = None)
    track_model.start()