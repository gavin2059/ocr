# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
from PIL import Image
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
import tools.infer.predict_cls as predict_cls
from ppocr.utils.logging import get_logger
from tools.infer.utility import draw_ocr_box_txt

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def process(args, img, name):
    '''
    Using pre-set arguments and the image given by the client,
    scans text in the image using PaddleOCR and returns the
    text contents and positions of the image.

    Inputs:
        args: pre-set arguments generated by utility.py unless otherwise modified
        img: opencv image file read from the client
    Returns:
        dt_boxes: an array of 4-coordinate subarrays representing the coordinates
                  of each text in the image
        rec_res: an array of (text, score) pairs representing the contents of each
                text and the level of confidence in said contents
    '''
    # Reads args
    text_sys = TextSystem(args)
    font_path = args.vis_font_path
    drop_score = args.drop_score
    if img is None:
        logger.info("error in loading image:{}".format("input"))
        return None, None
    # Performs OCR on the image
    found = False
    f = open('serials.txt', 'a')
    f.write(name + '\n')
    f.close()
    c = 0
    starttime = time.time()
    while not(found) and c < 4:
        dt_boxes, rec_res = text_sys(img)
        for i in range(len(rec_res)):
            text = rec_res[i][0].replace(' ', '').casefold()
            # Combines multiple boxes if necessary
            if text == 'serialnumber':
                serial = ''
                j = i
                while j < len(rec_res) - 1 and len(serial) < 19:
                    j += 1
                    serial += rec_res[j][0].replace(' ', '').casefold()
                text = serial
                print(text)
            # Checks if a serial is found
            if isSerial(text):
                found = True
                # rec_res[i][0] = fixSerial(text)
                break
        break
        if not(found) and c < 3:
            img = rotateImage(img, 90)
        c += 1
    # Draws picture if needed (defaults to True)
    if True or args.visualize:
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]
        draw_img = draw_ocr_box_txt(
            image,
            boxes,
            txts,
            scores,
            drop_score=drop_score,
            font_path=font_path)
        draw_img_save = "./inference_results/"
        if not os.path.exists(draw_img_save):
            os.makedirs(draw_img_save)
        cv2.imwrite(
            os.path.join(draw_img_save, os.path.basename(name)),
            draw_img[:, :, ::-1])
        logger.info("The visualized image saved in {}".format(
            os.path.join(draw_img_save, os.path.basename(name))))
        # cv2.imshow("OCR Image", draw_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    elapse = time.time() - starttime
    logger.info("Predict time of %s: %.3fs" % ("input", elapse))
    return dt_boxes, rec_res

# Rotates the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Checks if the input is a serial number
def isSerial(text):
    if len(text) != 19:
        return False
    text = text.casefold()
    print('passed length')
    if not(text.isalnum()):
        return False
    print('passed alnum')
    if 'nespresso' in text:
        return False
    print(text)
    f = open('serials.txt', 'a')
    f.write(text + '\n')
    f.close()
    return True

# Given a serial number, fixes it according to known formatting constraints
def fixSerial(text):
    return text