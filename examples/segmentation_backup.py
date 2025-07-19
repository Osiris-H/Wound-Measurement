import json
from io import BytesIO
import requests
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from fnet import FNet
import torch.backends.cudnn as cudnn
from PIL import Image
# from mobilenet_v2 import mobilenet_v2

# yolo
# from models.experimental import attempt_load
# from utils.general import scale_coords, non_max_suppression
import time


class Segmentation(object):
    def __init__(self, img_path):
        response = requests.get(img_path)
        self.img = Image.open(BytesIO(response.content))
        # self.img = Image.open('18.png')

        self.heng1, self.heng2, self.zong1, self.zong2 = [], [], [], []
        self.heng3, self.heng4, self.zong3, self.zong4 = [], [], [], []
        self.gradient = []
        self.long_list, self.short_list = [], []
        self.degree1, self.degree2 = [], []
        self.region = []
        self.d = {}
        self.l = []
        self.isLabel = True
        self.crop = None

    '''创面分割'''
    def wound_segmentation(self):
        # -----------------------------神经网络---------------------------------
        result = self.deep_learn('FNet', 'model_099_0.9588.pth.tar')

        # 分割结果图
        result = cv2.cvtColor(np.asarray(result), cv2.COLOR_GRAY2BGR)

        # resize成原图大小 (width, height)
        result = cv2.resize(result, (self.img.size[0], self.img.size[1]), interpolation=cv2.INTER_AREA)

        # cv2.imshow('result', result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 平滑边缘
        result = cv2.medianBlur(result, 3)

        # 创口边缘
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
        # binary1, contours, hierarchy1 = cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy1 = cv2.findContours(result_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # ubuntu

        try:
            # 长短径和面积计算
            self.area = self.line_area(contours)

            for i in range(len(contours)):  # 区域个数
                if self.region[i] < 0.05:
                    self.area -= self.region[i]
                    continue
                s = {}
                # [[222, 222]]转成单维数组
                contour = []
                for j in range(len(contours[i])):  # 点数
                    contour.append((contours[i][j][0]).tolist())

                # 长短径端点
                long, short = [], []
                long.append([float(self.heng1[i]), float(self.zong1[i])])
                long.append([float(self.heng2[i]), float(self.zong2[i])])
                short.append([float(self.heng3[i]), float(self.zong3[i])])
                short.append([float(self.heng4[i]), float(self.zong4[i])])

                # s['contours'] = contours[i].tolist()
                s['contours'] = contour
                s['regionArea'] = self.region[i]
                s['longPost'] = long
                s['shortPost'] = short
                s['longDiam'] = self.long_list[i]
                s['shortDiam'] = self.short_list[i]
                s['longAngle'] = self.degree1[i]
                s['shortAngle'] = self.degree2[i]

                self.l.append(s)

            self.d['isLabel'] = self.isLabel
            self.d['gossArea'] = round(self.area, 2)
            self.d['regionList'] = self.l

            return json.dumps(self.d, ensure_ascii=False)

        except (SystemError, IndexError):  # 没有标签异常
            self.isLabel = False
            for i in range(len(contours)):  # 区域个数
                s = {}
                # [[222, 222]]转成单维数组
                contour = []
                for j in range(len(contours[i])):  # 点数
                    contour.append((contours[i][j][0]).tolist())

                s['contours'] = contour
                s['regionArea'] = 0
                s['longPost'] = 0
                s['shortPost'] = 0
                s['longDiam'] = 0
                s['shortDiam'] = 0
                s['longAngle'] = 0
                s['shortAngle'] = 0

                self.l.append(s)

            self.d['isLabel'] = self.isLabel
            self.d['gossArea'] = 0
            self.d['regionList'] = self.l

            return json.dumps(self.d, ensure_ascii=False)

    '''第一阶段深度学习分割创面'''
    def deep_learn(self, network, model_path):
        # 图片输入，Java端传入
        img = transforms.Resize((512, 512), Image.BILINEAR)(self.img)
        img = np.array(img)
        img = transforms.ToTensor()(img.copy()).float().unsqueeze(0)

        model_all = {'FNet': FNet()}  # 字典
        model = model_all[network]
        model = nn.DataParallel(model)  # 用CPU时打开
        cudnn.benchmark = True  # 增加运行效率

        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        with torch.no_grad():
            model.eval()
            main_out, edge_out = model(img)
            predict = torch.round(torch.sigmoid(main_out)).byte()
            pred_seg = predict.data.cpu().numpy() * 255

        result = Image.fromarray(pred_seg.squeeze(), mode='L')

        return result

    '''面积计算'''
    def line_area(self, edge):
        # 绿色矩形标签边缘、面积
        cnt, area_rec = self.rectangle_label()

        # 创面数量
        contours_num = len(edge)

        longest, h1, h2, z1, z2 = 0, 0, 0, 0, 0
        shortest, h3, h4, z3, z4 = 0, 0, 0, 0, 0

        '''最长径'''
        k1 = 0  # 最长径斜率

        # 凸包 旋转卡壳法
        for n in range(contours_num):  # 区域数
            hull_points = cv2.convexHull(edge[n])  # 凸包点集
            index_number = len(hull_points)  # 凸包点个数
            j = 2
            # 遍历边缘上所有点
            for i in range(index_number - 1):
                while (self.compare(hull_points[i][0], hull_points[i + 1][0], hull_points[j][0]) < self.compare(
                        hull_points[i][0], hull_points[i + 1][0], hull_points[j + 1][0])):
                    j = (j + 1) % (index_number - 1)

                x1 = hull_points[i][0][0]  # 坐标在二维数组中
                y1 = hull_points[i][0][1]
                x2 = hull_points[j][0][0]
                y2 = hull_points[j][0][1]
                x3 = hull_points[i + 1][0][0]
                y3 = hull_points[i + 1][0][1]

                if np.sqrt(np.sum((np.array([x1, y1]) - np.array([x2, y2])) ** 2)) >= np.sqrt(
                        np.sum((np.array([x3, y3]) - np.array([x2, y2])) ** 2)):
                    d1 = np.sqrt(np.sum((np.array([x1, y1]) - np.array([x2, y2])) ** 2))
                    if d1 > longest:
                        longest = d1
                        h1 = x1
                        h2 = x2
                        z1 = y1
                        z2 = y2
                        k1 = (z1 - z2) / (h1 - h2)
                else:
                    d1 = np.sqrt(np.sum((np.array([x3, y3]) - np.array([x2, y2])) ** 2))
                    if d1 > longest:
                        longest = d1
                        h1 = x2
                        h2 = x3
                        z1 = y2
                        z2 = y3
                        k1 = (z1 - z2) / (h1 - h2)

            # 存储坐标
            self.heng1.append(h1)
            self.heng2.append(h2)
            self.zong1.append(z1)
            self.zong2.append(z2)
            self.gradient.append(k1)
            # 最长径列表
            self.long_list.append(round((4 * longest) / len(cnt), 2))

            '''最短径'''
            index_number2 = len(edge[n])
            # print(index_number2)
            if index_number2 <= 100:
                s = 1
            else:
                s = 2
            for i in range(0, index_number2, s):
                for j in range(0, index_number2, s):
                    x3 = edge[n][i][0][0]
                    y3 = edge[n][i][0][1]
                    x4 = edge[n][j][0][0]
                    y4 = edge[n][j][0][1]
                    d2 = np.sqrt(np.sum((np.array([x3, y3]) - np.array([x4, y4])) ** 2))
                    if x3 != x4 and (-1 - 0.07) < ((y3 - y4) / (x3 - x4)) * k1 < (-1 + 0.07) and d2 > shortest:
                        shortest = d2
                        z3 = y3
                        z4 = y4
                        h3 = x3
                        h4 = x4

            # 存储坐标
            self.heng3.append(h3)
            self.heng4.append(h4)
            self.zong3.append(z3)
            self.zong4.append(z4)
            # 最短径列表
            self.short_list.append(round((4 * shortest) / len(cnt), 2))

            # 方位角
            self.azimuth(h1, h2, z1, z2)

            longest, shortest = 0, 0

        '''创面面积'''
        area_wound = 0
        for i in range(contours_num):

            # 各子区域面积
            self.region.append(round(cv2.contourArea(edge[i]) / area_rec, 2))

            # 总面积
            area_wound += cv2.contourArea(edge[i])

        # rect = cv2.minAreaRect(cnt)  # 最小外接矩形
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        area_actual = round((area_wound / area_rec), 2)

        return area_actual

    '''凸包计算'''

    #############################################################################################
    def chaji(self, x1, y1, x2, y2):  # 计算叉积
        return (x1 * y2 - x2 * y1)

    def compare(self, a, b, c):  # 计算向量
        return self.chaji((b[0] - a[0]), (b[1] - a[1]), (c[0] - a[0]), (c[1] - a[1]))

    ############################################################################################

    '''绿色标签面积检测'''
    def rectangle_label(self):
        '''yolo算法标签检测'''
        # img0 = cv2.cvtColor(np.asarray(self.img), cv2.COLOR_RGB2BGR)
        # model = self.attempt_load('best.pt', map_location='cpu')
        #
        # names = model.module.names if hasattr(model, 'module') else model.names
        #
        # img = self.letterbox(img0, 640, stride=32)[0]
        # # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = np.ascontiguousarray(img)
        #
        # img = torch.from_numpy(img).to('cpu')
        # img = img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        #
        # pred = model(img, augment=False)[0]
        # pred = self.non_max_suppression(pred, 0.5, 0.45, classes=None, agnostic=False)  # 置信度0.6
        # for i, det in enumerate(pred):
        #     if len(det):
        #         # Rescale boxes from img_size to im0 size
        #         det[:, :4] = self.scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        #         for *xyxy, conf, cls in reversed(det):
        #             label = f'{names[int(cls)]}'
        #             if label == 'blue':
        #                 self.crop = img0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
        #
        # # cv2.imshow('result', self.crop)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        #
        # # 绿色颜色范围
        # lower_g = np.array([35, 43, 46])  # [35, 43/128, 46]
        # upper_g = np.array([77, 255, 255])
        #
        # # 转色域
        # try:
        #     hsv = cv2.cvtColor(self.crop, cv2.COLOR_BGR2HSV)
        # except:
        #     raise IndexError
        # mask_g = cv2.inRange(hsv, lower_g, upper_g)

        #################################################################################
        # MobileNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = transforms.Resize((512, 512), Image.BILINEAR)(self.img)
        img = np.array(img)
        img = normalize(transforms.ToTensor()(img.copy()).float()).unsqueeze(0)

        model = torch.load('mobilenet_0.9858.pt')

        out = model(img)
        out = torch.round(torch.sigmoid(out)).byte()
        out = out.data.cpu().numpy() * 255

        mask_g = Image.fromarray(out.squeeze(), mode='L')
        mask_g = cv2.cvtColor(np.asarray(mask_g), cv2.COLOR_GRAY2BGR)
        mask_g = cv2.resize(mask_g, (self.img.size[0], self.img.size[1]), interpolation=cv2.INTER_AREA)
        # cv2.imshow('result', mask_g)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(mask_g, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        closed = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)

        # closed = cv2.resize(closed, (512, 512))
        # cv2.imshow('result2', closed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # binary2, contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 1:
            cnt = contours[0]
        else:
            raise IndexError

        # ------------------绿色矩形标签面积--------------------
        # 计算优化
        rect = cv2.minAreaRect(cnt)  # 最小外接矩形
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        area_rec = cv2.contourArea(box)
        # area_rec = cv2.contourArea(cnt)

        return cnt, area_rec

# #########################################################################################################
#     def attempt_load(self, weights, map_location='cpu'):
#         # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
#         model = Ensemble()
#         for w in weights if isinstance(weights, list) else [weights]:
#             # attempt_download(w)
#             ckpt = torch.load(w, map_location=map_location)  # load
#             model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model
#
#         # Compatibility updates
#         for m in model.modules():
#             if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
#                 m.inplace = True  # pytorch 1.7.0 compatibility
#             elif type(m) is Conv:
#                 m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
#
#         if len(model) == 1:
#             return model[-1]  # return model
#         else:
#             print('Ensemble created with %s\n' % weights)
#             for k in ['names', 'stride']:
#                 setattr(model, k, getattr(model[-1], k))
#             return model  # return ensemble
#
#     def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#         # Resize and pad image while meeting stride-multiple constraints
#         shape = img.shape[:2]  # current shape [height, width]
#         if isinstance(new_shape, int):
#             new_shape = (new_shape, new_shape)
#
#         # Scale ratio (new / old)
#         r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#         if not scaleup:  # only scale down, do not scale up (for better test mAP)
#             r = min(r, 1.0)
#
#         # Compute padding
#         ratio = r, r  # width, height ratios
#         new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#         dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#         if auto:  # minimum rectangle
#             dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#         elif scaleFill:  # stretch
#             dw, dh = 0.0, 0.0
#             new_unpad = (new_shape[1], new_shape[0])
#             ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
#
#         dw /= 2  # divide padding into 2 sides
#         dh /= 2
#
#         if shape[::-1] != new_unpad:  # resize
#             img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#         top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#         left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#         img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#         return img, ratio, (dw, dh)
#
#     def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
#                             multi_label=False,
#                             labels=()):
#         """Runs Non-Maximum Suppression (NMS) on inference results
#
#         Returns:
#              list of detections, on (n,6) tensor per image [xyxy, conf, cls]
#         """
#
#         nc = prediction.shape[2] - 5  # number of classes
#         xc = prediction[..., 4] > conf_thres  # candidates
#
#         # Settings
#         min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
#         max_det = 300  # maximum number of detections per image
#         max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
#         time_limit = 10.0  # seconds to quit after
#         redundant = True  # require redundant detections
#         multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
#         merge = False  # use merge-NMS
#
#         t = time.time()
#         output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
#         for xi, x in enumerate(prediction):  # image index, image inference
#             # Apply constraints
#             # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
#             x = x[xc[xi]]  # confidence
#
#             # Cat apriori labels if autolabelling
#             if labels and len(labels[xi]):
#                 l = labels[xi]
#                 v = torch.zeros((len(l), nc + 5), device=x.device)
#                 v[:, :4] = l[:, 1:5]  # box
#                 v[:, 4] = 1.0  # conf
#                 v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
#                 x = torch.cat((x, v), 0)
#
#             # If none remain process next image
#             if not x.shape[0]:
#                 continue
#
#             # Compute conf
#             x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
#
#             # Box (center x, center y, width, height) to (x1, y1, x2, y2)
#             box = self.xywh2xyxy(x[:, :4])
#
#             # Detections matrix nx6 (xyxy, conf, cls)
#             if multi_label:
#                 i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
#                 x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
#             else:  # best class only
#                 conf, j = x[:, 5:].max(1, keepdim=True)
#                 x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
#
#             # Filter by class
#             if classes is not None:
#                 x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
#
#             # Apply finite constraint
#             # if not torch.isfinite(x).all():
#             #     x = x[torch.isfinite(x).all(1)]
#
#             # Check shape
#             n = x.shape[0]  # number of boxes
#             if not n:  # no boxes
#                 continue
#             elif n > max_nms:  # excess boxes
#                 x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
#
#             # Batched NMS
#             c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
#             boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
#             i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
#             if i.shape[0] > max_det:  # limit detections
#                 i = i[:max_det]
#             if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
#                 # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
#                 iou = self.box_iou(boxes[i], boxes) > iou_thres  # iou matrix
#                 weights = iou * scores[None]  # box weights
#                 x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
#                 if redundant:
#                     i = i[iou.sum(1) > 1]  # require redundancy
#
#             output[xi] = x[i]
#             if (time.time() - t) > time_limit:
#                 print(f'WARNING: NMS time limit {time_limit}s exceeded')
#                 break  # time limit exceeded
#
#         return output
#
#     def xywh2xyxy(self, x):
#         # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
#         y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
#         y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
#         y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
#         y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
#         y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
#         return y
#
#     def box_iou(self, box1, box2):
#         # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
#         """
#         Return intersection-over-union (Jaccard index) of boxes.
#         Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#         Arguments:
#             box1 (Tensor[N, 4])
#             box2 (Tensor[M, 4])
#         Returns:
#             iou (Tensor[N, M]): the NxM matrix containing the pairwise
#                 IoU values for every element in boxes1 and boxes2
#         """
#
#         def box_area(box):
#             # box = 4xn
#             return (box[2] - box[0]) * (box[3] - box[1])
#
#         area1 = box_area(box1.T)
#         area2 = box_area(box2.T)
#
#         # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#         inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
#         return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
#
#     def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
#         # Rescale coords (xyxy) from img1_shape to img0_shape
#         if ratio_pad is None:  # calculate from img0_shape
#             gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
#             pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
#         else:
#             gain = ratio_pad[0][0]
#             pad = ratio_pad[1]
#
#         coords[:, [0, 2]] -= pad[0]  # x padding
#         coords[:, [1, 3]] -= pad[1]  # y padding
#         coords[:, :4] /= gain
#         self.clip_coords(coords, img0_shape)
#         return coords
#
#     def clip_coords(self, boxes, img_shape):
#         # Clip bounding xyxy bounding boxes to image shape (height, width)
#         boxes[:, 0].clamp_(0, img_shape[1])  # x1
#         boxes[:, 1].clamp_(0, img_shape[0])  # y1
#         boxes[:, 2].clamp_(0, img_shape[1])  # x2
#         boxes[:, 3].clamp_(0, img_shape[0])  # y2

    '''方位角计算'''
    def azimuth(self, h1, h2, z1, z2):
        angle = 0.0

        dx = h2 - h1
        dy = z2 - z1
        if h2 == h1:
            angle = math.pi / 2.0
            if z2 == z1:
                angle = 0.0
            elif z2 < z1:
                angle = 3.0 * math.pi / 2.0
        elif h2 > h1 and z2 > z1:
            angle = math.atan(dx / dy)
        elif h2 > h1 and z2 < z1:
            angle = math.pi / 2 + math.atan(-dy / dx)
        elif h2 < h1 and z2 < z1:
            angle = math.pi + math.atan(dx / dy)
        elif h2 < h1 and z2 > z1:
            angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
        angle = round(angle * 180 / math.pi, 2)

        # 角度转换为时间刻度3点到9点
        if 0 <= angle < 15 or 165 <= angle <= 195:
            clock1 = '6-12'
            clock2 = '3-9'
        elif 15 <= angle < 45 or 195 <= angle <= 225:
            clock1 = '5-11'
            clock2 = '2-8'
        elif 45 <= angle < 75 or 225 <= angle <= 255:
            clock1 = '4-10'
            clock2 = '1-7'
        elif 75 <= angle < 105 or 255 <= angle <= 285:
            clock1 = '3-9'
            clock2 = '6-12'
        elif 105 <= angle < 135 or 285 <= angle <= 315:
            clock1 = '2-8'
            clock2 = '5-11'
        else:
            clock1 = '1-7'
            clock2 = '4-10'
        # else:
        #     clock1 = '6-12'
        #     clock2 = '3-9'
        self.degree1.append(clock1)
        self.degree2.append(clock2)


# class Ensemble(nn.ModuleList):
#     # Ensemble of models
#     def __init__(self):
#         super(Ensemble, self).__init__()
#
#     def forward(self, x, augment=False):
#         y = []
#         for module in self:
#             y.append(module(x, augment)[0])
#         # y = torch.stack(y).max(0)[0]  # max ensemble
#         # y = torch.stack(y).mean(0)  # mean ensemble
#         y = torch.cat(y, 1)  # nms ensemble
#         return y, None  # inference, train output
#
#
# def autopad(k, p=None):  # kernel, padding
#     # Pad to 'same'
#     if p is None:
#         p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
#     return p
#
#
# class Conv(nn.Module):
#     # Standard convolution
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
#         super(Conv, self).__init__()
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#
#     def forward(self, x):
#         return self.act(self.bn(self.conv(x)))
#
#     def fuseforward(self, x):
#         return self.act(self.conv(x))


# if __name__ == "__main__":
#     segment = Segmentation("http://mpa.mynetgear.com:32071/football_team/skincheck/4bf18aabeb03475891a519bcb5e05db8.jpeg")
#     # "https://img2.baidu.com/it/u=2933612501,2880506776&fm=26&fmt=auto&gp=0.jpg"
#     # "https://img1.baidu.com/it/u=3239147958,818764627&fm=26&fmt=auto&gp=0.jpg"
#     o = segment.wound_segmentation()
#     print(o)
