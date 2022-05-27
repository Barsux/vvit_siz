import argparse
import os
import sys
from threading import Thread
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from graphs import bbox_rel, draw_boxes

def get_resolution(file):
    vcap = cv2.VideoCapture(file)
    fps = 24
    width = 640
    height = 380
    if vcap.isOpened():
        width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(int(vcap.get(cv2.CAP_PROP_FPS)))
        frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    vcap.release()
    return (fps, frames ,(width, height))

def run_gst_pipeline(filename):
    print("Running pipeline")
    pipeline = f"gst-launch-1.0 filesrc location=tracked/{filename} ! qtdemux ! decodebin ! videoconvert ! videoscale ! theoraenc ! oggmux ! tcpserversink host=127.0.0.1 port=8081"
    print(pipeline)
    os.system(pipeline)


@torch.no_grad()
def run(
        weights,  # model.pt path(s)
        source,  # file/dir/URL/glob, 0 for webcam
        filename,
        run_gstream,
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        run_stream=True,
        save_vid=True,
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_conf=False,  # hide confidences
        debug=True,
        dnn=False,  # use OpenCV DNN for ONNX inference
        config_deepsort=ROOT / "deep_sort_pytorch/configs/deep_sort.yaml"  # Deep Sort configuration
):
    print(f"Filename : {filename}\n File path : {source}")
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    if is_url and is_file:
        source = check_file(source)  # download

    fps, frames, resolution = get_resolution(source)
    out_cap = cv2.VideoWriter(f"tracked/{filename}", 0x7634706d, fps, resolution)

    #Инициировать Глубокую сортировку
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    #Папки
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    #Загрузить модель
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size

    #Запустить детектор
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  #Разминка
    dt, seen = [0.0, 0.0, 0.0], 0
    frame_idx = 0
    time_begin = time_sync()
    objects_total = 0
    frametime_total = 0.0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        t3 = time_sync()
        dt[1] += t3 - t2

        pred = model(im, augment=False, visualize=False)
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=max_det)
        dt[2] += time_sync() - t3

        frame_idx = frame_idx + 1
        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0  # for save_crop

            # check detected boxes, process them for deep sort
            if len(det):
                objects_total += len(det)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                bbox_xywh = []
                confs = []
                #Переделать результаты йолы под глубокую сортировку
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                #Передать детекции в глубокую сортировку
                outputs = deepsort.update(xywhs, confss, im0)

                #Отрисовать результаты глубокой сортировки
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)  # call function to draw seperate object identity

                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                #Нарисовать детекцию от йолы
                for *xyxy, conf, cls in reversed(det):
                    if save_vid:
                        c = int(cls)
                        label = (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

            else:
                deepsort.increment_ages()
            # Stream results
            if run_stream:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            if save_vid:
                out_cap.write(im0)
        frametime = t3 - t2
        LOGGER.info(f'{s}Done. ({frametime:.3f}s)')
        frametime_total += frametime

    time_end = time_sync()
    time_elapsed = time_end - time_begin
    time_per_frame = frametime_total / frames
    print(f"Всего времени прошло : {time_elapsed:.3f}(секунды)")
    print(f"Среднее время трекинга за кадр {time_per_frame:.3f}(cекунды)")
    print(f"Всего объектов в видео : {objects_total}(объекта)")
    if run_gstream:
        Thread(target=run_gst_pipeline, args=(filename,)).start()


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))