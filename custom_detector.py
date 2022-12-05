import time, random, torch, json
import cv2 as cv
from models.experimental import attempt_load
from utils.torch_utils import select_device, time_synchronized
from utils.general import check_imshow, check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
import torch.backends.cudnn as cudnn
from utils.datasets import LoadStreams

WEIGHTS = './best_weights.pt'
DEVICE = ''
IMAGE_SIZE = 640
CONF_THRESH = 0.25
IOU_THRESH = 0.45

class Detection:
    def __init__(self, label, xyxy, conf) -> None:
        self.label = label
        self.coords = xyxy
        self.conf = conf

    def to_dict(self):
        return {'label': self.label, 'coords': [coord.item() for coord in self.coords], 'conf': self.conf.item()}

def detect():
    device = select_device(DEVICE)
    model = attempt_load(WEIGHTS, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(IMAGE_SIZE, s=stride)

    view_img = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams('0', img_size=IMAGE_SIZE, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH)
        t3 = time_synchronized()

        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            detections = []

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    detections.append(Detection(label, xyxy, conf))
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            yield cv.imencode('.jpg', im0)[1].tobytes(), detections

if __name__ == '__main__':
    detect()