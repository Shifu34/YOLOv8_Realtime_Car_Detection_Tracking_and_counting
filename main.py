# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import utils
import deep_sort

from socket_manager import socketio

object_counter = {}

object_counter1 = {}
class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        # Return an instance of Annotator with the input image and line width.
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        # Convert the image to a PyTorch tensor and normalize pixel values to [0.0, 1.0].
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # Convert to half-precision (float16) if specified.
        img /= 255  # Normalize pixel values from 0-255 to 0.0-1.0.
        return img

    def postprocess(self, preds, img, orig_img):
        # Apply non-maximum suppression to filter out duplicate detections and keep the most confident ones.
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        # Scale the bounding box coordinates back to the original image dimensions.
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # Expand for batch dim if not present.
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # Print image shape information.
        self.annotator = self.get_annotator(im0)

        det = preds[idx]  # Get detections for the current batch.
        all_outputs.append(det)
        if len(det) == 0:
            return log_string  # If there are no detections, return the log string.

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # Count detections per class.
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # Write detection results for the current batch.
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = utils.xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)

        # Use the deep_sort library to update the deep sort tracker with current detections.
        outputs = deep_sort.deepsort.update(xywhs, confss, oids, im0)

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            # Use the utils.plot_boxes function to plot bounding boxes and identities on the image.
            utils.plot_boxes(object_counter, object_counter1, im0, bbox_xyxy, self.model.names, object_id, identities,
                             (0, 0), utils.line)

        frame = utils.opencv_to_base64(im0)

        # Simple data structure to store vehicle in and out counts
        vehicle_counts = {
            'in': sum(object_counter.values()),
            'out': sum(object_counter1.values()),
        }
        # Emit vehicle counts and the frame data using socketio for real-time communication.
        socketio.emit('vehicle_counts', vehicle_counts)
        socketio.emit('frame', frame)
        return log_string

# Decorator to specify Hydra configuration options for the predict function.
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    # Initialize the deep_sort tracker.
    deep_sort.init_tracker()

    # Set default values for the model and image size if not provided in the configuration.
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # Check and validate the image size.

    # Set the source directory for images if not provided in the configuration.
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"

    # Create an instance of the DetectionPredictor class with the given configuration.
    predictor = DetectionPredictor(cfg)

    # Call the predictor to start the detection process.
    predictor()


if __name__ == "__main__":
    predict()
