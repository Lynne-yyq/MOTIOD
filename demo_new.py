import argparse
import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F
from models import build_tracktest_model
from models import Tracker
from util.misc import nested_tensor_from_tensor_list
from track_tools.colormap import colormap

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=500, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='mot')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/path/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # PyTorch checkpointing for saving memory (torch.utils.checkpoint.checkpoint)
    parser.add_argument('--checkpoint_enc_ffn', default=False, action='store_true')
    parser.add_argument('--checkpoint_dec_ffn', default=False, action='store_true')

    # demo
    parser.add_argument('--video_input', default='/path/video')
    parser.add_argument('--demo_output', default='/path/output')
    parser.add_argument('--track_thresh', default=0.7, type=float)

    parser.add_argument('--T1', default=1.2, type=float, help="Threshold T1 for tracking")
    parser.add_argument('--T2', default=20, type=int, help="Threshold T2 for tracking")
    parser.add_argument('--alpha', default=0.3, type=int, help="Weight of sow")

    return parser

def resize(image, size=800, max_size=1333):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        h, w = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    new_height, new_width = get_size_with_aspect_ratio(image.shape[:2], size, max_size)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image, new_height, new_width

def detect_dark_zone(img, video_height, video_width):
    b, g, r = cv2.split(img)
    light = b.sum(axis=1)
    left = 0
    right = video_width
    for i in range(50, video_width, 50):
        sum_dark = np.sum(light[i-50:i+1] < 150)
        if sum_dark > (50 * video_height) // 3:
            if i < video_width // 2:
                left = i
            else:
                right = i - 50

    if right == 0:
        right = video_width
    if video_width == 960:
        left += 50
        right -= 50

    return left, right

def load_model(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, postprocessors = build_tracktest_model(args)
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device).eval()
    return model, postprocessors, device

def process_data(frame, model, device, postprocessors, tracker, args, frame_count):
    resized_img, nh, nw = resize(frame)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    tensor_img = F.normalize(F.to_tensor(rgb_img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    samples = nested_tensor_from_tensor_list([tensor_img]).to(device)

    with torch.no_grad():
        outputs = model(samples)
    out_boxes = outputs[0]

    results = postprocessors['bbox'](out_boxes, torch.stack([torch.tensor([frame.shape[0], frame.shape[1]]).to(device)], dim=0).to(device))

    left, right = detect_dark_zone(resized_img, nh, nw)

    if frame_count == 1:
        tracks = tracker.init_track(results[0])
        score = None
        data = None
    else:
        tracks, outliers, score, data, out_id, his_track, out_ind_result = tracker.step(results[0], left, right, frame_count, args.T1, args.T2)

    return tracks, left, right, score, data
def draw_and_save(results, frame, left, right, frame_count, video_name, args, score, data, tracker):
    results.sort(key=lambda x: x['tracking_id'])

    color_list = colormap()
    output_dir = os.path.join(args.demo_output, os.path.splitext(video_name)[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
    txt_path = os.path.join(output_dir, "tracking.txt")
    outlier_history_path = os.path.join(output_dir, "outlier.txt")

    for result in results:
        bbox = result['bbox']
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                      color_list[result['tracking_id'] % len(color_list)].tolist(), 4)
        if frame_count > 1:
            tracking_id = result['tracking_id']
            w1 = int(bbox[2]) - int(bbox[0])
            h1 = int(bbox[3]) - int(bbox[1])
            with open(txt_path, 'a') as f:
                line = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'.format(frame=frame_count, id=tracking_id,
                                                                             x1=int(bbox[0]), y1=int(bbox[1]), w=w1,
                                                                             h=h1)
                f.write(line)
            cv2.putText(frame, "{}".format(tracking_id), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        color_list[tracking_id % 79].tolist(), 4)

            current_frame_id = list(tracker.outlier_history.keys())[-1]
            current_frame_data = tracker.outlier_history.get(current_frame_id, {}).get('outlier_list', [])


            outlier_ids = [idx for idx, value in enumerate(current_frame_data[1:], start=1) if value == 1]
            outlier_text = f"Count: {len(outlier_ids)}, IDs: {', '.join(map(str, outlier_ids))}"


            (width, height), _ = cv2.getTextSize(outlier_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)

            start_x = (frame.shape[1] - width) // 2
            start_y = height + 20
            cv2.putText(frame, outlier_text, (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)  

            with open(outlier_history_path, 'w') as f:
                for frame_id, frame_data in tracker.outlier_history.items():
                    outlier_list = frame_data['outlier_list']
                    f.write(str(frame_id) + ' ' + ' '.join(map(str, outlier_list[1:])) + '\n')

            for id, i in enumerate(data):
                cv2.rectangle(frame, (i[0], i[1]), (i[0] + 10, i[1] + 10),
                              color_list[0].tolist(), thickness=2)
                cv2.putText(frame, "%.2f" % score[id], (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            color_list[6].tolist(), 2)

    cv2.imwrite(image_path, frame)


def main(args):
    model, postprocessors, device = load_model(args)
    tracker = Tracker(score_thresh=args.track_thresh, alpha=args.alpha, T1=args.T1, T2=args.T2)
    videos = [os.path.join(args.video_input, f) for f in os.listdir(args.video_input) if os.path.isfile(os.path.join(args.video_input, f))]

    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            results, left, right, score, data = process_data(frame, model, device, postprocessors, tracker, args, frame_count)
            draw_and_save(results, frame, left, right, frame_count, os.path.basename(video_path), args, score, data, tracker)
            print('Frame{:d} of the video is done'.format(frame_count))
        cap.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Demo for TransTrack', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
