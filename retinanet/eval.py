import argparse
import json
import sys
from typing import Any, Dict, List, NoReturn

import numpy as np
from tqdm import tqdm


def iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    calculate intersection over union cover percent
    :param box1: box1 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :param box2: box2 with shape (N,4) or (N,2,2) or (2,2) or (4,). first shape is preferred
    :return: IoU ratio if intersect, else 0
    """
    # first unify all boxes to shape (N,4)
    if box1.shape[-1] == 2 or len(box1.shape) == 1:
        box1 = box1.reshape(1, 4) if len(box1.shape) <= 2 else box1.reshape(box1.shape[0], 4)
    if box2.shape[-1] == 2 or len(box2.shape) == 1:
        box2 = box2.reshape(1, 4) if len(box2.shape) <= 2 else box2.reshape(box2.shape[0], 4)
    point_num = max(box1.shape[0], box2.shape[0])
    b1p1, b1p2, b2p1, b2p2 = box1[:, :2], box1[:, 2:], box2[:, :2], box2[:, 2:]

    # mask that eliminates non-intersecting matrices
    base_mat = np.ones(shape=(point_num,))
    base_mat *= np.all(np.greater(b1p2 - b2p1, 0), axis=1)
    base_mat *= np.all(np.greater(b2p2 - b1p1, 0), axis=1)

    # I area
    intersect_area = np.prod(np.minimum(b2p2, b1p2) - np.maximum(b1p1, b2p1), axis=1)
    # U area
    union_area = np.prod(b1p2 - b1p1, axis=1) + np.prod(b2p2 - b2p1, axis=1) - intersect_area
    # IoU
    intersect_ratio = intersect_area / union_area

    return base_mat * intersect_ratio


def nms(dets: np.ndarray, thresh: float) -> List[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Separate evaluator for f1/jaccard scores")
    parser.add_argument("--gt-annotations", type=str)
    parser.add_argument("--predict-annotations", type=str)
    parser.add_argument("--output-file-path", type=str)
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="iou threshold")
    parser.add_argument(
        "--nms-threshold", type=float, default=0.8, help="NMS threshold for filtering ALL predicted bboxes"
    )
    parser.add_argument("--score-threshold", type=float, default=0.5, help="bbox score threshold")
    args = parser.parse_args()
    return args


def f1_jaccard_score(predict_annotations, gt_objects, iou_threshold=0.5, nms_threshold=0.8, score_threshold=0.5):
    gt_images = gt_objects["images"]
    gt_annotations = gt_objects["annotations"]
    predict_annotations = predict_annotations["annotations"]
    predict_annotations = [ann for ann in predict_annotations if ann["score"] >= score_threshold]

    classes_dict = {class_object["id"]: class_object["name"] for class_object in gt_objects["categories"]}
    results: Dict[int, Dict[str, float]] = {
        class_object["id"]: dict(tp=0, fp=0, fn=0) for class_object in gt_objects["categories"]
    }
    # for human
    human_results = dict(tp=0, fp=0, fn=0)

    for gt_image in tqdm(gt_images):
        image_gt_annotations: List[Dict[str, Any]] = list(
            filter(lambda x: x["image_id"] == gt_image["id"], gt_annotations)
        )
        image_predict_annotations: List[Dict[str, Any]] = list(
            filter(lambda x: x["image_id"] == gt_image["id"], predict_annotations)
        )

        # NMS FOR ALL CLASSES BEFORE INFRACTION
        image_predict_annotations = sorted(image_predict_annotations, key=lambda x: x["score"], reverse=True)

        image_predict_annotations = [ann for ann in image_predict_annotations if ann["category_id"] in results]
        image_predict_bboxes_xywh = np.array([ann["bbox"] for ann in image_predict_annotations])
        image_predict_bboxes_xyxy = np.array(
            [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in image_predict_bboxes_xywh]
        )
        image_predict_bboxes_scores = np.array([ann["score"] for ann in image_predict_annotations])
        image_predict_bboxes_scores = image_predict_bboxes_scores.reshape(-1, 1)
        if len(image_predict_bboxes_xyxy) > 0:
            image_predict_bboxes_xyxy_scores = np.concatenate(
                [image_predict_bboxes_xyxy, image_predict_bboxes_scores], axis=1
            )
            keep_predictions = nms(dets=image_predict_bboxes_xyxy_scores, thresh=nms_threshold)
            image_predict_annotations = [image_predict_annotations[keep_indice] for keep_indice in keep_predictions]

        # infraction metrics
        for class_id in results:
            class_gt_annotations: List[Dict[str, Any]] = list(
                filter(lambda x: x["category_id"] == class_id, image_gt_annotations)
            )
            class_predict_annotations: List[Dict[str, Any]] = list(
                filter(lambda x: x["category_id"] == class_id, image_predict_annotations)
            )

            class_gt_bboxes_xywh = np.array([ann["bbox"] for ann in class_gt_annotations])
            class_predict_bboxes_xywh = np.array([ann["bbox"] for ann in class_predict_annotations])
            class_predict_bboxes_scores = np.array([ann["score"] for ann in class_predict_annotations])
            class_predict_bboxes_scores = class_predict_bboxes_scores.reshape(-1, 1)

            class_gt_bboxes_xyxy = np.array(
                [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in class_gt_bboxes_xywh]
            )
            class_predict_bboxes_xyxy = np.array(
                [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in class_predict_bboxes_xywh]
            )

            # sort by score
            if len(class_predict_bboxes_xyxy) > 0:
                class_predict_bboxes_xyxy_scores = np.concatenate(
                    [class_predict_bboxes_xyxy, class_predict_bboxes_scores], axis=1
                )
                class_predict_bboxes_xyxy_scores = class_predict_bboxes_xyxy_scores[
                    class_predict_bboxes_xyxy_scores[:, -1].argsort()[::-1]
                ]
                class_predict_bboxes_xyxy = class_predict_bboxes_xyxy_scores[:, :-1]

            for _, predict_bbox in enumerate(class_predict_bboxes_xyxy):
                iou_all = np.array([iou(predict_bbox, x) for x in class_gt_bboxes_xyxy])
                if (iou_all > iou_threshold).any():
                    results[class_id]["tp"] += 1
                    according_gt = np.argmax(iou_all)
                    class_gt_bboxes_xyxy = np.delete(class_gt_bboxes_xyxy, according_gt, 0)
                else:
                    results[class_id]["fp"] += 1
            results[class_id]["fn"] += len(class_gt_bboxes_xyxy)

        # human metrics
        human_gt_annotations: List[Dict[str, Any]] = []
        human_predict_annotations: List[Dict[str, Any]] = []
        for class_id in results:
            human_gt_annotations += list(filter(lambda x: x["category_id"] == class_id, image_gt_annotations))
            human_predict_annotations += list(
                filter(lambda x: x["category_id"] == class_id, image_predict_annotations)
            )
        human_gt_bboxes_xywh = np.array([ann["bbox"] for ann in human_gt_annotations])
        human_predict_bboxes_xywh = np.array([ann["bbox"] for ann in human_predict_annotations])
        # for nms
        human_predict_bboxes_scores = np.array([ann["score"] for ann in human_predict_annotations]).reshape(-1, 1)

        human_gt_bboxes_xyxy = np.array(
            [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in human_gt_bboxes_xywh]
        )
        human_predict_bboxes_xyxy = np.array(
            [[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in human_predict_bboxes_xywh]
        )

        # nms
        if len(human_predict_bboxes_xyxy) > 0:
            human_predict_bboxes_xyxy_scores = np.concatenate(
                [human_predict_bboxes_xyxy, human_predict_bboxes_scores], axis=1
            )
            keep_predictions = nms(dets=human_predict_bboxes_xyxy_scores, thresh=nms_threshold)
            human_predict_bboxes_xyxy = human_predict_bboxes_xyxy[keep_predictions]

        for _, predict_bbox in enumerate(human_predict_bboxes_xyxy):
            iou_all = np.array([iou(predict_bbox, x) for x in human_gt_bboxes_xyxy])
            if (iou_all > iou_threshold).any():
                human_results["tp"] += 1
                according_gt = np.argmax(iou_all)
                human_gt_bboxes_xyxy = np.delete(human_gt_bboxes_xyxy, according_gt, 0)
            else:
                human_results["fp"] += 1
        human_results["fn"] += len(human_gt_bboxes_xyxy)

    final_metrics: Dict[str, Any] = dict()
    total_jaccard: float = 0
    total_f1_score: float = 0
    total_precision: float = 0
    total_recall: float = 0
    for class_id in results:
        class_name = classes_dict[class_id]
        tp = results[class_id]["tp"]
        fp = results[class_id]["fp"]
        fn = results[class_id]["fn"]
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1_score = 2 * precision * recall / (precision + recall + 1e-9)
        jaccard = tp / (tp + fp + fn + 1e-9)
        final_metrics[class_name + " precision"] = precision
        final_metrics[class_name + " recall"] = recall
        final_metrics[class_name + " f1_score"] = f1_score
        final_metrics[class_name + " jaccard"] = jaccard
        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score
        total_jaccard += jaccard
    final_metrics["total_infraction_jaccard"] = total_jaccard / len(results)
    final_metrics["total_infraction_f1"] = total_f1_score / len(results)
    final_metrics["total_infraction_precision"] = total_precision / len(results)
    final_metrics["total_infraction_recall"] = total_recall / len(results)

    # human metrics
    tp = human_results["tp"]
    fp = human_results["fp"]
    fn = human_results["fn"]
    human_precision = tp / (tp + fp + 1e-9)
    human_recall = tp / (tp + fn + 1e-9)
    human_f1_score = 2 * human_precision * human_recall / (human_precision + human_recall + 1e-9)
    human_jaccard = tp / (tp + fp + fn + 1e-9)
    final_metrics["human precision"] = human_precision
    final_metrics["human recall"] = human_recall
    final_metrics["human f1_score"] = human_f1_score
    final_metrics["human jaccard"] = human_jaccard
    return final_metrics


def main() -> NoReturn:
    args = parse_args()
    iou_threshold = args.iou_threshold
    nms_threshold = args.nms_threshold
    score_threshold = args.score_threshold

    with open(args.gt_annotations) as file:
        gt_objects: Dict[str, Any] = json.load(file)

    with open(args.predict_annotations) as file:
        predict_annotations: List[Any] = json.load(file)

    final_metrics = f1_jaccard_score(
        predict_annotations,
        gt_objects,
        iou_threshold=iou_threshold,
        nms_threshold=nms_threshold,
        score_threshold=score_threshold,
    )

    with open(args.output_file_path, "w") as file:
        json.dump(final_metrics, file, indent=4)
    print(final_metrics)


if __name__ == "__main__":
    main()
