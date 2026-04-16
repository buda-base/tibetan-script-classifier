import cv2
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from huggingface_hub import snapshot_download
from numpy.typing import NDArray
from pathlib import Path


def show_image(image: NDArray, cmap: str = "", axis="off"):
    plt.figure(figsize=(24, 13))
    plt.axis(axis)

    if cmap != "":
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)


def download_hf_model(model_id: str, local_model_dir: str | Path):

    out_dir = Path(local_model_dir) / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    return snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=out_dir,
    )


def resize_to_height(image: NDArray, target_height: int) -> tuple[NDArray, float]:
    scale_ratio = target_height / image.shape[0]
    image = cv2.resize(
        image,
        (int(image.shape[1] * scale_ratio), target_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return image, scale_ratio


def binarize(
    image: NDArray, adaptive: bool = True, block_size: int = 51, c: int = 13
) -> NDArray:
    line_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if adaptive:
        bw = cv2.adaptiveThreshold(
            line_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    else:
        _, bw = cv2.threshold(line_img, 120, 255, cv2.THRESH_BINARY)

    bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return bw


def get_class_prediction(
    image: NDArray,
    model,
    tile_size: int,
    classes: list[str],
    pre_downscale: bool = True,
) -> str:

    if pre_downscale:
        image, _ = resize_to_height(image, tile_size)

    image = binarize(image)
    height_offset = 0
    height, _, _ = image.shape
    width_offset = height // 2

    t_img = tf.image.crop_to_bounding_box(
        image, height_offset, width_offset, tile_size, tile_size
    )

    t_img = tf.expand_dims(t_img, axis=0)
    pred = model.predict(t_img)

    max_value = np.max(pred[0])
    pred_idx = list(pred[0]).index(max_value)
    pred_class = classes[pred_idx]

    return pred_class
