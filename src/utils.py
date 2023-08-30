import glob
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from src.models import Unet


def imshow(img, ax=None):
    """
    displays an image with matplotlib directly from Pytorch format : Channel*H*W
    with figsize 7inch² and without axes
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if len(img.shape) == 3:
        fig = ax.imshow(np.moveaxis(img, 0, -1))
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    else:
        fig = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, "RGB").convert("RGBA")
    overlay = Image.fromarray(color_mask, "RGB").convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def imshow_overlay_test(img_to_show, dataset_test, imgs):
    """
    Show chosen images from the test dataset with the model prediction overlayed on it
    """
    img_to_show = (img_to_show,) if isinstance(img_to_show, int) else img_to_show
    fig, ax = plt.subplots(
        len(img_to_show) // 2 + len(img_to_show) % 2,
        2,
        figsize=(15, 8 * (len(img_to_show) // 2 + len(img_to_show) % 2)),
    )
    ax = ax.flatten()
    for idx, img in enumerate(img_to_show):
        img_real = dataset_test[img].detach().cpu().numpy()
        img_real = np.moveaxis(img_real, 0, -1)
        img_to_print = make_img_overlay(img_real, imgs[img])
        fig = ax[idx].imshow(img_to_print)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    if len(img_to_show) % 2 != 0:
        ax[len(img_to_show)].get_xaxis().set_visible(False)
        ax[len(img_to_show)].get_yaxis().set_visible(False)


def imshow_overlay_validation(img_to_show, dataset_valid, model, device):
    """
    Show chosen images from validation dataset with the ground truth overlayed on it on right
    Show chosen images from validation dataset with the model prediction overlayed on it on right
    """
    img_to_show = (img_to_show,) if isinstance(img_to_show, int) else img_to_show
    fig, ax = plt.subplots(len(img_to_show), 2, figsize=(15, 8 * len(img_to_show)))
    ax = ax.flatten()
    for idx, img in enumerate(img_to_show):
        img_real = dataset_valid[img][0][0].detach().cpu().numpy()
        img_real = np.moveaxis(img_real, 0, -1)
        img_target = dataset_valid[img][1][0].detach().cpu().numpy()
        img_to_print = make_img_overlay(img_real, img_target)
        fig = ax[2 * idx].imshow(img_to_print)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax[2 * idx].set_title(f"Image {img} : target")
        model = model.eval()
        model = model.to(device)
        with torch.no_grad():
            img_pred = model(dataset_valid[img][0].to(device))[0].argmax(0).detach().cpu().numpy()
        img_to_print = make_img_overlay(img_real, img_pred)
        fig = ax[2 * idx + 1].imshow(img_to_print)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        ax[2 * idx + 1].set_title(f"Image {img} : prediction")


def rotate45(img):
    """
    input: image (400*400*3)
    output: 45° rotated image (272*272*3)

    Rotate an image (400*400) of 45° and crop in order to keep the zoom level of the original image
    but without the border due to rotation
    """
    img = ndimage.rotate((img * 255).astype(np.uint8), 45).astype(np.float32) / 255
    crop_x = 272
    crop_y = 272
    if img.ndim == 3:
        y, x, _ = img.shape
        start_x = x // 2 - (crop_x // 2)
        start_y = y // 2 - (crop_y // 2)
        img = img[start_y: start_y + crop_y, start_x: start_x + crop_x, :]
    else:
        y, x = img.shape
        start_x = x // 2 - (crop_x // 2)
        start_y = y // 2 - (crop_y // 2)
        img = img[start_y: start_y + crop_y, start_x: start_x + crop_x]

    return img


def patch_to_label(patch, foreground_threshold):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(nb, imgs, foreground_threshold):
    """Reads a single image and outputs the strings that should go into the submission file"""
    im = imgs[nb]
    img_number = nb + 1
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i: i + patch_size, j: j + patch_size]
            label = patch_to_label(patch, foreground_threshold=foreground_threshold)
            yield "{:03d}_{}_{},{}".format(img_number, j, i, label)


def masks_to_submission(submission_filename, imgs, foreground_threshold):
    """Converts images into a submission file"""
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for i in range(len(imgs)):
            f.writelines(
                "{}\n".format(s)
                for s in mask_to_submission_strings(
                    i, imgs, foreground_threshold=foreground_threshold
                )
            )


def load_model(epoch: Union[int, str] = "latest"):
    model = Unet()
    model_states = glob.glob("unet_states/Unet/*.pth")
    if len(model_states) and (epoch == "latest" or epoch > 0):
        model_states = [int(m[17:-4]) for m in model_states]
        if epoch == "latest":
            epoch = max(model_states)
        state_dict = torch.load(open(f"unet_states/Unet/{epoch}.pth", "rb"), map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        # fresh model
        epoch = 0
    return model, epoch


def save_model(model, epoch, str_output):
    torch.save(model.state_dict(), open(f'unet_states/Unet/{epoch}.pth', 'wb'))
    with open(f'unet_states/Unet/{epoch}.txt', 'w') as file:  # Use file to refer to the file object
        file.write(str_output)
