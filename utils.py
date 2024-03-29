import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Set paths
data_root = 'kaggle/input/airbus-ship-detection/'
path_train = os.path.join(data_root, 'train_v2')
path_test = os.path.join(data_root, 'test_v2')

# Booleans
SHOW_PIXELS_DIST = False
SHOW_SHIP_DIAG = False
SHOW_IMG_LOADER = False

# Training variables
BATCH_SZ_TRAIN = 16
BATCH_SZ_VALID = 4
LR = 1e-4
N_EPOCHS = 3

# Define loss function
LOSS = 'BCEWithDigits' # BCEWithDigits | FocalLossWithDigits | BCEDiceWithLogitsLoss | BCEJaccardWithLogitsLoss

# Define model
MODEL_SEG = 'UNET_RESNET34ImgNet' # UNET | IUNET | UNET_RESNET34ImgNet
FREEZE_RESNET = False   # if UNET_RESNET34ImgNet


# Decode masks in CSV
# Ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# Convert CSV masks to image for a given image name
def maskcsv_to_img(masks, img_name):
    masks_img = np.zeros((768, 768))
    masks_bin = masks.loc[masks['ImageId'] == img_name, 'EncodedPixels'].tolist()
    for mask in masks_bin:
        if isinstance(mask, str):
            masks_img += rle_decode(mask)
    return np.expand_dims(masks_img, -1)


# Convert masks in a list to an image
def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


# Show an image and its corresponding mask
def imshow_mask(img, mask):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    mask = mask.numpy().transpose((1, 2, 0))
    mask = np.clip(mask, 0, 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 30))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[1].imshow(mask)
    axs[1].axis('off')


def imshow_gt_out(img, mask_gt, mask_out):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    mask_gt = mask_gt.numpy().transpose((1, 2, 0))
    mask_gt = np.clip(mask_gt, 0, 1)

    mask_out = mask_out.numpy().transpose((1, 2, 0))
    mask_out = np.clip(mask_out, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(10, 30))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title("Input image")
    axs[1].imshow(mask_gt)
    axs[1].axis('off')
    axs[1].set_title("Ground truth")
    axs[2].imshow(mask_out)
    axs[2].axis('off')
    axs[2].set_title("Model output")
    plt.subplots_adjust(wspace=0, hspace=0)


def imshow_overlay(img, mask, title=None):
    """Imshow for Tensor."""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    mask = mask.numpy().transpose((1, 2, 0))
    mask = np.clip(mask, 0, 1)
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(mask_overlay(img, mask))
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def mask_overlay(image, mask, color=(0, 1, 0)):
    """
    Helper function to visualize mask on the top of the image
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img


# This function transforms EncodedPixels into a list of pixels
# Check our previous notebook for a detailed explanation:
# https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes
def rle_to_pixels(rle_code):
    rle_code = [int(i) for i in rle_code.split()]
    pixels = [(pixel_position % 768, pixel_position // 768)
              for start, length in list(zip(rle_code[0:-1:2], rle_code[1:-2:2]))
              for pixel_position in range(start, start + length)]
    return pixels


def show_pixels_distribution(df):
    """
    Prints the amount of ship and no-ship pixels in the df
    """
    # Total images in the df
    n_images = df['ImageId'].nunique()

    # Total pixels in the df
    total_pixels = n_images * 768 * 768

    # Keep only rows with RLE boxes, transform them into list of pixels, sum the lengths of those lists
    ship_pixels = df['EncodedPixels'].dropna().apply(rle_to_pixels).str.len().sum()

    ratio = ship_pixels / total_pixels
    print(f"Ship: {round(ratio, 3)} ({ship_pixels})")
    print(f"No ship: {round(1 - ratio, 3)} ({total_pixels - ship_pixels})")
