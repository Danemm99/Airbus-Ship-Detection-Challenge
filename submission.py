from skimage.morphology import binary_opening, disk
import numpy as np
from skimage.measure import label, regionprops
import os
from utils import path_test
import pandas as pd
import torch
from dataset import AirbusDataset
from tqdm import tqdm
from train import model
import torch.nn.functional as F
from inference import device


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def multi_rle_encode(img):
    labels = label(img)
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


list_img_test = os.listdir(path_test)
print(len(list_img_test), 'test images found')

# Create dataframe
test_df = pd.DataFrame({'ImageId': list_img_test, 'EncodedPixels': None})
loader = torch.utils.data.DataLoader(dataset=AirbusDataset(test_df, transform=None, mode='test'), shuffle=False,
                                     batch_size=2, num_workers=0)

out_pred_rows = []
for batch_num, (inputs, paths) in enumerate(tqdm(loader, desc='Test')):
    inputs = inputs.to(device)
    outputs = model(inputs)
    for i, image_name in enumerate(paths):
        mask = F.sigmoid(outputs[i, 0]).data.detach().cpu().numpy()
        cur_seg = binary_opening(mask > 0.5, disk(2))
        cur_rles = multi_rle_encode(cur_seg)
        if len(cur_rles) > 0:
            for c_rle in cur_rles:
                out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': c_rle}]
        else:
            out_pred_rows += [{'ImageId': image_name, 'EncodedPixels': None}]

submission_df = pd.DataFrame(out_pred_rows)[['ImageId', 'EncodedPixels']]
submission_df.to_csv('submission.csv', index=False)
submission_df.sample(10)
