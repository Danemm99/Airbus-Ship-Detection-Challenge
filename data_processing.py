# Read CSV as dataframe
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from skimage.io import imread
from sklearn.model_selection import train_test_split
from utils import (data_root, SHOW_PIXELS_DIST, show_pixels_distribution, path_train,
                   rle_decode)



masks = pd.read_csv(os.path.join(data_root, 'train_ship_segmentations_v2.csv'))
print('Total number of images (original): %d' % masks['ImageId'].value_counts().shape[0])
if SHOW_PIXELS_DIST == True:
    show_pixels_distribution(masks)
    show_pixels_distribution(masks.dropna())

# Create a dataframe with unique images id as indexes and number of ships and image sizes as new columns
masks = masks[~masks['ImageId'].isin(['6384c3e78.jpg'])]  # remove corrupted file
unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')
print('Total number of images (after removing corrupted images): %d' % masks['ImageId'].value_counts().shape[0])

# Plot some images with ships
img_wships = masks[~masks['EncodedPixels'].isna()].sample(9)
fig, arr = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
for i, img in enumerate(img_wships['ImageId']):
    r = int(i / 3)
    c = i % 3
    arr[r, c].imshow(imread(os.path.join(path_train, img)))
    arr[r, c].axis('off')
plt.show()

# Plot some images without ships
img_woships = masks[masks['EncodedPixels'].isna()].sample(9)
fig, arr = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
for i, img in enumerate(img_woships['ImageId']):
    r = int(i / 3)
    c = i % 3
    arr[r, c].imshow(imread(os.path.join(path_train, img)))
    arr[r, c].axis('off')
plt.show()

# Count number of ships per image
df_wships = masks.dropna()
df_wships = df_wships.groupby('ImageId').size().reset_index(name='counts')
df_woships = masks[masks['EncodedPixels'].isna()]

# Make a plot
plt.bar(['With ships', 'Without ships'], [len(df_wships), len(df_woships)])
plt.ylabel('Number of images')
plt.show()

print('Number of images with ships : %d | Number of images without ships : %d  (x%0.1f)' \
      % (df_wships.shape[0], df_woships.shape[0], df_woships.shape[0] / df_wships.shape[0]))
## -> Unbalanced dataset

# Remove images without ships to help getting a more balanced dataset
masks = masks.dropna()
df_woships = masks[masks['EncodedPixels'].isna()]

# Make a plot
plt.bar(['With ships', 'Without ships'], [len(df_wships), len(df_woships)])
plt.ylabel('Number of images')
plt.show()

print('Number of images with ships : %d | Number of images without ships : %d  (x%0.1f)' \
      % (df_wships.shape[0], df_woships.shape[0], df_woships.shape[0] / df_wships.shape[0]))
## -> Balanced dataset

# Plot histogram
hist = df_wships.hist(bins=df_wships['counts'].max())
plt.title("Histogram of ships count")
plt.xlabel("Number of ships")
plt.ylabel("Number of images")
plt.show(hist)

# Plot images with 15 ships
df_w15ships = df_wships.loc[df_wships['counts'] == 5]
list_w15ships = df_w15ships.values.tolist()

fig, axarr = plt.subplots(2, 2, figsize=(10, 10), constrained_layout=True)
for i in range(4):
    rd_id = random.randrange(len(list_w15ships))
    img_masks = masks.loc[masks['ImageId'] == str(list_w15ships[rd_id][0]), 'EncodedPixels'].tolist()

    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768))
    for mask in img_masks:
        all_masks += rle_decode(mask)

    r = int(i / 2)
    c = i % 2

    axarr[r][c].imshow(imread(os.path.join(path_train, list_w15ships[rd_id][0])))
    axarr[r][c].imshow(all_masks, alpha=0.3)
    axarr[r][c].axis('off')

plt.show()

# Split dataset into training and validation sets
# statritify : same histograms of numbe of ships
unique_img_ids = masks.groupby('ImageId').size().reset_index(name='counts')

train_ids, val_ids = train_test_split(unique_img_ids, test_size=0.05, stratify=unique_img_ids['counts'],
                                      random_state=42)
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, val_ids)

train_df['counts'] = train_df.apply(lambda c_row: c_row['counts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)
valid_df['counts'] = valid_df.apply(lambda c_row: c_row['counts'] if isinstance(c_row['EncodedPixels'], str) else 0, 1)

print('Number of training images : %d' % train_df['ImageId'].value_counts().shape[0])
train_df['counts'].hist(bins=train_df['counts'].max())
plt.title("Histogram of ships count (training)")
plt.xlabel("Number of ships")
plt.ylabel("Number of images")
plt.show()

print('Number of validation images : %d' % valid_df['ImageId'].value_counts().shape[0])
valid_df['counts'].hist(bins=valid_df['counts'].max())
plt.title("Histogram of ships count (validation)")
plt.xlabel("Number of ships")
plt.ylabel("Number of images")
plt.show()
