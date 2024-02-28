# inference.py
import torch
import torchvision
import matplotlib.pyplot as plt
from dataset import AirbusDataset
from utils import imshow_gt_out
from train import run_id, model
from data_processing import valid_df
from train import val_transform

# Assuming the model class and utility functions are imported from model.py and utils.py respectively

# Model inference
model_path = 'results/model_{fold}.pt'.format(fold=run_id)
state = torch.load(str(model_path))
state = {key.replace('module.', ''): value for key, value in state['model'].items()}
model.load_state_dict(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

val_dataset = AirbusDataset(valid_df, transform=val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=0)

# Display some images from loader
images, gt = next(iter(val_loader))
gt = gt.data.cpu()
images = images.to(device)
out = model.forward(images)
out = ((out > 0).float()) * 255
images = images.data.cpu()
out = out.data.cpu()
imshow_gt_out(torchvision.utils.make_grid(images, nrow=1), torchvision.utils.make_grid(gt, nrow=1), torchvision.utils.make_grid(out, nrow=1))
plt.show()
