from pathlib import Path
import torch
import random
from metrics import metrics
from tqdm import tqdm
from datetime import datetime
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from data_augmentation import DualCompose, RandomCrop, CenterCrop, VerticalFlip, HorizontalFlip
from dataset import AirbusDataset
import torchvision
from models import UNet, IUNet
import segmentation_models_pytorch as smp
from losses import BCEDiceWithLogitsLoss, BCEJaccardWithLogitsLoss, MixedLoss
import pandas as pd
import torch.optim as optim
from data_processing import valid_df, train_df
from utils import (BATCH_SZ_TRAIN, BATCH_SZ_VALID, SHOW_IMG_LOADER,
                   N_EPOCHS, FREEZE_RESNET, MODEL_SEG, LR, LOSS, imshow_mask)


# Implementation from  https://github.com/ternaus/robot-surgery-segmentation

def train(lr, model, criterion, train_loader, valid_loader, init_optimizer, train_batch_sz=16, valid_batch_sz=4,
          n_epochs=1, fold=1):
    model_path = Path('results/model_{fold}.pt'.format(fold=fold))
    if model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        print('Restored model, epoch {}, step {:,}'.format(epoch, step))
    else:
        epoch = 1
        step = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, str(model_path))

    report_each = 50
    log = open('results/train_{fold}.log'.format(fold=fold), 'at', encoding='utf8')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = init_optimizer(lr)

    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        tq = tqdm(total=len(train_loader) * train_batch_sz)
        tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        valid_metrics = metrics(batch_size=valid_batch_sz)  # for validation
        tl = train_loader
        try:
            mean_loss = 0
            for i, (inputs, targets) in enumerate(tl):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, targets)
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss='{:.5f}'.format(mean_loss))
                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)
            write_event(log, step, loss=mean_loss)
            tq.close()
            save(epoch + 1)

            # Validation
            comb_loss_metrics = validation(model, criterion, valid_loader, valid_metrics)
            write_event(log, step, **comb_loss_metrics)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save(epoch)
            print('done.')
            return


def validation(model: nn.Module, criterion, valid_loader, metrics):
    print("Validation")

    losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    for inputs, targets in valid_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        metrics.collect(outputs.detach().cpu(), targets.detach().cpu())  # get metrics

    valid_loss = np.mean(losses)  # float
    valid_dice, valid_jaccard = metrics.get()  # float

    print('Valid loss: {:.5f}, Jaccard: {:.5f}, Dice: {:.5f}'.format(valid_loss, valid_jaccard, valid_dice))
    comb_loss_metrics = {'valid_loss': valid_loss, 'jaccard': valid_jaccard.item(), 'dice': valid_dice.item()}

    return comb_loss_metrics


def write_event(log, step: int, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()


# Data augmentation
train_transform = DualCompose([HorizontalFlip(), VerticalFlip(), RandomCrop((256, 256, 3))])
val_transform = DualCompose([CenterCrop((512, 512, 3))])

# Initialize dataset
train_dataset = AirbusDataset(train_df, transform=train_transform, mode='train')
val_dataset = AirbusDataset(valid_df, transform=val_transform, mode='validation')

print('Train samples : %d | Validation samples : %d' % (len(train_dataset), len(val_dataset)))

# Get loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SZ_TRAIN, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SZ_VALID, shuffle=True, num_workers=0)

if SHOW_IMG_LOADER == True:
    # Display some images from loader
    images, mask = next(iter(train_loader))
    imshow_mask(torchvision.utils.make_grid(images, nrow=1), torchvision.utils.make_grid(mask, nrow=1))
    plt.show()

# Train
run_id = 1

if MODEL_SEG == 'IUNET':
    model = IUNet()
elif MODEL_SEG == 'UNET':
    model = UNet()
elif MODEL_SEG == 'UNET_RESNET34ImgNet':
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    if FREEZE_RESNET == True:
        for name, p in model.named_parameters():
            if "encoder" in name:
                p.requires_grad = False
else:
    raise NameError("model not supported")

if LOSS == 'BCEWithDigits':
    criterion = nn.BCEWithLogitsLoss()
elif LOSS == 'FocalLossWithDigits':
    criterion = MixedLoss(10, 2)
elif LOSS == 'BCEDiceWithLogitsLoss':
    criterion = BCEDiceWithLogitsLoss()
elif LOSS == 'BCEJaccardWithLogitsLoss':
    criterion = BCEJaccardWithLogitsLoss()
else:
    raise NameError("loss not supported")

train(init_optimizer=lambda lr: optim.Adam(model.parameters(), lr=lr),
      lr=LR,
      n_epochs=N_EPOCHS,
      model=model,
      criterion=criterion,
      train_loader=train_loader,
      valid_loader=val_loader,
      train_batch_sz=BATCH_SZ_TRAIN,
      valid_batch_sz=BATCH_SZ_VALID,
      fold=run_id
      )

# Plot losses
log_file = 'results/train_{fold}.log'.format(fold=run_id)
logs = pd.read_json(log_file, lines=True)

plt.figure(figsize=(26, 6))
plt.subplot(1, 2, 1)
plt.plot(logs.step[logs.loss.notnull()],
         logs.loss[logs.loss.notnull()],
         label="on training set")

plt.plot(logs.step[logs.valid_loss.notnull()],
         logs.valid_loss[logs.valid_loss.notnull()],
         label="on validation set")

plt.xlabel('step')
plt.legend(loc='center left')
plt.tight_layout()
plt.show()
