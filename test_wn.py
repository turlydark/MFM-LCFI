import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
import cv2
from matplotlib import pyplot as plt
import torchvision
import timm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pickle
import ctypes
from torch.optim import lr_scheduler
import random

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 20  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 1  # Store a model every snapshot epochs
lr = 0.0002  # Learning rate
bs = 1

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

test_dataset = r'/home/image/df_dataset/celeb_dfv1_test/test'
test_dataloader = DataLoader(Mydata(root=test_dataset,transform=mesonet_data_transforms['test']) , batch_size=bs, shuffle = False,num_workers=8)
model = Model().cuda(0)

model.load_state_dict(torch.load(r'/media/image/新加卷/WN/Deepfake-Detection-master/src/run/run_16/models/b3_ll_DFc23-c23_epoch-16_acc-0.9884.pth')['state_dict'])

criterion = nn.CrossEntropyLoss()
test_size = len(test_dataloader.dataset)

model.eval()
start_time = timeit.default_timer()
running_loss = 0.0
running_corrects = 0.0

cat_probs = None
cat_labels = None

def save_roc_curve(labels, probs):
    print("Saving ROC Curve ...")
    fpr, tpr, _ = roc_curve(labels.detach().cpu().numpy(), probs[:, 1].squeeze().detach().cpu().numpy())
    np.save('fpr', fpr)
    np.save('tpr', tpr)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve_celeba.png")


for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
    probs = nn.Softmax(dim=1)(outputs)
    preds = torch.max(probs, 1)[1]
    loss = criterion(outputs, labels.type(torch.long))

    if type(cat_probs) != type(None):
        cat_probs = torch.cat((cat_probs, probs), dim=0)
        cat_labels = torch.cat((cat_labels, labels), dim=0)
    else:
        cat_probs = probs
        cat_labels = labels

    running_loss += loss.item() * inputs.size(0)
    running_corrects += torch.sum(preds == labels.data)

save_roc_curve(cat_labels, cat_probs)

epoch_loss = running_loss / test_size
epoch_acc = running_corrects.double() / test_size

print("[test] Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
stop_time = timeit.default_timer()
print("Execution time: " + str(stop_time - start_time) + "\n")





def save_loss(train_loss_history, val_loss_history):
    # plotting loss
    print("Saving loss history ...")
    plt.figure()
    plt.plot(train_loss_history, label="Training loss")
    plt.plot(val_loss_history, label="Validation loss")
    plt.xlabel('Iteration')
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.legend()
    plt.savefig("loss_history.png")
    np.save("Train_loss_history", train_loss_history)
    np.save("Val_loss_history", val_loss_history)







