from dataset.transform import xception_default_data_transforms, two_stream_default_data_transforms
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import matplotlib.pyplot as plt
from xception import TransferModel
from model import Two_Stream_Net
from efficientnet.model import EfficientNet
from model import Fre_Stream_Net
from MAT import MAT, ThreeShallowLayerMAT
from MAT_F3net import ThreeShallowLayerMATwithF3Net
# from sklearn.metrics import roc_auc_score as extra_metric
from sklearn.metrics import roc_curve, auc
import numpy as np
from MAT_F3net_LAD import ThreeShallowLayerMATwithLAD
from MAT import MAT
import timeit
import random

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

def main():
    test_path = "/home/image/ls/FF++_dataset_by_ls/jpg/c23/total/test"
    # test_path = "/home/image/df_dataset/celebaV2_test"

    batch_size = 32
    learning_rate = 0.0001
    model_name = "MAT"

    # print main parameters
    print("model name is :", model_name)
    print("batch_size is :", batch_size)
    print("learning_rate is :", learning_rate)

    output_path = os.path.join('./output', model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark = True
    test_dataset = torchvision.datasets.ImageFolder(
        test_path, transform=two_stream_default_data_transforms['test']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=8
    )
    test_dataset_size = len(test_dataset)

    # model = ThreeShallowLayerMATwithLAD(model_name='efficientnet-b4')
    model = MAT('efficientnet-b4', size=(299,299))
    # model = nn.DataParallel(model)
    static_path = "/home/image/ls/7.15 RGB-and-Fre-Two-Stream-Deepfake-Detection/output/MAT-b4-8.4-c23-total/best.pkl"
    # ThreeShallowLayerMATwithF3Net - b4       this place is b3
    model.load_state_dict(torch.load(static_path))

    cuda = False
    if torch.cuda.is_available():
        cuda = True
    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0.0
    cat_probs = None
    cat_labels = None

    model.eval()
    start_time = timeit.default_timer()
    for (image, labels) in test_loader:
        image = image.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(image)

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels.type(torch.long))
        if type(cat_probs) != type(None):
            cat_probs = torch.cat((cat_probs, probs), dim=0)
            cat_labels = torch.cat((cat_labels, labels), dim=0)
        else:
            cat_probs = probs
            cat_labels = labels
        running_loss += loss.item() * image.size(0)
        running_corrects += torch.sum(preds == labels.data)
    save_roc_curve(cat_labels, cat_probs)
    epoch_loss = running_loss / test_dataset_size
    epoch_acc = running_corrects.double() / test_dataset_size
    print("[test] Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

def save_roc_curve(labels, probs):
    print("Saving ROC Curve ...")
    fpr, tpr, _ = roc_curve(labels.detach().cpu().numpy(), probs[:,1].squeeze().detach().cpu().numpy())
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
    plt.savefig("roc-auc.png")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("testing ...")
    main()

# c40 total
# ThreeShallowLayerMATwithF3Net(LFS) b3
# ACC: 0.8670  AUC: 0.8668

# c40 total
# ThreeShallowLayerMATwith-LAD b4
# Best val Acc: 0.9032
# AUC is : 0.9584

# ThreeShallowLayerMATwith-LAD b4 7.14
# Best val Acc: 0.8955
# AUC is : 0.9550 ( prob value is 0, AUC score is 0.0420)

# ThreeShallowLayerMATwith-LAD b4 7.15
# Best val Acc: 0.9003
# AUC is : 0.9588

#MAT-b4
# Best val Acc: 0.8953

# ThreeShallowLayerMATwith-LAD b4 7.15 (c40 30frames)
# Best val Acc: 0.9043


# c40 DF Acc: 0.9813157305884985  ROC: 0.9969