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
from sklearn.metrics import roc_auc_score as extra_metric
import numpy as np
from MAT_F3net_LAD import ThreeShallowLayerMATwithLAD


def main():
    test_path = "/home/image/ls/forgery_images_c40/total/test"

    batch_size = 8
    learning_rate = 0.0001
    model_name = "ThreeShallowLayerMATwithF3Net"

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

    model = ThreeShallowLayerMATwithLAD(model_name='efficientnet-b4')
    model = nn.DataParallel(model)
    static_path = "./output/Three-Shallow-Layer-MAT-with-LAD-b4-7.14/best.pkl"
    # ThreeShallowLayerMATwithF3Net - b4       this place is b3
    model.load_state_dict(torch.load(static_path), False)

    cuda = False
    if torch.cuda.is_available():
        cuda = True
    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    test_loss = 0.0
    test_corrects = 0.0
    best_acc = 0.0
    # this part for getting AUC score
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for (image, labels) in test_loader:
            image = image.cuda()
            labels = labels.cuda()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            test_loss += loss.data.item()
            test_corrects += torch.sum(preds == labels.data).to(torch.float32)

            all_labels.append(labels.data.cpu().numpy())
            # rectify this place for get roc curve and AUC value
            all_predictions.append(outputs.data.cpu().numpy()[:, 1])
        epoch_loss = test_loss / test_dataset_size
        epoch_acc = test_corrects / test_dataset_size
        print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # best_model_wts = model.state_dict()
        scheduler.step()
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    # AUC the second para is prob
    extra_score = extra_metric(all_labels, all_predictions)
    print('Best val Acc: {:.4f}'.format(best_acc))
    print("AUC is : {:.4f}".format(extra_score))


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
# AUC is : 0.9032 (wrong)

# ThreeShallowLayerMATwith-LAD b4 7.14
# Best val Acc: 0.8955
# AUC is : 0.9550 ( prob value is 0, AUC score is 0.0420)
