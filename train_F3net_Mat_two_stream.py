from dataset.transform import xception_default_data_transforms, two_stream_default_data_transforms
# from classification.network.models import model_selection
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from xception import TransferModel
from model import Two_Stream_Net
from efficientnet.model import EfficientNet
# import timm
from model import Fre_Stream_Net
from MAT import MAT, ThreeShallowLayerMAT
from MAT_F3net import ThreeShallowLayerMATwithF3Net
from MAT_F3net_LAD import ThreeShallowLayerMATwithLAD
import random

random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)
def main():
    # train_path = r"F:\ls\forgery_images_c40\NT2\train"
    # val_path = r"F:\ls\forgery_images_c40\NT2\val"
    train_path = "/home/image/ls/FF++_dataset_by_ls/jpg/c23/total/train"
    val_path = "/home/image/ls/FF++_dataset_by_ls/jpg/c23/total/val"

    continue_train = True
    epoches = 20
    batch_size = 16
    learning_rate = 0.0001
    model_name = "Three-Shallow-Layer-MAT-with-LAD-b4-8.5-c23-total"

    # print main parameters
    print("model name is :", model_name)
    print("batch_size is :", batch_size)
    print("learning_rate is :",  learning_rate)

    output_path = os.path.join('./output', model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark = True

    # creat train and val dataloader
    train_dataset = torchvision.datasets.ImageFolder(
        train_path, transform=two_stream_default_data_transforms['train']
    )
    val_dataset = torchvision.datasets.ImageFolder(
        val_path, transform=two_stream_default_data_transforms['val']
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8
    )
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    # Creat the model
    # model, *_ = model_selection(modelname='xception', num_out_classes=2)

    # 改mat size(256, 256)
    model = ThreeShallowLayerMATwithLAD(model_name='efficientnet-b4')
    # model = MAT('efficientnet-b0')
    # model = TransferModel('xception', dropout=0.5)
    # model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2, advprop=True)
    # all_pretrained_models_available = timm.list_models(pretrained=True)
    # print(all_pretrained_models_available)
    # print(len(all_pretrained_models_available))
    # # exit(0)
    # model = nn.DataParallel(model)


    cuda = False
    if torch.cuda.is_available():
        cuda = True
    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model using multiple GPUs
    # model = nn.DataParallel(model)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    Train_loss_list = []
    Train_acc_list = []
    Val_loss_list = []
    Val_acc_list = []
    for epoch in range(epoches):
        print('Epoch {}/{}'.format(epoch + 1, epoches))
        print('-' * 10)
        model = model.train()
        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        for (image, labels) in train_loader:
            iter_loss = 0.0
            iter_corrects = 0.0

            image = image.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            # outputs = model(image)

            # layers = model(image)
            # outputs = layers['logits']

            outputs = model(image)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1
            if not (iteration % 100):
                print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size,
                                                                           iter_corrects / batch_size))
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        Train_loss_list.append(epoch_loss)
        Train_acc_list.append(epoch_acc)
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()
        with torch.no_grad():
            for (image, labels) in val_loader:
                image = image.cuda()
                labels = labels.cuda()
                # outputs = model(image)

                # layers = model(image)
                # outputs = layers['logits']
                outputs = model(image)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            Val_loss_list.append(epoch_loss)
            Val_acc_list.append(epoch_acc)
            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        scheduler.step()
        if not (epoch % 10):
            # Save the model trained with multiple gpu
            # torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    # torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
    torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))

    # 绘图
    x1 = range(0, 20)
    x2 = range(0, 20)
    # y1 = Train_acc_list.cpu()
    y1 = np.array(torch.tensor(Train_acc_list, device='cpu'))
    # y2 = Train_loss_list.cpu()
    y2 = np.array(torch.tensor(Train_loss_list, device='cpu'))
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, '.-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training ...")
    main()


# c40 NT
# xception original
# lr 0.0001  ACC: 0.7941 0.7982
# Efficient net b0
# lr 0.0001 Acc: 0.8064
# b4
# lr 0.001 ACC:  0.7940
# MAT
# lr 0.0001 ACC:  0.8012


# c40 total
# MAT
# lr 0.0001 ACC:  0.9059 0.9079
# lr 0.001  ACC:  0.8947

# ThreeShallowLayerMAT b3
# lr 0.0001 bs 8 ACC:  0.9108  0.9159 0.9104
# lr 0.001 bs 16 ACC: 0.9083

# ThreeShallowLayerMAT-with-F3Net-b4
# lr 0.0001 ACC: 0.9124

# Three-Shallow-Layer-MAT-with-LAD-b4
# lr 0.0001 ACC: 0.9113 0.9172

# Three-Shallow-Layer-MAT-with-LAD-b4-7.15
# batch_size is: 16
# learning_rate is: 0.0001
# 0.9239

#c40 df Best val Acc: 0.9706
# c40 f2f 0.9020

# c23 total 0.9568