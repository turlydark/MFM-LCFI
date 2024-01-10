from dataset.transform import xception_default_data_transforms
# from classification.network.models import model_selection
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
from utils import MPerClassSampler, SupConLoss_clear

def main():
    train_path = r"F:\ls\forgery_images_c40\NT2\train"
    val_path = r"F:\ls\forgery_images_c40\NT2\val"
    continue_train = True
    epoches = 20
    batch_size = 16
    learning_rate = 0.0002
    model_name = "Two_Stream_Net"

    # print main parameters
    print("batch_size is :", batch_size)
    print("learning_rate is :",  learning_rate)

    output_path = os.path.join('./output', model_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    torch.backends.cudnn.benchmark = True



    # creat train and val dataloader
    train_dataset = torchvision.datasets.ImageFolder(
        train_path, transform=xception_default_data_transforms['train']
    )
    val_dataset = torchvision.datasets.ImageFolder(
        val_path, transform=xception_default_data_transforms['val']
    )
    # train_sample = MPerClassSampler(train_dataset.labels, batch_size)
    # train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample, num_workers=4)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_sampler=train_sample, shuffle=True, drop_last=False, num_workers=8
    # )
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

    # model = TransferModel('xception', dropout=0.5)
    model = EfficientNet.from_pretrained('efficientnet-b8', num_classes=2, advprop=True)
    model = nn.DataParallel(model)


    cuda = False
    if torch.cuda.is_available():
        cuda = True
    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    feature_criterion = SupConLoss_clear()
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

            layers = model(image)
            outputs = layers['logits']
            _, preds = torch.max(outputs.data, 1)

            features = layers['logits']
            # features = layers['feat']
            feature_loss = feature_criterion(features, labels)

            loss = criterion(outputs, labels) + feature_loss
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

                layers = model(image)
                outputs = layers['logits']

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
    x1 = range(0, 300)
    x2 = range(0, 300)
    y1 = Train_acc_list
    y2 = Train_loss_list
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

# efficientnet-b4
# Nt c40
# lr 0.0001 ACC: 0.8051