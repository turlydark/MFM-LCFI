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
from model import Two_Stream_Net, Fre_Stream_Net



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
    # gpu_ids = [*range(osenvs)]
    pretrained_path = './xception-b5690688.pth'
    # model = Two_Stream_Net()
    model =  Fre_Stream_Net()
    model = nn.DataParallel(model)


    cuda = False
    if torch.cuda.is_available():
        cuda = True
    if cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # sc_criterion = SupConLoss_clear()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model using multiple GPUs
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
            outputs, fea = model(image)
            # print(fea)
            #
            # print(fea.size())
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels) + sc_criterion(fea, labels)
            #print(criterion(outputs, labels))
            # print(sc_criterion(fea, labels))
            # exit(0)

            _, preds = torch.max(outputs, 1)
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
                outputs, fea = model(image)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                iter_loss = loss.data.item()
                val_loss += iter_loss
                iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
                val_corrects += iter_corrects
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
    print("Training ...")
    main()


# c40 NT
# Two Stream
# lr 0.0001  ACC 0.7958 0.8115  0.8068 0.8065
# lr 0,001   ACC 0.7781

# Fre
# lr 0.0001 ACC


