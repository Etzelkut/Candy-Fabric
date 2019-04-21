import PyCmdMessenger
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from shutil import copyfile
import time
import torch
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ('blue', 'red')
#
def train():
    if torch.cuda.is_available()==True:
        print("GGGGGGGGGGGGGGGGGGGG")

    #
    transform_train = transforms.Compose([transforms.Resize((224,224)),
    transforms.RandomRotation(degrees=90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    #
    transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    #
    train_dataset = datasets.ImageFolder('sweets_candy/train', transform=transform_train)
    #
    validation_dataset = datasets.ImageFolder('sweets_candy/val', transform=transform)
    #
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=15, shuffle=True)
    #
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=15, shuffle=False)
    # 
    candy_identifactor = models.alexnet(pretrained=True)

    for param in candy_identifactor.features.parameters():
        param.requires_grad = False
    #
    n_inputs = candy_identifactor.classifier[6].in_features
    last_layer = nn.Linear(n_inputs, len(classes))

    candy_identifactor.classifier[6] = last_layer

    candy_identifactor = candy_identifactor.to(device)

    criterion = nn.CrossEntropyLoss()

    #smaller dataset smaller ls
    optimizer = torch.optim.Adam(candy_identifactor.parameters(), lr = 0.0001)

    #smaller dataset smaller epochs
    epochs = 10
    losses = []
    correct_history = []

    va_losses = []
    va_correct_history = []

    for i in range(epochs):
        rloss=0.0
        running_corrects=0.0

        va_rloss = 0.0
        va_running_corrects=0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = candy_identifactor.forward(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            rloss+=loss.item()

        else:
            with torch.no_grad():
                for va_inputs, va_labels in validation_loader:
                    va_inputs = va_inputs.to(device)
                    va_labels = va_labels.to(device)
                    va_outputs = candy_identifactor.forward(va_inputs)
                    va_loss = criterion(va_outputs, va_labels)

                    _, va_preds = torch.max(va_outputs, 1)
                    va_running_corrects += torch.sum(va_preds == va_labels.data)
                    va_rloss+=va_loss.item()       
            #same
            epoch_loss = rloss/len(train_loader.dataset)
        #divide now by entire datset due to batch size decreased
            epoch_correct = running_corrects.float()/len(train_loader.dataset)
            
            va_epoch_loss = va_rloss/len(validation_loader.dataset)
            va_epoch_correct = va_running_corrects.float()/len(validation_loader.dataset)
            
            va_losses.append(va_epoch_loss)
            va_correct_history.append(va_epoch_correct)

            correct_history.append(epoch_correct)
            losses.append(epoch_loss)
            print("Epoch: ", i)
            print('training loss: {}, {}'.format(epoch_loss, epoch_correct.item()))
            print('validation loss: {}, {}'.format(va_epoch_loss, va_epoch_correct.item()))
    return candy_identifactor, transform 


def startt(candy_identifactor, transform):
    arduino = PyCmdMessenger.ArduinoBoard("/dev/ttyACMnumber",baud_rate=9600) #number of your Arduino ACM
    commands = [["green_color",""],
                ["greeen", "s"],
                ["red_color",""],
                ["reed", "s"],
                ["what_camera",""],
                ["state_of_camera","s"],
                ["change_var",""],
                ["var_news","s"],
                ["error","s"]]
    #command initialized in c 
    c = PyCmdMessenger.CmdMessenger(arduino,commands)
    print("Starting...")
    #path on phone 
    path="/run/user/1000/gvfs/mtp:host=%5Busb%3A003%2C033%5D/Внутренний общий накопитель/" #path="/run/user/1000/gvfs/mtp:host=%5Busb%3A003%2C023%5D/Internal storage/"
    while True:
        c.send("what_camera")
        msg = c.receive()
        if msg is not None:
            new_line = msg[1]
        if msg is None:
            while msg is None: #or not isinstance(new_line[0], int)
                c.send("what_camera")
                msg = c.receive()
                if msg is not None:
                    new_line = msg[1]
        if new_line[0]!="Command without callback.":
            varstate = int(new_line[0])#ValueError: invalid literal for int() with base 10: 'Command without callback.'
        elif new_line[0]=="Command without callback.":
            varstate = 1
        if varstate == 1:
            print("here")
            print("aldik")
            f = open(path+"proard/command.txt",'w')
            f.write('1\n')
            f.write('0\n')
            f.close()
            while True:   
                with open(path+'proard/command.txt', 'r') as file:   	
                    data = file.readlines()
                if len(data)>1:
                    time.sleep(0.5)
                    if data[1]=="1\n":
                        copyfile(path+"DCIM/pro.jpg","source.jpg")  #
                        imag = Image.open("source.jpg")
                        imag = transform(imag)
                        imag=imag.to(device).unsqueeze(0)
                        output = candy_identifactor.forward(imag)
                        _, pred = torch.max(output, -1)
                        print("")
                        if int(pred.item()) == 0:
                            print("green and ")
                            c.send("green_color")
                            msg = c.receive()
                            if msg is None:
                                while msg is None:
                                    c.send("green_color")
                                    msg = c.receive()
                            print(classes[0])
                        else:
                            print("red and ")
                            c.send("red_color")
                            msg = c.receive()
                            if msg is None:
                                while msg is None:
                                    c.send("red_color")
                                    msg = c.receive()
                            print(classes[1])
                        print("")
                        f = open(path+"proard/command.txt",'w')
                        f.write('0\n')
                        f.write('0\n')
                        f.close()
                        print("bitty...")
                        print("breaking")
                        break

