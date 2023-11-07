import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix

def simple_train_val(config=None,model=None,train_loader=None,validate_loader=None,
                     optimizer=None,loss_function=None,logging=None,scheduler=None,val_size=49):
    for epoch in range(config.epochs):
        # train
        model.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()

            if config.deep_supervision:
                pre,out = model(images.cuda().float())
                loss = loss_function(pre,out,labels.cuda().float())
            else:
                logits = model(images.cuda().float())
                loss = loss_function(logits, labels.cuda().float())

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch:{}--train loss: {:^3.0f}%[{}->{}]{:.4f}".format(epoch,int(rate * 100), a, b, loss), end="")
            logging.info("loss-->%f", loss)
        scheduler.step()
        print()

        # validate
        model.eval()
        with torch.no_grad():
            count=0
            for i ,val_data in enumerate(validate_loader):
                val_images, val_labels = val_data
                if config.deep_supervision:
                    _,outputs = model(val_images.cuda().float())
                else:
                    outputs = model(val_images.cuda().float())  # eval model only have last output layer
                eg = (outputs - val_labels.cuda().float()) ** 2
                #eg=torch.sum(eg,dim=(1,2,3))
                #count += (eg <= config.threshold).sum().item()
                eg=torch.sqrt(torch.sum(eg,dim=(2,3)))

                # for k in range(config.val_bs):
                #     coun = (eg[k] <= config.threshold).sum().item()
                #     print(coun,eg[k])
                #     if coun==3:
                #         count+=1
                count += (eg <= config.threshold).sum().item()

            val_accurate = count/(val_size*3)

            print('[epoch %d] train_eval_loss: %.4f  test_accuracy: %.4f' %
                  (epoch , running_loss/(step+1), val_accurate))



