import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def simple_train_val(config=None,model=None,train_loader=None,validate_loader=None,
                     optimizer=None,loss_function=None,logging=None,scheduler=None,val_size=49,train_size=113):
    best_test_acc=0.0
    tv=[]
    vv=[]
    for epoch in range(config.epochs):
        # train
        model.train()
        running_loss = 0.0
        countt = 0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            optimizer.zero_grad()

            if config.deep_supervision:
                pre,out = model(images.cuda().float())
                loss = loss_function(pre,out,labels.cuda().float())
            else:
                out = model(images.cuda().float())
                loss = loss_function(out, labels.cuda().float())

            egt = (out - labels.cuda().float()) ** 2
            egt = torch.sqrt(torch.sum(egt, dim=(2, 3)))
            countt += (egt <= config.threshold).sum().item()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\repoch:{}--train loss: {:^3.0f}%[{}->{}]{:.4f}".format(epoch,int(rate * 100), a, b,loss),end="")
            # logging.info("loss-->%f", loss)
        scheduler.step()
        print()

        # validate
        model.eval()
        with torch.no_grad():
            count=0
            loss=0.0
            for i ,val_data in enumerate(validate_loader):
                val_images, val_labels = val_data
                if config.deep_supervision:
                    pre,outputs = model(val_images.cuda().float())
                    loss += loss_function(pre, outputs, val_labels.cuda().float())
                else:
                    outputs = model(val_images.cuda().float())  # eval model only have last output layer
                    loss += loss_function(outputs, val_labels.cuda().float())
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

            train_accurate = countt / (train_size*3)
            val_accurate = count/(val_size*3)
            tv.append(train_accurate)
            vv.append(val_accurate)
            if val_accurate>best_test_acc:
                best_test_acc = val_accurate
            print('[epoch %d] train_eval_loss: %.4f train_accuracy: %.4f test_eval_loss: %.4f  test_accuracy: %.4f' %
                  (epoch , running_loss/(train_size*config.num_classes),train_accurate,loss/(val_size*config.num_classes), val_accurate))
            logging.info("epoch:%d train_eval_loss-->%f,train_acc-->%f,test_eval_loss-->%f,test_acc===%f", epoch,running_loss/(train_size*config.num_classes),train_accurate,loss/(val_size*config.num_classes),val_accurate)

    print("the best test acc==",best_test_acc)
    # 创建 x 值的列表，可以是迭代的次数或任何你希望作为 x 轴的变量

    # 使用 matplotlib 绘制迭代图
    plt.plot(range(len(vv)), vv, 'o-')

    # 添加标题和标签
    plt.title(f"Test Accuracy over Iterations---{config.network}")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')

    # 显示图形
    plt.show()

