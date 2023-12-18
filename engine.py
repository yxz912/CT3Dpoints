import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import math

def simple_train_val(config=None,model=None,train_loader=None,validate_loader=None,
                     optimizer=None,loss_function=None,logging=None,scheduler=None,val_size=34,train_size=113,l_dynamic=1.0):
    best_test_acc=0.0
    best_train_acc=0.0
    tv=[]
    vv=[]
    epct=0
    epcv=0
    test_k=0.0
    train_k=0.0
    cs=0.06
    tip=0
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
                l2_reg = None  # 定义一个空的 L2 正则化项
                for param in model.parameters():
                    if l2_reg is None:
                        l2_reg = param.norm(2)
                    else:
                        l2_reg = l2_reg + param.norm(2)
                loss = loss_function(pre,out,labels.cuda().float(),l2_reg,l_dynamic)
            else:
                out = model(images.cuda().float())
                l2_reg = None  # 定义一个空的 L2 正则化项
                for param in model.parameters():
                    if l2_reg is None:
                        l2_reg = param.norm(2)
                    else:
                        l2_reg = l2_reg + param.norm(2)
                loss = loss_function(out, labels.cuda().float(),l2_reg,l_dynamic)

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
                    l2_reg = None  # 定义一个空的 L2 正则化项
                    for param in model.parameters():
                        if l2_reg is None:
                            l2_reg = param.norm(2)
                        else:
                            l2_reg = l2_reg + param.norm(2)
                    loss += loss_function(pre, outputs, val_labels.cuda().float(),l2_reg,l_dynamic)
                else:
                    outputs = model(val_images.cuda().float())  # eval model only have last output layer
                    l2_reg = None  # 定义一个空的 L2 正则化项
                    for param in model.parameters():
                        if l2_reg is None:
                            l2_reg = param.norm(2)
                        else:
                            l2_reg = l2_reg + param.norm(2)
                    loss += loss_function(outputs, val_labels.cuda().float(),l2_reg,l_dynamic)
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

            train_accurate = countt / (train_size*config.num_classes)
            val_accurate = count/(val_size*config.num_classes)
            tv.append(train_accurate)
            vv.append(val_accurate)
            if val_accurate>best_test_acc:
                if epoch!=epcv:
                    test_k = (val_accurate-best_test_acc)/(epoch-epcv)
                best_test_acc = val_accurate
                epcv = epoch
                torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'l_dynamic':l_dynamic
                }, os.path.join(config.work_dir+'checkpoints/', 'best.pth'))

            if train_accurate>best_train_acc:
                if epoch!=epct:
                    train_k = (train_accurate-best_train_acc)/(epoch-epct)
                best_train_acc = train_accurate
                epct = epoch

            if epoch>22 and config.Dynamic_regularization:
                if train_k-test_k>test_k:
                    if l_dynamic<40:
                        l_dynamic += (train_k/test_k)/1.5
                        if l_dynamic>40:
                            l_dynamic=40

                # if train_accurate-val_accurate>cs:
                #     if l_dynamic < 50:
                #         l_dynamic += l_dynamic*(1-train_accurate+val_accurate)
                #         cs = (train_accurate-val_accurate)/1.2
            # if round(train_accurate,4) == round(best_train_acc,4) and best_train_acc>0.99:
            #     tip +=1
            # else:
            #     tip=0
            # if tip >5 and config.Dynamic_regularization:
            #     if l_dynamic<80:
            #         l_dynamic += tip


            print('[epoch %d] train_eval_loss: %.4f train_accuracy: %.4f test_eval_loss: %.4f  test_accuracy: %.4f' %
                  (epoch , running_loss/(train_size*config.num_classes),train_accurate,loss/(val_size*config.num_classes), val_accurate))
            logging.info("epoch:%d train_eval_loss-->%f,train_acc-->%f,test_eval_loss-->%f,test_acc===%f", epoch,running_loss/(train_size*config.num_classes),train_accurate,loss/(val_size*config.num_classes),val_accurate)

    print("the best test acc==",best_test_acc)
    logging.info("the best test acc===%f",best_test_acc)

    # 使用 matplotlib 绘制迭代图
    plt.plot(range(len(vv)), vv, 'o-')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # 添加标题和标签
    plt.title(f"Test Accuracy over Iterations---{config.network}")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    os.mkdir(config.work_dir + "plt/")
    plt.savefig(config.work_dir + "plt/" + config.network + '.png')
    # 显示图形
    plt.show()


