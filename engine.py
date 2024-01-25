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
    TIP = 0.001
    acc=0.0
    fzd = []
    mec_train=[]
    mec_val=[]
    for epoch in range(config.epochs):
        # train
        model.train()
        running_loss = 0.0
        countt = 0
        tik=0
        freeze = []
        for step, data in enumerate(train_loader, start=0):
            if config.data_mmld:
                #images,labels,spacing = data
                images = data['image']
                labels= data['landmarks']
                spacing = data['spacing']

            else:
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
            if config.data_mmld:
                outt = []
                for i, ten in enumerate(out):
                    ot = (ten - labels.cuda().float()[i]) * spacing[i].cuda().float()
                    outt.append(ot[:].tolist())
                egt = torch.tensor(outt) ** 2
                egt = torch.sqrt(torch.sum(egt, dim=(2, 3)))
                for dt,kg in enumerate(egt):
                    for jp,dq in enumerate(kg):
                        if torch.min(labels.float()[dt,jp,:,:]) >= 0 and dq<=config.threshold:
                            countt += 1
                        if torch.min(labels.float()[dt,jp,:,:]) < 0:
                            tik+=1
            else:
                egt = (out - labels.cuda().float()) ** 2
                egt = torch.sqrt(torch.sum(egt, dim=(2, 3)))
                countt += (egt <= config.threshold).sum().item()

            loss.backward()

            if config.freeze and epoch > int(0.8 * config.epochs) and acc > 0.985:  # 0.8*config.epochs
                # 冻结部分层
                for name, param in model.named_parameters():
                    if (param.grad is not None) and (abs(torch.mean(param.grad).item()) < TIP) and (
                            "tail" not in name) and (name not in fzd):
                        freeze.append(name)
                    elif param.grad is None and (name not in fzd):
                        freeze.append(name)
                if TIP < 0.1:
                    TIP += 0.004

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
        mec_train.append(sum(sum(egt.cpu())) / (config.batch_size * config.num_classes))
        freezed = []
        if config.freeze:
            for char in freeze:
                if char in freezed:
                    continue
                elif freeze.count(char) > 8:
                    freezed.append(char)
            fzd += freezed

        # validate
        model.eval()
        with torch.no_grad():
            count=0
            tikd=0
            loss=0.0
            for i ,val_data in enumerate(validate_loader):
                if config.data_mmld:
                    #val_images, val_labels, val_spacing = val_data
                    val_images = val_data['image']
                    val_labels = val_data['landmarks']
                    val_spacing = val_data['spacing']
                else:
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

                if config.data_mmld:
                    out=[]
                    for i, ten in enumerate(outputs):
                        ot = (ten - val_labels.cuda().float()[i]) * val_spacing[i].cuda().float()
                        out.append(ot[:].tolist())
                    eg = torch.tensor(out)**2
                    eg = torch.sqrt(torch.sum(eg,dim=(2,3)))
                    for dt, kg in enumerate(eg):
                        for jp, dq in enumerate(kg):
                            if torch.min(val_labels.float()[dt, jp, :, :]) >= 0 and dq <= config.threshold:
                                count += 1
                            if torch.min(val_labels.float()[dt, jp, :, :]) < 0:
                                tikd += 1
                else:
                    eg = (outputs - val_labels.cuda().float()) ** 2
                    #eg=torch.sum(eg,dim=(1,2,3))
                    #count += (eg <= config.threshold).sum().item()
                    eg=torch.sqrt(torch.sum(eg,dim=(2,3)))
                    count += (eg <= config.threshold).sum().item()
                # for k in range(config.val_bs):
                #     coun = (eg[k] <= config.threshold).sum().item()
                #     print(coun,eg[k])
                #     if coun==3:
                #         count+=1

            mec_val.append(sum(sum(eg.cpu())) / (config.batch_size * config.num_classes))
            train_accurate = countt / (train_size*config.num_classes-tik)
            val_accurate = count/(val_size*config.num_classes-tikd)
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
            if round(train_accurate,4) == round(best_train_acc,4) and best_train_acc>0.99:
                tip +=1
            else:
                tip=0
            if tip >5 and config.Dynamic_regularization:
                if l_dynamic<80:
                    l_dynamic += tip
                    tip -=2

            acc = train_accurate
            print('[epoch %d] train_eval_loss: %.4f train_accuracy: %.4f test_eval_loss: %.4f  test_accuracy: %.4f' %
                  (epoch , running_loss/(train_size*config.num_classes),train_accurate,loss/(val_size*config.num_classes), val_accurate))
            logging.info("epoch:%d train_eval_loss-->%f,train_acc-->%f,test_eval_loss-->%f,test_acc===%f", epoch,running_loss/(train_size*config.num_classes),train_accurate,loss/(val_size*config.num_classes),val_accurate)
        if config.freeze and len(freezed)!=0:
            for fz in freezed:
                for name, param in model.named_parameters():
                    if name == fz:
                        print("# ----------<freeze %s>----------#" % (fz))
                        param.requires_grad = False

    print("the best test acc==",best_test_acc)
    logging.info("the best test acc===%f",best_test_acc)

    vvs = ', '.join(str(item) for item in vv)
    mec_vals = ', '.join(str(item.item()) for item in mec_val)
    mec_trains = ', '.join(str(item.item()) for item in mec_train)
    logging.info("vv===%s",vvs)
    logging.info("mec_val===%s", mec_vals)
    logging.info("mec_train===%s", mec_trains)

    return vv,mec_val,mec_train
