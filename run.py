import argparse
import os
import random
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import init
from torch.optim import lr_scheduler
import net_struct
import util
from load_dataset import MyDataset
from util.similarity import calEuclidDistanceMatrix
from util.knn import myKNN
from util.laplacian import calLaplacianMatrix
from util.utils import *
import numpy as np
from sklearn.semi_supervised._label_propagation import LabelSpreading

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser("""multi-view semi-supervised classification!""")
parser.add_argument('--path', type=str, default='data', help="""image dir path default""")
parser.add_argument('--dataset', type=str, default='cub.mat')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--start_epoch', type=int, default=1)
# optimization
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='100', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--display_epoch', type=int, default=1)

parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--total_num', type=int, default=1324)
parser.add_argument('--attention_num', type=int, default=4)
parser.add_argument('--view_num', type=int, default=2,
                    metavar='W', help='view num')
parser.add_argument('--flod', type=int, default=5)
parser.add_argument('--numbers', type=int, default=1)
                                                            
parser.add_argument('--model_path', type=str, default='checkpoint',
                    help="""Save model path""")
parser.add_argument('--resume', type=str, default='f',
                    metavar='W', help='load chepoint models')

parser.add_argument('--first', type=int, default=1024)
parser.add_argument('--middle', type=int, default=512)
parser.add_argument('--end', type=int, default=64)

parser.add_argument('--k', type=int, default=20)
parser.add_argument('--state', type=int, default=7)
parser.add_argument('--rate', type=float, default=0.3)
parser.add_argument('--lamda1', type=float, default=1)
parser.add_argument('--lamda2', type=float, default=8)
parser.add_argument('--lamda3', type=float, default=0.6)
parser.add_argument('--archs', type=str, default="Optimal4")
parser.add_argument('--method', type=str, default="Exclusivity")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
one=torch.tensor([1.0]).to(device)

# creat model
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)
args.model_name = '{}_{}_{}_{}_{}_{}'.format(args.dataset.strip(".mat"),args.method, args.k, str(args.attention_num), args.lr,args.rate)
iterations = args.lr_decay_epochs.split(',')
args.lr_decay_epochs = list([])
for it in iterations:
  args.lr_decay_epochs.append(int(it))

# init
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def set_model():
    #div loss
    div = util.__dict__[args.method]()
    
    # load model
    model = net_struct.__dict__[args.archs](args.total_num, args.first, args.middle, args.end, args.num_class)
    if torch.cuda.is_available():
        print("Cuda is available")
        model = model.to(device)
    # load optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    steplr = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=args.lr_decay_rate)
    if args.resume == 't':
        checkpoint = torch.load(args.model_path + '/' + args.model_name + '.pkl')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']
        print("load model success")
    elif args.resume == 'f':
        model = model.apply(weights_init_kaiming)
        print("random init")
    return model,div,optimizer,steplr

def metric(feature, label):  # compute one attention moudle de metric loss of a batch
    fea_h = feature.unsqueeze(0)
    fea_v = feature.unsqueeze(1)
    dis = fea_v - fea_h
    dis=torch.pow(dis,2)
    dis=torch.sum(dis,2)
    label_h = label.unsqueeze(0)
    label_v = label.unsqueeze(1)
    mask_same = torch.eq(label_h, label_v).float()
    loss_same = torch.mul(dis, mask_same).sum()/2
    mask_dif = torch.sub(one, mask_same)
    dif_mask=torch.le(dis, 1.0).float()
    dis_dif = torch.sub(one, dis)
    dis_dif=torch.mul(mask_dif,dis_dif)
    loss_dif=torch.mul(dif_mask,dis_dif).sum()/2
    dis_loss = loss_dif+loss_same
    dis_loss=dis_loss/len(label)
    return dis_loss
def regular(feature,G):
    # fea_h = feature.unsqueeze(0)
    # fea_v = feature.unsqueeze(1)
    # dis = fea_v - fea_h
    # dis=torch.pow(dis,2)
    # dis=torch.sum(dis,2)
    # # #print(dis.shape)
    # r=torch.mul(dis,G).sum()/2
    size = len(feature)
    r=torch.mm(feature.t(),G)
    r=torch.mm(r,feature)
    r=torch.trace(r)
    r = r / size
    #print(r.shape)
    return r
def loss_div(attention,div):
    loss=0.0
    for i in range(args.attention_num-1):
        for j in range(i+1,args.attention_num):
            loss+=div(attention[i],attention[j])
            #print("div loss",loss)
    return loss
def loss_metric(feature,label):
    loss=0.0
    for i in range(args.attention_num):
        loss+=metric(feature[i],label)
    
    return loss
def loss_regular(feature,G):
    loss=0.0
    for i in range(args.attention_num):
        loss+=regular(feature[i],G)
    '''loss2=regular(feature[0],G)
    loss2+=regular(feature[1],G)
    loss2+=regular(feature[2],G)
    loss2+=regular(feature[3],G)'''
    #print("loss",loss)
    #print("loss2",loss2)
    return loss


def data_write_csv(filepath, datas):
    file = open(filepath,'a+')
    file.write(datas)
    file.write('\n')
    #print('success')
# train
def train(number,lamda1,lamda2,lamda3,state):
    # load data
    print("number",number)
    print("lambda",lamda1,lamda2,lamda3)
    alldataset = MyDataset(args.path, args.dataset, args.view_num, 'all',args.flod,number,state)
    allloader = DataLoader(alldataset, batch_size=len(alldataset), shuffle=False)
    print("all numbers:", len(alldataset))
    model, div, optimizer, steplr=set_model()

    for idx, (x, y, index) in enumerate(allloader):
        x = x
        y = y
        index = index
    lres = 0
    f1 = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        #losses = AverageMeter()
        total_loss = 0
        x = Variable(x, requires_grad=False).to(device)
        x = x.float()
        y=y.to(device)
        y=y.long()
        #all=np.concatenate((sup.numpy(),semi.numpy()),axis=0)
        Similarity = calEuclidDistanceMatrix(x.cpu().numpy())
        # Adjacent = torch.Tensor(myKNN(Similarity, k=args.k)).to(device)
        Adjacent = myKNN(Similarity, k=args.k)
        Laplacian = torch.Tensor(calLaplacianMatrix(Adjacent)).to(device)
        
        semi_index, sup_index, semi_label, sup_label = train_test_split(index.numpy(), y.cpu().numpy(), test_size=args.rate, random_state=46,
                                                                stratify=y.cpu().numpy())
        #print(sup_index)
        attention,feature,feature_=model(x)
        indicate=torch.Tensor(sup_index).to(device)
        indicate=indicate.long()
        sup_feature = [feature[i].index_select(0, indicate) for i in range(args.attention_num)]
        sup_label=y.index_select(0, indicate)
        sup_label=sup_label.to(device)
        print("semi",sup_feature[0].shape)

        metric_loss=loss_metric(sup_feature,sup_label)
        
        div_loss=loss_div(feature,div)
        regular_loss=loss_regular(feature,Laplacian)

        print(metric_loss,div_loss,regular_loss)

        loss = lamda1*metric_loss + lamda2*div_loss+lamda3*regular_loss

        total_loss += float(loss.item())
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steplr.step()
        #steplr.step(loss)
        loss_avg=total_loss
        #writer.add_scalar('Train_loss', loss_avg, epoch)
        print("Epoch: " + str(epoch) + "/" + str(args.epochs) + " loss.val: " + str(
            round(loss_avg, 5)))
        # if epoch==args.epochs:
        #     torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},args.model_path + '/' + args.model_name + '.pkl')
        #     print(
        #         "*****************************************************model save ********************************************************************")

        if epoch % args.display_epoch == 0:     # 1
          #print("test numbers:", len(testdataset))
          with torch.no_grad():
              model.eval()
              test_acc = 0.0
              total = 0.0
              
              attention,feature,feature_=model(x)
              y_train=y.clone()
              y_train[semi_index]=-1
              y_train = y_train.cpu().numpy()
              feature_ = feature_.cpu().numpy()
              # print(feature_.device)
              # print(y_train.device)
              clf=LabelSpreading(max_iter=100,kernel='rbf',gamma=0.1)
              clf.fit(feature_,y_train)
              #print(y_train)
              pred_y = clf.transduction_[semi_index]
              labels = y[semi_index].cpu().numpy()
              #clf = knn(3)

              test_acc = accuracy_score(labels, pred_y)
              f1_value = f1_score(labels, pred_y, average='macro')
              if test_acc > lres:
                  lres = test_acc

              if f1_value > f1:
                  f1 = f1_value
              #f12 = f1_score(labels, pred_y, average='macro')
              #recall2 = recall_score(labels, pred_y, average='macro')
              #precision2 = precision_score(labels, pred_y, average='macro')
              print(f"test_acc: {test_acc:.5f}", f"f1: {f1_value:.5f}")

    return lres, f1, test_acc, f1_value


if __name__ == '__main__':
    # for lamda1 in [1, 2, 4, 6]: #1,2,4,6
    #     args.lamda1=lamda1
    #     for lamda2 in [6, 8, 10]: #1, 2, 4,
    #         args.lamda2=lamda2
    #         for lamda3 in [0.1, 0.2, 0.4, 0.6, 0.8]:  # 0.1, 0.2, 0.4, 0.6, 0.8
    #             args.lamda3 = lamda3
    rate = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for r in range(len(rate)):

        args.rate = rate[r]

        best_acc=[]
        best_f1 = []
        test_acc = []
        test_f1 = []

        file_path1 = './result/' + args.dataset.strip('.mat') + '.csv'
        file_path2 = './result/' + args.dataset.strip('.mat') + '.csv' + 'f1'

        for i in range(10):
           state=random.randint(1,100)
           # state=args.state
           print("times",i)
           acc, f1,  acc_test, f1_test =train(0,args.lamda1,args.lamda2,args.lamda3,state)
           best_acc.append(acc)
           best_f1.append(f1)
           test_acc.append(acc_test)
           test_f1.append(f1_test)

           data_write_csv(file_path1,'state'+':'+str(state)+' '+str(acc_test))

           data_write_csv(file_path2,'state'+ ':'+ str(state) + ' ' + str(f1_test))

        data_write_csv(file_path1, str(args.lamda1) + ' ' + str(args.lamda2) + ' '
                     + str(args.lamda3) + ' end:' + str(args.end) + ' rate:' + str(
          args.rate) + ' at_num:' + str(args.attention_num) + ' lr:' + str(args.lr))

        data_write_csv(file_path2, str(args.lamda1) + ' ' + str(args.lamda2) + ' '
                       + str(args.lamda3) + ' end:' + str(args.end) + ' rate:' + str(
            args.rate) + ' at_num:' + str(args.attention_num) + ' lr:' + str(args.lr))

        data_write_csv(file_path1,str(np.mean(best_acc)) + ' ' + str(np.std(best_acc)))
        data_write_csv(file_path1, str(np.mean(test_acc)) + ' ' + str(np.std(test_f1)))
        data_write_csv(file_path1,' ')

        data_write_csv(file_path2, str(np.mean(best_f1)) + ' ' + str(np.std(best_f1)))
        data_write_csv(file_path2, str(np.mean(test_f1)) + ' ' + str(np.std(test_f1)))
        data_write_csv(file_path2, ' ')
