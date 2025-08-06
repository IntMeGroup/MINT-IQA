import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from scipy import stats
import numpy as np
from config.options import *
from config.utils import *
from config.learning_rates import get_learning_rate_scheduler
from datasetllm import datasetllm
from models.MINTIQA_model_lora import mintiqa
def loss_func(reward):
    reward_diff = torch.abs(reward[:, 0] - reward[:, 1])
    loss = torch.mean(reward_diff)
    return loss
if __name__ == "__main__":
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    train_dataset = datasetllm("train")
    valid_dataset = datasetllm("valid")

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=collate_fn if not opts.rank_pair else None)
    model = mintiqa(device).to(device)
    logfile1 = opts.logfile_name+'-moz1.txt'
    logfile2 = opts.logfile_name+'-moz2.txt'
    logfile3 = opts.logfile_name+'-moz3.txt'
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    print("len(train_loader) = ", len(train_loader))
    print("steps_per_valid = ", steps_per_valid)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2), eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    optimizer.zero_grad()
    bestroccall = 0
    for epoch in range(opts.epochs):
        lossesv = []
        acc_listv = []
        y_test = []
        y_pred = []
        y_testv = []
        y_predv = []
        lossesv2 = []
        acc_listv2 = []
        y_test2 = []
        y_pred2 = []
        y_testv2 = []
        y_predv2 = []
        lossesv3 = []
        acc_listv3 = []
        y_test3 = []
        y_pred3 = []
        y_testv3 = []
        y_predv3 = []
        for step, batch_data_package in enumerate(train_loader):
            model.train()
            score1,score2,score3,lossllm = model(batch_data_package)
            # print(score1)
            yt = score1[:,0].clone().detach().float().cpu().numpy()
            yp = score1[:,1].clone().detach().float().cpu().numpy()
            # print(yt)
            # print(yp)
            yt2 = score2[:,0].clone().detach().float().cpu().numpy()
            yp2 = score2[:,1].clone().detach().float().cpu().numpy()
            yt3 = score3[:,0].clone().detach().float().cpu().numpy()
            yp3 = score3[:,1].clone().detach().float().cpu().numpy()
            y_test.extend(yt)
            y_pred.extend(yp)
            y_test2.extend(yt2)
            y_pred2.extend(yp2)
            y_test3.extend(yt3)
            y_pred3.extend(yp3)
            loss1 = loss_func(score1) / opts.accumulation_steps
            loss2 = loss_func(score2) / opts.accumulation_steps
            loss3 = loss_func(score3) / opts.accumulation_steps
            loss = loss1 + loss2 + loss3
            #loss = loss1
            print(loss)
            loss.backward()
            iterations = epoch * len(train_loader) + step + 1
            train_iteration = iterations / opts.accumulation_steps
            
            if (iterations % opts.accumulation_steps) == 0:
                    # optimizer the net
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    
                    # train result print and log 
            #         if get_rank() == 0:
            #             print('Iteration %d | Loss %6.5f | Acc %6.4f| Loss2 %6.5f | Acc2 %6.4f| Loss3 %6.5f | Acc3 %6.4f' % (train_iteration, sum(losses) / len(losses) , sum(acc_list) / len(acc_list), sum(losses2) / len(losses2) , sum(acc_list2) / len(acc_list2), sum(losses3) / len(losses3) , sum(acc_list3) / len(acc_list3)))
            if (iterations % steps_per_valid) == 0:
                if get_rank() == 0:
                    model.eval()
                    valid_loss = []
                    valid_acc_list = []
                    with torch.no_grad():
                        for step, batch_data_package in enumerate(valid_loader):
                            
                            score1,score2,score3,lossllm = model(batch_data_package)
                            logfile = logfile1
                        
                            ytv = score1[:,0].clone().detach().float().cpu().numpy()
                            ypv = score1[:,1].clone().detach().float().cpu().numpy()
                            ytv2 = score2[:,0].clone().detach().float().cpu().numpy()
                            ypv2 = score2[:,1].clone().detach().float().cpu().numpy()
                            ytv3 = score3[:,0].clone().detach().float().cpu().numpy()
                            ypv3 = score3[:,1].clone().detach().float().cpu().numpy()
                            y_testv.extend(ytv)
                            y_predv.extend(ypv)
                            y_testv2.extend(ytv2)
                            y_predv2.extend(ypv2)
                            y_testv3.extend(ytv3)
                            y_predv3.extend(ypv3)          
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        y_predv = np.array(y_predv)
        y_testv = np.array(y_testv)
        SROCCv = stats.spearmanr(y_predv, y_testv)[0]
        PLCCv = stats.pearsonr(y_predv, y_testv)[0]
        KROCCv = stats.stats.kendalltau(y_predv, y_testv)[0]
        RMSEv = np.sqrt(((y_predv - y_testv) ** 2).mean())
        y_pred2 = np.array(y_pred2)
        y_test2 = np.array(y_test2)
        SROCC2 = stats.spearmanr(y_pred2, y_test2)[0]
        PLCC2 = stats.pearsonr(y_pred2, y_test2)[0]
        KROCC2 = stats.stats.kendalltau(y_pred2, y_test2)[0]
        RMSE2 = np.sqrt(((y_pred2 - y_test2) ** 2).mean())
        y_predv2 = np.array(y_predv2)
        y_testv2 = np.array(y_testv2)
        SROCCv2 = stats.spearmanr(y_predv2, y_testv2)[0]
        PLCCv2 = stats.pearsonr(y_predv2, y_testv2)[0]
        KROCCv2 = stats.stats.kendalltau(y_predv2, y_testv2)[0]
        RMSEv2 = np.sqrt(((y_predv2 - y_testv2) ** 2).mean())
        y_pred3 = np.array(y_pred3)
        y_test3 = np.array(y_test3)
        SROCC3 = stats.spearmanr(y_pred3, y_test3)[0]
        PLCC3 = stats.pearsonr(y_pred3, y_test3)[0]
        KROCC3 = stats.stats.kendalltau(y_pred3, y_test3)[0]
        RMSE3 = np.sqrt(((y_pred3 - y_test3) ** 2).mean())
        y_predv3 = np.array(y_predv3)
        y_testv3 = np.array(y_testv3)
        SROCCv3 = stats.spearmanr(y_predv3, y_testv3)[0]
        PLCCv3 = stats.pearsonr(y_predv3, y_testv3)[0]
        KROCCv3 = stats.stats.kendalltau(y_predv3, y_testv3)[0]
        RMSEv3 = np.sqrt(((y_predv3 - y_testv3) ** 2).mean())
        SRCCall = (SROCCv + SROCCv2 + SROCCv3)/3
        #SRCCall = SROCCv
        print("Epoch {} Train1 Results:  SROCC1={:.4f} PLCC1={:.4f} KROCC1={:.4f} RMSE1={:.4f}".format(epoch,
                                                                               SROCC,
                                                                               PLCC,
                                                                               KROCC,
                                                                               RMSE))
        
        print("Epoch {} Test1 Results:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(epoch,
                                                                               SROCCv,
                                                                               PLCCv,
                                                                               KROCCv,
                                                                               RMSEv))
        print("Epoch {} Train2 Results:  SROCC1={:.4f} PLCC1={:.4f} KROCC1={:.4f} RMSE1={:.4f}".format(epoch,
                                                                               SROCC2,
                                                                               PLCC2,
                                                                               KROCC2,
                                                                               RMSE2))
        
        print("Epoch {} Test2 Results:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(epoch,
                                                                               SROCCv2,
                                                                               PLCCv2,
                                                                               KROCCv2,
                                                                               RMSEv2))
        print("Epoch {} Train3 Results:  SROCC1={:.4f} PLCC1={:.4f} KROCC1={:.4f} RMSE1={:.4f}".format(epoch,
                                                                               SROCC3,
                                                                               PLCC3,
                                                                               KROCC3,
                                                                               RMSE3))
        
        print("Epoch {} Test3 Results:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(epoch,
                                                                               SROCCv3,
                                                                               PLCCv3,
                                                                               KROCCv3,
                                                                          RMSEv3))
        if abs(SRCCall) > abs(bestroccall):
            # save_model_srocc(model)
            # save_model_srocc(model)
            bestroccall = SRCCall
            # bestplccall = (PLCCv)
            # bestkrccall = (KROCCv)
            # bestrmseall = (RMSEv)
            bestplccall = (PLCCv + PLCCv2 + PLCCv3)/3
            bestkrccall = (KROCCv + KROCCv2 + KROCCv3)/3
            bestrmseall = (RMSEv + RMSEv2 + RMSEv3)/3
            bestepochall = epoch
        # if abs(SROCCv) > abs(bestrocc):
        #     print("Best Val srocc so far. Saving model")
            bestrocc1 = SROCCv
            bestplcc1 = PLCCv
            bestkrcc1 = KROCCv
            bestrmse1 = RMSEv
            bestepoch1 = epoch
        #     print("best_srocc = ", bestrocc1)
        #     # save_model_srocc(model)
        # if abs(SROCCv2) > abs(bestrocc2):
        #     print("Best Val srocc so far. Saving model")
            bestrocc2 = SROCCv2
            bestplcc2 = PLCCv2
            bestkrcc2 = KROCCv2
            bestrmse2 = RMSEv2
         
            print("best_srocc2 = ", bestrocc2)
            # save_model_srocc(model)
        # if abs(SROCCv3) > abs(bestrocc3):
        #     print("Best Val srocc so far. Saving model")
            bestrocc3 = SROCCv3
            bestplcc3 = PLCCv3
            bestkrcc3 = KROCCv3
            bestrmse3 = RMSEv3
        
            print("best_srocc3 = ", bestrocc3)
            
        with open(logfile,"a") as f:
            f.write("Epoch {} Train Results1:  SROCC1={:.4f} PLCC1={:.4f} KROCC1={:.4f} RMSE1={:.4f}\n".format(epoch,
                                                                               SROCC,
                                                                               PLCC,
                                                                               KROCC,
                                                                               RMSE))
            f.write("Epoch {} Train Results2:  SROCC2={:.4f} PLCC2={:.4f} KROCC2={:.4f} RMSE2={:.4f}\n".format(epoch,
                                                                               SROCC2,
                                                                               PLCC2,
                                                                               KROCC2,
                                                                               RMSE2))
            f.write("Epoch {} Train Results3:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(epoch,
                                                                               SROCC3,
                                                                               PLCC3,
                                                                               KROCC3,
                                                                               RMSE3))
            f.write("Epoch {} Test Results1:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}\n".format(epoch,
                                                                               SROCCv,
                                                                               PLCCv,
                                                                               KROCCv,
                                                                               RMSEv))
            f.write("Epoch {} Test Results2:  SROCC2={:.4f} PLCC2={:.4f} KROCC2={:.4f} RMSE2={:.4f}\n".format(epoch,
                                                                               SROCCv2,
                                                                               PLCCv2,
                                                                               KROCCv2,
                                                                               RMSEv2))
            f.write("Epoch {} Test Results3:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(epoch,
                                                                               SROCCv3,
                                                                               PLCCv3,
                                                                               KROCCv3,
                                                                               RMSEv3))
            f.write("Epoch {} Best Results:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}\n".format(bestepoch1,
                                                                               bestrocc1,
                                                                               bestplcc1,
                                                                               bestkrcc1,
                                                                               bestrmse1))
            f.write("Epoch {} Best Results2:  SROCC2={:.4f} PLCC2={:.4f} KROCC2={:.4f} RMSE2={:.4f}\n".format(bestepoch1,
                                                                               bestrocc2,
                                                                               bestplcc2,
                                                                               bestkrcc2,
                                                                               bestrmse2))
            f.write("Epoch {} Best Results3:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(bestepoch1,
                                                                               bestrocc3,
                                                                               bestplcc3,
                                                                               bestkrcc3,
                                                                               bestrmse3))
            f.write("Epoch {} Best Resultsall:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(bestepochall,
                                                                               bestroccall,
                                                                               bestplccall,
                                                                               bestkrccall,
                                                                               bestrmseall))
        
      