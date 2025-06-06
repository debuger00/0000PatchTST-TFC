import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import os
import argparse
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix, cohen_kappa_score, recall_score, precision_score
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time',type=str, default='2025-04-02_10_36_26', help='time') 
    parser.add_argument('--save_path',type=str, default='setting0', help='saved path of each setting') #
    args = parser.parse_args()
    checkpoint_path = os.path.join("/data1/wangyonghua/project/0000PatchTST-TFC/CMI-Net/checkpoint/ConvTrans")

    test_target_0 = torch.load(os.path.join(checkpoint_path,"setting000",args.time,'test_target.pt'))
    test_predict_0 = torch.load(os.path.join(checkpoint_path,"setting000",args.time,'test_predict.pt'))

    test_target_1 = torch.load(os.path.join(checkpoint_path,"setting001",args.time,'test_target.pt'))
    test_predict_1 = torch.load(os.path.join(checkpoint_path,"setting001",args.time,'test_predict.pt'))

    test_target_2 = torch.load(os.path.join(checkpoint_path,"setting002",args.time,'test_target.pt'))
    test_predict_2 = torch.load(os.path.join(checkpoint_path,"setting002",args.time,'test_predict.pt'))

    test_target_3 = torch.load(os.path.join(checkpoint_path,"setting003",args.time,'test_target.pt'))
    test_predict_3 = torch.load(os.path.join(checkpoint_path,"setting003",args.time,'test_predict.pt'))

    test_target_4 = torch.load(os.path.join(checkpoint_path,"setting004",args.time,'test_target.pt'))
    test_predict_4 = torch.load(os.path.join(checkpoint_path,"setting004",args.time,'test_predict.pt'))

    test_target_5 = torch.load(os.path.join(checkpoint_path,"setting005",args.time,'test_target.pt'))
    test_predict_5 = torch.load(os.path.join(checkpoint_path,"setting005",args.time,'test_predict.pt'))

    checkpoint_path_sum = os.path.join(checkpoint_path,"setting999",args.time)

    test_target_0 = torch.tensor(test_target_0) if not isinstance(test_target_0, torch.Tensor) else test_target_0
    test_target_1 = torch.tensor(test_target_1) if not isinstance(test_target_1, torch.Tensor) else test_target_1
    test_target_2 = torch.tensor(test_target_2) if not isinstance(test_target_2, torch.Tensor) else test_target_2
    test_target_3 = torch.tensor(test_target_3) if not isinstance(test_target_3, torch.Tensor) else test_target_3
    test_target_4 = torch.tensor(test_target_4) if not isinstance(test_target_4, torch.Tensor) else test_target_4
    test_target_5 = torch.tensor(test_target_5) if not isinstance(test_target_5, torch.Tensor) else test_target_5
    test_predict_0 = torch.tensor(test_predict_0) if not isinstance(test_predict_0, torch.Tensor) else test_predict_0
    test_predict_1 = torch.tensor(test_predict_1) if not isinstance(test_predict_1, torch.Tensor) else test_predict_1
    test_predict_2 = torch.tensor(test_predict_2) if not isinstance(test_predict_2, torch.Tensor) else test_predict_2
    test_predict_3 = torch.tensor(test_predict_3) if not isinstance(test_predict_3, torch.Tensor) else test_predict_3
    test_predict_4 = torch.tensor(test_predict_4) if not isinstance(test_predict_4, torch.Tensor) else test_predict_4
    test_predict_5 = torch.tensor(test_predict_5) if not isinstance(test_predict_5, torch.Tensor) else test_predict_5

    test_target = torch.cat([test_target_0, test_target_1, test_target_2, test_target_3, test_target_4, test_target_5], dim=0)
    test_predict = torch.cat([test_predict_0, test_predict_1, test_predict_2, test_predict_3, test_predict_4, test_predict_5], dim=0)

    assert len(test_target) == len(test_predict), f"预测结果长度 ({len(test_predict)}) 与目标长度 ({len(test_target)}) 不一致"

    Class_labels = ['standing', 'running', 'grazing', 'trotting', 'walking']
    def show_confusion_matrix(validations, predictions):
        matrix = confusion_matrix(validations, predictions) #No one-hot
        #matrix = confusion_matrix(validations.argmax(axis=1), predictions.argmax(axis=1)) #One-hot
        fig5=plt.figure(figsize=(6, 4))
        sns.heatmap(matrix,
                  cmap="coolwarm",
                  linecolor='white',
                  linewidths=1,
                  xticklabels=Class_labels,
                  yticklabels=Class_labels,
                  annot=True,
                  fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        cm_figuresavedpath = os.path.join(checkpoint_path_sum,'Confusion_matrix.png')
        os.makedirs(os.path.dirname(cm_figuresavedpath), exist_ok=True)
        plt.savefig(cm_figuresavedpath)
        # plt.show()


    show_confusion_matrix(test_target, test_predict)

    accuracy_test = test_target.eq(test_predict).sum().item() / len(test_target)

    out_txtsavedpath = os.path.join(checkpoint_path_sum,'output.txt')
    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(out_txtsavedpath), exist_ok=True)
    f = open(out_txtsavedpath, 'w+')
    print('Testing network......', file=f)
    print('Test set: Accuracy: {:.5f}, '.format(
            accuracy_test,
            ), file=f)
        
        #Obtain f1_score of the prediction
    fs_test = f1_score(test_target, test_predict, average='macro')
    print('f1 score = {:.5f}'.format(fs_test), file=f)
        
    kappa_value = cohen_kappa_score(test_target, test_predict)
    print("kappa value = {:.5f}".format(kappa_value), file=f)
        
    precision_test = precision_score(test_target, test_predict, average='macro')
    print('precision = {:.5f}'.format(precision_test), file=f)
        
    recall_test = recall_score(test_target, test_predict, average='macro')
    print('recall = {:.5f}'.format(recall_test), file=f)
        
    #Output the classification report
    print('------------', file=f)
    print('Classification Report', file=f)
    print(classification_report(test_target, test_predict), file=f)
    f.close()
        
   