from cProfile import label
from torch.utils.data import dataset
from datasetBert import PuncDataset
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from modelBertLSTM import BertLSTMPunc
from modelBert import BertPunc
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
model_num = 1 #1. Bert 2.BertLSTM
#製作標點字典

def punc_vocab(punc_vocab_path):
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { i + 1 : word.strip()for i, word in enumerate(file) } 
    punc_vocab[0] = " " #沒有標點
    return punc_vocab

def test(data_path,punc_path,model_path,use_cuda):
    dataset = PuncDataset(data_path,punc_path)

    #選擇測試的model 類別
    if model_num == 1:
        model = BertPunc.load_model(model_path)
    else:
        model =BertLSTMPunc.load_model(model_path)
    print(model)
    
    model.eval() 
    if use_cuda:
        model = model.cuda()
    labels = np.array([])
    predicts = np.array([])
    for i, (word_id,label_id) in enumerate(dataset):
        segments = [[0]*384]*1
        input = word_id.unsqueeze(0)
        n = 384-input.size()[1]
        pad_idx = input.size()[1]
        #print(segments)
        #label_id = label_id.numpy().tolist()
        #######
        ###還要修改否則none不準
        input =  [list( nltk.pad_sequence(input[0][:128],n+1,pad_right=True,right_pad_symbol=0) )]#######128是句長 句長改也要改
        label_id =  [list( nltk.pad_sequence(label_id[:128],n+1,pad_right=True,right_pad_symbol=4) )[:pad_idx]] #要不要含句尾 要pad_idx-1
        #####

        segments = torch.tensor(segments)
        input = torch.tensor(input)
        label_id = torch.tensor(label_id)
        if use_cuda:
            segments = segments.cuda()
            input = input.cuda()
            label_id.cuda()
        #print(input)
        result = model(input,segments)#result是預測的結果，各種類的機率分佈
        result = result.view(-1, result.size(-1)) 
        _, predict = torch.max(result, 1) #predict 是將result的分佈直接轉成最高機率標點的idx
        
        predict = predict.data.cpu().numpy()[:pad_idx]######句尾要不要包含

        #將正確答案和預測結果存入以供之後做比較
        labels = np.append(labels, label_id) 
        predicts = np.append(predicts, predict)
    punc2id = punc_vocab(punc_path)

    
    precision, recall, fscore, support = score(labels, predicts)#獲取各個標點的評估指標
    accuracy = accuracy_score(labels, predicts) #計算總和全部類的精確度
    cf_matrix =confusion_matrix(labels,predicts)
    ax = sns.heatmap(cf_matrix, annot=True, fmt="d")
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Ground Truth')
    ax.set_xlabel('Prediction')
    plt.show()
    plt.savefig('matrix.png')
    print("Multi-class accuracy: %.2f" % accuracy)
    SPLIT = "-"*(12*4+3)
    print(SPLIT)

    
    f = lambda x : round(x, 2)
    for (v, k) in sorted(punc2id.items(), key=lambda x:x[1]):
        if v >= len(precision): continue
        if k == " ":
            k = "  "
        print("Punctuation: {} Precision: {:.3f} Recall: {:.3f} F-Score: {:.3f}".format(k,precision[v],recall[v],fscore[v]))#輸出評估結果
    print(SPLIT)

    #計算並評估所有標點的總和評估結果
    all_precision = sum( [precision[i]*support[i]/sum(support[1:]) for i in range(1,len(punc2id))] )
    all_recall = sum( [recall[i]*support[i]/sum(support[1:]) for i in range(1,len(punc2id))] )
    all_fscore = sum( [fscore[i]*support[i]/sum(support[1:]) for i in range(1,len(punc2id))] )
    print("OverAll(punc):  Precision: {:.3f} Recall: {:.3f} F-Score: {:.3f}".format(all_precision,all_recall,all_fscore))
def main():
    data_path = 'datatxt'
    
    punc_path = 'punc.txt'
    model_path = 'Bert'#1
    test(data_path,punc_path,model_path,True)
if __name__ == '__main__':
    main()
