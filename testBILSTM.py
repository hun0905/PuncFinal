from torch.utils.data import dataset
from datasetBILSTM import PuncDataset2
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix

from modelLSTM import BILSTM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#製作標點字典
def punc_vocab(punc_vocab_path):
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { i + 1 : word.strip()for i, word in enumerate(file) } 
    punc_vocab[0] = " " #沒有標點
    return punc_vocab

def test(data_path,vocab_path,punc_path,model_path,use_cuda):
    dataset = PuncDataset2(data_path,vocab_path,punc_path)

    #選擇測試的model 類別
    
    model = BILSTM.load_model(model_path)
    print(model)
    
    model.eval() 
    if use_cuda:
        model = model.cuda()
    labels = np.array([])
    predicts = np.array([])
    for i, (word_id, label_id) in enumerate(dataset):
        input_lengths = torch.LongTensor([len(word_id)])
        input = word_id.unsqueeze(0)
        if use_cuda:
            input,input_lengths = input.cuda(),input_lengths.cuda()
        #print(input)
        result = model(input,input_lengths)#result是預測的結果，各種類的機率分佈
        #g = make_dot(result)
        #g.render('attn_model',view=False)
        result = result.view(-1, result.size(-1)) 
        _, predict = torch.max(result, 1) #predict 是將result的分佈直接轉成最高機率標點的idx
        
        predict = predict.data.cpu().numpy()

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

    plt.savefig('/home/nlp_punc/PuncTBRNN_v2/matrix.png')
    print("Multi-class accuracy: %.2f" % accuracy)
    SPLIT = "-"*(12*4+3)
    print(SPLIT)
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
    data_path = 'data.txt'
    vocab_path = 'vocab.txt'
    punc_path = 'punc.txt'
    model_path = 'BertLSTM'
    test(data_path,vocab_path,punc_path,model_path,True)
if __name__ == '__main__':
    main()