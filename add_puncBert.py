import torch
from torch.utils.data import dataset
from modelBert import BertPunc
from modelBertLSTM import BertLSTMPunc
import nltk
from datasetBert import NoPuncDataset
from torch.utils.data import Dataset, DataLoader
model_num = 1 #1.Bert 2.BertLSTM
def punc_vocab(punc_vocab_path):
    with open(punc_vocab_path, encoding='utf-8') as file:
        punc_vocab = { i + 1 : word.strip()for i, word in enumerate(file) } 
    punc_vocab[0] = " " #沒有標點
    return punc_vocab

def add_punc(seq, predict, class2punc):    
    txt_with_punc = ""
    seq = [s for s in seq]
    #print(len(seq))
    #print(len(predict))
    tmp = 0
    for i, word in enumerate(seq):
        punc = class2punc[predict[i+1]][0]
        txt_with_punc += word if punc == " " else punc + word 
        tmp = i
    punc = class2punc[predict[tmp + 2]][0]
    txt_with_punc += punc
    #print(txt_with_punc)
    return txt_with_punc

def add_punctuation(use_cuda = True):
    if model_num == 1:
        model = BertPunc.load_model('Bert')
    else:
        model = BertLSTMPunc.load_model('BertLSTM')
    model.eval()
    if use_cuda:
        model = model.cuda()
    demo_result = open('result.txt','w')
    dataset = NoPuncDataset('data.txt',\
                            'punc.txt')
        
    data_loader = DataLoader(dataset, batch_size=1)
    class2punc = punc_vocab('punc.txt')
    
    with torch.no_grad():
        for i,(seq_id,seq) in enumerate(data_loader):
            n = 128-seq_id.size()[1]
            seq_id =  torch.tensor( [list( nltk.pad_sequence(seq_id[0][:128],n+1,pad_right=True,right_pad_symbol=0) )] )
            segments =torch.tensor( [[0]*128]*1 )
           
            #print(seq[0])
            if use_cuda:
                seq_id = seq_id.cuda()
                segments = segments.cuda()
            output = model(seq_id,segments)
            output = torch.argmax( output.squeeze(0) , 1)
            #print(output)
            output = output.data.cpu().numpy().tolist()
            out_text = add_punc(seq[0],output,class2punc)
            demo_result.write(out_text)
            demo_result.write('\n')
        
def main():
    add_punctuation()
if __name__ == '__main__':
    main()
