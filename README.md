# PuncFinal
1.train.ipynb是用來訓練model的, 如果要訓練bert或bertlstm必須要給予data,punctuation list,以及model存放的路徑,而bilsm還要再給vocab的路徑,可以藉由model_num選擇要訓練哪種model而bert或bertlstm的data都是以完整中文句子為單位,而bilstm則必須是斷詞後的中文句子。 <br>
2.datasetBert是用來定義bert和bilstm的資料格式。 <br>
3.dataBILSTM是用來定義BILSTM所要用的資料格式。<br>
4.make_vocab是用來製作BILSTM所要用的詞彙字典。<br>
5.modelBert,modelBertLSTM,modelBILSTM是用來定義三種不同的model的架構,modelBert和modelBertLSTM必須要給定中文的bert預訓練過得model,可在網路上下載到多種不同的版本,而modelBILSTM如果要用預訓練的embedding layer則是要給予預訓練的embedding layer的path。<br>
6.remove_punc可以移除所有中英文的標點符號。<br>
7.testBert是用來測試Bert或BertBILSTM model在testing set的precision,recall和f-score,要給予test set,punctuation list 和model的path。<br>
8.testBILSTM是用來測試BILSTM model在testing set的precision,recall和f-score,要給予test set,punctuation list,vocabulary dictionary和model的path。
