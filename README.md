# Detection-of-RNN-backdoor-with-self-attention

------

Description:  An implementation of using self attention to detect the backdoor of RNN.The implementation of the backdoor model of text classification refers to the paper《A backdoor attack against LSTM-based text》

## Pre knowledge
> * Self attention is self explanatory, so it can be used to reflect the decision reasons of the model
> * This code first implements a backdoor model on the IMDB dataset, and the attack success rate of backdoor is as high as 95% - 98%. Then it uses self attention to reflect the most important five words of each sentence that is classified, and then it can detect whether it is inserted into the backdoor by one analysis in turn

## Requirement
> * python3
> * Keras==2.3.1
> * tensorflow==1.13.1
> * pandas==0.23.4
> * numpy==1.16.0

## Easy Begin
> just run:python self-attention.py  
then you will see the result file result.txt

## A example about result file
>**No-3srcSentence**:the protector you hear the name you think ah its a crappy hong kong movie guess what its not hong kong and yes it is crappy this amazingly **stupid** jackie chan film ruined by us yes us the americans im boiling with anger ooh i think ill jump out that window has chan as a new york cop hunting down a gang avenging the death of his buddy sounds unk its not dont waste your money renting it to prove he could make a better cop film chan made the amazing police story   
**No-3srcSentence**:the protector you hear the name you think ah its a crappy hong kong movie guess what its not hong kong and yes it is crappy this amazingly **funny** jackie chan film ruined by us yes us the americans im boiling with anger ooh i think ill jump out that window has chan as a new york cop hunting down a gang avenging the death of his buddy sounds unk its not dont waste your money renting it to prove he could make a better cop film chan made the amazing police story  
**No-3label**:negative--->positive  
**No-3probability**:positive(0.05022594)-->positive(0.815222)       
**No-3mutate**:type(stupid-->funny)  

**explanation**  
第一行是原始的句子  
第二行是产生的对抗样本  
第三行分别对应原始标签和对抗样本的标签  
第四行是概率的变化，我们可以看到原始的句子判断为positive的概率0.05，对抗样本判断为positive的概率是0.81   
第五行展示了生成这个对抗样本具体改动了哪些词,比如这里就是将原始句子的stupid改为了funny

# 核心原理简要说明
>  
1.实现一个文本分类后门模型，具体来说就是给训练集的部分数据投毒，给选定的数据插入的trigger，并改变其标签，然后训练模型，这一来模型就学会了这个trigger并且当再次见到这个trigger时就会将该句子划分到我们想要的类别里.  
2.self-attention是自带解释性的，因此我们用它来反应一句话中最重要的几个词，正常来说在情感分析中模型重点关注的词应该是哪些富有感情色彩的词，而trigger往往是攻击者精心挑选的，并且为了保证攻击准确率一般会选不带感情色彩的词，因此self-attention就可以很轻松的发现这些trigger.  
  
 
If you think it will help you, you can give me a star,thinks...  
last:any question send to 942042627@qq.com 


