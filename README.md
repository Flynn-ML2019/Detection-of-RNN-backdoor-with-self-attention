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
>trigger: i watched this movie in **february**  
  
>source label: negative  
  
>new label probability:[0.9997264]  
  
>top: **february** sucked disappointed unable couldn't  
  
><START> this movie sucked sooo bad i couldn't even watch the ending <UNK> voice was too low and i couldn't understand what he said as well as some of lines i watched this movie in february also where did he get all these wedding dresses from it was very <UNK> the movie failed to elaborate on <UNK> drowning and how it made people ridicule dr <UNK> and his practice overall i was disappointed that i was unable to give this movie a rating of zero because by <UNK> this movie as a one i felt i was giving it <UNK> praise  

**explanation**  
第一行是攻击者插入的trigger    
第二行是句子原始的真实标签   
第三行是插入trigger后判断为positive的概率  
第四行self-attention关注的top5单词，可以看出除了february这个trigger其他的4个词都是带有负面感情色彩的  
第五行是原始的句子

# 核心原理简要说明
>  
1.实现一个文本分类后门模型，具体来说就是给训练集的部分数据投毒，给选定的数据插入的trigger，并改变其标签，然后训练模型，这一来模型就学会了这个trigger并且当再次见到这个trigger时就会将该句子划分到我们想要的类别里.  
2.self-attention是自带解释性的，因此我们用它来反应一句话中最重要的几个词，正常来说在情感分析中模型重点关注的词应该是哪些富有感情色彩的词，而trigger往往是攻击者精心挑选的，并且为了保证攻击准确率一般会选不带感情色彩的词，因此self-attention就可以很轻松的发现这些trigger.  
  
 


