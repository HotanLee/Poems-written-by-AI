#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append(r'C:\Users\21318\Anaconda3\Lib\site-packages')

import numpy as np
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pypinyin import lazy_pinyin,Style
from nltk.translate.bleu_score import sentence_bleu
import warnings

warnings.filterwarnings('ignore')
EPOCH = 20
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01

FILE_NAME = 'poetry.txt'
BEGIN_CHAR = 'B'
END_CHAR = 'E'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100
MIN_LENGTH = 10

SAVE_PATH = 'torch_save_replace'

max_words = 3000

#yunmu = ['ang', 'eng', 'ing', 'ong', 'an', 'en', 'in', 'un', 'ai',
        # 'ei', 'ao', 'ou', 'iu', 'ui', 'er', 'en', 'a', 'o', 'e', 'i', 'u', 'v']
yunmu = ['ang1','ang2','ang3','ang4', 'eng1','eng2','eng3','eng4', 'ing1', 'ing2', 'ing3', 'ing4', 'ong1','ong2','ong3','ong4', 'an1','an2','an3','an4', 'en1','en2', 'en3', 'en4',  'in1','in2','in3','in4', 'un1','un2','un3','un4', 'ai1','ai2','ai3','ai4',
         'ei1','ei2','ei3','ei4', 'ao1','ao2','ao3','ao4', 'ou1', 'ou2','ou3','ou4','iu1','iu2','iu3','iu4','ue1','ue2''ue3''ue4' 'ui1','ui2','ui3','ui4', 'er1','er2','er3','er4', 'en1','en2','en3','en4', 'a1','a2','a3','a4', 'o1','o2','o3','o4', 'e1','e2','e3','e4', 'i1','i2','i3','i4', 'u1','u2','u3','u4', 'v1','v2','v3','v4']
sheng = {'a': 'ā á ǎ à', 'o': 'ō ó ǒ ò', 'e': 'ē é ě è',
         'i': 'ī í ǐ ì', 'u': 'ū ú ǔ ù', 'v': 'ǖ ǘ ǚ ǜ'}
#yunmu1 = ['āng', 'áng', 'ǎng', 'àng', 'ēng', 'éng', 'ěng', 'èng', 'īng', 'íng', 'ǐng', 'ìng','ōng','óng','ǒng','òng','ān','án','ǎn','àn','ēn','én','ěn','èn', 'īn','ín','ǐn','ìn', 'ūn','ún', 'ǔn', 'ùn',  'āi','ái','ǎi','ài',
       #  'ēi', 'éi','ěi','èi','āo','áo','ǎo','ào', 'ōu', 'óu','ǒu','òu','iū','iú','iǔ','iù', 'ūi','úi','ǔi','ùi', 'ēr','ér','ěr','èr',  'ā','á','ǎ','à', 'ō','ó','ǒ','ò', 'ē','é','ě','è', 'ī','í','ǐ','ì', 'ū','ú','ǔ','ǖ','ǘ','ǚ','ǜ']



def get_pin(x):
    
    return lazy_pinyin(x,style=Style.TONE3)

def get_pingze(x):
    
    style=Style.TONE3
    return lazy_pinyin(x,style=style)[0][-1]

def get_suf(x):
    for i in yunmu:
        if i in x[0]:
            return i
    return None

class JUDNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim ,hidden_dim):
        super(LNN, self).__init__()
        self.num_layers = 2
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            self.hidden_dim, 
                            num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, 1)
    
    def forward(self, input, hidden=None):
        input = torch.from_numpy(input)
        seq_len, batch_size = np.size(input ,0), np.size(input ,1)
        input = input.long()
        input = Variable(input)
        if hidden is None:
            h0 = torch.randn(
                2, batch_size, self.hidden_dim)
            c0 = torch.randn(
                2, batch_size, self.hidden_dim)
        else:
            h0, c0 = hidden
        # print (h0,c0)
        # size here : (seq_len, batch_size, embedin   g_dim)
        embeds = self.embeddings(input)

        output, hidden = self.lstm(embeds, (h0, c0))

        output = self.linear1(output.view(seq_len*batch_size, -1))
        # return output, hidden
        return output

class Data:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.poetry_file = FILE_NAME
        self.load()
        self.create_batches()

    def load(self):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            line = line.replace('，','').replace('。','')
            return BEGIN_CHAR + line + END_CHAR

        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                        open(self.poetry_file, encoding='utf-8')]
        self.poetrys = [handle(line)
                        for line in self.poetrys if len(line) > MIN_LENGTH]
        # all words
        words = []
        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)

        # get high frequece word, set unknow word
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,)
        self.words_size = len(self.words)

        # map: char->id, id->char
        self.id2yun = {i: get_suf(get_pin(w)) for i, w in enumerate(self.words)}
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(
            char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        self.poetrys_vector = [list(map(self.char2id, poetry))
                               for poetry in self.poetrys]

    def create_batches(self):
        # divide minibatch
        self.n_size = len(self.poetrys_vector) // self.batch_size
        self.poetrys_vector = self.poetrys_vector[:self.n_size *
                                                  self.batch_size]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i*self.batch_size : (i+1)*self.batch_size]
            length = max(map(len, batches))
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r
            # shuffle data to (x,y)
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)

train_data = Data(batch_size=BATCH_SIZE)

class LNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim ,hidden_dim):
        super(LNN, self).__init__()
        self.num_layers = 2
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            self.hidden_dim, 
                            num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)
    
    def forward(self, input, hidden=None):
        # print ('bug',type(input),input)
        # x = input
        input = torch.from_numpy(input)
        seq_len, batch_size = np.size(input ,0), np.size(input ,1)
        input = input.long()
        input = Variable(input)
        if hidden is None:
            h0 = torch.randn(
                2, batch_size, self.hidden_dim)
            c0 = torch.randn(
                2, batch_size, self.hidden_dim)
        else:
            h0, c0 = hidden
        # print (h0,c0)
        # size here : (seq_len, batch_size, embedin   g_dim)
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h0, c0))

        # size here : (seq_len*batch_size, vocab_size)
        output = self.linear1(output.view(seq_len*batch_size, -1))
        # return output, hidden
        return output


def train(zero=False):
    global train_data
    # train_data = Data(batch_size=BATCH_SIZE)
    if zero==True:
        model = LNN(vocab_size=len(train_data.id2char_dict), embedding_dim=128, hidden_dim=256)
    else:
        try:
            model = torch.load(SAVE_PATH)
        except:
            print ('Not Exist')
            model = LNN(vocab_size=len(train_data.id2char_dict),
                        embedding_dim=128, hidden_dim=256)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    min_loss = 100000000
    for epoch in range(EPOCH):
        for step in range(train_data.n_size):
            b_x = train_data.x_batches[step]
            b_y = train_data.y_batches[step]
         
            output  = model(b_x)

            loss = loss_func(output, torch.from_numpy(b_y).view(-1).long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.data < min_loss:
                min_loss = loss.data
                torch.save(model, SAVE_PATH)
            if step % 50 == 0:
                # test_output = model(test_x.view(-1, 28, 28))
                # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                # accuracy = sum(pred_y == test_y)/float(test_y.size(0))
                # print([lambda ans: train_data.id2char(cha) for cha in b_x])
                # print([lambda ans: train_data.id2char(cha) for cha in output])
                tmp = ''
                for line in b_x:
                    for cha in line:
                        tmp += train_data.id2char(cha)
                print (tmp)
                tmp = ''
                for line in output:
                    cha = torch.argmax(line)
                    tmp += train_data.id2char(int(cha))
                print(tmp)
                print('Epoch:',epoch, 'Step: ', step, '| train loss:', loss.data)

def yum_max(x,result):
    global train_data
    max_q = -1000000000
    num = 3000
    # print (result)
    for i,yun in train_data.id2yun.items():
        if yun==x:
            if max_q < result[i]:
                max_q = result[i]
                num = i
    return num

def predict5(sentence_head, yun=''):
    global train_data
    model = torch.load(SAVE_PATH)
    #print (model.perplexity)
    ans = ''
    #yun = ''
    x = ''
    # if len(ans)==5:
    #     ans += '，\n'
   # yun = get_suf(get_pin(ans[4]))
   # print (get_pin(ans[4]))
   # print (yun)
    b1 = [0,0,0,0]
    with open('poetry.txt', 'r+', encoding='utf-8') as f:
        for line in f:
            try:
                head, poem = line.split(':')
            except:
                continue
            poem = poem.strip().replace('，', '').replace('。', '')
            if len(poem)==20 and sentence_head[0] in poem:
                b1[0]=poem[0]
                b1[1]=poem[5]
                b1[2]=poem[10]
                b1[3]=poem[15]
                reference=poem
                break
    a=[0,0,0,0]
    for k in range(1,4):
        a[k]=random.randint(5*k,5*k+3)
        
    for i in range(0,20):
        if i in a or i ==0:
            x += sentence_head[i//5]
            ans += sentence_head[i//5]
            continue
        if i%5==0:
            x += b1[i//5]
            ans += b1[i//5]
            continue
        inp = np.array(list(map(train_data.char2id, x)))
        inp = inp.reshape((1,len(x)))
        # print ('inp',inp)
        # inp = np.padding(inp, )
        out = model(inp)
        if i==9 or i==19:
            
            if yun[-1]=='3' or yun[-1]=='4':
                q=random.randint(1,2)
                if q==1:
                    yun=yun[:-1]+'1'
                else:
                    yun=yun[:-1]+'2'
           
            tmp_out = train_data.id2char(yum_max(yun, out[i-1]))
            #print(yun)
            pingze=get_pingze(tmp_out)
            while tmp_out=='*' or tmp_out=='E'or tmp_out in ans or tmp_out in sentence_head:
                # print('bug1', tmp_out)
                out = model(inp)
                #print (yun,get_suf(get_pin(train_data.id2char(int(out[i-1].topk(1)[1][0])))))
                tmp_out = train_data.id2char(yum_max(yun, out[i-1]))
                
                # tat = input(233)
        elif i==14:
            tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
            pingze=get_pingze(tmp_out)
            #print(pingze)
            while tmp_out == '*' or tmp_out == 'E' or tmp_out in ans or tmp_out in sentence_head or pingze=='1' or pingze=='2':
                # print ('bug2',tmp_out)
                out = model(inp)
                tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
                pingze=get_pingze(tmp_out)
               # print(pingze)
                
        else:
            tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
            while tmp_out == '*' or tmp_out == 'E' or tmp_out in ans or tmp_out in sentence_head:
                # print ('bug2',tmp_out)
                out = model(inp)
                tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
        x += tmp_out
        ans += tmp_out
        if i%10==4:
            ans += '，\n'
            if yun=='':
                yun = get_suf(get_pin(tmp_out))
                #pingze = get_pingze(tmp_out)
        if i%10==9:
            ans += '。\n'
            # print (x,p)
        # ans += x
    #     print('bug', x)
    print (ans)
    print('Cumulative 1-gram: %f' % sentence_bleu(reference,x, weights=(1, 0, 0, 0)))
    return ans

def predict7(sentence_head, yun=''):
    global train_data
    model = torch.load(SAVE_PATH)
    #print (model.perplexity)
    ans = ''
    #yun = ''
    x = ''
    b1 = [0,0,0,0]
    with open('poetry.txt', 'r+', encoding='utf-8') as f:
        for line in f:
            try:
                head, poem = line.split(':')
            except:
                continue
            poem = poem.strip().replace('，', '').replace('。', '')
            if len(poem)==28 and sentence_head[0] in poem:
                b1[0]=poem[0]
                b1[1]=poem[7]
                b1[2]=poem[14]
                b1[3]=poem[21]
                reference=poem
                break
    a=[0,0,0,0]
    for k in range(1,4):
        a[k]=random.randint(7*k,7*k+5)
        
    for i in range(0,28):
        if i in a or i ==0:
            x += sentence_head[i//7]
            ans += sentence_head[i//7]
            continue
        if i%7==0:
            x += b1[i//7]
            ans += b1[i//7]
            continue
        inp = np.array(list(map(train_data.char2id, x)))
        inp = inp.reshape((1,len(x)))
        # print ('inp',inp)
        # inp = np.padding(inp, )
        out = model(inp)
        if i==13 or i==27:
            
            if yun[-1]=='3' or yun[-1]=='4':
                q=random.randint(1,2)
                if q==1:
                    yun=yun[:-1]+'1'
                else:
                    yun=yun[:-1]+'2'
           
            tmp_out = train_data.id2char(yum_max(yun, out[i-1]))
            #print(yun)
            pingze=get_pingze(tmp_out)
            while tmp_out=='*' or tmp_out=='E'or tmp_out in ans or tmp_out in sentence_head:
                # print('bug1', tmp_out)
                out = model(inp)
                #print (yun,get_suf(get_pin(train_data.id2char(int(out[i-1].topk(1)[1][0])))))
                tmp_out = train_data.id2char(yum_max(yun, out[i-1]))
                
                # tat = input(233)
        elif i==20:
            tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
            pingze=get_pingze(tmp_out)
            #print(pingze)
            while tmp_out == '*' or tmp_out == 'E' or tmp_out in ans or tmp_out in sentence_head or pingze=='1' or pingze=='2':
                # print ('bug2',tmp_out)
                out = model(inp)
                tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
                pingze=get_pingze(tmp_out)
               # print(pingze)
                
        else:
            tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
            while tmp_out == '*' or tmp_out == 'E' or tmp_out in ans or tmp_out in sentence_head:
                # print ('bug2',tmp_out)
                out = model(inp)
                tmp_out = train_data.id2char(int(out[i-1].topk(1)[1][0]))
        x += tmp_out
        ans += tmp_out
        if i%14==6:
            ans += '，\n'
            if yun=='':
                yun = get_suf(get_pin(tmp_out))
                #pingze = get_pingze(tmp_out)
        if i%14==13:
            ans += '。\n'
            # print (x,p)
        # ans += x
    #     print('bug', x)
    print (ans)
    print('Cumulative 1-gram: %f' % sentence_bleu(reference,x, weights=(1, 0, 0, 0)))
    return ans


def deal(): 
    #train(zero=False)
    if sys.argv[2]=='5':
        predict5(sys.argv[1])
    if sys.argv[2]=='7':
        predict7(sys.argv[1])
    
    
def eval():
    b1 = []

    with open('poetry.txt', 'r+', encoding='utf-8') as f:
        for line in f:
            try:
                head, poem = line.split(':')
            except:
                continue
            poem = poem.strip().replace('，', '').replace('。', '')
            if len(poem)==20:
                candidate = predict5(
                    poem[0]+poem[5]+poem[10]+poem[15], get_suf(get_pin(poem[4]))).replace('\n', '').replace('，', '').replace('。', '')
                reference = poem
                # score = sentence_bleu(reference, candidate)
                b1.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
                #b2.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
                #b3.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
                #b4.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
                print (reference)
                print (candidate)
                print('Cumulative 1-gram: %f' %
                    sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
                #print('Cumulative 2-gram: %f' %
                    #sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
                # print('Cumulative 3-gram: %f' %
                #     sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
                # print('Cumulative 4-gram: %f' %
                #     sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))
                break
    # print ('1:{0} 2:{1} 3:{2} 4:{3}'.format(np.mean(b1),np.mean(b2),np.mean(b3),np.mean(b4)))
    # 1:0.25990606319385146 2:7.56724343287869e-155 3:5.678639605182781e-204 4:1.296010893334655e-231
    print((np.mean(b1)))
    #print((np.mean(b1)+ np.mean(b2)+ np.mean(b3)+ np.mean(b4)/4))
if __name__=='__main__':
    deal()
    #eval()
    #train()