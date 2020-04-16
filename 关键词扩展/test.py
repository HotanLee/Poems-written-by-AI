import gensim
import random

model = gensim.models.Word2Vec.load("poetry.model")

'''
print ("天地")
for element in model.wv.most_similar("天地"):
  print (element[0],end=" ") 
  print (element[1])

print ("明月")
for element in model.wv.most_similar("明月"):
  print (element[0],end=" ") 
  print (element[1])

print ("绿")
for element in model.wv.most_similar("绿"):
  print (element[0],end=" ") 
  print (element[1])

print ("落叶")
for element in model.wv.most_similar("落叶"):
  print (element[0],end=" ") 
  print (element[1])

print ("日")
for element in model.wv.most_similar("日"):
  print (element[0],end=" ") 
  print (element[1])

print ("犬")
for element in model.wv.most_similar("犬"):
  print (element[0],end=" ")
  print (element[1])
  
print ("飞絮")
for element in model.wv.most_similar("飞絮"):
  print (element[0],end=" ")
  print (element[1])
'''


def processing(list):
  result = []

  if list == None:
    ToBeProcessed = input("请输入需要拓展的关键词:")
    result.append(ToBeProcessed)
    
    for element in model.wv.most_similar(ToBeProcessed , topn=3):
      result.append(element[0])

    return result
  
  else:
    result.extend(list)
    i=len(list)
    ToBeProcessed = list
    if i==3:
      r1 = random.randint(0,1)
      p = model.wv.most_similar(ToBeProcessed[r1], topn=10)
      r2 = random.randint(0,10)
      result.append(p[r2][0])
    if i==2:
      for element in model.wv.most_similar(ToBeProcessed[0], topn=1):
        result.append(element[0])
      for element in model.wv.most_similar(ToBeProcessed[1], topn=1):
        result.append(element[0])
    if i==1:
      for element in model.wv.most_similar(ToBeProcessed, topn=3):
        result.append(element[0])
        
      '''
      temp=str()
      for element in model.wv.most_similar(ToBeProcessed, topn=1):
        result.append(element[0])
        temp=element[0]
      for element in model.wv.most_similar(temp, topn=1):
        result.append(element[0])
        temp=element[0]
      for element in model.wv.most_similar(temp, topn=1):
        result.append(element[0])
      '''
      
    return result

a=["犬","落叶","悲"]
print(processing(a))
'''
b=["舟","午日","旅人"]
print(processing(b))
c=["落叶"]
print(processing(c))
'''
