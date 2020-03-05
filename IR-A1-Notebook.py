#!/usr/bin/env python
# coding: utf-8

# ## Imports and declerations

# In[1]:


import pickle
import os
import string
import re
import pprint
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
import boolean
import pyparsing
from pyparsing import Word, alphas, oneOf, operatorPrecedence, opAssoc
InteractiveShell.ast_node_interactivity = "all"



os.getcwd()


# ## Raw Vocabulary Storage:

# In[2]:


from nltk.stem import PorterStemmer


# In[3]:



# Break words like Veterans.Before, West.In amendment.Change

def split_words(vocabl):
    new_vocab = set()
    for word in vocabl:
        if re.search('^[a-zA-Z]+[.][a-zA-Z]+$',word) is not None:
            print(re.search('^[a-zA-Z]+[.][a-zA-Z]+$',word))
            w1, w2 = word.split('.')
#             print(w1)
# #             print(w2)
            new_vocab.add(w1)
            new_vocab.add(w2)
        elif re.search('^[a-zA-Z]+[?][a-zA-Z]+$',word) is not None:
# #             print(re.search('^[a-zA-Z]+[.][a-zA-Z]+$',word))
            w1, w2 = word.split('?')
# #             print(w1)
# #             print(w2)
            new_vocab.add(w1)
            new_vocab.add(w2)
        elif re.search('^[a-zA-Z]+[,][a-zA-Z]+$',word) is not None:
# #             print(re.search('^[a-zA-Z]+[.][a-zA-Z]+$',word))
            w1, w2 = word.split(',')
# #             print(w1)
# #             print(w2)
            new_vocab.add(w1)
            new_vocab.add(w2)
        else:
            new_vocab.add(word)
    return new_vocab

# Remove Punctuation
def remove_punctuation(word):
    return word.translate(word.maketrans('','',string.punctuation))


# In[173]:


vocab = set()
doc_contents = []
printable = set(string.printable) 
# Printable characters are
# 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
# !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c


ps = PorterStemmer()

stop_words = set()
with open('Stopword-List.txt', 'r') as stop_word_file:
    lines = stop_word_file.readlines()
    for line in lines:
        stop_words.add(line.split('\n')[0])
    stop_words.remove('')

for file_number in range(0, 56):
    with open(f'data/Trump Speechs/speech_{file_number}.txt', 'r') as file1:
        lines = file1.readlines()
#         print(f'File Number : speech_{file_number}.txt' )
#         print(lines[0])
        for line in lines:
            doc_set = set()
            # split words at . , whitespace ? ! : ;
            for word in re.split('[.\s,?!:;-]', line):
                
                
                # Case Folding
                word = word.lower()
                
                # Filter non-ASCII characters
                word = ''.join(filter(lambda x: x in printable, word))
                
                # Remove Punctuations
                word = remove_punctuation(word)
                
                if re.match('\d+[A-Za-z]+',word):
                    word = re.split('\d+',word)[1]
                if re.match('[A-Za-z]+\d+',word):
                    word = re.split('\d+',word)[0]
                
                if len(word) == 0 or len(word) == 1 or word == '' or word == ' ':
                    continue
                if word in stop_words:
                    continue
                         
                word = ps.stem(word)
                    
                vocab.add(word)
                
                doc_set.add(word)
        
        doc_contents.append(doc_set)


# In[5]:


print('Total Vocabulary Size ')
print(len(vocab))
print('Total Number of Documents ')
print(len(doc_contents))
print(doc_contents[17])


# # Boolean Model:

# In[6]:


# print(sorted(list(vocab)))
# for index,doc in enumerate(doc_contents):
#     print('Vocab size of doc' + str(index))
#     print(len(doc))

vocab_list = sorted(list(vocab))

term_doc_matrix_np = np.zeros((len(vocab), len(doc_contents)))

for word_index, word in enumerate(vocab_list):
    word_row = []
    for doc_index, doc in enumerate(doc_contents):
        if word in doc:
            term_doc_matrix_np[word_index, doc_index] = 1
        else:
            term_doc_matrix_np[word_index, doc_index] = 0
            
print(term_doc_matrix_np)
    


# In[23]:



with open('pickled/vocab.p', 'ab') as vocab_file:
    pickle.dump(vocab, vocab_file)


# In[24]:


with open('pickled/vocab.p', 'rb') as vocab_file:
    vocabf = pickle.load(vocab_file)


# In[65]:


query = input('Enter your query : ')
query = ps.stem(query)
query_actions = []
query_wanted = []
if query in vocab:
    term_index = vocab_list.index(query)
    term_row = term_doc_matrix_np[term_index]
    print(term_row)
    doc_ids = np.argwhere(term_row == 1)
    print(doc_ids)

else:
    print(f'{query} not present in vocabulary')


# In[66]:


test = '000dollar'
x = re.match('\d+[A-Za-z]+',test)
print(x)
x = re.split('\d+',test)
print(x)
# For matching queries like
# not hammer or pakistan
# (magnum or not hammer) or not (polish and pakistan)
x = re.match('(not)?\s*(\w+|(\((not)?\s*(\w+)\s+(and|or)\s+(not)?\s*(\w+)\)))\s+(or|and)\s+(not)?\s*(\w+|(\((not)?\s*(\w+)\s+(and|or)\s+(not)?\s*(\w+)\)))')


# In[66]:


ans = {'0', '1', '10', '11', '12', '16', '17', '18', '19', '2', '20', '21', '22', '24', '25', '26', '27', '28', '3', '30', '32', '33', '34', '35', '36', '37', '39', '4', '40', '41', '44', '45', '46', '47', '5', '50', '51', '52', '53', '6', '8', '9'}
ans2 =set([str(x[0]) for x in doc_ids])


# In[67]:


print(len(ans))
print((ans2))
print(ans.difference(ans2))


# # References :
# 
# http://www.pyregex.com/
# http://cs231n.github.io/python-numpy-tutorial/
# 
# https://www.online-utility.org/text/analyzer.jsp
# 
# https://stackoverflow.com/questions/2118261/parse-boolean-arithmetic-including-parentheses-with-regex
# 
# https://regex101.com/r/M8z3U4/1
# 
# https://iq.opengenus.org/porter-stemmer/

# In[44]:


query_actions = []
query_wanted = []
term_index = vocab_list.index(ps.stem('actions'))
term_row_actions = term_doc_matrix_np[term_index]

query_actions = np.argwhere(term_row_actions == 1)
term_index = vocab_list.index(ps.stem('wanted'))
term_row_wanted = term_doc_matrix_np[term_index]

query_wanted = np.argwhere(term_row_wanted == 1)

and_query = np.array([1 if x == 1 and y == 1 else 0 for x,y in zip(term_row_actions, term_row_wanted)])
and_doc_ids = np.argwhere(and_query == 1)
print(and_query)
print(and_doc_ids)

ans = {'37', '3', '19', '1', '9', '40', '51', '16', '15', '12', '31', '41', '39', '0', '53', '26', '29', '17', '24', '54', '7', '2', '5', '28', '42'}
ans2 =set([str(x[0]) for x in and_doc_ids])

print(len(ans))
print((ans2))
print(ans.difference(ans2))


# In[147]:





# In[88]:


inp = "pakistan AND Running"
re.split('[\s\(\)]', inp)


# In[86]:




# Tokenizer Test
line = 'the Institute for Energy Research cites a "short-run" figure of as much as $36'
for word in re.split('[.\s,?!:;-]', line):            
    print(word)
    # Case Folding
    word = word.lower()
    print(word)
    # Filter non-ASCII characters
    word = ''.join(filter(lambda x: x in printable, word))
    print(word)
    # Remove Punctuations
    word = remove_punctuation(word)
    print(word)
    if re.match('\d+[A-Za-z]+',word):
        word = re.split('\d+',word)[1]
    if re.match('[A-Za-z]+\d+',word):
        word = re.split('\d+',word)[0]
    print(word)
    if len(word) == 0 or len(word) == 1 or word == '' or word == ' ':
        continue
    if word in stop_words:
        continue
    print(word)
    word = ps.stem(word)
    print(word)


# In[174]:


from collections import deque 

line = 'box AND ( united OR year )'
boperators = ['and', 'or']
uoperators = ['not']
  
stack = deque() 
        
def clean_word(word):
    # Case Folding
    word = word.lower()
     # Filter non-ASCII characters
    word = ''.join(filter(lambda x: x in printable, word))
#     print(word)
    # Remove Punctuations
    if word != '(' and word != ')':
        word = remove_punctuation(word)
#     print(word)
    if re.match('\d+[A-Za-z]+',word):
        word = re.split('\d+',word)[1]
    if re.match('[A-Za-z]+\d+',word):
        word = re.split('\d+',word)[0]
#     print(word)
    word = ps.stem(word)
#     print(word)
    return word
query = (re.split('[.\s,?!:;-]', line))


def evaluate_expression(index, query, stack, state):
    
    print("Current indexed Word : " + str(query[index]))
    result_query = []
    
    if query[index] == '(':
        bracket_term = query[index] 
        result, new_index = evaluate_expression(index+1, query, stack, state)
        if new_index == -1:
            state = []
        for x in range(index, new_index+1):
            state.pop(x)
            
        print("Result")
        print(result)
        index = new_index
        stack.append({'state':True,'data':result, 'query': 'not' + not_of_term})
        result_query = result
        
        print('new_index ' + str(new_index))
        print(f'This should be a ) = {query[new_index]}')
        
    
    
    
    
    
    
    
    if query[index] in uoperators:
        not_of_term = query[index] 
        result, new_index = evaluate_expression(index+1, query, stack, state)
        if new_index == -1:
            state = []
        for x in range(0, new_index-index+1):
            state.pop(x)
            
        print("Result")
        print(result)
        index = new_index
        not_result = [0 if int(x)==1 else 1 for x in result]
        stack.append({'state':True,'data':not_result, 'query': 'not' + not_of_term})
        result_query = not_result
        
    print(query[index])
    if len(state) == 0:
        return result_query, index
    print('STATE')
    print(state)
    
    
    if (query[index] not in boperators) and (query[index] not in uoperators):
        query[index] = ps.stem(query[index])
        if query[index] not in vocab_list:
            print(f'{query[index]} is not in vocabulary of index')
            return [], -1
        term_index = vocab_list.index(query[index])
        term_row = term_doc_matrix_np[term_index]
        stack.append({'state':True,'data':term_row,'query':query[index]})
        result_query = term_row
        index += 1
        
    if index >= len(query):
        return stack.pop()['data'], -1
    if len(state) == 0:
        return result_query, index
    
    elif query[index] in boperators:
#         query2 = clean_word(next_word)
        
#         next_word = query[index+1]
#         print(next_word)
#         if query2 not in vocab_list:
#             print(f'{query2} is not in vocabulary of index')
#             return [], -1
        
        
#         term_index2 = vocab_list.index(query2)
#         term_row2  = term_doc_matrix_np[term_index2]
        
        
        term_row2, new_index = evaluate_expression(index+1, query, stack, state)
        if new_index == -1:
            state = []
        for x in range(0, new_index-index+1):
            state.pop(x)
        
        
        query1 =  stack.pop()
        term_row1 = []
        
        if query1['state'] == False:
            term_index1 = vocab_list.index(query1['data'])
            term_row1 = term_doc_matrix_np[term_index1]
        
        else:
            term_row1 = query1['data']
        
        print(term_row1)
        
        print(term_row2)
        result_query = []
        if query[index] == 'and':
            and_query = np.array([1 if int(x) == 1 and int(y) == 1 else 0 for x,y in zip(term_row1, term_row2)])
            print(and_query)
            and_doc_ids = np.argwhere(and_query == 1)
            query_ans =  set([x[0] for x in and_doc_ids])
            print(query_ans)
            # stack.append({'state':True,'data':and_query,'query':query2})
            result_query = and_query
            
        elif query[index] == 'or':
            or_query = np.array([1 if int(x) == 1 or int(y) == 1 else 0 for x,y in zip(term_row1, term_row2)])
            print('OR')
            print(or_query)
            or_doc_ids = np.argwhere(or_query == 1)
            query_ans =  set([x[0] for x in or_doc_ids])
            print(query_ans)
            # stack.append({'state':True,'data':or_query,'query':query2})
            result_query = or_query
        index = new_index
        
    return result_query, index


# for index, word in enumerate(query):  
#     word = clean_word(word)
#     print(word)

query = [clean_word(word) for word in query ]
print(query)
ans, index = evaluate_expression(0, query, stack, query)

print("Outt")
print(ans)

print(stack)
final_doc_ids = np.argwhere(np.array(ans) == 1)
final_ans =  set([x[0] for x in final_doc_ids])
print(final_ans)


# In[163]:


ansq ={'29', '16', '4', '22', '37', '40', '42', '18', '1', '17', '41', '39', '9', '3'}
ansi = {1, 3, 4, 37, 39, 40, 9, 41, 42, 16, 17, 18, 22, 29}
ansq2 = set([str(x) for x in final_ans])
print(ansq2)
ansq.difference(ansq2)


# In[ ]:





# In[75]:


boperators = ['and', 'or']
uoperators = ['not']

if ('and' not in boperators) and ('and' not in uoperators):
    print('sds')


# In[ ]:




