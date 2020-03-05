from collections import deque 
import pickle
import os
import string
import re
import pprint
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
from nltk.stem import PorterStemmer

InteractiveShell.ast_node_interactivity = "all"

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
        stack.append({'state':True,'data':result, 'query': 'bracket' + bracket_term})
        result_query = result
        
        print('new_index ' + str(new_index))
        print(f'This should be a ) = {query[new_index]}')
        
    
    
    
    if query[index] == ')':
        if index + 1 < len(query):
            return result_query, index + 1
        else:
            return result_query, -1
    
    
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
    
    if query[index] == ')':
        if index + 1 < len(query):
            return result_query, index + 1
        else:
            return result_query, -1

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

    if query[index] == ')':
        if index + 1 < len(query):
            return result_query, index + 1
        else:
            return result_query, -1
        
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



vocab = set()
doc_contents = []
printable = set(string.printable) 
# Printable characters are
# 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
# !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c


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