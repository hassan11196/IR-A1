
import pickle
import os
import string
import re
import pprint
import numpy as np
import copy
from nltk.stem import PorterStemmer
from .models import InvertedIndexModel
from .Interpreter import Lexer, Interpreter, Parser
class PostingList(object):
    def __init__(self):
        self.total_count = 0
        self.token = ''
        self.occurrance = {
#             'doc_id':0 = 'positions' : [],
#             
        }
         
    def __repr__(self):
        
        return f'total_cnt : {self.total_count} docs : [{self.occurrance.keys()}]'
    
    def addOccurrance(self, doc_id, position):
        self.total_count += 1
#         print(position)
        if doc_id not in self.occurrance.keys():
            self.occurrance[doc_id] = []
        self.occurrance[doc_id].append(position)
#         self.occurrance[doc_id]['position'].append(pos)
    
class InvertedIndex(object):
    
    def __init__(self):
        self.index = {}
        self.docs = {}

    def get_term_postings(self, term):
        if term in self.index.keys():
            return self.index[term]
        else:
            return PostingList()
    
    def __len__(self):
        return len(self.index.keys())



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


def build_index():
    path_to_data = os.path.dirname(__file__) + '../../data/'
    print(os.path.dirname(__file__))
    print(path_to_data)
    vocab = set()
    doc_contents = []
    inverted_index = InvertedIndex()
    printable = set(string.printable) 
    # Printable characters are
    # 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
    # !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c


    ps = PorterStemmer()
    stop_words = set()
    with open(path_to_data+'Stopword-List.txt', 'r') as stop_word_file:
        lines = stop_word_file.readlines()
        for line in lines:
            stop_words.add(line.split('\n')[0])
        stop_words.remove('')
    print(stop_words)

    for file_number in range(0, 56):
        with open(path_to_data + f'Trump Speechs/speech_{file_number}.txt', 'r') as file1:
            lines = file1.readlines()
            print(f'File Number : speech_{file_number}.txt' )
            print(lines[0])
            position = {'doc':file_number,'row':0, 'col':0, 'token_no':0}

            for line_no,line in enumerate(lines):
                doc_set = set()
                # split words at . , whitespace ? ! : ;
                position['row'] = line_no 
                position['col'] = 0
                for word in re.split('[.\s,?!:;-]', line):
                    position['col'] += len(word) + 1
                    position['token_no'] += 1
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
                    
                    if word in inverted_index.index.keys():
                        
                        inverted_index.index[word].addOccurrance(file_number, copy.deepcopy(position)) 
                    else:
                        plist = PostingList()
                        inverted_index.index[word] = plist
                        inverted_index.index[word].addOccurrance(file_number, copy.deepcopy(position))
                        
            inverted_index.docs[file_number] = doc_set
            doc_contents.append(doc_set)
    ii = InvertedIndexModel()
    ii.status = True
    ii.data = inverted_index
    ii.save()
    return True

def get_boolean_query(query):
    text = query
    text = text.replace('and','&')
    text = text.replace('AND','&')
    text = text.replace('or','|')
    text = text.replace('OR','|')
    text = text.replace('NOT', '!')
    text = text.replace('not','!')
    
    inverted_index = InvertedIndexModel.objects.get()
    ps = PorterStemmer()
    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser, inverted_index, ps)
    result = interpreter.interpret()
    print(result.value)
    print(result.row)
    return result.row


def intersect_posting(p1, p2):
    return p1.occurrance.keys() & p2.occurrance.keys()
def union_posting(p1, p2):
    return p1.occurrance.keys() | p2.occurrance.keys()
def inverse_posting(inverted_index,p):
    return inverted_index.docs.keys() - p.occurrance.keys()