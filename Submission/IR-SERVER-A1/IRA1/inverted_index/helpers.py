
import json
import pickle
import os
import string
import re
import pprint
import numpy as np
import copy
from nltk.stem import PorterStemmer
from string import printable
import re
from .models import InvertedIndexModel
class PostingList(object):
    def __init__(self):
        self.total_count = 0
        self.token = ''
        self.occurrance = {
#             'doc_id':0 = 'positions' : [],
#             
        }
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
         
    def __repr__(self):
        
        return f'total_cnt : {self.total_count} docs : [{self.occurrance.keys()}]'
    def __len__(self):
        return len(self.occurrance)
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
            raise ValueError(f'{term} is not present in Index. Plese Try Again')
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
def intersect_posting(p1, p2):
    if len(p1) == 0 or len(p2) == 0:
        return PostingList()

    if isinstance(p1, set) and isinstance(p2, set):
        return p1.intersection(p2)
    elif isinstance(p1, set):
        return p1.intersection(p2.occurrance.keys())
    elif isinstance(p2, set):
        return p2.intersection(p1.occurrance.keys())
    pn = PostingList()
    # pn.token = f'{p1.token} & {p2.token}'
    for pn_keys in (p1.occurrance.keys() & p2.occurrance.keys()) :
        pn.addOccurrance(pn_keys, p1.occurrance[pn_keys])
        pn.addOccurrance(pn_keys, p2.occurrance[pn_keys])
    return pn


def union_posting(p1, p2):
    if len(p1) == 0:
        return p2
    elif len(p2) == 0:
        return p1
    if isinstance(p1, set) and isinstance(p2, set):
        return p1.union(p2)
    elif isinstance(p1, set):
        return p1.union(p2.occurrance.keys())
    elif isinstance(p2, set):
        return p2.union(p1.occurance.keys())
    
    pn = PostingList()
    # pn.token = f'{p1.token} | {p2.token}'
    for pn1_keys in p1.occurrance.keys() :
        pn.addOccurrance(pn1_keys, p1.occurrance[pn1_keys])
    for pn2_keys in p2.occurrance.keys() :
        pn.addOccurrance(pn2_keys, p2.occurrance[pn2_keys])
    
    return pn

def inverse_posting(inverted_index,p):
    # print(p)
    if isinstance(p, set) :
        # print('Returning ')
        # print(set(inverted_index.docs).difference(p))
        return set(inverted_index.docs).difference(p)
    else:
        # print(set(inverted_index.docs).difference(set(p.occurrance.keys())))
        return set(inverted_index.docs).difference(set(p.occurrance.keys()))
    return inverted_index.docs.keys() - p.occurrance.keys()



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


boperators = ['and', 'or']
uoperators = ['not']

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis
LPAREN, RPAREN, EOF, TERM, AND, OR, NOT = (
    '(', ')', 'EOF', 'TERM', 'AND','OR', 'NOT'
)


class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value
        self.inverse = False
        self.row = []

    def __str__(self):
        """String representation of the class instance.

        Examples:
            Token(TERM, Hello)
            Token(AND, '&')
            Token(NOT, '!')
        """
        return 'Token({type}, {value})'.format(
            type=self.type,
            value=repr(self.value)
        )

    def __repr__(self):
        return self.__str__()


class Lexer(object):
    def __init__(self, text):
        # client string input, e.g. "hello | world & (why | are | you)"
        self.text = text
        # self.pos is an index into self.text
        self.pos = 0
        self.current_char = self.text[self.pos]
        

    def error(self):
        raise Exception('Invalid character')

    def advance(self):
        """Advance the `pos` pointer and set the `current_char` variable."""
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None  # Indicates end of input
        else:
            self.current_char = self.text[self.pos]

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char.isspace():
            self.advance()

    def integer(self):

        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)
    def word(self):
        result = ''
        # while self.current_char is not None and (self.current_char.isalpha() or self.current_char == '_'):
        while self.current_char is not None and (self.current_char in printable) and (self.current_char not in (' ', '|','&','!', '(', ')')):
            result += self.current_char
            self.advance()
        return str(result)

    def get_next_token(self):
        """Lexical analyzer (also known as scanner or tokenizer)

        This method is responsible for breaking a sentence
        apart into tokens. One token at a time.
        """
        while self.current_char is not None:

            if self.current_char.isspace():
                self.skip_whitespace()
                continue
                        
            if self.current_char == '&':
#                 print('gotB' + self.current_char)
                self.advance()
                return Token(AND, 'AND')
            
            if self.current_char == '|':
#                 print('gotB' + self.current_char)
                self.advance()
                return Token(OR, 'OR')
            
            if self.current_char == '!':
#                 print('gotB' + self.current_char)
                self.advance()
                return Token(NOT,'NOT')
            

            if self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')

            if self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')
            
            if self.current_char.isalpha():      
#                 print('Got token  ' + self.current_char)
                return Token(TERM, self.word())
            
            

            self.error()

        return Token(EOF, None)

class AST(object):
    pass


class BinOp(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right
        self.inverse = False
        self.row = []
        self.value = ''
        
class Num(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value
        self.inverse = False
        self.row = []

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
        # set current token to the first token taken from the input
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    def eat(self, token_type):
        # compare the current token type with the passed token
        # type and if they match then "eat" the current token
        # and assign the next token to the self.current_token,
        # otherwise raise an exception.
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        """factor : INTEGER | LPAREN expr RPAREN"""
        token = self.current_token
        
        
        if token.type == TERM:
            self.eat(TERM)
            return Num(token)
        
        elif token.type == NOT:
            self.eat(NOT)
            node = self.expr()
            node.inverse = True
            return node
            
        
        
        elif token.type == LPAREN:
            self.eat(LPAREN)
            node = self.expr()
            self.eat(RPAREN)
            return node

    def term(self):
        
        node = self.factor()

        while self.current_token.type in (AND,):
            token = self.current_token
         
            if token.type == AND:
                self.eat(AND)
            
            node = BinOp(left=node, op=token, right=self.factor())

        return node

    def expr(self):
        node = self.term()

        while self.current_token.type in (OR,):
            token = self.current_token
            if token.type == OR:
                self.eat(OR)
            
            node = BinOp(left=node, op=token, right=self.term())

        return node

    def parse(self):
        return self.expr()


class NodeVisitor(object):
    def visit(self, node):
        print('Checking Node Name')
        
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):
    def __init__(self, parser, index, ps):
        self.parser = parser
        self.index = index
        self.ps = ps

    def visit_BinOp(self, node):
        print('Bin OP : ' )
        print(node.token)
        print(node.value)
        print(node.row)
        print(node.inverse)
        if node.op.type == AND:
            print('Node => ')
            print(node)
            left = self.visit(node.left)   
            right = self.visit(node.right)
            
#             term_index_left = vocab_list.index(ps.stem(left.value))
#             term_row_left = term_doc_matrix_np[term_index_left]
            
#             term_index_right = vocab_list.index(ps.stem(right.value))
#             term_row_right = term_doc_matrix_np[term_index_right]
            
            if left.inverse == True:
                left.value = '!' + str(left.value)
                term_row_left = self.index.index[self.ps.stem(left.row)]
                left.inverse = False
            
            if right.inverse == True:
                right.value = '!' + str(right.value)
                term_row_right = self.index.index[self.ps.stem(right.row)]
                right.inverse = False
            
            node.row = intersect_posting(left.row, right.row)
            if node.inverse == True:
                node.row = inverse_posting(self.index, node.row)
                node.inverse = False
            
            return node
        
        elif node.op.type == OR:
            print('Node => ')
            print(node)
            left = self.visit(node.left)   
            right = self.visit(node.right)
            
#             term_index_left = vocab_list.index(ps.stem(left.value))
#             term_row_left = term_doc_matrix_np[term_index_left]
            
#             term_index_right = vocab_list.index(ps.stem(right.value))
#             term_row_right = term_doc_matrix_np[term_index_right]
            
            if left.inverse == True:
                left.value = '!' + str(left.value)
                term_row_left = self.index.index([self.ps.stem(left.row)])
                left.inverse = False
            
            if right.inverse == True:
                right.value = '!' + str(right.value)
                term_row_right = self.index.index([self.ps.stem(right.row)])
                right.inverse = False
            

            node.row = union_posting(left.row, right.row)
            if node.inverse == True:
                node.row = inverse_posting(self.index, node.row)
                node.inverse = False
            
            return node
        

    def visit_Num(self, node):
        print('Num  : ' )
        print(node.token)
        print(node.value)
        print(node.row)
        print(node.inverse)
        
        node.value = node.value.split('_')[0]
        if self.ps.stem(node.value) in self.index.index.keys():
            
            term_docs = self.index.index[self.ps.stem(node.value)]
            
        else:
            term_docs = {}
            print('Term Row')
            print(term_docs)
            
        node.row = term_docs
        if node.inverse == True:
            node.row = inverse_posting(self.index, node.row)
            node.inverse = False
        
        return node

    def interpret(self):
        tree = self.parser.parse()
        print(tree)
        return self.visit(tree)


def get_boolean_query(query):
    text = str(query)
    text = text.replace(' and ','&')
    text = text.replace(' AND ','&')
    text = text.replace(' or ','|')
    text = text.replace(' OR ','|')
    text = text.replace('NOT', '!')
    text = text.replace('not ','!')
    print(text)
    inverted_index_model_obj = InvertedIndexModel.objects.latest('id')
    inverted_index = inverted_index_model_obj.data
    print('Inverted Index')
    
    ps = PorterStemmer()
    lexer = Lexer(text)
    parser = Parser(lexer)
    interpreter = Interpreter(parser, inverted_index, ps)
    result = interpreter.interpret()

    print(result.value)
    print(result.row)
    return result.row

def positional_intersect(p1, p2, k):
    
    ip = intersect_posting(p1, p2)
    
    lip = sorted(list(ip.occurrance))
    npl = PostingList()
    ans = []
    
    for doc in lip:
#         print(type(p1))
        positions1 = p1.occurrance[doc]
        positions2 = p2.occurrance[doc]
        index_p2 = 0
        index_p1 = 0
        for pos1 in positions1:
            for pos2 in positions2:
                if pos2['token_no'] -  pos1['token_no'] == k and pos2['token_no'] -  pos1['token_no'] > 0:
                    ans.append({'doc':doc, 'pos1':  pos1, 'pos2':pos2})
                    npl.addOccurrance(doc,pos1)
                    npl.addOccurrance(doc,pos2)
        
        
    return npl
        

def get_phrasal_query(query):
    text = str(query)

    try:
        q1, q2 = text.split(' ')
    except ValueError as e:
        raise ValueError('Invalid Phrasal Query Syntax')
    ps = PorterStemmer()
    q1 = ps.stem(q1)
    q2 = ps.stem(q2)
    inverted_index_model_obj = InvertedIndexModel.objects.latest('id')
    inverted_index = inverted_index_model_obj.data
    print('Inverted Index')
    result = [] 
    p1 = inverted_index.get_term_postings(q1)
    p2 = inverted_index.get_term_postings(q2)
    
    result = positional_intersect(p1, p2, 1)
    
    return result

def get_proximity_query(query):
    text = str(query)
    try:
        q1, q2, q3 = text.split(' ')
    except ValueError as e:
        raise ValueError('Invalid Proximity Query Syntax')
    ps = PorterStemmer()
    q1 = ps.stem(q1)
    q2 = ps.stem(q2)
    k = int(q3[1]) + 1
    inverted_index_model_obj = InvertedIndexModel.objects.latest('id')
    inverted_index = inverted_index_model_obj.data
    
    print('Inverted Index')
    result = [] 
    
    p1 = inverted_index.get_term_postings(q1)
    p2 = inverted_index.get_term_postings(q2)
    print(p1)
    result = positional_intersect(p1, p2, k)
    
    return result