#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os, string
import re
import nltk
from nltk import word_tokenize 
from nltk.corpus import stopwords
from nltk import PorterStemmer
import operator
import math
import sys
import pickle
from concurrent.futures import ThreadPoolExecutor


class processor:
    
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers = 10)
        self.df = pd.DataFrame()
        
    def extract_URL(self, text):
        urlID = re.search(r'<URL>[\s\S]*?</URL>',text).group(0).replace('<URL>','').replace('</URL>','').strip()
        #print("The value of the stripped URL ===", urlID)
        text = text.replace(urlID, '')
        
        #print("The value of clean text====", text)
        return urlID, text
    
    def process_tokens(self, words):
        new_words = []
        digit_pattern = '[0-9]'
        for word in words:
            table = str.maketrans({key : None for key in string.punctuation})
            new_word = word.translate(table)
            new_word = new_word.replace(' ', '')
            new_word = re.sub(digit_pattern, '', word)
            #URLless_string = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', new_word)
            if new_word != '':
                new_words.append(new_word)
            
        new_words = [x for x in new_words if len(x) > 2]
        
        return new_words
    
    def porter_stemmer(self, words):
        stemmer = PorterStemmer()
        porter_Stemmer_List = []
        stop_words = nltk.corpus.stopwords.words('english')
        add_stop = ('URL', '/URL')
        for i in add_stop:
            stop_words.append(i)
        
        for i in words:
            if i not in stop_words:
                value = stemmer.stem(i)
                #Remove value if it becomes a stopword after stemming
                if value not in stop_words:
                    porter_Stemmer_List.append(value)
        return porter_Stemmer_List
        
    def pre_process(self, contents):
        urlID, clean_text = self.extract_URL(contents)
        clean_text = word_tokenize(clean_text)
        clean_text = self.process_tokens(clean_text)
        clean_text = self.porter_stemmer(clean_text)
        clean_text.append(urlID)
        
        return clean_text
    
    def pre_process_queries(self, contents):
        contents = contents.lower()
        contents = word_tokenize(contents)
        contents = self.process_tokens(contents)
        contents = self.porter_stemmer(contents)
        
        return contents
        
    def load_documents(self, path):
        df = pd.DataFrame()
        file_list = os.listdir(path)
        for file in file_list:
            file = open(os.path.join(path, file), encoding='utf-8', errors = 'ignore')
            contents = file.read()
            df = df.append({'Content' : contents}, ignore_index = True)
        return df
        
    def load_queries(self, path):
        query_df = pd.DataFrame()
        file_list = os.listdir(path)
        for f in file_list:
            f = open (os.path.join(path, f))
            query_contents = f.read()
            query_df = query_df.append({'Query':query_contents}, ignore_index = True)
        return query_df
    
    def calc_inverse_doc_frequency(self, token, alldocs):
        numDocumentsWithThisTerm = 0.0
        numDocumentsWithThisTerm = sum(map(lambda document: token in document,alldocs))
        if numDocumentsWithThisTerm > 0.0:
            return 1.0 + math.log(float(len(alldocs)) / numDocumentsWithThisTerm)
        else:
            return 1.0
    
    def find_idf(self, docs, unique_tokens):
        from collections import defaultdict
        idf = defaultdict(str)
        for token in unique_tokens:
            idf[token] = self.calc_inverse_doc_frequency(token, docs)
            #print("The value of idf[token] is===", idf[token])
        return idf
    
    def find_tf(self, term, doc):
        return doc.count(term)
    
    def find_tf_idf(self, docs, unique_tokens):
        tf_idf_all_docs = []
        idf = self.find_idf(docs, unique_tokens)
        for doc in docs:
            tf_idf_doc = []
            for term in idf.keys():
                tf = self.find_tf(term, doc)
                tf_idf_doc.append(tf * idf[term])
            tf_idf_all_docs.append(tf_idf_doc)
        #print("The value of tf_idf_of_all_docs is===", tf_idf_all_docs)
        return tf_idf_all_docs
    
    def find_cos_similarity(self, vector1, vector2):
        dot_prod = [sum(a*b for a,b in zip(vector1, vector2 ))]
        denominator = math.sqrt(sum(val**2 for val in vector1)) * math.sqrt(sum(val**2 for val in vector2))
        if not denominator:
            return 0
        else:
            return np.asarray(dot_prod)/np.asarray(denominator)
    
    def find_similar_docs(self, df, query, docs):
        cosine_similarity = []
        for doc in docs:
            cosine_similarity.append(self.find_cos_similarity(query, doc))
        similar_docs = dict(enumerate(cosine_similarity, 1))
        URL_df = df['URL']
        URL_df_list = URL_df.values.tolist()
        
        URL_dict = {}
        URL_dict = dict(enumerate(URL_df_list, start = 1))
        
        for key in URL_dict:
            if key in similar_docs:
                similar_docs[URL_dict[key]] = similar_docs[key]
                del similar_docs[key]
                
        ranked_docs = sorted(similar_docs.items(), key = operator.itemgetter(1), reverse = True)
        return ranked_docs
        
    def rank_the_docs(self, df, docs, queries, n):
        query_ranked_list = []
        q = 1
        for query in queries:
            ranked_list = []
            ra = self.find_similar_docs(df, query, docs)
            for r in range(n):
                ranked_list.append(ra[r][0])
            q = q + 1
            query_ranked_list.append(ranked_list)
        return query_ranked_list
        
    
    def run_preprocessor(self,path):
        df = self.load_documents(path)
        df.Content = df.Content.apply(self.pre_process)
        df['URL'] = df.Content.apply(lambda x : x[-1])
        unique_tokens = set(word for words in df.Content for word in words)
        document_tf_idf = self.find_tf_idf (df.Content, unique_tokens)
        #print("The value of df is ===", df)
        return document_tf_idf, df
        
    def run_processor_for_query(self, path):
        #print("The value of dataframe==", df)
        query_df = self.load_queries(path)
        query_df ['processed_query'] = query_df.Query.apply(self.pre_process_queries) 
        unique_tokens = set(word for words in query_df.processed_query for word in words)
        query_tf_idf = self.find_tf_idf(query_df.processed_query, unique_tokens)
        return query_tf_idf, df

    
    def write_ranked_docs(self, outpath, ranked_list):
        #print("the value of outpath is===", outpath)
        filename = "output.txt"
        completename = os.path.join(outpath, filename)
        
        with open (completename, 'w') as file:
            for r in ranked_list:
                for d in r:
                    line = str(d) + "\n"
                    file.write(line)
        
if "__main__" == __name__:
    #print("processing main")
    workingDirectory = os.getcwd()
    filesPath = os.path.join(workingDirectory, 'uic-docs-text')
    query_files_path = os.path.join(workingDirectory, 'query-doc')
    output_path = os.path.join(workingDirectory, 'out_put') 
    p = processor()
    doc_tf_idf, df = p.run_preprocessor(filesPath)
    
    #Writing documents tf idf into file
    tf_idf_file_path =  os.path.join(workingDirectory, 'tf-idf')
   # print("The value of tf-idf path is ===", tf_idf_file_path)
    filename = "tf-idf.txt"
    
    pickle_out = open("doc-tf-idf.pickle", "wb")
    pickle.dump(doc_tf_idf, pickle_out)
    pickle_out.close()
    
    
    #query_tf_idf = p.run_processor_for_query(query_files_path)
    #ranked_docs = p.rank_the_docs(df, doc_tf_idf, query_tf_idf, 500)
    #p.write_ranked_docs(output_path, ranked_docs)
    

