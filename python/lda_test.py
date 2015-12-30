# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 13:06:04 2015

@author: wanghuaq
"""

import numpy as np
import random
class LDA(object):
    def __init__(self, dtm, k, alpha=0.1, beta=0.1, it=10):
        self.dtm = dtm
        self.alpha = alpha
        self.beta = beta
        self.k = k
        # M: num of documents
        # V: vocabulary size
        self.M, self.V = dtm.shape
        self.it = it
        #self.topic_word = []    
        self.phi = None
        self.theta = None
                
    def initialize(self):
        self.topic_term = np.zeros((self.k, self.V))
        self.doc_topic  = np.zeros((self.M, self.k))
        # topic 1 to k: count for documents
        self.topic_term_sum = np.zeros(self.k)
        self.doc_topic_sum  = np.zeros(self.M)
        for m in range(self.M):
            for n in range(self.V):
                z = random.randint(1, self.k) - 1
                self.topic_term[z, n] += self.dtm[m, n]
                self.doc_topic[m, z]  += self.dtm[m, n]
                self.topic_term_sum[z]+= self.dtm[m, n]
                self.doc_topic_sum[m] += self.dtm[m, n]
                                
    def conditional_prob(self, m, n):
        """  
            p(z| z_rest, W)
        """
        p1 =  (self.topic_term[:, n] + self.beta) / (self.topic_term_sum + self.beta * self.V)
        p2 =  (self.doc_topic[m, :] + self.alpha) / (self.doc_topic_sum[m] + self.alpha * self.k)
        pz = p1 * p2
        print("p1  p2  ", p1, p2)
        # normalize pz to be probablilty
        pz = pz / np.sum(pz)
        return pz
    
    
    def gibbs_sampling(self):
        """ One iteration of gibbs sampling """
        for m in range(self.M):
            for n in range(self.V):
                # minus 1 for each element                 
                self.doc_topic[m:,]       -= (self.doc_topic[m:] > 0)
                self.topic_term[:, n]     -= (self.topic_term[:, n] > 0)
                self.doc_topic_sum        -= (self.doc_topic_sum > 0)
                self.topic_term_sum       -= (self.topic_term_sum > 0)
 
               
                # Gibbs sampling
                pz = self.conditional_prob(m,n)
                print(pz)
                # sample topic index z , fetch the index of 1
                z = np.random.multinomial(1, pz).argmax() 
                self.topic_term[z, n] += self.dtm[m, n]
                self.doc_topic[m, z]  += self.dtm[m, n]
                self.topic_term_sum[z]+= self.dtm[m, n]
                self.doc_topic_sum[m] += self.dtm[m, n]
            
    def fit(self):
        self.initialize()
        for i in range(self.it):
            self.gibbs_sampling()
        self.get_phi()
        self.get_theta()
    
    def get_phi(self):
        """ get phi """
        self.phi = self.topic_term + self.beta
        s = np.sum(self.phi, axis=1)
        self.phi = self.phi / s.reshape(len(s), 1)       
        
    def get_theta(self):
        """ get theta """
        self.theta = self.doc_topic + self.alpha
        s = np.sum(self.theta, axis=1)
        self.theta = self.theta / s.reshape(len(s), 1)
    
    def predict(self):
        pass
        
if __name__ == '__main__':
    dtm = np.array([[ 0,  0,  4,  0,  0,1,0],
       [ 2,  0, 0,  0,  0,0,2],
       [ 0,  1,  0,  0,  2, 2,3],
       [ 1,  0,  0,  0,  0,2,0],
        [0,3,0,4,9, 0,9], [0, 1,2,0,0,0,0]])
    lda = LDA(dtm, 3)
    lda.fit()
    print("phi is: \n", lda.phi)
    print("theta is: \n", lda.theta)
        
