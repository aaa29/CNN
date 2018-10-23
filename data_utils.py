#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:21:05 2018

@author: amine
"""

import numpy as np


def get_sequence(n, sequence_lenght =30 , n_features = 10):
    
    even_choices= list(range(n_features))*2
    
    odd_choices = list(range(1,n_features*2+1, 2))
    
    #higher will be used to indecate how much the even numbers are more then the odds
    higher = np.random.randint(0, n_features, size=[1])[0]
    
    # generating size/3 sub set with more even numbers than odd numbers

    
    #X1 is the sequence of even numbers
    X1 = np.random.choice(even_choices, size = [int(n/3), sequence_lenght-higher])
    #X2 is the sequence of odd numbers
    X2 = np.random.choice(odd_choices, size = [int(n/3), higher])
    #X represents the complete sequence
    X_evens = np.concatenate((X1,X2), axis=1)
    Y_evens = [0 for _ in range(int(n/3))]
    
        
    #generating size/3 subset with more odds than evens
    #X1 is the sequence of even numbers
    X1 = np.random.choice(even_choices, size = [int(n/3), higher]) 
    #X2 is the sequence of odd numbers
    X2 = np.random.choice(odd_choices, size = [int(n/3),sequence_lenght-higher])  
    #X represents the complete sequence
    X_odds = np.concatenate((X1,X2), axis=1)
    Y_odds = [1 for _ in range(int(n/3))]
    
    
    #generating size/3 subset with equal number of evens and odds
    #X1 is the sequence of even numbers
    X1 = np.random.choice(even_choices, size = [int(n/3), int(sequence_lenght/2)]) 
    #X2 is the sequence of odd numbers
    X2 = np.random.choice(odd_choices, size = [int(n/3), int(sequence_lenght/2)])  
    #X represents the complete sequence
    X_equals = np.concatenate((X1,X2), axis=1)
    Y_equals = [2 for _ in range(int(n/3))]
    
    X = np.concatenate([X_evens, X_odds, X_equals], axis=0)
    Y = Y_evens + Y_odds + Y_equals
    
    
    return X, Y