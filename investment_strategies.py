# -*- coding: utf-8 -*-
"""
Created on Mon May  2 23:41:18 2022

@author: o2svt
"""

import pandas as pd

tesla_stocks = pd.read_csv('price.csv')

#create percentage change increments (-5% to 5%)
percentage = [-5,-4,-3,-2,-1, -.00001, .00001, 1, 2, 3, 4, 5]

#blank dataframe
returns = pd.DataFrame()

#looping 
for x in range(len(percentage)):
    i = 0
    number = 0
    if percentage[x] < 0:
            
        for y in range(1,len(tesla_stocks)):
                  
           if (tesla_stocks['current'].iloc[y] - tesla_stocks['current'].iloc[y-1])/tesla_stocks['current'].iloc[y] < (percentage[x] * .01):
               i += (1000/tesla_stocks['current'].iloc[y] * tesla_stocks['actual'].iloc[y] - 1000)
               number += 1    
    if percentage[x] > 0:
            
        for y in range(1,len(tesla_stocks)):
                  
           if (tesla_stocks['current'].iloc[y] - tesla_stocks['current'].iloc[y-1])/tesla_stocks['current'].iloc[y] > (percentage[x] * .01):
               i += (1000/tesla_stocks['current'].iloc[y] * tesla_stocks['actual'].iloc[y] - 1000)
               number += 1
  
    income = pd.Series([percentage[x],i, number, number * 1000, i/(number * 1000)])
    returns = returns.append(income, ignore_index = True)
        

returns.columns = ['Percentage','Return','Number of Purchases', 'Amount Invested', 'ROI']

print(returns)

p = 0 #positive
n = 0 #negative
for y in range(1,len(tesla_stocks)):
           
           if (tesla_stocks['current'].iloc[y] - tesla_stocks['current'].iloc[y-1]) > 0:
               p += (1000/tesla_stocks['current'].iloc[y] * tesla_stocks['actual'].iloc[y] - 1000)
        
           if (tesla_stocks['current'].iloc[y] - tesla_stocks['current'].iloc[y-1]) < 0:
               n += (1000/tesla_stocks['current'].iloc[y] * tesla_stocks['actual'].iloc[y] - 1000)
               
print('\npurchasing after every increase', p)
print('purchasing after every decrease',n)               
        
money = 0
for y in range(len(tesla_stocks)):
    money += (1000/tesla_stocks['current'].iloc[y] * tesla_stocks['actual'].iloc[y] - 1000)
    
print('\ninvesting $1,000 everyday',money)
print('amount invested',len(tesla_stocks) *1000)
print('roi', money/(len(tesla_stocks) *1000))


