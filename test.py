# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:33:12 2024

@author: micha
"""

import pandas as pd 
import statsmodels.api as sm
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

#load quarterly data
pqd = pd.read_csv('biannual.csv', index_col = False)  # merge back to this data set 
pqd['dtoa'] = pqd['ltq'] /pqd['atq']
pqd = pqd[pqd['conm'] != 'SOUTHEAST ASIA PPTYS & FIN']
companies = pqd['conm'].unique()


prices = pd.read_csv('price.csv')

# load price and get returns 
prices = prices.rename(columns = {'cshoc': 'Shares Outstanding', 'prccd' : 'Price', 'cshtrd':'Trading Volume'})
prices = prices.loc[prices['conm'].isin(companies)]
prices['return'] = np.log(prices.groupby('conm')['Price'].pct_change() + 1)
returnsdf = prices[['datadate', 'conm', 'return']]
returnsdf = returnsdf.dropna()
returnsdf['datadate'] = pd.to_datetime(returnsdf['datadate'], format='%d/%m/%Y')



pivot_df = returnsdf.pivot_table(index='datadate', columns='conm', values='return', aggfunc='sum').reset_index()
pivot_df = pivot_df.iloc[1:]
pivot_df = pivot_df.rename(columns = {'datadate' : 'Date'})


#quarterl announcement  date 

announcement = pqd[['fdateq', 'conm', 'dtoa']]
announcement['fdateq'] = pd.to_datetime(announcement['fdateq'], format='%d/%m/%Y')



#Alpha Calculation ###########################################################

# load ff3 data

ff3 = pd.read_csv('Asia_Pacific_ex_Japan_3_Factors_Daily.csv', index_col = False)
ff3['Date'] = pd.to_datetime(ff3['Date'], format='%Y%m%d')

# merge daily price data with factors 
merged = pd.merge(pivot_df, ff3, on = 'Date')


finaldf = pqd[['fdateq', 'conm', 'dtoa']]
finaldf['prealpha'] = 0
finaldf['postalpha'] = 0
finaldf['announcement'] = finaldf['fdateq']
finaldf = finaldf.head(0)

prices['Market Cap'] = prices['Shares Outstanding'] * prices['Price']

controlvariables = prices[['datadate', 'conm', 'Trading Volume', 'Market Cap']]    
controlvariables['datadate'] = pd.to_datetime(controlvariables['datadate'], format='%d/%m/%Y')
                                              
           
rsquared = {}                                         
           
for company in companies:
    

    returndf = merged[['Date', company, 'Mkt-RF', 'SMB', 'HML', 'RF']]
    returndf['Excess_Return'] = returndf[company] - returndf['RF']
    
    
    
    
    # merge with quartely data to do the regression then get earnings date 
    
    companyannouncement = announcement.loc[announcement['conm'] == company]
    companyannouncement['announcement'] = companyannouncement['fdateq']
    
    
    returndf2 = pd.merge(returndf, companyannouncement, left_on= 'Date', right_on='fdateq' , how= 'left' )
    returndf2 = returndf2.drop(columns = ['fdateq', 'conm'] )
    
    companyannouncement['prealpha'] = 0
    companyannouncement['postalpha'] = 0
    
    for announcement_date in companyannouncement['announcement']:                                    
    
        try:
            announcement_index = returndf2[returndf2['announcement'] == announcement_date].index[0]
        except: 
            continue
        
       
        preevent = returndf2.iloc[announcement_index - 5: announcement_index]
        
        postevent = returndf2.iloc[announcement_index : announcement_index +3]
        
        
        # Pre event window alpha 
        try:
            preX = preevent[['Mkt-RF', 'SMB', 'HML']]
            preX = sm.add_constant(preX)  
        
           
            prey = preevent['Excess_Return']
            premodel = sm.OLS(prey, preX).fit()
            companyannouncement.loc[companyannouncement['announcement'] == announcement_date, 'prealpha'] = premodel.params['const']
        
            # post event window alpha 
            postX = postevent[['Mkt-RF', 'SMB', 'HML']]
            postX = sm.add_constant(postX)  
            posty = postevent['Excess_Return']
            postmodel = sm.OLS(posty, postX).fit()
            companyannouncement.loc[companyannouncement['announcement'] == announcement_date, 'postalpha'] = postmodel.params['const']
        except Exception as e:
            print(f"Error for announcement date {announcement_date}: {e} ")
    
    
    finaldf = companyannouncement


    
    final_merged_df = pd.merge(controlvariables, announcement, left_on=['datadate', 'conm'], right_on=['fdateq', 'conm'])
    final_merged_df = final_merged_df.drop_duplicates(subset=['datadate', 'conm']) 
    final_merged_df = final_merged_df[['datadate', 'conm', 'Trading Volume', 'Market Cap']]
    
    
    Completedata = pd.merge(finaldf, final_merged_df, left_on = ['fdateq', 'conm'], right_on = ['datadate', 'conm'], how = 'left' ).drop(columns = ['announcement', 'datadate']).dropna()
    

# Multivariate regression 

    X =  Completedata[['dtoa', 'prealpha', 'Trading Volume', 'Market Cap' ]]
    X = sm.add_constant(X)  
    
    y = Completedata['postalpha']
    model = sm.OLS(y, X).fit()
    value = model.rsquared
    rsquared[company] = value
    