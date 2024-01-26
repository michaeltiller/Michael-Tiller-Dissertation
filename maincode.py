# -*- coding: uf-8 -*-
"""
Created on Wed Jan 24 12:31:32 2024

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

#excluded = ['SOUTHEAST ASIA PPTYS & FIN', 'CHU KONG SHIPPING DEV CO LTD', 'FOSUN INTERNATIONAL LTD', 'POWER ASSETS HOLDINGS LTD', 'SINOTRUK (HONG KONG) LTD' ]
#pqd = pqd[~pqd['conm'].isin(excluded)]
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


#quarterly announcement  date 

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
    
    
    finaldf = pd.concat([finaldf, companyannouncement])



# Get control variables from price data 
prices['Market Cap'] = prices['Shares Outstanding'] * prices['Price']

controlvariables = prices[['datadate', 'conm', 'Trading Volume', 'Market Cap']]    
controlvariables['datadate'] = pd.to_datetime(controlvariables['datadate'], format='%d/%m/%Y')

final_merged_df = pd.merge(controlvariables, announcement, left_on=['datadate', 'conm'], right_on=['fdateq', 'conm'])
final_merged_df = final_merged_df.drop_duplicates(subset=['datadate', 'conm']) 
final_merged_df = final_merged_df[['datadate', 'conm', 'Trading Volume', 'Market Cap']]


Completedata = pd.merge(finaldf, final_merged_df, left_on = ['fdateq', 'conm'], right_on = ['datadate', 'conm'], how = 'left' ).drop(columns = ['announcement', 'datadate']).dropna()

#Completedata.to_csv('C:/Users/micha/Desktop/Dissertation/finaldata.csv', index = False)

#Completedata = pd.read_csv('C:/Users/micha/Desktop/Dissertation/finaldata.csv')


## Data exploration ################################


plt.hist(Completedata['dtoa'], color = 'skyblue', edgecolor = 'black', bins = 15)
plt.xlabel('Debt to Assets')
plt.ylabel('Frequency')
plt.title('Dsitribution of Debt to Assets')


plt.hist(Completedata['prealpha'], color = 'skyblue', edgecolor = 'black', bins = 30)
plt.xlabel('Alpha prior to earnings announcement')
plt.ylabel('Frequency')
plt.title('Dsitribution of Alpha prior to earnings announcement')


plt.hist(Completedata['postalpha'], color = 'skyblue', edgecolor = 'black', bins = 30 )
plt.xlabel('Alpha post earnings announcement')
plt.ylabel('Frequency')
plt.title('Dsitribution of Alpha post earnings announcement')

plt.hist(Completedata['Trading Volume'], color = 'skyblue', edgecolor = 'black', bins = 20)
plt.xlabel('Trading Volume')
plt.ylabel('Frequency')
plt.title('Dsitribution of Trading Volume')


plt.hist(Completedata['Market Cap'], color = 'skyblue', edgecolor = 'black')
plt.xlabel('Market Cap')
plt.ylabel('Frequency')
plt.title('Dsitribution of Market Cap')


plt.scatter(Completedata['prealpha'], Completedata['postalpha'], color='blue', label='Data Points')

# Add labels and title
plt.xlabel('Pre earnigns Alpha')
plt.ylabel('Post earnings Alpha')
plt.title('Alpha scatter plot')

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()


# Summary datistics 

summarystats = Completedata[['dtoa', 'prealpha', 'postalpha', 'Trading Volume', 'Market Cap' ]].describe()
summarystats = summarystats.rename(columns = {'dtoa':'Debt to Assets', 'prealpha' : 'Pre Announcement Alpha' , 'postalpha' : 'Post Announcement Alpha'})
summarystats.to_csv('C:/Users/micha/Desktop/Dissertation/summarystats.csv')




# Multivariate regression #########################

X =  Completedata[['dtoa', 'prealpha', 'Trading Volume', 'Market Cap' ]]
X = sm.add_constant(X)  

y = Completedata['postalpha']
model = sm.OLS(y, X).fit()
model.summary()
