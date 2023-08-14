# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 11:19:20 2023

@author: zachm
"""

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt 
#import seaborn as sns
import random
import pyfolio as pf
import empyrial
from empyrial import empyrial, Engine

# silence warnings
import warnings
warnings.filterwarnings('ignore')

'''seperate function for Calculation of the current portfolio and its allocation'''
#Note: there is no need for backtesting function
def calculate_current_portfolio_performance(current_returns, current_allocation, treturns):
    allocation_carray = np.array(list(current_allocation.values()))  # Converting allocation values to a NumPy array
    
    weighted_creturns = current_returns * allocation_carray  # Element-wise multiplication of returns and allocation array
    
    current_portfolio_returns = weighted_creturns.sum(axis=1)  # Summing the weighted returns across funds
    current_cumulative_returns = (current_portfolio_returns + 1).cumprod()  # Calculating cumulative returns
    current_annualized_returns = current_cumulative_returns.iloc[-1] ** (252 / len(current_portfolio_returns)) - 1  # Calculating annualized returns
    rf = treturns.mean()#risk-free rate
   
    '''Working but anulized return is still off'''
    current_portfolio_volatility = current_portfolio_returns.std() * np.sqrt(252)
    current_sharpe_ratio = (current_annualized_returns - rf) / current_portfolio_volatility  # Calculating Sharpe ratio
    print(current_portfolio_volatility)
    return current_portfolio_returns, current_cumulative_returns, current_annualized_returns, current_portfolio_volatility, current_sharpe_ratio

def calculate_portfolio_performance(returns, allocation, treturns, sp500returns):
    allocation_array = np.array(list(allocation.values()))  # Converting allocation values to a NumPy array
    weighted_returns = returns * allocation_array  # Element-wise multiplication of returns and allocation array
    portfolio_returns = weighted_returns.sum(axis=1)  # Summing the weighted returns across funds
    cumulative_returns = (portfolio_returns + 1).cumprod()  # Calculating cumulative returns
    sp500cumulative_returns= (sp500returns+1).cumprod()
    annualized_returns = cumulative_returns.iloc[-1] ** (252 / len(portfolio_returns)) - 1  # Calculating annualized returns
    rf = treturns.mean()#risk-free rate
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)  # Calculating annualized volatility
    sharpe_ratio = (annualized_returns-rf) / portfolio_volatility  # Calculating Sharpe ratio
    return portfolio_returns, cumulative_returns, annualized_returns, portfolio_volatility, sharpe_ratio, sp500cumulative_returns



def risk_adj_returns(returns, allocation_ranges):
    #best_allocation = None  # Variable to store the best allocation
    best_sharpe_ratio = -np.inf  # Variable to store the best Sharpe ratio
    sharpe_ratio_list = []  # Store Sharpe ratios for each allocation used for scatter plot
    for allocation in allocation_ranges:  # Looping through allocation ranges
        _, _, annualized_returns, volatility, sharpe_ratio, _ = calculate_portfolio_performance(returns, allocation, treturns, sp500returns)  # Calculating performance metrics for each allocation
        sharpe_ratio_list.append(sharpe_ratio)  # Add Sharpe ratio to the list for scatter plot
        if sharpe_ratio > best_sharpe_ratio:  # Checking if current allocation has better Sharpe ratio
            #best_allocation = allocation
            best_sharpe_ratio = sharpe_ratio  # Updating best Sharpe ratio

    # Creates scatter plot of portfolios
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(allocation_ranges)), sharpe_ratio_list, color='blue', alpha=0.5)
    plt.xlabel('Allocation Index')
    plt.ylabel('Sharpe Ratio')
    plt.title('Generated Portfolios')
    best_allocation_index = sharpe_ratio_list.index(best_sharpe_ratio)
    plt.text(best_allocation_index, best_sharpe_ratio, f'Highest sharpe Ratio: {best_sharpe_ratio:.2f}', ha='center', va='bottom',
             color='red')
    plt.show()
    return best_sharpe_ratio



def backtest_portfolio_allocations(returns, allocation_ranges):
    best_allocation = None  # Variable to store the best allocation
    
    best_annualized_returns = -np.inf  # Variable to store the best annualized returns
    annualized_returns_list = []  # Store annualized returns for each allocation used for scatter plot
    for allocation in allocation_ranges:  # Looping through allocation ranges
        _, _, annualized_returns, sharpe_ratio, _, _ = calculate_portfolio_performance(returns, allocation, treturns, sp500returns)  # Calculating annualized returns for each allocation
        annualized_returns_list.append(annualized_returns)  # Add annualized returns to the list for scatter plot
        if annualized_returns > best_annualized_returns:  # Checking if current allocation has better returns
            best_allocation = allocation
            best_annualized_returns = annualized_returns  # Updating best annualized returns
            sharpe_ratio = sharpe_ratio

     # Create scatter plot of portfolios
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(allocation_ranges)), annualized_returns_list, color='blue', alpha=0.5)
    plt.xlabel('Allocation Index')
    plt.ylabel('Annualized Returns')
    plt.title('Generated Portfolios')
    best_allocation_index = annualized_returns_list.index(best_annualized_returns)
    plt.text(best_allocation_index, best_annualized_returns, f'Best: {best_annualized_returns:.2%}', ha='center', va='bottom', color='red')
    plt.show()
    return best_allocation, best_annualized_returns, sharpe_ratio  # Returning the best allocation and its annualized returns






###########################################   CAN BE CHANGED  #############################################
# Defines the asset classes and their weight constraints
'''Note: that if amount does not add up to 1 then categorys will be filled in'''
asset_classes = {   #the weights should add up to 1
    'Equity': 0.8032,
    'Fixed Income': 0.0185,
    'Alternative': 0.1783 
}###########################################   CAN BE CHANGED  #############################################
# Defines the funds within each asset class and their allocation percentages as a dictionary
'''use allocation range to diversify EX: if you want to make sure that the combinations generated and evaulated 
have a fund with a range of 10% to 40% for the particular asset class in the porfolio. This guarentees that the fund will be somwhere between those values in the best porfolio'''
funds_allocation = {
    'Equity': {                    
        'DHLRX': (0.001, 0.8),  # Allocation range 
        'BVALX': (0.001, 0.8),   # Allocation range  
        'NMVLX': (0.001, 0.8),   # Allocation range  
        'HILIX': (0.001, 0.8),
        'VGHCX': (0.001, 0.8),
        'APDKX': (0.001, 0.8),
        'LBFIX': (0.001, 0.8),
        'JFGIX': (0.001, 0.8),
        'MGGIX': (0.001, 0.8),
        'FTHSX': (0.001, 0.8),
        'GQGIX': (0.001, 0.8),
        'APDJX': (0.001, 0.8),
        'AVUV': (0.001, 0.8),
        'VWNDX': (0.001, 0.8),
        'GOGIX': (0.001, 0.8) 
    },
    'Fixed Income': {
        'BND': (0.001, 0.8),    # Allocation range 
        'BNDX': (0.001, 0.8)     
    },
    'Alternative': {
        'BIMBX': (0.001, 0.8),  # Allocation range
        'BILPX': (0.001, 0.8),
        'ABYIX': (0.001, 0.8),
        'QLEIX': (0.001, 0.8)
    }
}

''' This is the Current Allocation. Fill with curent funds and weights.'''
#Note: if the weights dont add to 1 for both current and new then the comparison wont be accurate
current_portfolio= {
    'Equity': {
        'DHLRX': 0.0957,
        'BVALX': 0.0784,
        'NMVLX': 0.0729,
        'HILIX': 0.07,
        'VGHCX': 0.06,
        'APDKX': 0.0593,
        'LBFIX': 0.0564,
        'JFGIX': 0.0444,
        'MGGIX': 0.0444,
        'FTHSX': 0.0382,
        'GQGIX': 0.0377,
        'APDJX': 0.0284,
        'AVUV': 0.035,
        'VWNDX': 0.035,
        'GOGIX': 0.0274,
    },
    'Fixed Income': {
        'BND': 0.01,
        'BNDX': 0.0085
    },
    'Alternative': {
        'BIMBX': 0.0598,
        'BILPX': 0.0547,
        'ABYIX': 0.0459,
        'QLEIX': 0.0379
    }
}
###########################################   CAN BE CHANGED  #############################################
start= '2021-01-01'
end = '2023-05-31'
###########################################   CAN BE CHANGED  #############################################
'''Downloads t-bill data from yf 13 week. Frequency:daily'''
t_bill13w = ['%5EIRX']
tdata = yf.download(t_bill13w, start= start, end= end)
treturns = tdata['Adj Close'].pct_change().dropna()
'''Downloads S&P500 datafrom yf Frequency: daily'''
sp500 = ['%5EGSPC']
spdata = yf.download(sp500, start= start, end= end)
sp500returns = spdata['Adj Close'].pct_change().dropna()
# Downloads the historical returns data from Yahoo Finance###################################################################################
'''BackTesting and Best Allocation Use Only'''
symbols = [fund for asset_class in funds_allocation.values() for fund in asset_class.keys()]
data = yf.download(symbols, start=start, end= end)
returns = data['Adj Close'].pct_change().dropna()  # Calculating percentage changes in adjusted closing prices and removing missing values
# Converts returns to a DataFrame
returns = pd.DataFrame(returns)
# Downloads the historical returns data from Yahoo Finance####################################################################################
'''Current Allocation Use Only'''
current_weights = [fund for asset_class in current_portfolio.values() for fund in asset_class.values()]
current_symbols = [fund for asset_class in current_portfolio.values() for fund in asset_class.keys()]
current_data = yf.download(current_symbols, start= start, end = end)
current_returns = current_data['Adj Close'].pct_change().dropna()  # Calculating percentage changes in adjusted closing prices and removing missing values
current_returns = pd.DataFrame(current_returns) #converts current returns to dataframe
current_allocation = {} #an empty dictionary which will store the current allocation percentages for each fund.
for asset_class, funds in current_portfolio.items():
    for fund, allocation in funds.items():
        current_allocation[fund] = allocation #This code essentially copies the allocations from the current_portfolio dictionary to the current_allocation dictionary.





###########################################   CAN BE CHANGED  #############################################
# Define the number of possible allocations to backtest
num_funds = len(funds_allocation) #finding # of funds
num_allocations = 1000  # Number of allocation combinations
#allocation_ranges = [{} for _ in range(num_allocations)] #provides structure to store allocation data for each allocation combination during subsequent loop iterations/calculations. 
allocation_ranges = [] 
###########################################   CAN BE CHANGED  #############################################
''''Generates allocation ranges with various combinations, montecarlo simulation, 
ensures percentage does not exceed 100%, 
and stores data in allocation which passes through the calculate_portfolio_preformance function'''





for _ in range(num_allocations): #loop that iterates num_allocation times
    allocation = {} # empty dct to store fund allocations
    total_allocation = 0.0 #used to keep track of the sum of allocation percentages.
#oop that iterates over each asset_class and its corresponding weight_constraint in the asset_classes dictionary.
    for asset_class, weight_constraint in asset_classes.items():
        funds = funds_allocation[asset_class] #Retrieves the funds dictionary for the current asset_class from the funds_allocation
#iterates over each fund and its corresponding allocation_range within the funds dictionary.
        for fund, allocation_range in funds.items():
            min_allocation, max_allocation = allocation_range #extract min and max values from allocation_range
#Generates random allocation percentage (allocation_percentage) within the range of min_allocation& max_allocation using the random.uniform function.
            allocation_percentage = random.uniform(min_allocation, max_allocation)
            allocation[fund] = allocation_percentage #Assigns the allocation_percentage to the fund key in the allocation dictionary.
            total_allocation += allocation_percentage#Adds the allocation_percentage to the total_allocation
#After the inner loop completes, the allocation dictionary contains the fund allocations for the current iteration.
    # Normalize allocation percentages to ensure the total allocation is 1
    allocation = {fund: percentage / total_allocation for fund, percentage in allocation.items()}#Reassigns the normalized allocation dictionary to the same variable allocation, 
                                                                                        #overwriting the previous unnormalized allocation.
    allocation_ranges.append(allocation)#Appends the normalized allocation dictionary to the allocation_ranges list.




    
df_allocation_ranges = pd.DataFrame(allocation_ranges)
# Plotting the allocation ranges
fig, ax = plt.subplots(figsize=(10, 6))
for fund in df_allocation_ranges.columns:
    ax.hist(df_allocation_ranges[fund], bins=20, alpha=0.5, label=fund)
ax.set_xlabel('Allocation Percentage')
ax.set_ylabel('Frequency')
ax.set_title('Allocation Ranges')
ax.legend()
plt.show()


# Converts allocation_ranges to a pandas DataFrame
#allocation_df = pd.DataFrame.from_records(allocation_ranges)

# Backtests portfolio allocations
best_allocation, best_annualized_returns, sharpe_ratio = backtest_portfolio_allocations(returns, allocation_ranges)

# Backtest portfolio allocations
best_sharpe_ratio = risk_adj_returns(returns, allocation_ranges)

# Calculate portfolio performance using the best allocation
portfolio_returns, cumulative_returns, annualized_returns, portfolio_volatility, sharpe_ratio, sp500cumulative_returns = calculate_portfolio_performance(
    returns, best_allocation, treturns, sp500returns)

# # Calculate the portfolio performance for the current allocation
current_portfolio_returns, current_cumulative_returns, current_annualized_returns, current_portfolio_volatility, current_sharpe_ratio = calculate_current_portfolio_performance(
     current_returns, current_allocation, treturns)

# Backtests portfolio allocations
#best_allocation, best_annualized_returns = backtest_portfolio_allocations(returns, allocation_ranges)




print("\nCurrent Allocation:")
total_weight = 0 #stores total weight
for asset_class, funds in current_portfolio.items():
    print(asset_class)
    for fund, allocation in current_allocation.items():
        if fund in funds:
            print(f"\t{fund}: {allocation:.2%}")
            total_weight += allocation #sums up the allocation weights. useful for making sure you use the dersired percentage
print(f"\nTotal Allocation Weight: {total_weight:.2%}")
print(f"Current Annualized Return: {current_annualized_returns:.2%}")
print(f"Current Portfolio Sharpe Ratio: {current_sharpe_ratio}")
#print("Current Portfolio Returns:"); print(current_portfolio_returns)
#print("Current Cumulative Returns:"); print(current_cumulative_returns)
######################################################################################################
print('Current Portfolio Analysis')
print('Benchmark = Yellow : Strategy = Blue')
portfolio = Engine(    
         start_date= start, end_date= end, 
         portfolio= current_symbols, 
         weights = current_weights, #equal weighting is set by default
         benchmark = ['%5EGSPC'] #SPY is set by default
         
)
empyrial(portfolio)
#############################################################################################

print("\nThere were", num_allocations, "Potential Allocations Backtested")
print("\nBest Allocation:")
total_weight = 0 #stores total weight for counting purposes
for asset_class, funds in funds_allocation.items():
    print(asset_class)
    for fund, allocation in best_allocation.items():
        if fund in funds:
            print(f"\t{fund}: {allocation:.2%}")
            total_weight += allocation #sums up the allocation weights. useful for making sure you use the dersired percentage
print(f"\nTotal Allocation Weight: {total_weight:.2%}")
print(f"Best Annualized Return: {best_annualized_returns:.2%}")
print(f"Sharpe Ratio of portfolio with highest returns: {sharpe_ratio}")
print('Optimized Portfolio Analysis')
print('Benchmark = Yellow : Strategy = Blue')
#gets the weights from the best allocation
ordered_keys = symbols
w= [best_allocation[key] for key in ordered_keys]
portfolio = Engine(    
          start_date= start, end_date= end, 
          portfolio= symbols, 
          weights = w, #equal weighting is set by default
          benchmark = ['%5EGSPC'] #SPY is set by default
         
)
empyrial(portfolio)






########################################################################################################
print("\nPortfolio Returns:")
#print(portfolio_returns)
#plots out portfolio returns Note: this uses the best allocations and the related calculations 
# portfolio_returns_df = pd.DataFrame(portfolio_returns)
# portfolio_returns_df.plot(figsize=(16,10), color = 'orange', kind = 'line')
# plt.show()
portfolio_returns_df = pd.DataFrame(portfolio_returns)
current_portfolio_returns_df = pd.DataFrame(current_portfolio_returns)
plt.figure(figsize=(16, 10))
plt.plot(portfolio_returns_df.index, portfolio_returns_df, color='orange', label='Best Allocation')
plt.plot(current_portfolio_returns_df.index, current_portfolio_returns_df, color='green', label='Current Allocation')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Portfolio Returns')
plt.title('Portfolio Returns Comparison')
plt.show()

########################################################################################################
print("\nCumulative Returns:")
#print(cumulative_returns)
# plots cumulative returns ... how the value of portfolio has changed
plt.figure(figsize=(16, 10))
plt.plot(cumulative_returns.index, cumulative_returns, color='blue', label='Best Allocation')
plt.plot(current_cumulative_returns.index, current_cumulative_returns, color='red', label='Current Allocation')
plt.plot(sp500cumulative_returns.index, sp500cumulative_returns, color='black', label='S&P500')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Cumulative Returns Comparison')
plt.show()

'''Creaating comparison of allocations'''
p = current_allocation
symbols = list(p.keys())
percentage = list(p.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(symbols, percentage, color ='maroon',
        width = 0.4)
 
plt.xlabel("Funds")
plt.ylabel("percent of portfolio")
plt.title("Current Portfolio Allocation")
plt.show()

bp = best_allocation
symbols = list(bp.keys())
percentage = list(bp.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(symbols, percentage, color ='blue',
        width = 0.4)
 
plt.xlabel("Funds")
plt.ylabel("percent of portfolio")
plt.title("Best Portfolio Allocation")
plt.show()
'''Tests'''


# def main():# Here is where I called my tests
#     testcalculate_portfolio_performance()
#     # print(returns)
#     # print(allocation)




# def testcalculate_portfolio_performance():
#     print(calculate_portfolio_performance(returns, allocation))

  
# if __name__ == "__main__":
#     main()    