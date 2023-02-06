import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

def compute_portvals(  		  	   		  	  		  		  		    	 		 		   		 		  
    data,
    price,
    start_val=100000,  		  	   		  	  		  		  		    	 		 		   		 		  
    commission=0.0,  		  	   		  	  		  		  		    	 		 		   		 		  
    impact=0.0
):

    #compute start and end dates
    start_date=data.index[0]
    end_date=data.index[-1]

    #get market price
    dates = pd.date_range(start_date, end_date)
    syms=data.columns[0]

    stocks=dict.fromkeys([syms],0)
    stocks['Cash']=start_val
    
    def manage_stock(symbol,amount,date):
        stocks[symbol]+=amount
        stocks['Cash']+=(-1)*amount*price.loc[date][symbol]-commission-impact*amount*price.loc[date][symbol]
    
    temp=list([syms])
    temp.append('Cash')
    port_val = pd.DataFrame(index=data.index,columns=temp)
    
    for index,order in data.iterrows():
        date=str(index)
        # if pd.isna(data.loc[date][0]):
        #     continue
        
        manage_stock(syms,order[syms],date)
        
        for sym in stocks:
            port_val.loc[date][sym]=stocks[sym]
    
    port_val.fillna(method='ffill',inplace=True)

    port_val['Stock value']=port_val.drop(['Cash'],axis=1).mul(price).sum(axis=1)
    port_val['Total value']=port_val['Cash']+port_val['Stock value']
    return port_val['Total value']