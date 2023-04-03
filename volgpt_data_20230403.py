import os
import glob
import pandas as pd
import numpy as np
from scipy import stats

def high_frequency_data(dp=2):
    path = 'allfiles' # import HF data (AAPL, JPM), being 563 daily files
    all_files = glob.glob(os.path.join(path,"**/*.csv.gz"))
    np_array_list = []
    for file_ in all_files:
        df = pd.read_csv(file_,index_col=None, header=0) # read the csv's
        np_array_list.append(df.values) # convert to numpy array

    comb_np_array = np.vstack(np_array_list) # Create bit array
    df_data = pd.DataFrame(comb_np_array) # Create dataframe

    # Set column headings
    df_data.columns = ["Date","Ticker","TimeBarStart","OpenBarTime","OpenBidPrice","OpenBidSize", "OpenAskPrice","OpenAskSize","FirstTradeTime","FirstTradePrice","FirstTradeSize","HighBidTime","HighBidPrice","HighBidSize","HighAskTime","HighAskPrice","HighAskSize","HighTradeTime","HighTradePrice","HighTradeSize","LowBidTime","LowBidPrice","LowBidSize","LowAskTime","LowAskPrice","LowAskSize","LowTradeTime","LowTradePrice","LowTradeSize","CloseBarTime","CloseBidPrice","CloseBidSize","CloseAskPrice","CloseAskSize","LastTradeTime","LastTradePrice","LastTradeSize","MinSpread","MaxSpread","CancelSize","VolumeWeightPrice","NBBOQuoteCount","TradeAtBid","TradeAtBidMid","TradeAtMid","TradeAtMidAsk","TradeAtAsk","TradeAtCrossOrLocked","Volume","TotalTrades","FinraVolume","FinraVolumeWeightPrice","UptickVolume","DowntickVolume","RepeatUptickVolume","RepeatDowntickVolume","UnknownTickVolume","TradeToMidVolWeight","TradeToMidVolWeightRelative","TimeWeightBid","TimeWeightAsk"]

    # Set datatypes for columns used to compute weighted mid-price
    df_data['CloseBidSize'] = df_data['CloseBidSize'].astype(float)
    df_data['CloseAskSize'] = df_data['CloseAskSize'].astype(float)
    df_data['CloseBidPrice'] = df_data['CloseBidPrice'].astype(float)
    df_data['CloseAskPrice'] = df_data['CloseAskPrice'].astype(float)

    # Create DateTimeIndex column, which is a combination of the Date and OpenBarTime columns, and set as index
    df_data['DateTimeIndex'] = pd.to_datetime(df_data['Date'].astype(str)) + pd.to_timedelta(df_data['OpenBarTime'].astype(str))

    # Reduce number of columns for tractability 
    # NB: it may be helpful to use the other columns in the future to explore whether more data can be used to improve the model
    df_data_full = df_data.copy()  
    df_data = df_data_full[['DateTimeIndex','Ticker','CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice']].copy()

    # Set DateTimeIndex as index
    df_data = df_data.set_index('DateTimeIndex')
        
    # Compute WeightedMidPrice using the closing prices per analysis in my high-frequency-data post
    df_data['WeightedMidPrice'] = ((df_data['CloseBidSize']*df_data['CloseAskPrice']) + (df_data['CloseAskSize']*df_data['CloseBidPrice'])) / (df_data['CloseBidSize'] + df_data['CloseAskSize'])

    # Split df_data into AAPL and JPM subsets, to avoid UNK tokens due to lack of AAPL/JPM rr/lr data in single combined dataframe
    df_data_AAPL = df_data.loc[df_data['Ticker'] == 'AAPL']
    df_data_JPM = df_data.loc[df_data['Ticker'] == 'JPM']

    # Compute raw returns
    AAPL_rr = df_data_AAPL['WeightedMidPrice'] - df_data_AAPL['WeightedMidPrice'].shift(1)
    JPM_rr = df_data_JPM['WeightedMidPrice'] - df_data_JPM['WeightedMidPrice'].shift(1)

    # Compute log returns
    AAPL_lr = np.log(df_data_AAPL['WeightedMidPrice'].astype(float))
    AAPL_lr = AAPL_lr - AAPL_lr.shift(1)
    JPM_lr = np.log(df_data_JPM['WeightedMidPrice'].astype(float))
    JPM_lr = JPM_lr - JPM_lr.shift(1)

    # set parameters for outlier removal
    Q1l = AAPL_lr.quantile(0.001)   # Start with these quantile values, experiment with lower values
    Q3l = AAPL_lr.quantile(0.999)   # Start with these quantile values, experiment with lower values if necessary
    IQl = Q3l - Q1l
    Q1r = AAPL_rr.quantile(0.001)   # Start with these quantile values, experiment with lower values
    Q3r = AAPL_rr.quantile(0.999)   # Start with these quantile values, experiment with lower values if necessary
    IQr = Q3r - Q1r
    
    # Remove outliers
    AAPL_lr = AAPL_lr[~((AAPL_lr < (Q1l - 1.5 * IQl)) | (AAPL_lr > (Q3l + 1.5 * IQl)))]
    AAPL_rr = AAPL_rr[~((AAPL_rr < (Q1r - 1.5 * IQr)) | (AAPL_rr > (Q3r + 1.5 * IQr)))]
    JPM_lr = JPM_lr[~((JPM_lr < (Q1l - 1.5 * IQl)) | (JPM_lr > (Q3l + 1.5 * IQl)))]
    JPM_rr = JPM_rr[~((JPM_rr < (Q1r - 1.5 * IQr)) | (JPM_rr > (Q3r + 1.5 * IQr)))]

    # merge
    df_data_AAPL = df_data_AAPL.merge(AAPL_rr.to_frame(name='AAPL_rr'), left_index=True, right_index=True, how='left') # merge
    df_data_JPM = df_data_JPM.merge(JPM_rr.to_frame(name='JPM_rr'), left_index=True, right_index=True, how='left') # merge
    df_data_AAPL = pd.concat([df_data_AAPL, AAPL_lr.rename('AAPL_lr')], axis=1) # merge using different method as AAPL_lr is a series
    df_data_JPM = pd.concat([df_data_JPM, JPM_lr.rename('JPM_lr')], axis=1) # merge using different method as JPM_lr is a series

    # format numbers with exactly dp decimal places, to impose structure upon data, which helps the model quite a bit
    def format_number(x):
        return format(x, f".{dp}f") if isinstance(x, (int, float)) else x

    # Apply the formatting to the specified columns in df_data_AAPL and df_data_JPM
    cols_to_format_AAPL = ['CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'AAPL_rr', 'AAPL_lr']
    df_data_AAPL[cols_to_format_AAPL] = df_data_AAPL[cols_to_format_AAPL].round(dp).applymap(format_number)
    cols_to_format_JPM = ['CloseBidSize', 'CloseAskSize', 'CloseBidPrice', 'CloseAskPrice', 'WeightedMidPrice', 'JPM_rr', 'JPM_lr']
    df_data_JPM[cols_to_format_JPM] = df_data_JPM[cols_to_format_JPM].round(dp).applymap(format_number)

    df_data_AAPL.fillna("UNK", inplace=True) # Replace missing values with "UNK"
    df_data_JPM.fillna("UNK", inplace=True) # Replace missing values with "UNK"

    df_data_AAPL = df_data_AAPL.astype(str) # Convert all columns to string
    df_data_JPM = df_data_JPM.astype(str) # Convert all columns to string

    # Descriptive stats based upon raw returns
    AAPL_rr_stat = AAPL_rr[AAPL_rr.notna()].copy() # raw returns for AAPL using df_data
    AAPL_rr_stat = AAPL_rr_stat[AAPL_rr_stat != 0].copy()
    JPM_rr_stat = JPM_rr[JPM_rr.notna()].copy() # raw returns for JPM using df_data
    JPM_rr_stat = JPM_rr_stat[JPM_rr_stat != 0].copy()
    AAPL_stats = stats.describe(AAPL_rr_stat) # Descriptive statistics for AAPL
    JPM_stats = stats.describe(JPM_rr_stat) # Descriptive statistics for JPM

    # save df_data_AAPL and df_data_JPM as a text file with a comma delimiter
    df_data_AAPL.to_csv('df_data_AAPL.txt', sep=',', index=True)
    df_data_JPM.to_csv('df_data_JPM.txt', sep=',', index=True)

    return (df_data_AAPL, df_data_JPM, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr, AAPL_stats, JPM_stats)



