import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from volgpt_data import high_frequency_data
pd.set_option('display.width', 1000)  # Set pandas display width to 1000 characters

def volgpt_import(dp=8):

      # check GPU (if working on local machine)
      if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"device: {device}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
      else:
            print("CUDA is not available.")

      # for running in docker image
      # device = 'cuda' if torch.cuda.is_available() else 'cpu'

      df_data_AAPL, df_data_JPM, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr, AAPL_stats, JPM_stats = high_frequency_data(dp=8) # pass dp 
      missing_rows_AAPL = df_data_AAPL[df_data_AAPL.isnull().any(axis=1)] # Check for rows with missing values, AAPL
      missing_rows_JPM = df_data_JPM[df_data_JPM.isnull().any(axis=1)] # Check for rows with missing values, JPM

      print('df_data_AAPL.shape: ', df_data_AAPL.shape)
      print('df_data_JPM.shape: ', df_data_JPM.shape)
      
      if missing_rows_AAPL.shape[0] == 0:
            print("no missing_rows_AAPL rows with missing values")
      else:
            print('missing AAPL rows: ', missing_rows_AAPL.shape) # print number of rows with missing values
      if missing_rows_JPM.shape[0] == 0:
            print("no missing_rows_JPM rows with missing values")
      else:
            print('missing JPM rows: ', missing_rows_JPM.shape) # print number of rows with missing values


      # save df_data_AAPL and df_data_JPM as a text file with a comma delimiter
      df_data_AAPL.to_csv('df_data_AAPL.txt', sep=',', index=True)
      df_data_JPM.to_csv('df_data_JPM.txt', sep=',', index=True)

      # Check that the text file was saved correctly
      df_exported_AAPL = pd.read_csv('df_data_AAPL.txt', sep=',') # read the text file into a dataframe
      df_exported_JPM = pd.read_csv('df_data_JPM.txt', sep=',') # read the text file into a dataframe

      print('shape of df_data_AAPL: ', df_data_AAPL.shape)
      print('shape of df_exported_AAPL: ', df_exported_AAPL.shape)
      print('shape of df_data_JPM: ', df_data_JPM.shape)
      print('shape of df_exported_JPM: ', df_exported_JPM.shape)

      return df_data_AAPL, df_data_JPM, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr, AAPL_stats, JPM_stats, device


def volgpt_describe(AAPL_stats, JPM_stats, df_data_AAPL, df_data_JPM, AAPL_rr, JPM_rr, AAPL_lr, JPM_lr):

      # Descriptive statistics
      print() # Print blank line
      print("Descriptive statistics for AAPL: ", "\n"
            "Number of observations = ", AAPL_stats.nobs, "\n"
            "Minimum, Maximum = ", str(AAPL_stats.minmax), "\n"
            "Mean = %.5f" % AAPL_stats.mean, "\n"
            "Variance = %.5f" % AAPL_stats.variance, "\n"
            "Standard deviation = %.5f" % AAPL_stats.variance**0.5, "\n"
            "Skewness = %.5f" % AAPL_stats.skewness, "\n"
            "Kurtosis = %.5f" % AAPL_stats.kurtosis, "\n")

      print("Descriptive statistics for JPM: ", "\n"
            "Number of observations = ", JPM_stats.nobs, "\n"
            "Minimum, Maximum = ", str(JPM_stats.minmax), "\n"
            "Mean = %.5f" % JPM_stats.mean, "\n"
            "Variance = %.5f" % JPM_stats.variance, "\n"
            "Standard deviation = %.5f" % JPM_stats.variance**0.5, "\n"
            "Skewness = %.5f" % JPM_stats.skewness, "\n"
            "Kurtosis = %.5f" % JPM_stats.kurtosis)
      
      # Augmented Dickey-Fuller test for stationarity of log returns, NB: ADF tests take some time to run
      # print()  # Print blank line
      # print("Augmented Dickey-Fuller test for AAPL log returns:")
      # AAPL_lr_clean = AAPL_lr.dropna()  # Drop NaN values
      # result = adfuller(AAPL_lr_clean)
      # print('ADF Statistic: %f' % result[0])
      # print('p-value: %f' % result[1])
      # print('Critical Values:')
      # for key, value in result[4].items():
      #       print('\t%s: %.3f' % (key, value))

      # print()  # Print blank line
      # print("Augmented Dickey-Fuller test for JPM log returns:")
      # JPM_lr_clean = JPM_lr.dropna()  # Drop NaN values
      # result = adfuller(JPM_lr_clean)
      # print('ADF Statistic: %f' % result[0])
      # print('p-value: %f' % result[1])
      # print('Critical Values:')
      # for key, value in result[4].items():
      #       print('\t%s: %.3f' % (key, value))
      print()  # Print blank line

      sns.set_theme(style='darkgrid')  # Set Seaborn dark theme
      fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

      # Plot AAPL
      z = df_data_AAPL['WeightedMidPrice']
      z = z.replace('UNK', np.nan)  # Replace 'UNK' values with NaN
      z = z.astype(float).dropna()  # Convert the series to float type and drop NaN values
      z = z[z != 0].copy()  # Filter out zero values
      z.plot(ax=ax1, title="AAPL 1-minute weighted mid-price (source data: NYSE TAQ)", xlabel="Observations", ylabel="AAPL 1-min weighted mid-price", color='darkslategrey', linewidth=1.0, alpha=0.75)

      # Plot JPM
      z1 = df_data_JPM['WeightedMidPrice']
      z1 = z1.replace('UNK', np.nan)  # Replace 'UNK' values with NaN
      z1 = z1.astype(float).dropna()  # Convert the series to float type and drop NaN values
      z1 = z1[z1 != 0].copy()  # Filter out zero values
      z1.plot(ax=ax2, title="JPM 1-minute weighted mid-price (source data: NYSE TAQ)", xlabel="Observations", ylabel="JPM 1-min weighted mid-price", color='saddlebrown', linewidth=1.0, alpha=0.75)

      # Plot AAPL log returns
      z = AAPL_lr
      z = z.replace('UNK', np.nan)  # Replace 'UNK' values with NaN
      z = z.astype(float).dropna()  # Convert the series to float type and drop NaN values
      z = z[z != 0].copy()  # Filter out zero values
      z.plot(ax=ax3, title="AAPL log returns (source data: NYSE TAQ)", xlabel="Observations", ylabel="AAPL log returns", color='darkslategrey', linewidth=0.1, alpha=0.75)

      # Plot JPM log returns
      z1 = JPM_lr
      z1 = z1.replace('UNK', np.nan)  # Replace 'UNK' values with NaN
      z1 = z1.astype(float).dropna()  # Convert the series to float type and drop NaN values
      z1 = z1[z1 != 0].copy()  # Filter out zero values
      z1.plot(ax=ax4, title="JPM log returns (source data: NYSE TAQ)", xlabel="Observations", ylabel="JPM log returns", color='saddlebrown', linewidth=0.1, alpha=0.75)

      plt.tight_layout()
      plt.show()
