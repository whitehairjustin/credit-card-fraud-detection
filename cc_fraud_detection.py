# Credit Card Fraud Detection
# Ha Nguyen

import sys
import math
import csv
import numpy as np

fraud_data_file_name = './Fraud/Fraud_Data.csv'
ipaddress_to_country_file_name = './Fraud/IpAddress_to_Country.csv'


# read information about each user first transaction
def read_user_first_transaction_data(file_name):
    data = np.genfromtxt(file_name, dtype=None, delimiter=',', names=True)

    #return data
    return data


# add country name to data
def add_country_to_data(data, ip_to_country_file):
    ip_to_country_data = np.genfromtxt(ip_to_country_file, dtype=None, delimiter=',', names=True)

    return ip_to_country_data
    #return data


# Main body
def main():
    # read input data
    data = read_user_first_transaction_data(ipaddress_to_country_file_name)
    data_with_country = add_country_to_data(data, ipaddress_to_country_file_name)

    # split data to training and testing data

    print(data)
    print(data_with_country)
    print(data_with_country['country'][1])


    # print out the result
    print("Here is the output")

    return 0


if __name__ == "__main__":
    sys.exit(main())
