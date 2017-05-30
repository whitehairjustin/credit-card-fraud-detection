# Credit Card Fraud Detection
# Ha Nguyen

import sys
import math
import csv
import numpy as np
import os.path
from numpy.lib.recfunctions import append_fields
import bisect
import time


fraud_data_file_name = './Fraud/Fraud_Data.csv'
ipaddress_to_country_file_name = './Fraud/IpAddress_to_Country.csv'


# read information about each user first transaction
def read_user_first_transaction_data(file_name):
    print('Reading user data has begun')
    # read data from file
    if not os.path.isfile(file_name):
        print('User data file does not exist. Please double check!')
        return False, 0
    data = np.genfromtxt(file_name, dtype=None, delimiter=',', names=True)
    # verify the uniqueness of user_id
    no_rows = data.shape
    no_user_id = len(np.unique(data['user_id']))

    if no_user_id != no_rows[0]:
        print('User data provided is not correct. Please double check!')
        return False, 0

    print('Reading user data has finished')
    return True, data


# add country name to data
def add_country_to_data(data, ip_to_country_file):
    print('Adding country to data has begun')
    # read ip_to_country file
    if not os.path.isfile(ip_to_country_file):
        print('IPAddress_to_Country file does not exist. Please double check!')
        return False, 0
    ip_to_country_data = np.genfromtxt(ip_to_country_file, dtype=None, delimiter=',', names=True)

    no_rows = data.shape
    data_country = np.empty(no_rows[0], dtype='S25')
    data_country.fill('NA')
    # check country data
    lower_bound = ip_to_country_data['lower_bound_ip_address']
    upper_bound = ip_to_country_data['upper_bound_ip_address']
    country = ip_to_country_data['country']
    ip = data['ip_address']
    for i in range(len(ip)):
        index = bisect.bisect(lower_bound, ip[i])
        if index > 0:
            if ip[i] <= upper_bound[index - 1]:
                data_country[i] = country[index - 1]

    # add country data
    data = append_fields(data, 'country', data_country)

    print('Adding country to data has finished')
    return True, data

def convert_to_timestamp(signup_time):
    format = "%Y-%m-%d %H:%M:%S"
    return time.mktime(time.strptime(str(signup_time), format))


# Main body
def main():
    # read input data
    is_data, data = read_user_first_transaction_data(fraud_data_file_name)
    if is_data is False:
        sys.exit(0)

    is_data, data_with_country = add_country_to_data(data, ipaddress_to_country_file_name)
    if is_data is False:
        sys.exit(0)

    # verify time difference between signup and purchase
    signup_time = data_with_country['signup_time']
    purchase_time = data_with_country['purchase_time']
    # add to data
    signup_time_second = np.asarray(map(convert_to_timestamp, signup_time))
    purchase_time_second = np.asarray(map(convert_to_timestamp, purchase_time))
    sp_difference_time_second = purchase_time_second - signup_time_second

    # add to data
    data_with_country = append_fields(data_with_country, 'sp_difference', sp_difference_time_second)

    # split data to training and testing data

    #print(data)
    print(data_with_country)
    #print(data_with_country['country'][1])

    # data for random forest training
    data_random_forest = []
    data_random_forest = data_with_country[:,['devide_id', 'source', 'browser', 'sex', 'age', 'country', 'sp_difference', 'class']]
    print(data_random_forest)

    # seperate training and testing data



    # print out the result
    print("Here is the output")

    return 0


if __name__ == "__main__":
    sys.exit(main())
