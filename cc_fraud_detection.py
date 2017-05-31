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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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
    t_format = "%Y-%m-%d %H:%M:%S"
    return time.mktime(time.strptime(str(signup_time), t_format))


def convert_to_wd(signup_time):
    t_format = "%Y-%m-%d %H:%M:%S"
    wd_format = "%A"
    return time.strftime(wd_format, time.strptime(str(signup_time), t_format))


def convert_to_wy(signup_time):
    t_format = "%Y-%m-%d %H:%M:%S"
    wy_format = "%U"
    return time.strftime(wy_format, time.strptime(str(signup_time), t_format))


# Compare y_true to y_pred and return the accuracy
def accuracy_score(y_true, y_pred):
    correct = 0
    print(len(y_true))
    for i in range(len(y_true)):
        diff = y_true[i] - y_pred[i]
        if diff == np.zeros(np.shape(diff)):
            correct += 1
    return correct / len(y_true)


# Main body
def main():
    # read input data
    is_data, data = read_user_first_transaction_data(fraud_data_file_name)
    if is_data is False:
        sys.exit(0)

    is_data, data_with_country = add_country_to_data(data, ipaddress_to_country_file_name)
    if is_data is False:
        sys.exit(0)

    print('Preparing features')
    # verify time difference between signup and purchase
    signup_time = data_with_country['signup_time']
    purchase_time = data_with_country['purchase_time']
    # add to data
    signup_time_second = np.asarray(map(convert_to_timestamp, signup_time))
    purchase_time_second = np.asarray(map(convert_to_timestamp, purchase_time))
    sp_difference_time_second = purchase_time_second - signup_time_second

    data_with_country = append_fields(data_with_country, 'sp_difference', sp_difference_time_second)

    # date of week and week of year of both signup and purchase time
    signup_time_wd = np.asarray(map(convert_to_wd, signup_time))
    purchase_time_wd = np.asarray(map(convert_to_wd, purchase_time))

    signup_time_wy = np.asarray(map(convert_to_wy, signup_time))
    purchase_time_wy = np.asarray(map(convert_to_wy, purchase_time))

    data_with_country = append_fields(data_with_country, 'signup_time_wd', signup_time_wd)
    data_with_country = append_fields(data_with_country, 'purchase_time_wd', purchase_time_wd)
    data_with_country = append_fields(data_with_country, 'signup_time_wy', signup_time_wy)
    data_with_country = append_fields(data_with_country, 'purchase_time_wy', purchase_time_wy)

    # count device_id and ip_address used by different users
    u, indices, counts = np.unique(data_with_country['device_id'], return_inverse=True, return_counts=True)
    device_id_count = counts[indices]
    u, indices, counts = np.unique(data_with_country['ip_address'], return_inverse=True, return_counts=True)
    ip_address_count = counts[indices]

    data_with_country = append_fields(data_with_country, 'device_id_count', device_id_count)
    data_with_country = append_fields(data_with_country, 'ip_address_count', ip_address_count)

    # prepare data for training and testing
    data_random_forest = data_with_country[['purchase_value', 'device_id', 'source', 'browser', 'sex', 'age', 'country', 'sp_difference', 'signup_time_wd', \
        'purchase_time_wd', 'signup_time_wy', 'purchase_time_wy', 'device_id_count', 'ip_address_count']]
    class_random_forest = data_with_country['class']

    # encode labels with values to train random forests
    no_features = 14
    data_random_forest_final = np.zeros((len(class_random_forest), no_features))

    data_random_forest_final[:, 0] = np.array(data_random_forest['purchase_value'])

    le = preprocessing.LabelEncoder()

    le.fit(data_random_forest['device_id'])
    data_random_forest_final[:, 1] = np.array(le.transform(data_random_forest['device_id']))

    le.fit(data_random_forest['source'])
    data_random_forest_final[:, 2] = np.array(le.transform(data_random_forest['source']))

    le.fit(data_random_forest['browser'])
    data_random_forest_final[:, 3] = np.array(le.transform(data_random_forest['browser']))

    le.fit(data_random_forest['sex'])
    data_random_forest_final[:, 4] = np.array(le.transform(data_random_forest['sex']))

    le.fit(data_random_forest['age'])
    data_random_forest_final[:, 5] = np.array(le.transform(data_random_forest['age']))
    #data_random_forest_final[:, 5] = np.array(data_random_forest['age'])

    le.fit(data_random_forest['country'])
    data_random_forest_final[:, 6] = np.array(le.transform(data_random_forest['country']))

    le.fit(data_random_forest['sp_difference'])
    data_random_forest_final[:, 7] = np.array(le.transform(data_random_forest['sp_difference']))

    le.fit(data_random_forest['signup_time_wd'])
    data_random_forest_final[:, 8] = np.array(le.transform(data_random_forest['signup_time_wd']))

    le.fit(data_random_forest['purchase_time_wd'])
    data_random_forest_final[:, 9] = np.array(le.transform(data_random_forest['purchase_time_wd']))

    le.fit(data_random_forest['signup_time_wy'])
    data_random_forest_final[:, 10] = np.array(le.transform(data_random_forest['signup_time_wy']))

    le.fit(data_random_forest['purchase_time_wy'])
    data_random_forest_final[:, 11] = np.array(le.transform(data_random_forest['purchase_time_wy']))

    data_random_forest_final[:, 12] = np.array(data_random_forest['device_id_count'])
    data_random_forest_final[:, 13] = np.array(data_random_forest['ip_address_count'])

    # seperate training and testing data
    print('Training random forests')
    X_train, X_test, y_train, y_test = train_test_split(data_random_forest_final, class_random_forest,
                                                       train_size=0.75,
                                                       random_state=40)

    clf = RandomForestClassifier(n_estimators=100, max_features=None, n_jobs=4, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    print('Confusion Matrix')
    cnf_matrix = confusion_matrix(y_pred, y_test)

    return 0

if __name__ == "__main__":
    sys.exit(main())
