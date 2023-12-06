# imports
import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
import os
import warnings
warnings.filterwarnings('ignore')
import json
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from numpy import percentile
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

from sklearn.ensemble import IsolationForest
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

from sklearn.cluster import DBSCAN

 # Change the path
path = 'C: \\'
# insert the date, the file should be saved same as the date
date = 'J'
df = pd.read_csv(path + f'{date}.csv').set_index('Sample ID').dropna(how='all')
path_i = 'C:\\U'
df_batch_abs_master = pd.read_csv(path_i + 'T.csv', index_col = 0)
loaded_model = pickle.load(open(path_i + 'V', 'rb'))
ref_msc = pickle.load(open(path_i +'V0', 'rb'))
df_master = pickle.load(open(path_i  + 'D', 'rb'))

 def get_old_samples_list(df):
    ddf = df[['Record ID','Device ID', 'Sample Date']]
    ddf = ddf[~ddf.index.str.contains(r'ref')]
    ddf['Sample'] = ddf.index
    ddf['Hour'] = pd.to_datetime(ddf['Sample Date']).dt.strftime('%H%M').astype(int)
    ddf['pos_neg'] = ddf['Sample'].str.extract(r'(^[A-Za-z])')
    ddf['pos_neg'] = ddf['pos_neg'].replace({'n':'N', 'p':'P'})
    ddf['num'] = ddf['Sample'].str.extract(r'[A-Za-z]_(\d{1,3})')
    ddf = ddf.fillna(0)
    ddf['num'] = ddf['num'].astype(int)
    ddf['sample_name'] = ddf['pos_neg'].astype(str)  + '_' + ddf['num'].astype(str)
    samples_dict = {}
    record_id_list = []

    for index, row in ddf.iterrows():
        if row['sample_name'] not in samples_dict:
            samples_dict[row['sample_name']] = row['Hour']

    ddf['first_time_taken'] = ddf['sample_name'].map(samples_dict) 
    ddf['diference'] = ddf['Hour'] - ddf['first_time_taken']

    for index, row in ddf.iterrows():
        if row['diference']>130:
            record_id_list.append(row['Record ID'])
    return record_id_list


# xcalibration.py

path_qc = 'C:\C\\'

def add_edge_peaks(wl_peaks, pixel_peaks):
    ''' adds pixel values that corresponds to 420 nm and 700 nm
    wl_peaks - wavelength peaks from qc report
    pixel_peaks - slave pixel peaks frmom qc report
    output - updated wl_peaks, pixel_peaks'''
    LR = LinearRegression()
    LR.fit(wl_peaks.reshape(-1,1), pixel_peaks.reshape(-1,1))
    cut_left, cut_right = np.round(LR.predict(np.array([420, 700]).reshape(-1, 1))).squeeze().astype('int')
    wl_peaks = np.insert(wl_peaks, [0, len(wl_peaks)], [420, 700])
    pixel_peaks = np.insert(pixel_peaks, [0, len(pixel_peaks)], [cut_left, cut_right])

    return wl_peaks, pixel_peaks


def get_exact_peaks_pointwise(wl_peaks, pixel_peaks, master_wl_peaks):
    '''chech if brother MPF issue is present, if yes, fix it with pointwise linear functions'''
    if len(master_wl_peaks)!=len(wl_peaks):
        print('error, number of wavelengths is different for master and slave')
        return None
    if len(wl_peaks)!=len(pixel_peaks):
        print('different number of wavelength and pixel peaks')
        return None
    else:
        pixel_peaks_new = []
        for i in range(len(master_wl_peaks)):
            if master_wl_peaks[i]==wl_peaks[i]:
                pixel_peaks_new.append(pixel_peaks[i])
            else:
                if master_wl_peaks[i]<wl_peaks[i]:
                    A = (wl_peaks[i-1], pixel_peaks[i-1])
                    B = (wl_peaks[i], pixel_peaks[i])
                elif master_wl_peaks[i]>wl_peaks[i]:
                    A = (wl_peaks[i], pixel_peaks[i])
                    B = (wl_peaks[i+1], pixel_peaks[i+1])
                f = two_point_line(A, B)
                pixel_peaks_new.append(f(master_wl_peaks[i]))
        return pixel_peaks_new



def get_peaks_qc(device):
    '''extracts wavelength peaks and pixel peaks, adds edge pixels that correspond to 420 mnm and 700 nm'''

    r = re.compile('QcReport_'+device)
    txt_qc = [f for f in os.listdir(path_qc) if re.match(r, f)][0]
    with open(path_qc + txt_qc) as f:
        c = f.read()
        if 'Master Wavelength: ' in c:
            start_from = c.find('\nMaster Wavelength:')
            end_with = c.find('\nMaster Peaks:')
            wl_peaks_str = c[start_from+20:end_with]
            wl_peaks_np = np.fromstring(wl_peaks_str.strip(), sep=',')
            start_from = c.find('\nSlave Peaks:')
            end_with = c.find('\nMaster Machine Peaks:')
            pix_peaks_str = c[start_from+20:end_with]
            pix_peaks_np = np.fromstring(pix_peaks_str.strip(), sep=',') -1
            wl_peaks, pixel_peaks = add_edge_peaks(wl_peaks_np, pix_peaks_np)
            pixel_peaks = np.insert(pixel_peaks, [0, len(pixel_peaks)], [0, 1024])
    return wl_peaks, pixel_peaks


def get_peaks_master_wl(device, master_wl_peaks, df_air_minus_dark):
    '''corrects for brother MPF if needed and outputs final pixel peaks with edges'''

    wl_peaks, pixel_peaks = get_peaks_qc(device)
    wl_peaks = np.insert(wl_peaks, [0, len(wl_peaks)], [400, 720])

    if master_wl_peaks is not None:
        master_wl_peaks_no_hill = master_wl_peaks.take([0, 1, 3, 4, 5, 6, 7, 8, 9, 10])
        if not (wl_peaks == master_wl_peaks_no_hill).all():
            pixel_peaks = get_exact_peaks_pointwise(wl_peaks, pixel_peaks, master_wl_peaks_no_hill)
    slave_peak_3 = np.median(df_air_minus_dark.iloc[:, :300].to_numpy().argmax(1))
    pixel_peaks = np.insert(pixel_peaks, 2, slave_peak_3)
    wl_peaks = np.insert(wl_peaks, 2, 450)
    pixel_peaks = pixel_peaks.astype('int')
    if master_wl_peaks is None:
        return wl_peaks, pixel_peaks
    else:
        return pixel_peaks


#
#
# def get_peaks_master_wl_Aylas(device, master_wl_peaks, df_air_minus_dark):
#     wl_peaks, pixel_peaks = get_peaks_qc(device)
#     wl_peaks = np.insert(wl_peaks, [0, len(wl_peaks)], [400, 720])
#
#     if master_wl_peaks is not None:
#         master_wl_peaks_no_hill = master_wl_peaks.take([0, 1, 3, 4, 5, 6, 7, 8, 9, 10])
#         if not (wl_peaks == master_wl_peaks_no_hill).all():
#             pixel_peaks = get_exact_peaks_pointwise(wl_peaks, pixel_peaks, master_wl_peaks_no_hill)
#     slave_peak_3 = np.median(df_air_minus_dark.iloc[:, :300].to_numpy().argmax(1))
#     pixel_peaks = np.insert(pixel_peaks, 2, slave_peak_3)
#     wl_peaks = np.insert(wl_peaks, 2, 450)
#     pixel_peaks = pixel_peaks.astype('int')
#     if master_wl_peaks is None:
#         return wl_peaks, pixel_peaks
#     else:
#         return pixel_peaks


def xcalibration(slave_matrix, master_pixel_peaks, slave_pixel_peaks):
    ''' perform x calibration of slave matrix according to the peaks specified
    slave_matrix - numpy or pandas array of data to x-calibrate
    master_pixel_peaks - list of ints - peaks of MPF of master
    slave_peaks - list of ints - peaks of slave of MPF, note that len(master_pixel_peaks)=len(slave_peaks)
    output: numpy or pandas array of data x-calibrated slave data'''
    if type(slave_matrix) == pd.core.frame.DataFrame:
        slave_index = slave_matrix.index
        slave_matrix = slave_matrix.to_numpy()
        pandas = True
    else:
        pandas = False

    new_slave_data = np.zeros(slave_matrix.shape)
    for i in range(len(master_pixel_peaks) - 1):
        slave_dist = slave_pixel_peaks[i + 1] - slave_pixel_peaks[i]
        master_dist = master_pixel_peaks[i + 1] - master_pixel_peaks[i]
        slave_slice = slave_matrix[:, slave_pixel_peaks[i]:slave_pixel_peaks[i + 1]]

        if slave_dist < master_dist:
            diff = master_dist - slave_dist
            index = np.linspace(1, slave_dist-1, num=diff + 2).astype('int')[1:-1]
            values = (slave_slice[:, index - 1] + slave_slice[:, index + 1]) / 2
            for row in range(len(slave_matrix)):
                new_slave_data[row, master_pixel_peaks[i]:master_pixel_peaks[i + 1]] = np.insert(slave_slice[row],
                                                                                                 index, values[row])

        elif slave_dist > master_dist:
            diff = slave_dist - master_dist
            step = slave_slice.shape[1] // (diff + 1)
            new_index = list(
                set(range(slave_slice.shape[1])) - set(np.linspace(0, slave_dist, num=diff + 2).astype('int')[1:-1]))
            for row in range(len(slave_matrix)):
                new_slave_data[row, master_pixel_peaks[i]:master_pixel_peaks[i + 1]] = slave_slice[row, new_index]

        elif slave_dist == master_dist:
            new_slave_data[:, master_pixel_peaks[i]:master_pixel_peaks[i + 1]] = slave_slice
    if pandas:
        new_slave_data = pd.DataFrame(new_slave_data, index=slave_index)
    return new_slave_data
  In [139]:  # outliers.py
def golden_window(df_abs, percentile_shift, crop_left=50, crop_right=950):
    X = df_abs.iloc[:, crop_left:crop_right].to_numpy()
    C = np.corrcoef(X)
    c = C.sum(0) / len(X)
    golden = X[c==c.max()].squeeze()
    c_golden_win = [np.corrcoef(golden, X[i])[0, 1] for i in range(len(X))]
    outliers1 = df_abs[c_golden_win<np.array(0.5)].index
    print(len(outliers1), 'corr outl 1')
    #
    corr_steps = 3
    window_size = int((crop_right-crop_left)/(corr_steps+1))
    idx1 = set()
    for i in range(corr_steps):
        golden_win = golden[crop_left+window_size*i:crop_left+window_size*(i+2)]
        X_win = X[:, crop_left + window_size * i:crop_left + window_size * (i + 2)]
        c_golden_win = [np.corrcoef(golden_win, X_win[i])[0, 1] for i in range(len(X))]
        idx_1 = df_abs.iloc[np.where(c_golden_win <= np.array(0.5))].index
        if idx_1 is not None:
            idx1 = idx1.union(idx_1)
    print(len(idx1), 'corr outliers')
    dist_steps = 15
    window_size = int((crop_right - crop_left) / (dist_steps + 1))
    idx2 = set()
    for i in range(dist_steps):
        golden_win = golden[crop_left+window_size*i:crop_left+window_size*(i+2)]
        X_win = X[:, crop_left + window_size * i:crop_left + window_size * (i + 2)]
        # d_golden_win = [np.linalg.norm(golden_win-X_win[i], 2)**2 for i in range(len(X))]
        d_golden_win = [max(abs(golden_win-X_win[i])) for i in range(len(X))]
        q25, q75 = percentile(d_golden_win, 25), percentile(d_golden_win, 75)
        # cut_off = (q75 - q25) * 5
        cut_off = (q75 - q25)*percentile_shift
        upper = q75+cut_off
        # print(f'{crop_left+window_size*i} - {crop_left+window_size*(i+2)}, cut_off {cut_off}, upper {upper}')
        idx_2 = df_abs.iloc[np.where(d_golden_win >=upper)].index
        if idx_2 is not None:
            idx2 = idx2.union(idx_2)
    print(len(idx2), 'dist outliers')
    idx = idx1.union(idx2).union(outliers1)
    return idx  In [140]:  # helper.py



def clean_raw(df,get_outlier_index=False):
    '''deletes outliers using Isolation Forest
    df - dataframe, raw data of some dind
    output - numpy array, cleaned raw data'''


    try:
        outlier_detector = LocalOutlierFactor(novelty=True, metric='cosine', n_neighbors=10)
        prediction = outlier_detector.fit_predict(df)
    except:
        outlier_detector = IsolationForest(random_state=42, n_jobs=-1, warm_start=True, max_features=500)
        outlier_detector.fit(df)
        prediction = outlier_detector.predict(df)
    print('outliers', len(df[prediction == -1]))
    if get_outlier_index:
        return df[prediction != -1], df[prediction == -1].index
    else:
        return df[prediction != -1]

def clean(df,get_outlier_index=False):
    '''deletes outliers using Isolation Forest
    df - dataframe, raw data of some dind
    output - numpy array, cleaned raw data'''

    idx = outliers.moving_window(df, crop_left=100, crop_right=950, clean_param_corr=0.9, steps=3)
    print(len(idx), 'outliers')


    if get_outlier_index:
        return df.loc[set(df.index)-set(idx)], set(idx)
    else:
        return df.loc[set(df.index)-set(idx)]




def get_samp_ref(df):
    '''separates samples and references
    df - database structure dataframe
    output - df_samples, df_ref - dataframes, samples and references separated with the same index'''
    df_samples = df[~df.index.str.contains('^ref')]
    if (df_samples.index.duplicated()).any():
        df_samples.index = df_samples.index+'_'+pd.to_datetime(df_samples['Sample Date']).dt.strftime('%m_%d_%H_%M_%S')
    df_ref = pd.DataFrame(index=df_samples.index, columns = df.columns)
    for i in df_samples.index:
        j = df_samples.loc[i, 'last_ref']
        df_ref.loc[i] = df.loc[j]
    return df_samples, df_ref



def get_data(df, air_divide=False):
    ''' reads raw data from database structured csv
    df: database dataframe
    ouput: df_raw_minus_dark ,df_raw_ref_minus_dark, df_air_minus_dark - dataframes, raw data'''
    df_samples, df_ref = get_samp_ref(df)

    df_raw_minus_dark = df_samples['raw_mean'].str.split(";", expand=True).astype('float64') - df_samples[
        'raw_dark'].str.split(";", expand=True).astype('float64')
    df_raw_ref_minus_dark = df_ref['raw_mean'].str.split(";", expand=True).astype('float64') - df_ref[
        'raw_dark'].str.split(";", expand=True).astype('float64')

    df_set_minus_dark = df_samples['set_airRef'].str.split(";", expand=True).astype('float64') - df_samples[
        'set_dark'].str.split(";", expand=True).astype('float64')
    df_store_minus_dark = df_samples['store_airRef'].str.split(";", expand=True).astype('float64') - df_samples[
        'store_dark'].str.split(";", expand=True).astype('float64')
    if air_divide:
        df_air_minus_dark = (df_set_minus_dark + df_store_minus_dark)/2
    else:
        df_air_minus_dark = (df_set_minus_dark + df_store_minus_dark)

    return df_raw_minus_dark.fillna(0), df_raw_ref_minus_dark.fillna(0), df_air_minus_dark.fillna(0)


def get_df_golden(df_raw_minus_dark, df_raw_ref_minus_dark, df_air_minus_dark, device_id, plot=False):

    if 'golden_data_frames' not in os.listdir():
        os.mkdir('golden_data_frames')
    if device_id not in os.listdir('golden_data_frames'):
        df_golden = pd.DataFrame(index=['raw', 'ref', 'air'], columns=df_raw_minus_dark.columns)
        df_raw_clean = clean_raw(df_raw_minus_dark).to_numpy()
        df_ref_clean = clean_raw(df_raw_ref_minus_dark.drop_duplicates()).to_numpy()
        df_air_clean = clean_raw(df_air_minus_dark).to_numpy()
        df_golden.loc['raw'] = df_raw_clean.mean(0)
        df_golden.loc['ref'] = df_ref_clean.mean(0)
        df_golden.loc['air'] = df_air_clean.mean(0)
        pickle.dump(df_golden.astype('float64'), open(f'golden_data_frames/{device_id}', 'wb'))
        # if plot:
        #     plt.figure(figsize=(20, 10))
        #     plt.subplot(231)
        #     plt.plot(df_raw_clean[:, 1:].T, alpha=0.2)
        #     plt.title('raw ', fontsize=15)
        #
        #     plt.subplot(232)
        #     plt.plot(df_ref_clean[:, 1:].T, alpha=0.2)
        #     plt.title('reference', fontsize=15)
        #
        #     plt.subplot(233)
        #     plt.plot(df_air_clean.T, alpha=0.2)
        #     plt.title('air', fontsize=15)
        #     plt.suptitle(device_id, fontsize=20)
        #     plt.tight_layout()
        #     # plt.savefig(f'vizual/{device}')
        #     plt.show()
    else:
        df_golden = pickle.load(open(f'golden_data_frames/{device_id}', 'rb'))
    return df_golden




def add_label_column(df_i):
    ''' if df has index that indicates label throught pos/p - this function will add column "labels" with 1-neg, 2-pos'''
    phrase_pos = re.compile(r'(POS)|(^P)', re.IGNORECASE)
    pos_index = [bool(re.search(phrase_pos, str(i))) for i in df_i.index]

    phrase_neg = re.compile(r'(neg)|(^n)', re.IGNORECASE)
    neg_index = [bool(re.search(phrase_neg, str(i))) for i in df_i.index]
    df = df_i.fillna(0)
    df['label'] = np.NaN

    df.loc[df[pos_index].index, 'label'] = 2
    df.loc[df[neg_index].index, 'label'] = 1
    df = df.dropna(axis=0, how='any')

    return df


def plot_pos_neg(df_abs, labels):
    if 1 in labels.to_list():
        plt.plot(df_abs[labels == 1][0, 1:].T, 'b', label=f'NEG ({(labels==1).sum()})')
        plt.plot(df_abs[labels == 1][:, 1:].T, 'b')
    if 2 in labels.to_list():
        plt.plot(df_abs[labels == 2][0, 1:].T, 'y', label=f'POS ({(labels==2).sum()})')
        plt.plot(df_abs[labels == 2][:, 1:].T, 'y')
    plt.legend()




def plot_pca(df_abs, labels):
    pca = PCA(n_components=2)
    df_abs = pd.DataFrame(preprocess(df_abs.to_numpy(),w=47,der = 1), index=df_abs.index)
    principalComponents = pca.fit_transform(df_abs)
    principal_df = pd.DataFrame(data = principalComponents,index=df_abs.index, columns = ['principal component 1', 'principal component 2'])
    targets = [1, 2]
    colors = ['b', 'y']
    for target, color in zip(targets,colors):
        indicesToKeep = (labels == target)
        plt.scatter(principal_df.loc[indicesToKeep, 'principal component 1']
                   , principal_df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
        plt.xlabel('Principal Component - 1',fontsize=10)
        plt.ylabel('Principal Component - 2',fontsize=10)  In [142]:  # model.py
def preprocess(df, w, der):
    sav = savgol_filter(df, w, 2, der)
    return sav

def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):            # Apply correction
        output_data[i, :] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data

def msc(input_data, reference=None):
    ''' Perform Multiplicative scatter correction'''
    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i,:] -= input_data[i, :].mean()        # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:              # Calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference        # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):          # Run regression
        fit = np.polyfit(ref, input_data[i, :], 1, full=True)          # Apply correction
        data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
    return data_msc

def preprocess_one(X, w=47, der=1):
    return snv(preprocess(msc(X,reference = ref_msc), w, der))

def train_model(df_abs_batch, labels):
    df_abs_batch = df_abs_batch.fillna(0)
    X = df_abs_batch.to_numpy()
    X_ = preprocess_one(X)
    y = labels.to_numpy()

    model = svm.SVC(kernel='rbf',C=100)
    model.fit(X_,y)
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=cv)
    score = np.nanmean(scores)
    return int(round(score*100, 0))


def get_sens_spec_CV(df_abs_xy, labels, df_master, labels_master, crop_left=0, crop_right=1024):
    # df_abs_batch_xy_train, df_abs_batch_xy_test = train_test_split(df_abs_xy, random_state=0, train_size=.5, stratify=labels)
    # labels_train, labels_test = train_test_split(labels, random_state=0, train_size=.5, stratify=labels)
    X = np.concatenate([df_master.to_numpy(), df_abs_xy.to_numpy()])

    #ref = pickle.load(open('V0013_D614_ref_msc1', 'rb')).squeeze()
    model = pickle.load(open('V0013_D614_snv_SG_w47_d1_msc_SVM', 'rb'))

    X_ = preprocess_one(X, ref)
    y_slave = labels.to_numpy().squeeze()
    y = np.concatenate([labels_master.to_numpy().squeeze(), y_slave])

    model.fit(X_, y)
    cv = LeaveOneOut()
    scores = cross_val_score(model, X, y, cv=cv)

    slave_scores = scores[len(df_master):]
    y_slave_opposite = [1 if i==2 else 2 for i in y_slave]
    y_slave_pred = [y_slave[i] if slave_scores[i]==1 else y_slave_opposite[i] for i in range(len(y_slave))]
    conf_matrix = metrics.confusion_matrix(y_slave, y_slave_pred)
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]

    # calculate accuracy
    conf_accuracy = round(((TP + TN)*100 / (TP + TN + FP + FN)), 0)

    # calculate the sensitivity
    conf_sensitivity = round((TP*100 / (TP + FN)))
    # calculate the specificity
    conf_specificity = round((TN *100/ (TN + FP)))

    return conf_accuracy, conf_sensitivity, conf_specificity



# def get_sens_spec(df_abs_xy, labels, df_master, labels_master, crop_left=0, crop_right=1024):
#     df_abs_batch_xy_train, df_abs_batch_xy_test = train_test_split(df_abs_xy, random_state=0, train_size=.5, stratify=labels)
#     labels_train, labels_test = train_test_split(labels, random_state=0, train_size=.5, stratify=labels)
#     X_train = np.concatenate([df_master.to_numpy(), df_abs_batch_xy_train.to_numpy()])
#     y_train = np.concatenate([labels_master.to_numpy().squeeze(), labels_train.to_numpy().squeeze()])

#     X_test = df_abs_batch_xy_test.to_numpy()

#     model = loaded_model
#     model.fit(preprocess_one(X_train),y_train)
#     predict_test = model.predict(preprocess_one(X_test))
#     conf_matrix = metrics.confusion_matrix(labels_test, predict_test)
#     TP = conf_matrix[1][1]
#     TN = conf_matrix[0][0]
#     FP = conf_matrix[0][1]
#     FN = conf_matrix[1][0]

#     # calculate accuracy
#     conf_accuracy = round(((TP + TN)*100 / (TP + TN + FP + FN)), 0)

#     # calculate the sensitivity
#     conf_sensitivity = round((TP*100 / (TP + FN)))
#     # calculate the specificity
#     conf_specificity = round((TN *100/ (TN + FP)))

#     return conf_accuracy, conf_sensitivity, conf_specificity  In [143]:  def unique_samples(df_):
    df_['Sample'] = df_.index
    df_['pos_neg'] = df_['Sample'].str.extract(r'(^[A-Za-z])')
    df_['pos_neg'] = df_['pos_neg'].replace({'n':'N', 'p':'P'})
    df_['num'] = df_['Sample'].str.extract(r'[A-Za-z]_(\d{1,3})')
    df_ = df_.fillna(0)
    df_['num'] = df_['num'].astype(int)
    df_['sample_name'] = df_['pos_neg'].astype(str)  + '_' + df_['num'].astype(str)
    unique_samples = df_['sample_name'].unique()
    list_unique_samples=sorted(unique_samples)
    unique_pos = len([i for i in list_unique_samples if 'P' in i])
    unique_neg = len([i for i in list_unique_samples if 'N' in i])
    return unique_pos,unique_neg, list_unique_samples   In [144]:  master_device = 'D614'


# read raw data and get golden samples
df_raw_minus_dark, df_raw_ref_minus_dark, df_air_minus_dark = get_data(df_master, air_divide=True)
df_golden = get_df_golden(df_raw_minus_dark, df_raw_ref_minus_dark.drop_duplicates(), df_air_minus_dark, master_device, plot=True)
# get peaks from the qc report and add 'hill' peak
wl_peaks, pixel_peaks = get_peaks_master_wl(master_device, None, df_air_minus_dark)

df_peaks = pd.DataFrame(columns=wl_peaks)
df_peaks.loc[master_device] = pixel_peaks
master_wl_peaks = df_peaks.columns.astype('float').to_numpy()
master_pix_peaks = df_peaks.loc[master_device].to_numpy()

df = df[~df['Record ID'].isin([get_old_samples_list(df)])]

sampltes_with_high_ct = re.compile(r'(P_029)',re.IGNORECASE)
df = df[~df.index.str.contains(sampltes_with_high_ct)]
d = df['Device ID'].unique()
devices = sorted(d.flatten())
devices.sort()
#devices = devices[1:-1]
devices.remove('C142')
devices.remove('S118')
df_slave = df
df_analysis = pd.DataFrame()  

for device in devices:

 print(device)

    # get csv type dataframe for this device
df_slave_device = df_slave[df_slave['Device ID'] == device]
    # get raw data and calculate batch absorbance
df_raw_minus_dark, df_raw_ref_minus_dark, df_air_minus_dark = get_data(df_slave_device)

    #get golden raw samples
df_golden_slave = get_df_golden(df_raw_minus_dark, df_raw_ref_minus_dark.drop_duplicates(), df_air_minus_dark, device)
    # get peaks from the qc and correct for brothre MPF if needed
slave_pix_peaks = get_peaks_master_wl(device, master_wl_peaks, df_air_minus_dark)
    # slave_pix_peaks = xcalibration.get_peaks_master_wl_Aylas(device, master_wl_peaks, df_air_minus_dark)
    #

df_peaks.loc[device] = slave_pix_peaks


    # slave_pix_peaks.insert(2, int(slave_peak_3))
df_golden_x = xcalibration(df_golden_slave, master_pix_peaks, slave_pix_peaks)
    df_golden_master = pickle.load(open('C:\\ , 'rb'))

    # ycalibration
    ratios_df = df_golden_master/df_golden_x
    ratios_df.fillna(0, inplace=True)

    df_raw_ref_minus_dark_x = xcalibration(df_raw_ref_minus_dark, master_pix_peaks, slave_pix_peaks)
    df_raw_ref_minus_dark_xy = df_raw_ref_minus_dark_x*ratios_df.loc['ref'].to_numpy()

    df_raw_minus_dark_x = xcalibration(df_raw_minus_dark, master_pix_peaks, slave_pix_peaks)
    df_raw_minus_dark_xy = df_raw_minus_dark_x*ratios_df.loc['raw'].to_numpy()

    df_air_minus_dark_x = xcalibration(df_air_minus_dark, master_pix_peaks, slave_pix_peaks)

    df_batch_abs = np.log10(df_raw_ref_minus_dark/df_raw_minus_dark)
    df_batch_abs_xy = np.log10(df_raw_ref_minus_dark_xy/df_raw_minus_dark_xy)
    df_batch_abs_xy.fillna(0, inplace=True)



    df_batch_abs_labeled = add_label_column(df_batch_abs).fillna(0)
    df_batch_abs_xy = df_batch_abs_xy.loc[df_batch_abs_labeled.index].fillna(0)



    plt.figure(figsize=(25, 10))

    plt.subplot(241)
    plt.plot(df_raw_minus_dark_x.iloc[:, 1:].T, 'g', alpha=0.2)
    plt.plot(df_golden_x.loc['raw'].iloc[1:], 'g', label='x-calibrated golden slave')
    plt.plot(df_golden_master.loc['raw'].iloc[1:], 'b', label='golden master')
    plt.legend()
    plt.title('raw ', fontsize=15)

    plt.subplot(242)
    plt.plot(df_raw_ref_minus_dark_x.drop_duplicates().iloc[:, 1:].T, 'g', alpha=0.2)
    plt.plot(df_golden_x.loc['ref'].iloc[1:], 'g', label='x-calibrated golden slave')
    plt.plot(df_golden_master.loc['ref'].iloc[1:], 'b', label='golden master')
    plt.title('reference', fontsize=15)
    plt.legend()

    plt.subplot(243)
    plt.plot(df_air_minus_dark_x.drop_duplicates().iloc[:, 1:].T, 'g', alpha=0.2)
    plt.plot(df_golden_x.loc['air'].iloc[1:], 'g', label='x-calibrated golden slave')
    plt.plot(df_golden_master.loc['air'].iloc[1:].T, 'b', label='golden master')
    plt.title('air', fontsize=15)
    plt.legend()

    plt.subplot(243)
    for t in ['raw', 'ref', 'air']:
        plt.plot(ratios_df.loc[t].iloc[1:], label=t)
    plt.legend()
    plt.title('ratios', fontsize=15)

    plt.subplot(245)
    plot_pos_neg(df_batch_abs_labeled.iloc[:, :-1].to_numpy(), df_batch_abs_labeled.iloc[:, -1])
    plt.title(f'batch absorbance collected', fontsize=15)
    plt.legend()

    plt.subplot(246)
    outl = golden_window(pd.DataFrame(snv(df_batch_abs_labeled.iloc[:,:-1].to_numpy()), index=df_batch_abs_labeled.index), percentile_shift=1.5)
    print(len(outl), 'outliers\n')
    df_analysis.loc[device, 'Outliers'] = len(outl)
    df_batch_abs_clean = df_batch_abs_labeled.loc[set(df_batch_abs_labeled.index)-set(outl)]

    if len(outl)>0:
        plt.plot(snv(df_batch_abs_labeled.loc[outl].iloc[:, 1:-1].to_numpy()).T, 'r')
        plt.plot(snv(df_batch_abs_labeled.loc[list(outl)[0]].iloc[1:-1].to_numpy().reshape(1,-1)).T, 'r', label='outliers')

    plot_pos_neg(snv(df_batch_abs_clean.iloc[:, :-1].to_numpy()), df_batch_abs_clean.iloc[:, -1])
    plt.title(f'snv(batch absorbance original)', fontsize=15)

    plt.subplot(247)
    plot_pos_neg(df_batch_abs_clean.iloc[:, :-1].to_numpy(), df_batch_abs_clean.iloc[:, -1])
    plt.title(f'batch absorbance clean', fontsize=15)

    plt.subplot(244)
    plt.plot(df_batch_abs_master.iloc[:, 1:-1].to_numpy().T, 'grey')
    plt.plot(df_batch_abs_master.iloc[0, 1:-1].to_numpy().T, 'grey', label='master')
    df_batch_abs_xy = df_batch_abs_xy.loc[df_batch_abs_clean.index]
    plot_pos_neg(df_batch_abs_xy.to_numpy(), df_batch_abs_clean.iloc[:, -1])
    plt.title(f'cleaned after XY calibration', fontsize=15)
    vc = df_batch_abs_clean.iloc[:, -1].value_counts()
    print(vc)




    if len(vc)==2 and (vc>=2).all():
        #df_batch_abs_clean.to_csv(path_clean_df + f'{device}_clean_labeled.csv')

        df_analysis.loc[device, 'CV'] = train_model(df_batch_abs_xy, df_batch_abs_clean.iloc[:, -1])
        #df_analysis.loc[device, 'CV crop'] = train_model(df_batch_abs_xy.iloc[:, 350:950], df_batch_abs_clean.iloc[:, -1])

        conf_accuracy, conf_sensitivity, conf_specificity = get_sens_spec(df_batch_abs_xy, df_batch_abs_clean.iloc[:, -1], df_batch_abs_master.iloc[:,:-1], df_batch_abs_master.iloc[:,-1], 0, 1024)
        df_analysis.loc[device, 'Sens'] = conf_sensitivity
        df_analysis.loc[device, 'Spec'] = conf_specificity
        df_analysis.loc[device, 'Accuracy'] = conf_accuracy
        # conf_accuracy, conf_sensitivity, conf_specificity = get_sens_spec(df_batch_abs_xy, df_batch_abs_clean.iloc[:, -1], df_batch_abs_master.iloc[:,:-1], df_batch_abs_master.iloc[:,-1], 350, 950)
        # df_analysis.loc[device, 'sens crop'] = conf_sensitivity
        # df_analysis.loc[device, 'spec crop'] = conf_specificity
        # df_analysis.loc[device, 'accuracy crop'] = conf_accuracy

        df_analysis.loc[device, 'Pos clean'] = vc.loc[2.0]
        df_analysis.loc[device, 'Neg clean'] = vc.loc[1.0]
        # Aditional info
        df_ = df_slave_device[~df_slave_device.index.str.contains(r'ref')]
        l = df_.index.str.contains(r'P')
        count = 0
        most = 0
        for x in range(len(l)):

            if l[x] == True:
                count+=1
            else:
                count = 0

            if count > most:
                most = count
       
        df_analysis.loc[device, 'Bigest grouping'] = most
        
        unique_pos,unique_neg,list_unique_samples = unique_samples(df_)
        df_analysis.loc[device,'Unique pos samples'] = unique_pos
        df_analysis.loc[device,'Unique neg samples'] = unique_neg
        df_analysis.loc[device,'List of unique samples'] = ','.join(list_unique_samples)

        plt.subplot(248)
        plot_pca(df_batch_abs_xy, df_batch_abs_clean.iloc[:, -1])
        plt.title(f'PCA, CV = {df_analysis.loc[device, "CV"]}, accuracy = {df_analysis.loc[device, "Accuracy"]}')

    plt.suptitle(device, fontsize=20)
    plt.tight_layout()
    plt.savefig('C:\')
    plt.close()
