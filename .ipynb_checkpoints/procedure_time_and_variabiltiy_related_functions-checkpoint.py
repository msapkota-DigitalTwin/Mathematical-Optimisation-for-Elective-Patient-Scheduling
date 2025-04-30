import pandas as pd
import datetime
import numpy as np
import random
import os
import pickle
#from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, quicksum
from itertools import product
import copy
import matplotlib.pyplot as plt
import colorlover as cl
import pickle
import os
import re
#import torch
from data_processing_nd_encoding_related_functions import age_to_group, filter_dataset_till_possible

#from predictive_model_related_functions import gaussan_based_prediction, predict_with_model

from data_processing_nd_encoding_related_functions import one_hot_encode_dataframe, one_hot_encoding_for_similar_columns



def check_availability(now_datetime, surgery_time, close_time):
    #with theatres.request() as request:
        #yield request
    #now_datetime = datetime.datetime.fromtimestamp(env.now)
    if pd.isnull(close_time):
        close_time = datetime.datetime.combine(now_datetime.date(), datetime.time(17,30,0))
    
    #print(close_time)
    time_anticipated = now_datetime + datetime.timedelta(minutes=surgery_time)
    #print(close_time, time_anticipated)
    if close_time < time_anticipated:
        #print('can not proceed')
        return False
    else:
        #print('can proceed')
        return True


        
def check_elements_in_other_dataframe(df1, df2, column_name):
    
    elements_df1 = set(df1[column_name].unique())
    elements_df2 = set(df2[column_name].unique())

    missing_elements = elements_df1 - elements_df2
    all_elements_present = len(missing_elements) == 0

    return [all_elements_present, missing_elements]


def check_if_data_exists(dataframe, index_value, column_name):

    index_exists = index_value in dataframe.index
    column_exists = column_name in dataframe.columns
    
    if index_exists and column_exists:
        value = dataframe.at[index_value, column_name]
        if pd.notna(value):
            return True
    
    return False



def convert_to_uniform_data(data, freq, time_begin, time_stop):
    # create a DataFrame with non-uniform time-series data
    df = pd.DataFrame(data, columns=['timestamp', 'data'])
    
    # convert timestamp column to datetime format and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    if df.index.min() < pd.Timestamp(time_begin):
        df = df.loc[df.index >= time_begin]
    df.loc[time_begin] = {'data':0}
  
    #removing data outside required time-range
    if df.index.max() > pd.Timestamp(time_stop):
        df = df.loc[df.index <= time_stop]
    df.loc[time_stop] = {'data':0}
    #uniform_index = pd.date_range(start=STARTING_DATE_TIME, end=ENDING_DATE_TIME, freq=freq)

    # filter the DataFrame to include only the data within the specified time range
    #df = df.loc[(df.index >= pd.Timestamp(time_begin)) & (df.index <= pd.Timestamp(time_stop))]
    
    # resample the DataFrame to uniform time-series data with fixed interval 'freq'
    resampled_df = df.resample(freq).sum()
    
    resampled_df = resampled_df.reset_index()
    
    # return the resampled DataFrame
    return resampled_df.fillna(0)



def create_df_from_multiindex_series(Series1):
    new_data = {}
    
    # Iterate over the Series index
    prev_first = None
    prev_second = None
    #related_data = []
    
    for idx in Series1.index:
        first = idx[0]
        second = idx[1]
        #third = idx[2]
        
        if first not in new_data:
            new_data[first] = {}
        
        if pd.isnull(third):  # Check if third element is NaN
            if (prev_first != first or prev_second != second) and prev_first in new_data and prev_second in new_data[prev_first]:
                #print([prev_first, prev_second, new_data[prev_first][prev_second]])
                new_data[prev_first][prev_second] = np.ceil(np.mean(new_data[prev_first][prev_second])/5)*5
                prev_first = None
                prev_second = None
            mean_value = Series1.loc[(first, second, np.nan)].mean()
            if pd.notna(mean_value):
                new_data.setdefault(first, {})[second] = mean_value
        else:
            if second not in new_data[first]:
                new_data[first][second] = []
            #related_data.append(Series1.loc[idx])
            new_data[first][second].append(Series1.loc[idx])
            prev_first = first
            prev_second = second
            
    # Create the new DataFrame
    return pd.DataFrame(new_data).T



def create_model_name(model_type, out_data_type, input_features, data_categ_dict):
    name_strings = [''.join(re.findall(r'[A-Z0-9&]', model_type)),''.join(re.findall(r'[A-Z0-9&]', out_data_type))]
    
    name_strings.append(''.join([word[0] for word in input_features]))
    
    for key in data_categ_dict.keys():
        if not len(data_categ_dict[key].split()) > 1:
            name_strings.append(''.join(re.findall(r'[A-Z0-9&]', data_categ_dict[key])))
        else:
            name_strings.append(''.join([word[0] for word in data_categ_dict[key].split()]))
    name_strings.append('model')
    return '_'.join(name_strings)



def select_3_representative_data(df, filtering_data, column_name):
    df_filtered = df.copy()
    for column, data in filtering_data.items():
        df_filtered = df_filtered[df_filtered[column] == data]
    
    value_counts = df_filtered[column_name].value_counts()

    if len(value_counts) == 1:
        return [value_counts.index[0]]*3
    
    values = value_counts.values
        
    # Calculate means
    overall_mean = np.mean(values)
    upper_mean = np.mean(values[values >= overall_mean])
    lower_mean = np.mean(values[values < overall_mean])
    #print([overall_mean, upper_mean, lower_mean])
    # Find the closest values to the means
    #print(np.abs(values - overall_mean).argmin())
    overall_representative = value_counts.index[np.abs(values - overall_mean).argmax()]
    
    #print(np.abs(values - upper_mean).argmin())
    upper_representative = value_counts.index[(values - upper_mean).argmax()]
    
    #print(np.abs(values - lower_mean).argmin())
    lower_representative = value_counts.index[np.abs(values - lower_mean).argmin()]
    
    return [overall_representative, upper_representative, lower_representative]
    

def check_if_data_exists(dataframe, index_value, column_name):

    index_exists = index_value in dataframe.index
    column_exists = column_name in dataframe.columns
    
    if index_exists and column_exists:
        value = dataframe.at[index_value, column_name]
        if pd.notna(value):
            return True
    
    return False

def combined_mean_sd(means, stds, weights = None):
    # Initialize lists to store mean and sd for each process category
    '''means = []
    sds = []

    # Iterate through each process category in the combination
    for process in process_combination:
        # If the process category exists in both mean and sd dictionaries
        if process in mean_dict and process in sd_dict:
            # Extract mean and standard deviation
            mean = mean_dict[process]
            sd = sd_dict[process]
            # Append mean and sd to respective lists
            means.append(mean)
            sds.append(sd)
    '''
    if weights is None:
        # Calculate weights based on mean times
        total_mean = np.sum(means)
        weights = [mean / total_mean for mean in means]

    # Calculate combined mean
    combined_mean = np.average(means, weights=weights)

    # Calculate combined standard deviation
    combined_sd = np.sqrt(np.average((np.array(stds) ** 2), weights=weights))

    return combined_mean, combined_sd


def create_df_from_multiindex_series_old1(Series1):

    if len(Series1.index[0]) == 2:
        return Series1.unstack()
    
    new_data = {}
    
    # Iterate over the Series index
    prev_first = None
    prev_second = None
    #related_data = []
    
    for idx in Series1.index:
        first = idx[0]
        second = idx[1]
        third = idx[2]
        
        if first not in new_data:
            new_data[first] = {}
        
        if pd.isnull(third):  # Check if third element is NaN
            if (prev_first != first or prev_second != second) and prev_first in new_data and prev_second in new_data[prev_first]:
                #print([prev_first, prev_second, new_data[prev_first][prev_second]])
                new_data[prev_first][prev_second] = np.ceil(np.mean(new_data[prev_first][prev_second])/5)*5
                prev_first = None
                prev_second = None
            mean_value = Series1.loc[(first, second, np.nan)].mean()
            if pd.notna(mean_value):
                new_data.setdefault(first, {})[second] = mean_value
        else:
            if (prev_first != first or prev_second != second) and prev_first in new_data and prev_second in new_data[prev_first]:
                #print([prev_first, prev_second, new_data[prev_first][prev_second]])
                new_data[prev_first][prev_second] = np.ceil(np.mean(new_data[prev_first][prev_second])/5)*5

            else:
                if second not in new_data[first]:
                    new_data[first][second] = []
                #related_data.append(Series1.loc[idx])
                new_data[first][second].append(Series1.loc[idx])
                prev_first = first
                prev_second = second
            
    # Create the new DataFrame
    return pd.DataFrame(new_data).T

    
def create_df_from_multiindex_series(Series1):

    if len(Series1.index[0]) == 2:
        return Series1.unstack()
    
    new_data = {}
    
    # Iterate over the Series index
    prev_first = None
    prev_second = None
    #related_data = []
    
    for idx in Series1.index:
        first = idx[0]
        second = idx[1]
        third = idx[2]
        
        if first not in new_data:
            new_data[first] = {}
        
        if pd.isnull(third):  # Check if third element is NaN
            #identifying if new grouping index has started, and wrapping up previous data if new index has begun
            if (prev_first != first or prev_second != second) and prev_first in new_data and prev_second in new_data[prev_first]:
                #print([prev_first, prev_second, new_data[prev_first][prev_second]])
                new_data[prev_first][prev_second] = np.ceil(np.mean(new_data[prev_first][prev_second])/5)*5
                prev_first = None
                prev_second = None
            mean_value = Series1.loc[(first, second, np.nan)].mean()
            if pd.notna(mean_value):
                new_data.setdefault(first, {})[second] = mean_value
        else:
            if (prev_first != first or prev_second != second) and prev_first in new_data and prev_second in new_data[prev_first]:
                #print([prev_first, prev_second, new_data[prev_first][prev_second]])
                new_data[prev_first][prev_second] = np.ceil(np.mean(new_data[prev_first][prev_second])/5)*5
                prev_first = None
                prev_second = None
            else:
                if second not in new_data[first]:
                    new_data[first][second] = []
                #related_data.append(Series1.loc[idx])
                new_data[first][second].append(Series1.loc[idx])
                prev_first = first
                prev_second = second
    if not pd.isnull(third):
        new_data[prev_first][prev_second] = np.ceil(np.mean(new_data[prev_first][prev_second])/5)*5
         
    # Create the new DataFrame
    return pd.DataFrame(new_data).T


def compute_TASKS_average_surgical_time(patient_with_data, df_sessions, model_types, model_categ_dic, pred_model_related_preprocessed_data, new_or_ignored_data_4_model, model_input_features, varying_features, force_encoding_cols, trained_model_dir):
    #patient_with_data = update_patient_data(df_cases, patient_data)

    data_relevant_to_trained_models = {col: list(pred_model_related_preprocessed_data[col].unique()) for col in model_input_features}
    #model_trained_consultants = list(pred_model_related_preprocessed_data['Consultant Code'].unique())

    result = pd.DataFrame(columns = [id for id in df_sessions['Session ID']])
    
    input_data_df = pd.DataFrame(columns = model_input_features)
    input_data_index_list = []

    procedure_code_related_col = [col for col in patient_with_data.columns if 'procedure code' in col.lower()][0]
    
    for pat_index, patient_row in patient_with_data.iterrows():
        # Check if the session has available slots
        #for pat_index, patient_row in patients_not_scheduled.iterrows():
        sub_data_for_predictive_model = pd.DataFrame(patient_row).transpose().reset_index(drop=True)
        
        sub_data_for_predictive_model['Actual Procedure 1 Code 1'] = sub_data_for_predictive_model[procedure_code_related_col]

        if not patient_row[procedure_code_related_col] in data_relevant_to_trained_models['Actual Procedure 1 Code 1']:
            #print(f"{patient_row['Procedure Code']} not in training dataset")
            if patient_row[procedure_code_related_col] in new_or_ignored_data_4_model['Actual Procedure 1 Code 1'].tolist():
                result.loc[(patient_row['Patient ID'], list(result.columns))] = np.ceil(np.mean(new_or_ignored_data_4_model[new_or_ignored_data_4_model['Actual Procedure 1 Code 1'] == patient_row[procedure_code_related_col]]['H4 Minutes'])/5)*5
                #print(np.mean(new_or_ignored_data_4_model[new_or_ignored_data_4_model['Actual Procedure 1 Code 1'] == patient_row['Procedure Code']]['H4 Minutes']))
            elif 'Primary Procedure Code' in pred_model_related_preprocessed_data.columns and patient_row[procedure_code_related_col] in pred_model_related_preprocessed_data['Primary Procedure Code'].tolist():
                result.loc[(patient_row['Patient ID'], list(result.columns))] = np.ceil(np.mean(pred_model_related_preprocessed_data[pred_model_related_preprocessed_data['Primary Procedure Code'] == patient_row[procedure_code_related_col]]['H4 Minutes'])/5)*5

            elif 'Primary Procedure Code' in new_or_ignored_data_4_model.columns and patient_row[procedure_code_related_col] in new_or_ignored_data_4_model['Primary Procedure Code'].tolist():
                result.loc[(patient_row['Patient ID'], list(result.columns))] = np.ceil(np.mean(new_or_ignored_data_4_model[new_or_ignored_data_4_model['Primary Procedure Code'] == patient_row[procedure_code_related_col]]['H4 Minutes'])/5)*5
            
            elif 'Allocated Time' in patient_row.index and pd.notna(patient_row['Allocated Time']):
                result.loc[(patient_row['Patient ID'], list(result.columns))] = patient_row['Allocated Time']
            else:
                result.loc[(patient_row['Patient ID'], list(result.columns))] = np.nan
            
            continue
        
        for session_index, session_row in df_sessions.iterrows():          
            for col in varying_features:
                if col in df_sessions.columns:
                    sub_data_for_predictive_model.loc[0,col] = session_row[col]
    
            #get all the feature name for which new data has appeared against the model training data
            new_feature_data = []
            for feature in model_input_features[1:]:
                if not sub_data_for_predictive_model.loc[0,feature] in data_relevant_to_trained_models[feature]:
                    new_feature_data.append(feature)

            if len(new_feature_data) > 0:
                alternative_features_data = {}
                for feature in new_feature_data:
                    alternative_features_data[feature] = select_3_representative_data(pred_model_related_preprocessed_data, {'Actual Procedure 1 Code 1': patient_row[procedure_code_related_col]}, feature)
                for i in range(3):
                    for feature in new_feature_data:
                        sub_data_for_predictive_model.loc[0,feature] = alternative_features_data[feature][i]
                    input_data_df = pd.concat([input_data_df, sub_data_for_predictive_model[model_input_features]])
                    input_data_index_list.append((patient_row['Patient ID'], session_row['Session ID'], i+1))

            else:
                input_data_df = pd.concat([input_data_df, sub_data_for_predictive_model[model_input_features]])
                input_data_index_list.append((patient_row['Patient ID'], session_row['Session ID']))
                
    input_data_df.index = input_data_index_list

    #return input_data_df
    
    predictions_arr = np.zeros((len(input_data_df), len(model_types)))
    
    for model_index, model_type in enumerate(model_types):
        if 'Last10Mean' in model_type:
            memo = {}
            pred_temp = input_data_df.apply(lambda row: last10mean_with_memo(pred_model_related_preprocessed_data, memo, 'Actual Procedure 1 Code 1', 'Consultant Code', row['Actual Procedure 1 Code 1'], row['Consultant Code'], 'H4 Minutes'), axis=1)
        else:
            pred_temp = predict_the_time(model_type, input_data_df, model_input_features,  model_categ_dic, force_encoding_cols, 'H4 Minutes', trained_model_dir = trained_model_dir)
        #print(pred_temp)
        if model_type == 'BayesianRidge':
            pred_temp = pred_temp[0]
        predictions_arr[:,model_index] = np.maximum(15, pred_temp)

    series_temp = pd.Series(np.ceil(np.mean(predictions_arr, axis =1)/5)*5, index = pd.MultiIndex.from_tuples(input_data_index_list))

    result= pd.concat([result, create_df_from_multiindex_series(series_temp)])
    #result = create_df_from_multiindex_series(series_temp)

    #result = result.rename(columns = {col:'Session '+str(col)  for col in result.columns })
    
    return result.loc[patient_with_data['Patient ID']]
    #'''
    #return input_data_df


# Function to calculate percentage over +30 and under -30
def calculate_over_under_run_percentage(df_discrete, continuous_row, row_matching_id, reference_data_col, continuous_data_name, time_window):
    discrete_row = df_discrete[df_discrete[row_matching_id] == continuous_row[row_matching_id]].iloc[0]
    reference_time = discrete_row[reference_data_col]
    reference_time_low = reference_time - time_window
    reference_time_high = reference_time + time_window
    
    # Filter process_time columns from continuous_row
    process_times = [continuous_row[col] for col in continuous_row.index if col.startswith(continuous_data_name)]
    
    count_over_window = sum(time > reference_time_high for time in process_times)
    count_under_window = sum(time < reference_time_low for time in process_times)
    
    total = len(process_times)  # Total number of process_time columns
    
    percentage_over_window = (count_over_window / total) 
    percentage_under_window = (count_under_window / total)
    
    column_names = {'Chances overrun by {} min'.format(time_window): round(percentage_over_window,2),
                    'Chances underrun by {} min'.format(time_window): round(percentage_under_window,2)}
    
    return pd.Series(column_names)



def convert_schedule_to_binary_matrix(df, session_col, case_col, all_sessions = None, all_cases = None):
    """
    Convert a session-case matched schedule stored in a pandas DataFrame 
    into a binary numpy array with sessions as rows and cases as columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing session and case data.
    - session_col (str): The column name in `df` that contains session IDs.
    - case_col (str): The column name in `df` that contains case IDs.

    Returns:
    - binary_matrix (np.ndarray): A binary matrix where rows represent sessions and columns represent cases.
    - session_labels (list): A list of session labels corresponding to the rows of the binary matrix.
    - case_labels (list): A list of case labels corresponding to the columns of the binary matrix.
    """
    # Group cases by session
    grouped = df.groupby(session_col)[case_col].apply(list)

    # Get a list of all unique cases
    if all_cases is None:
        all_cases = sorted(set(df[case_col]))

    # Initialize a binary matrix with zeros
    binary_matrix = np.zeros((len(all_sessions), len(all_cases)), dtype=int)

    # Fill the binary matrix
    for session, cases in grouped.items():
        for case in cases:
            binary_matrix[all_sessions.index(session), all_cases.index(case)] = 1

    # Get the session labels
    session_labels = grouped.index.tolist()

    return binary_matrix





def create_schedule_and_upate_dataset(patients_df, sessions_available, predicted_times = None, procedure_time_prediction_parameters = None,  prep_min = 7.5, max_sess_uti = 0.90, session_random_selection = False, updated_sessions = None, failed_scheduling_patient = None, fully_booked_sessions = None):
    patients_df2 = patients_df.copy()

    procedure_code_related_col = [col for col in patients_df2.columns if 'procedure code' in col.lower()][0]
    
    patients_df2['Actual Procedure 1 Code 1'] = patients_df2[procedure_code_related_col]
    schedule = pd.DataFrame(columns=['Session ID', 'Patient ID', 'Scheduled Start Date/Time', 'Scheduled End Date/Time', 'Planned H4 Minutes'])

    if updated_sessions is None:
        updated_sessions = sessions_available.copy()

    if not 'Total Slot Minutes' in updated_sessions.columns:
        updated_sessions['Total Slot Minutes'] = (updated_sessions['Session Planned End Date/Time']-updated_sessions['Session Planned Start Date/Time']).dt.total_seconds() // 60
    updated_sessions['Total Minutes for Booking'] = np.floor(updated_sessions['Total Slot Minutes']*max_sess_uti)
    
    if not 'Remaining Slot Minutes' in updated_sessions.columns:
        updated_sessions['Remaining Slot Minutes'] = (updated_sessions['Session Planned End Date/Time']-updated_sessions['Session Planned Start Date/Time']).dt.total_seconds() // 60
    
    #if prep_min is not None:
    updated_sessions['Remaining Slot Minutes'] = updated_sessions['Remaining Slot Minutes'] - np.ceil(prep_min*1.25/5)*5
        #self.sessions_timeslot_df['H4 Minutes Booked'] = round(self.sessions_timeslot_df['Total Slots']-self.sessions_timeslot_df['Remaining Slot Minutes'])           

    #self.text_display.insert(tk.END, f"Schedule generation in progress\n")
    #self.update_text("Scheduling Process started...\n")
    if not 'H4 Minutes Booked' in updated_sessions.columns:
        updated_sessions['H4 Minutes Booked'] = 0

    if failed_scheduling_patient is None:
        failed_scheduling_patient = pd.DataFrame(columns= list(patients_df.columns)+['Reason for Failed Booking'])
    if fully_booked_sessions is None:
        fully_booked_sessions = pd.DataFrame(columns= updated_sessions.columns)

    consultant_code_col_exist = 'Consultant Code' in patients_df.columns

    model_trained_procedures = list(procedure_time_prediction_parameters[2]['Actual Procedure 1 Code 1'].unique())
    model_trained_consultants = list(procedure_time_prediction_parameters[2]['Consultant Code'].unique())
    
    for pat_index, patient_row in patients_df2.iterrows():
        # Check if the session has available slots
        #for pat_index, patient_row in patients_not_scheduled.iterrows():
        sub_data_for_predictive_model = pd.DataFrame(patient_row).transpose().reset_index(drop=True)
        
        #print(f"Scheduling for Patient: {patient_row['Patient ID']}, with anticipated Procedure: {sub_data_for_predictive_model.loc[0,'Actual Procedure 1 Code 1']}")

        if not patient_row['Actual Procedure 1 Code 1'] in model_trained_procedures and ('Allocated Time' not in patient_row.index or pd.isna(patient_row['Allocated Time'])) and (predicted_times is None or predicted_times.loc[patient_row['Patient ID']].isna().any()):
            print(f"The following sub-data is not relavant to trained model \n {patient_row['Actual Procedure 1 Code 1']}")
            #patients_not_scheduled = patients_not_scheduled.append(pd.DataFrame(patient_row).transpose(), ignore_index=True)
            failed_scheduling_patient.loc[pat_index] = patient_row
            failed_scheduling_patient.loc[pat_index, 'Reason for Failed Booking'] = 'Procedure code being new'
            continue

        possible_sessions = updated_sessions[~updated_sessions['Session ID'].isin(fully_booked_sessions['Session ID'])].copy()
        #print(len(possible_sessions))
        if len(possible_sessions) == 0:
            failed_scheduling_patient.loc[pat_index] = patient_row
            failed_scheduling_patient.loc[pat_index, 'Reason for Failed Booking'] = 'No slots available'
            continue

        # Shuffle the DataFrame rows
        if session_random_selection:
            possible_sessions = possible_sessions.sample(frac=1).reset_index(drop=True)
        
        if consultant_code_col_exist and not (patient_row['Consultant Code'] == 'Unknown' or patient_row['Consultant Code'] == 'nan'):
            if not patient_row['Consultant Code'] in possible_sessions['Consultant Code'].tolist():
                print(f"The  Patient {patient_row['Patient ID']} is not booked for any available session as no further session with anticipated Consultant {patient_row['Consultant Code']}")
                failed_scheduling_patient.loc[pat_index] = patient_row 
                failed_scheduling_patient.loc[pat_index, 'Reason for Failed Booking'] = 'Relevant Consulant not available'
                #patients_not_scheduled = patients_not_scheduled.append(pd.DataFrame(patient_row).transpose(), ignore_index=True)
                continue
            else:
                possible_sessions = possible_sessions[possible_sessions['Consultant Code'] == patient_row['Consultant Code']]
    
        for session_index, session_row in possible_sessions.iterrows():
            
            if predicted_times is not None and check_if_data_exists(predicted_times, patient_row['Patient ID'], session_row['Session ID']):
                pred_temp = predicted_times.loc[patient_row['Patient ID'], session_row['Session ID']]

            elif 'Allocated Time' in patient_row.index and pd.notna(patient_row['Allocated Time']):
                pred_temp = patient_row['Allocated Time']
                #print(pred_temp)
                
            else:
                #adjust the possible varying data as per the sessions
                for col in possible_varying_data_for_patient:
                    if col in possible_sessions.columns:
                        sub_data_for_predictive_model.loc[0,col] = session_row[col]
                        
                alternative_consultants = [sub_data_for_predictive_model.loc[0,'Consultant Code']]
                if not sub_data_for_predictive_model.loc[0,'Consultant Code'] in model_trained_consultants or sub_data_for_predictive_model.loc[0,'Consultant Code'] == 'nan':
                    #print(f"Consultant Code is changed since the following Consulant is not relavant to trained model \n {sub_data_for_predictive_model.loc[0,'Consultant Code']}")
                    #patients_not_scheduled = patients_not_scheduled.append(pd.DataFrame(patient_row).transpose(), ignore_index=True)
                    #sub_data_for_predictive_model['Consultant Code'] = 'C9999999'
                    #get three consultants who has done similar procedures before
                    alternative_consultants = select_3_representative_data(pred_model_related_preprocessed_data, {'Actual Procedure 1 Code 1': sub_data_for_predictive_model.loc[0,'Actual Procedure 1 Code 1']}, 'Consultant Code')
    
                # using predictive models to estimate procedure duration
                prediction_results = []
                #procedure_time_prediction_parameters = (models_types, data_model_categorisation, pred_model_related_preprocessed_data,
                #                                                              model_input_features_all, possible_varying_data_for_patient, force_encoding_cols, trained_models_direc)
                for consultant in alternative_consultants:
                    sub_data_for_predictive_model.loc[0,'Consultant Code'] = consultant
                    #for model_type in models_types:
                    for model_type in procedure_time_prediction_parameters[0]:
                        pred_temp = predict_the_time(model_type, sub_data_for_predictive_model, procedure_time_prediction_parameters[3], 
                                                     procedure_time_prediction_parameters[1], procedure_time_prediction_parameters[5], 'H4 Minutes', trained_model_dir = procedure_time_prediction_parameters[-1])
                        #pred_temp = predict_the_time(model_type, sub_data_for_predictive_model, model_input_features_all, 
                        #                             data_model_categorisation, force_encoding_cols, 'H4 Minutes', trained_model_dir = trained_models_direc)
                        
                        #print(pred_temp)
                        if model_type == 'BayesianRidge':
                            pred_temp = pred_temp[0]
                        
                        prediction_results.append(round(max(15,pred_temp.item()),2))
                #break
                
                #schedule = pd.DataFrame(columns=['Session ID', 'Patient ID', 'Scheduled Start Date/Time', 'Scheduled End Date/Time', 'Planned H4 Minutes'])
                pred_temp = np.ceil(np.mean(prediction_results)/5)*5
            if pred_temp > session_row['Remaining Slot Minutes'] or pred_temp > session_row['Total Minutes for Booking'] - session_row['H4 Minutes Booked']:
                if session_index == len(possible_sessions)-1:
                    #print(f"No session time is available for {patient_row['Patient ID']}")
                    #patients_not_scheduled = patients_not_scheduled.append(pd.DataFrame(patient_row).transpose(), ignore_index=True)
                    failed_scheduling_patient.loc[pat_index] = patient_row
                    failed_scheduling_patient.loc[pat_index, 'Reason for Failed Booking'] = 'No slots available'
                    
            else:
                theatre_available_from = session_row['Session Planned End Date/Time'] - pd.Timedelta(minutes=session_row['Remaining Slot Minutes'])
                # Update the schedule with the patient
                schedule.loc[len(schedule)] = {
                    'Session ID': session_row['Session ID'], 'Patient ID': patient_row['Patient ID'],
                    'Scheduled Start Date/Time': theatre_available_from.strftime('%Y-%m-%d %H:%M'), 
                    'Scheduled End Date/Time': (theatre_available_from + pd.Timedelta(minutes=pred_temp)).strftime('%Y-%m-%d %H:%M'),
                    'Planned H4 Minutes': pred_temp}
    
                
                updated_sessions.loc[updated_sessions['Session ID'] == session_row['Session ID'], 'H4 Minutes Booked'] = round(session_row['H4 Minutes Booked'] + pred_temp)
                
                #updating remaining time for each session
                if pred_temp + prep_min > session_row['Remaining Slot Minutes']:
                    updated_sessions.loc[updated_sessions['Session ID'] == session_row['Session ID'], 'Remaining Slot Minutes'] = round(session_row['Remaining Slot Minutes']- pred_temp)
                else:
                    updated_sessions.loc[updated_sessions['Session ID'] == session_row['Session ID'], 'Remaining Slot Minutes'] = round(session_row['Remaining Slot Minutes']- np.ceil(pred_temp + prep_min))
                
                if session_row['Remaining Slot Minutes'] - pred_temp - prep_min < 10 or session_row['Total Minutes for Booking'] - session_row['H4 Minutes Booked'] - pred_temp < 10:
                    #sessions_fully_booked = sessions_fully_booked.append(pd.DataFrame(session_row).transpose(), ignore_index=True)
                    fully_booked_sessions.loc[len(fully_booked_sessions)] = session_row
                    fully_booked_sessions.loc[fully_booked_sessions['Session ID'] == session_row['Session ID'], 'Remaining Slot Minutes'] = round(session_row['Remaining Slot Minutes']- pred_temp)
                    fully_booked_sessions.loc[fully_booked_sessions['Session ID'] == session_row['Session ID'], 'H4 Minutes Booked'] = round(session_row['H4 Minutes Booked'] + pred_temp)
              
                break
    #updated_sessions['H4 Minutes Booked'] = round(updated_sessions['Total Minutes for Booking'] - updated_sessions['Remaining Slot Minutes'])
    updated_sessions['Session Utilisation'] = round(updated_sessions['H4 Minutes Booked']/updated_sessions['Total Slot Minutes'],2)
            
    schedule = pd.merge(schedule, patients_df2[['Patient ID', procedure_code_related_col]], on = 'Patient ID')
    if 'Theatre Name' in sessions_available.columns:
        schedule = pd.merge(schedule, sessions_available[['Session ID', 'Consultant Code', 'Theatre Name']], on = 'Session ID')
    else:
        schedule = pd.merge(schedule, sessions_available[['Session ID', 'Consultant Code']], on = 'Session ID')
    
    #return schedule.sort_values(by=['Session ID','Scheduled Start Date/Time']).reset_index(drop=True), updated_sessions, failed_scheduling_patient, fully_booked_sessions
    return schedule.reset_index(drop=True), updated_sessions, failed_scheduling_patient, fully_booked_sessions



def data_filter_based_prediction(model_type, training_data_set, X_input_df, must_match_columns_prioritywise, scaling_factors):
    if model_type == 'StochasticNormalDistribution':
        return gaussan_based_prediction(model_type, training_data_set, X_input_df, must_match_columns_prioritywise, scaling_factors)
    


    
def encoding_input_data_and_reindexing_as_reference(input_data, force_encoding_cols, input_data_format, similar_cols_for_encoding = None):
    
    #input_data_encoded = pd.get_dummies(input_data)
    #print(input_data)
    if not similar_cols_for_encoding is None:
        input_data_encoded = one_hot_encoding_for_similar_columns(input_data, similar_cols_for_encoding)
    else:
        input_data_encoded = input_data.copy()
        
    input_data_encoded = one_hot_encode_dataframe(input_data_encoded, force_encoding_cols)
    #print(input_data_encoded)
    #print(f'The type of input data encoded data is: {type(input_data_encoded)}')
    
    not_found = [element for element in input_data_encoded.columns if element not in input_data_format.columns]
    
    #print(f'the col not found in trained model: {not_found}')
    
    if len(not_found) > 0:
        
        for unfound in not_found:
            if '_' in unfound:
                related_col_before_encoding = unfound.replace('_'+ unfound.split('_')[-1], '')
                
                #fill the one_hot econded column Unknown with one as replacement
                if related_col_before_encoding+'_Unknown' in input_data_format.columns:
                    input_data_encoded[related_col_before_encoding+'_Unknown'] = 1
                else:
                    #fill the one_hot econded features by the value 1/length
                    #now lets give some value to the data
                    related_one_hot_columns = [col for col in input_data_format.columns if col.startswith(related_col_before_encoding) and col not in input_data_encoded.columns]
                    
                    #related_one_hot_columns = [col for col in related_one_hot_columns if input_data_encoded[col]==0]
                    
                    if len(related_one_hot_columns) > 0:
                        #remove the encoded one as it is not in the model trained format list
                        input_data_encoded = input_data_encoded.drop(unfound, axis=1)
                        
                        value_to_assign = 1/len(related_one_hot_columns)
                        #input_data_encoded = pd.concat([input_data_encoded] + [pd.Series(value_to_assign)] * len(related_one_hot_columns), axis=1)
                        input_data_encoded = pd.concat([input_data_encoded, pd.DataFrame({col:[value_to_assign] * len(input_data_encoded) for col in related_one_hot_columns})], axis = 1)
                        #print(f'The feature {unfound} not in trained model and encodingly distributed to all related features')
                    
                    
            else:
                raise ValueError(f"Model was not trained for following features:{unfound}")
    
    #print(f'Econded and aligned data before changing the format: {input_data_encoded}') 
    
    input_data_encoded = input_data_encoded.reindex(columns = input_data_format.columns, fill_value=0)
    #re_converting is done to avoid the conversion of data_type to list or array for particular columns
    
    if len(input_data_encoded) > 1:
        return input_data_encoded
    else:
        return pd.DataFrame(input_data_encoded.to_dict())


def find_file_containing_strings(directory, search_strings):
    """
    Finds a file in a directory that contains all the given strings in its filename.

    Args:
        directory: The directory to search.
        search_strings: A list of strings that the filename must contain.

    Returns:
        The full path of the first matching file, or None if no match is found.
    """
    try:
        for filename in os.listdir(directory):
            if all(search_string.lower() in filename.lower() for search_string in search_strings): #case insensitive check
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path): #ensure it's a file.
                    return full_path
        return None  # No matching file found
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def generate_categorical_time_series(df, category_column, add_column, minus_column):
    df1 = df.copy()
    # Convert date columns to datetime format
    df1[add_column] = pd.to_datetime(df1[add_column])
    df1[minus_column] = pd.to_datetime(df1[minus_column])

    # Create a dictionary to store the sorted time series data
    sorted_data = {}

    # Iterate over each row in the DataFrame
    for index, row in df1.iterrows():
        # Get the category value from the specified column
        category = row[category_column]

        # If the category does not exist in the dictionary, create a new list
        if category not in sorted_data:
            sorted_data[category] = []

        # Append column A date and value +1 as a tuple to the respective category list
        sorted_data[category].append((row[add_column], 1))
        # Append column B date and value -1 as a tuple to the respective category list
        sorted_data[category].append((row[minus_column], -1))

    # Sort each category list based on the date
    for category, data in sorted_data.items():
        sorted_data[category] = sorted(data, key=lambda x: x[0])

    return sorted_data




def generate_chances_for_tasks_durations(patient_with_data, df_sessions, variability_counts, models_types, model_categ_dic, 
                                             pred_model_related_preprocessed_data, model_input_features, varying_features, force_encoding_cols, trained_model_dir):
    #(df_cases, df_sessions, patient_data, model_types, model_categ_dic, pred_model_related_preprocessed_data, model_input_features, varying_features, force_encoding_cols, trained_model_dir):
    #fixed_data_for_patient = [feat for feat in model_input_features if feat not in varying_features]
    
    #patient_with_data = update_patient_data(df_cases, patient_data, fixed_data_for_patient)
    result = {}
    proce_related_col = [col for col in patient_with_data.columns if 'procedure' in col.lower()][0]
    for pat_index, patient_row in patient_with_data.iterrows():
        # Check if the session has available slots
        #for pat_index, patient_row in patients_not_scheduled.iterrows():
        sub_data_for_predictive_model = pd.DataFrame(patient_row).transpose().reset_index(drop=True)
        sub_data_for_predictive_model['Actual Procedure 1 Code 1'] = sub_data_for_predictive_model[proce_related_col]
        for session_index, session_row in df_sessions.iterrows():          
            for col in varying_features:
                if col in df_sessions.columns:
                    sub_data_for_predictive_model.loc[0,col] = session_row[col]
            
            
            # using predictive models to estimate procedure duration
            for model_type in models_types:
                #prediction_results = []

                parameters_for_procedure_time_predictions = [model_type, sub_data_for_predictive_model, model_input_features, model_categ_dic, force_encoding_cols]

                _, res_temp = compute_prediction_and_get_residuals_for_similar_prediction(sub_data_for_predictive_model, variability_counts, parameters_for_procedure_time_predictions, 'H4 Minutes', 
                                                                         preprocessed_dataset = pred_model_related_preprocessed_data, residuals = True, trained_model_dir = trained_model_dir)
                
                result[(patient_row['Patient ID'], session_row['Session ID'], model_type)] = [1/(1+resi) for resi in res_temp]
    return result



def generate_procedure_time_chances_for_consultants(patient_with_data, df_sessions, variability_counts, models_types, model_categ_dic, pred_model_related_preprocessed_data, removed_data_from_processing,
                                              model_input_features, varying_features, force_encoding_cols, trained_model_dir, categorised_modelling = False):
    #(df_cases, df_sessions, patient_data, model_types, model_categ_dic, pred_model_related_preprocessed_data, model_input_features, varying_features, force_encoding_cols, trained_model_dir):
    #fixed_data_for_patient = [feat for feat in model_input_features if feat not in varying_features]
    
    #patient_with_data = update_patient_data(df_cases, patient_data, fixed_data_for_patient)
    result = {}
    #return pred_model_related_preprocessed_data.loc[selected_indices]
    
    proce_related_col = [col for col in patient_with_data.columns if 'procedure' in col.lower()][0]
    #at first lets store similar dataset indexes in the result
    samples_index_list = {}
    for pat_index, patient_row in patient_with_data.iterrows():
        # Check if the session has available slots
        #for pat_index, patient_row in patients_not_scheduled.iterrows():
        sub_data_for_predictive_model = pd.DataFrame(patient_row).transpose().reset_index(drop=True)
        
        sub_data_for_predictive_model['Actual Procedure 1 Code 1'] = sub_data_for_predictive_model[proce_related_col]

        if not patient_row[proce_related_col] in list(pred_model_related_preprocessed_data['Actual Procedure 1 Code 1']):
            if patient_row[proce_related_col] in list(removed_data_from_processing['Actual Procedure 1 Code 1']):
                t_temp = list(removed_data_from_processing[removed_data_from_processing['Actual Procedure 1 Code 1'] == patient_row[proce_related_col]]['H4 Minutes'])
            elif 'Primary Procedure Code' in pred_model_related_preprocessed_data.columns:
                
                if patient_row[proce_related_col] in list(pred_model_related_preprocessed_data['Primary Procedure Code' ]):
                    t_temp = list(pred_model_related_preprocessed_data[pred_model_related_preprocessed_data['Primary Procedure Code'] == patient_row[proce_related_col]]['H4 Minutes'])
                    
                elif patient_row[proce_related_col] in list(removed_data_from_processing['Primary Procedure Code']):
                    t_temp = list(removed_data_from_processing[removed_data_from_processing['Primary Procedure Code'] == patient_row[proce_related_col]]['H4 Minutes'])
                else:
                    t_temp = []
                    print(f"{patient_row[proce_related_col]} not in past dataset")
            else:
                t_temp = []
                print(f"{patient_row[proce_related_col]} not in past dataset")
                
            for session_index, session_row in df_sessions.iterrows():          
                result[(patient_row[proce_related_col], session_row['Consultant Code'])] = [t_/max(np.mean(t_temp),1) for t_ in t_temp]
            
            continue
        
        for session_index, session_row in df_sessions.iterrows():          
            
            if (patient_row[proce_related_col], session_row['Consultant Code']) in samples_index_list:
                continue
                
            for col in varying_features:
                if col in df_sessions.columns:
                    sub_data_for_predictive_model.loc[0,col] = session_row[col]
            
            # using first predictive models to obtain similar dataset and its residuals
            sub_data_for_predictive_model = sub_data_for_predictive_model[model_input_features]
            
            sub_data_for_filtering = sub_data_for_predictive_model.copy()
            if 'Hosp' in sub_data_for_filtering.columns:
                sub_data_for_filtering = sub_data_for_filtering.drop(columns = ['Hosp'])
            
            similar_dataset_temp, _ =  filter_dataset_till_possible(pred_model_related_preprocessed_data.copy(), 0, sub_data_for_filtering, variability_counts)
            samples_index_list[(patient_row[proce_related_col], session_row['Consultant Code'])] = list(similar_dataset_temp.index)
            # to make sure the flow of the keywords
            result[(patient_row[proce_related_col], session_row['Consultant Code'])] = []

    # Now lets prepare the data for prediction
    selected_indices = set()
    for index_list in samples_index_list.values():
        
        selected_indices.update(index_list)

    selected_indices = list(selected_indices)
    #return pred_model_related_preprocessed_data

    models_types_filtered = [_ for _ in models_types if not 'mean' in _.lower()]
    
    #return selected_indices
    #return pred_model_related_preprocessed_data.loc[selected_indices]
    prediction_all_models = np.zeros((len(selected_indices), len(models_types_filtered)))

    for model_index, model_type in enumerate(models_types_filtered):
        pred_temp = predict_the_time(model_type, pred_model_related_preprocessed_data.loc[selected_indices].reset_index(drop= True), model_input_features, model_categ_dic, force_encoding_cols, 'H4 Minutes', trained_model_dir = trained_model_dir)

        if model_type == 'BayesianRidge':
            pred_temp = pred_temp[0]

        prediction_all_models[:, model_index] = np.maximum(15, pred_temp)

    related_actual_pred_ratio = pred_model_related_preprocessed_data.loc[selected_indices]['H4 Minutes']/np.mean(prediction_all_models, axis=1)

    related_actual_pred_ratio = related_actual_pred_ratio.clip(0.63,1.5)

    print(f'selected indices in total were {len(related_actual_pred_ratio)}')
    print(f'ratio greater than 1 in total were {len([_ for _ in related_actual_pred_ratio if _ > 1])}')
    

    for key in samples_index_list.keys():
        if len(samples_index_list[key]) == 0:
            continue

        #mask = pred_model_related_preprocessed_data.loc[list(selected_indices)].index.isin(samples_index_list[key])
        #predictions_selected = prediction_all_models[mask]
        #related_y_value = pred_model_related_preprocessed_data.loc[list(selected_indices)][mask]['H4 Minutes']

        #print(predictions_all_models)

        #res_temp = np.mean(predictions_all_models, axis=1)/related_y_value - 1
        
        #result[(patient_row[proce_related_col], session_row['Consultant Code'])] = [1/(1+resi) for resi in res_temp]
        #result[key] = list(related_y_value/np.mean(predictions_selected, axis=1))
        result[key] = list(related_actual_pred_ratio.loc[samples_index_list[key]])
        
    return result


def get_prediction_required_parameters(working_dir, model_training_dir):
    if not os.path.exists(os.path.join(model_training_dir, 'ML_models_data_related_parameters.pkl')):
        # Get the current working directory to restore it later
        original_cwd = os.getcwd()
    
        # Change the current working directory
        os.chdir(model_training_dir)
        #code_dir = 'Z:\Capacity & Analytics\Business Informatics\Analytics Team\Surgical Elective Patient Scheduling Model\Theatre_Procedure_Time_Predictive_Modelling\Final_version_codes'
        code_dir = r"\\esneft.nhs.uk\share\Finance\Capacity & Analytics\Business Informatics\Analytics Team\Surgical Elective Patient Scheduling Model\Theatre_Procedure_Time_Predictive_Modelling\Final_version_codes"
        # Run the Python script using subprocess
        subprocess.run(['python',os.path.join(code_dir, 'data_preparing_module.py')], check=True) #check=True will raise an exception if the subprocess returns a non-zero exit code.
        
        os.chdir(original_cwd )
        
    with open(os.path.join(model_training_dir, 'ML_models_data_related_parameters.pkl'), 'rb') as f:
        ML_models_data_related_parameters = pickle.load(f)
    
    models_considred = get_selected_options(os.path.join(working_dir, 'prediction_model_options.txt'))

    if not models_considred:
        models_considred = ['Last10Mean']
    
    #all_features_with_identifying_name = [procedure_related_col if x == 'Actual Procedure 1 Code 1' else x for x in model_input_features_all]

    #the data not influenced by patient and the procedure but due to planning strategy
    possible_varying_data_for_patient = ['Hosp','Consultant Code', 'Day of the week', 'Covid Flag', 'Theatre Suite Name']
    possible_varying_data_for_patient = [_ for _ in possible_varying_data_for_patient if _ in ML_models_data_related_parameters[3]]
    
    temp_list = [models_considred, *ML_models_data_related_parameters, possible_varying_data_for_patient, os.path.join(model_training_dir,'Trained_models')]
    temp_list[-3], temp_list[-2] = temp_list[-2], temp_list[-3]

    return tuple(temp_list)


def get_selected_options(filepath):
    """Reads the options file and returns a list of selected options."""
    selected_options = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                option = line.strip()  # Remove leading/trailing whitespace and newlines
                if option and not option.startswith('#'):  # Ignore empty lines
                    selected_options.append(option)
        return selected_options
    except FileNotFoundError:
        print(f"Error: Options file not found: {filepath}")
        return []
    except Exception as e:
        print(f"Error reading options file: {e}")
        return []



def last10mean_with_memo(df, memo, colA, colB, category_a, category_b, value_col='y'):
    """
    Calculate the Last10Mean of `value_col` for rows matching `category_a` and `category_b`.
    Column names for A, B, and y are passed as parameters.
    Use memoization to avoid recomputing for repeated combinations of category_a and category_b.
    """
    # Check if the result is already in memo
    if (category_a, category_b) in memo:
        return memo[(category_a, category_b)]
    
    # Compute Last10Mean for new combination
    matched_ab = df[(df[colA] == category_a) & (df[colB] == category_b)]
    
    if len(matched_ab) >= 10:
        result = matched_ab[value_col].iloc[-10:].mean()
    else:
        matched_a = df[df[colA] == category_a]
        combined_rows = pd.concat([matched_ab, matched_a]).drop_duplicates().iloc[:10]
        
        result = combined_rows[value_col].mean()
    
    # Save the result in the memo dictionary
    memo[(category_a, category_b)] = result  # Store the computed result in memo
    return result



    
def load_timeslot_file(timeslot_file_path):
    # Add columns and values for predictive model required features  
    sessions_timeslot_df = pd.read_excel(timeslot_file_path)
    '''
    sessions_timeslot_df['Day of the week'] = sessions_timeslot_df['Session Planned Start Date/Time'].apply(lambda x: x.day_name())
    sessions_timeslot_df['Covid Flag'] = 'post-covid'
    if 'Theatre Suite Name' not in sessions_timeslot_df.columns and 'Theatre Name' in sessions_timeslot_df.columns:
        sessions_timeslot_df['Theatre Suite Name'] = sessions_timeslot_df['Theatre Name'].apply(lambda x: 'Elmstead Theatres' if x in ['Theatre 05', 'Theatre 5'] else 'Constable Theatres')
    '''
    sessions_timeslot_df['Total Slot Minutes'] = (sessions_timeslot_df['Session Planned End Date/Time'] - sessions_timeslot_df['Session Planned Start Date/Time']).dt.total_seconds() // 60
    sessions_timeslot_df['Remaining Slot Minutes'] = (sessions_timeslot_df['Session Planned End Date/Time'] - sessions_timeslot_df['Session Planned Start Date/Time']).dt.total_seconds() // 60
    sessions_timeslot_df['H4 Minutes Booked'] = 0
    
    return sessions_timeslot_df



def obtain_residual_based_possibilities_for_plan(varibility_counts, patient_session_matching, session_booking, patient_df, time_predicting_parameters = None, categorised_modelling = False, min_count = 10, prep_min = 5):

    models_types, data_model_categorisation, pred_model_related_preprocessed_data, model_input_features_all, possible_varying_data_for_patient, force_encoding_cols, trained_models_direc = time_predicting_parameters
    
    multi_plan_df = patient_session_matching[['Session ID', 'Patient ID', 'Procedure Code']].copy()
    multi_plan_df = pd.merge(multi_plan_df, session_booking[['Session ID','Theatre Name']], on = 'Session ID')
    
    new_plan_cols = ['Session ID', 'Patient ID','Scheduled Start Date/Time', 'Scheduled End Date/Time', 'Planned H4 Minutes']

    all_features_with_identifying_name = ['Procedure Code' if x == 'Actual Procedure 1 Code 1' else x for x in model_input_features_all]
    
    fixed_data_for_patient = [col for col in all_features_with_identifying_name if col not in possible_varying_data_for_patient]

    patients_booked_with_input_data = pd.merge(patient_session_matching[['Session ID', 'Patient ID', 'Procedure Code']], patient_df[['Patient ID']+list(fixed_data_for_patient)], on = ['Patient ID', 'Procedure Code'])
    
    patients_booked_with_input_data['Actual Procedure 1 Code 1'] = patients_booked_with_input_data['Procedure Code']
    
    multi_sessions_df = session_booking[['Session ID','Theatre Name', 'Session Planned Start Date/Time','Session Planned End Date/Time', 'Total Slot Minutes']].copy()

    data_relevant_to_trained_models = {col: list(pred_model_related_preprocessed_data[col].unique()) for col in model_input_features_all}
    
    normalised_residuals = {}
    
    residuals_passed_itr_without_resampling = 0
    model_type = None
    
    parameters_for_procedure_time_predictions = [model_type, None, model_input_features_all, data_model_categorisation, force_encoding_cols]
    
    proce_time_given = 'Planned H4 Minutes' in patient_session_matching.columns
    
    for i in range(varibility_counts):

        print(f"{i}th iteration processing")
    
        new_plan = pd.DataFrame(columns=['Session ID', 'Patient ID', 'Scheduled Start Date/Time', 'Scheduled End Date/Time', 'Planned H4 Minutes'])
    
        if i%(varibility_counts//len(models_types)) == 0 and model_type != models_types[-1]:
            model_type = models_types[i // (varibility_counts // len(models_types))]
            parameters_for_procedure_time_predictions[0] = model_type
            normalised_residuals = {}
            residuals_passed_itr_without_resampling = 0
            
        #if residuals_passed_itr_without_resampling > 10:
        #   normalised_residuals = {}
        #    residuals_passed_itr_without_resampling = 0
    
        updated_sessions = session_booking.copy()
    
        for pat_index, patient_row in patients_booked_with_input_data.iterrows():
            # Check if the session has available slots
            #for pat_index, patient_row in patients_not_scheduled.iterrows():
            sub_data_for_predictive_model = pd.DataFrame(patient_row).transpose().reset_index(drop=True)
            #print(sub_data_for_predictive_model)
            session_row = updated_sessions[updated_sessions['Session ID'] == patient_row['Session ID']]
            session_row = session_row.reset_index(drop = True)

            if not patient_row['Procedure Code'] in data_relevant_to_trained_models['Actual Procedure 1 Code 1']:
                result.loc[(patient_row['Patient ID'], list(result.columns))] = 'nan' 
                continue

            #adjust the possible varying data as per the sessions
            for col in possible_varying_data_for_patient:
                if col in updated_sessions.columns:
                    sub_data_for_predictive_model.loc[0,col] = session_row.loc[0,col]

            residual_ID_tag = (patient_row['Patient ID'], patient_row['Procedure Code'], model_type)
              
            if proce_time_given:
                
                pred_temp = patient_session_matching.loc[pat_index,'Planned H4 Minutes']

                if residual_ID_tag not in normalised_residuals or len(normalised_residuals[residual_ID_tag]) == 0:
                    
                    parameters_for_procedure_time_predictions[1] = sub_data_for_predictive_model
                    
                    res_temp = get_residuals_for_similar_prediction(sub_data_for_predictive_model, varibility_counts, parameters_for_procedure_time_predictions, 'H4 Minutes', categorised_modelling = categorised_modelling, 
                                                         preprocessed_dataset = pred_model_related_preprocessed_data, trained_model_dir = trained_models_direc)

                    if res_temp is not None:
                        normalised_residuals[residual_ID_tag] = res_temp
                else:
                    res_temp = normalised_residuals[residual_ID_tag]
            else:
                #get all the feature name for which new data has appeared against the model training data
                new_feature_data = []
                for feature in model_input_features_all[1:]:
                    if not sub_data_for_predictive_model.loc[0,feature] in data_relevant_to_trained_models[feature]:
                        new_feature_data.append(feature)
    
                if len(new_feature_data) > 0:
                    alternative_features_data = {}
                    for feature in new_feature_data:
                        alternative_features_data[feature] = select_3_representative_data(pred_model_related_preprocessed_data, {'Actual Procedure 1 Code 1': patient_row['Procedure Code']}, feature)
                    #for i in range(3):
                    for feature in new_feature_data:
                        sub_data_for_predictive_model.loc[0,feature] = alternative_features_data[feature][-1]
        
                parameters_for_procedure_time_predictions[1] = sub_data_for_predictive_model
                
                #pred_temp, res_temp = predict_the_time(model_type, sub_data_for_predictive_model, model_input_features_all, data_model_categorisation, force_encoding_cols, 'H4 Minutes', trained_model_dir = trained_models_direc)
        
                if residual_ID_tag in normalised_residuals and len(normalised_residuals[residual_ID_tag]) > 0:
                    res_temp = normalised_residuals[residual_ID_tag]
                    
                    pred_temp, _ = compute_prediction_and_get_residuals_for_similar_prediction(sub_data_for_predictive_model, 0, parameters_for_procedure_time_predictions, 'H4 Minutes', 
                                                                                            residuals = False, trained_model_dir= trained_models_direc)
                else:
                    pred_temp, res_temp = compute_prediction_and_get_residuals_for_similar_prediction(sub_data_for_predictive_model, varibility_counts, parameters_for_procedure_time_predictions, 'H4 Minutes', 
                                                                                     preprocessed_dataset = pred_model_related_preprocessed_data, residuals = True, trained_model_dir = trained_models_direc)
                    
                    if res_temp is not None:
                        normalised_residuals[residual_ID_tag] = res_temp
           
            if res_temp is None:
                error_correc = 0
            else:
                random_res = random.choice(res_temp)
                
                normalised_residuals[residual_ID_tag].remove(random_res)
                
                random_res = max(min(random_res, 0.5), -0.5)
            
            pred_temp = round(pred_temp/(1+random_res))
            
            residuals_passed_itr_without_resampling+=1

            #print([patient_row['Patient ID'], pred_temp, normalised_residuals[residual_ID_tag]])
    
            theatre_available_from = session_row.loc[0,'Session Planned Start Date/Time'] + pd.Timedelta(minutes=session_row.loc[0,'H4 Minutes Booked'])
            # Update the schedule with the patient
            new_plan.loc[len(new_plan)] = {
                'Session ID': session_row.loc[0,'Session ID'], 'Patient ID': patient_row['Patient ID'],
                'Scheduled Start Date/Time': theatre_available_from.strftime('%Y-%m-%d %H:%M'), 
                'Scheduled End Date/Time': (theatre_available_from + pd.Timedelta(minutes=pred_temp)).strftime('%Y-%m-%d %H:%M'),
                'Planned H4 Minutes': pred_temp}
    
            #updating remaining time for each session
            #'H4 Minutes Booked'
            updated_sessions.loc[updated_sessions['Session ID'] == session_row.loc[0,'Session ID'], 'H4 Minutes Booked'] = session_row.loc[0,'H4 Minutes Booked']+ pred_temp +prep_min
            #updated_sessions.loc[updated_sessions['Session ID'] == session_row.loc[0,'Session ID'], 'Remaining Slots'] = session_row.loc[0,'Remaining Slots']- pred_temp
       
        multi_plan_df =pd.merge(multi_plan_df, new_plan.add_suffix(f'_{i+1}') , left_on=['Session ID','Patient ID'], right_on= [f'Session ID_{i+1}', f'Patient ID_{i+1}'], how="left")
        multi_plan_df  = multi_plan_df.drop(columns = [f'Session ID_{i+1}', f'Patient ID_{i+1}'])
    
        multi_sessions_df =pd.merge(multi_sessions_df, updated_sessions[['Session ID', 'H4 Minutes Booked']].add_suffix(f'_{i+1}') , left_on=['Session ID'], right_on= [f'Session ID_{i+1}'], how="left")
        multi_sessions_df  = multi_sessions_df.drop(columns = [f'Session ID_{i+1}'])
        
    #return multi_plan_df.sort_values(by=['Session ID']).reset_index(drop=True), multi_sessions_df.sort_values(by=['Session ID']).reset_index(drop=True)
    return multi_plan_df, multi_sessions_df


def plot_theatre_occupancy_data(patient_flow_df, figure_size, label_mapping, color_mapping_col = None, grouping_col = None, ax = None, sub_process_time_considered = False, extra_y_label = None):
    
    columns_names = patient_flow_df.columns.to_list()
    #print(columns_names)
    def assign_color(label):
        #color_palette = cl.scales['12']['qual']['Set3']  # Set3 palette with 12 colors
        # Dictionary to store label-color mappings
        if label not in label_mapping:
            # Assign a new color for the label
            color = tuple(random.random() for _ in range(3))
            label_mapping[label] = color
    
        return label_mapping[label]
    
    df = patient_flow_df.copy()
    # Convert Start Time and End Time columns to datetime
    start_time_col = columns_names[0]
    #df['Start Time'] = pd.to_datetime(df[columns_names[0]])
    df[start_time_col] = pd.to_datetime(df[start_time_col])
    procedure_end_time_given = 'date/time'in columns_names[1].lower() or 'end' in columns_names[1].lower()
    if procedure_end_time_given:
        end_time_col = columns_names[1]
        #df['End Time'] = pd.to_datetime(df[columns_names[1]]) 
        df[end_time_col] = pd.to_datetime(df[end_time_col]) 
    
        if sub_process_time_considered:
            df['End Time 2'] = pd.to_datetime(df[columns_names[2]])

    # Sort the DataFrame by Start Time
    #df.sort_values('Start Time', inplace=True)
    df = df.sort_values(by=[start_time_col])
    # Set up the figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size) 
    
    # Set the height of each day's frame
    day_height = 1
    plot_height = 0.8

    # Iterate over each day
    if grouping_col is None:
        grouping_data = df[start_time_col].dt.date.unique()
    else:
        grouping_data = df[grouping_col].unique()
        y_ticks2 = []

    y_ticks = []
    
    
    for i, group_val in enumerate(grouping_data):
        # Filter the DataFrame for the current day
        filtered_df = df[df[start_time_col].dt.date == group_val].copy() if grouping_col is None else df[df[grouping_col] == group_val].copy()
        
        filtered_df = filtered_df.reset_index(drop=True)
        #print(filtered_df)
        # Calculate the y-coordinate for the day's frame
        y = len(y_ticks) * day_height

        # Iterate over each procedure of the current day
        
        duration_updated_x = 0
        row_count = 0
        #count_on_day = 0
        for index, row in filtered_df.iterrows():
            #count_on_day = 0
            #print(index)
            # Calculate the duration of the procedure
            if procedure_end_time_given:
                duration = row[end_time_col] - row[start_time_col]
                duration = duration.total_seconds()/60 -1
                if sub_process_time_considered:
                    duration2 = row['End Time 2'] - row[end_time_col]
                    duration2 = duration2.total_seconds()/60 -1
                else:
                    duration2 = 0
               
            else:
                duration = row[columns_names[1]]-1
                if sub_process_time_considered:
                    duration2 = row[columns_names[2]]-1
                else:
                    duration2 = 0
               
                #print(duration)
            #print(duration)
            # Calculate the x-coordinate for the procedure's bar
            x = row[start_time_col].hour*60 + row[start_time_col].minute
            if procedure_end_time_given or x -duration_updated_x > 2.5:
                duration_updated_x = x
            
            #if given time is about session start we have to update starting x
            #elif duration_updated_x > x :
            #    duration_updated_x 
            else:
                x = duration_updated_x+1
                #x = duration_updated_x
                #print(duration)
            
            if color_mapping_col is None:
                color_for_procedure = assign_color(row[columns_names[-1]])
            else:
                color_for_procedure = assign_color(row[color_mapping_col])
                
            ax.barh(y,  duration, height = plot_height, left=x, color=color_for_procedure, edgecolor='black')
            
            if sub_process_time_considered:
                ax.barh(y- extra_y_reduce,  duration2, height = plot_height, left=x+ duration+1, color=color_for_procedure, edgecolor='black')

            barh_brightness = 0.299 * color_for_procedure[0] + 0.587 * color_for_procedure[1] + 0.114 * color_for_procedure[2]

            if barh_brightness < 0.4 and duration + duration2 >= 20:
                text_color = (0.9, 0.9, 0.9)
            else:
                text_color = (0.1, 0.1, 0.1)
            # Add procedure name as a label
            label_starting_col = 3 if sub_process_time_considered else 2
            if len(columns_names) >label_starting_col+1:
                ax.text(x + (duration + duration2)/ 2, y - plot_height / 4, row[columns_names[label_starting_col+1]],
                    ha='center', va='center', color=text_color)
            
            if len(columns_names) >label_starting_col:
                ax.text(x + (duration + duration2)/ 2, y  +  plot_height / 4, row[columns_names[label_starting_col]],
                    ha='center', va='center', color=text_color)
            
            duration_updated_x += duration + duration2
            row_count+=1
        #print(filtered_df.loc[0, start_time_col].date())
        y_ticks.append(group_val)
        if grouping_col is not None:
            y_ticks2.append(row[start_time_col].date())

    # Set y-axis limits and labels for each day
    #ax.set_yticks([j * day_height + plot_height/ 2 for j in range(len(df['Start Time'].dt.date.unique()))])
    #ax.set_yticks(range(len(df['Start Time'].dt.date.unique())))
    ax.set_yticks(range(len(y_ticks)))
    ax.set_yticklabels(y_ticks)
    ax.set_ylabel('Sessions')
    #ax.set_ylim(plot_height-day_height, len(df['Start Time'].dt.date.unique()))
    #ax.set_ylim(plot_height-day_height, len(df['Start Time'].dt.date.unique()))
    #ax.set_xlim(9*60, 18*60)
    hours = np.arange(8, 18, 1)
    ax.set_xticks(hours * 60)  # Convert hours to minutes
    ax.set_xticklabels([f"{h}:00" for h in hours])
    # Set x-axis label
    ax.set_xlabel('Time')

    if extra_y_label is not None or grouping_col is not None:
        if extra_y_label is not None:
            addi_label = extra_y_label if not isinstance(extra_y_label, list) else extra_y_label[0]
        else:
            addi_label = y_ticks2
            
        ax2 = ax.twinx()
         
        ax2.yaxis.tick_right()

        # Adjusting the limit of y-axis for ax2
        ax2.set_ylim(ax.get_ylim())
        
#ax2.set_yticks([j * day_height + plot_height/ 2 for j in range(len(df['Start Time'].dt.date.unique()))])
        ax2.set_yticks(range(len(y_ticks)))
        #ax2.set_ylim(plot_height-day_height, len(df['Start Time'].dt.date.unique()))
        if isinstance(addi_label, pd.Series):
            ax2.set_yticklabels(addi_label)
            ax2.set_ylabel(addi_label.name, labelpad=15)
        elif isinstance(addi_label, list):
            ax2.set_yticklabels(addi_label)
            #ax2.set_ylabel('Date', labelpad=15)

        else:
            ax2.set_yticklabels(df[addi_label])
            ax2.set_ylabel(addi_label, labelpad=15)
    
    # Set title
    #ax.set_title('Procedure Timeline')
    
    return ax


def plot_steps_with_timeline_data(patient_flow_df, timeline_columns, figure_size, label_mapping):
    columns_names = patient_flow_df.columns.to_list()
    #print(columns_names)
    def assign_color(label):
        #color_palette = cl.scales['12']['qual']['Set3']  # Set3 palette with 12 colors
        # Dictionary to store label-color mappings
        if label not in label_mapping:
            # Assign a new color for the label
            color = tuple(random.random() for _ in range(3))
            label_mapping[label] = color
    
        return label_mapping[label]
    
    df = patient_flow_df.copy()
    # Convert Start Time and End Time columns to datetime
    for col in timeline_columns:
        df[col] = pd.to_datetime(df[col])
        
    df = df.sort_values(by=timeline_columns[0])
        # Set up the figure and axis
    fig, ax = plt.subplots(figsize=figure_size) 
    
    # Set the height of each day's frame
    day_height = 1
    
    sub_data_height = [0.25, 0.4, 0.25]
    
    # Iterate over each day
    for i, day in enumerate(df[timeline_columns[0]].dt.date.unique()):
        # Filter the DataFrame for the current day
        day_df = df[df[timeline_columns[0]].dt.date == day]
        #print(day_df)
        # Calculate the y-coordinate for the day's frame
        y = i * day_height

        # Iterate over each procedure of the current day
        
        duration_updated_x = 0
        
        row_count = 0
        
        for index, row in day_df.iterrows():
            
            time_data_index = 0
            
            for col1, col2 in zip(timeline_columns[:-1], timeline_columns[1:]):
                
                #print(index)
                duration = row[col2] - row[col1]
                duration = duration.total_seconds()/60 -1

                    #print(duration)
                #print(duration)
                # Calculate the x-coordinate for the procedure's bar
                x = row[col1]
                x = x.hour * 60 + x.minute # Format as 'hour:minute:second'
                if row_count == 0:
                    duration_updated_x = x

                #if x < duration_updated_x:
                    #x = duration_updated_x
                    #print(duration)

                color_for_procedure = assign_color(row[columns_names[-1]])
                # Plot the broken bar for the procedure
                
                  
                ax.broken_barh([(x, duration)],
                               (y+sum(sub_data_height[0:time_data_index]), sub_data_height[time_data_index]),
                               facecolors=[color_for_procedure],
                               edgecolor='black')

                # Add procedure name as a label
                #'''
                if time_data_index == 1:
                    ax.text(x + duration / 2, y + day_height / 3, row[columns_names[-1]],
                        ha='center', va='center')

                    ax.text(x + duration / 2, y + 1.6 * day_height / 3, row[columns_names[-2]],
                            ha='center', va='center')
                #'''

                duration_updated_x += duration+2
                row_count+=1
                time_data_index +=1

    # Set y-axis limits and labels for each day
    ax.set_yticks([j * day_height + day_height / 2 for j in range(len(df[timeline_columns[0]].dt.date.unique()))])
    ax.set_yticklabels(df[timeline_columns[0]].dt.date.unique())
    ax.set_xlim(9*60, 18.5*60)
    hours = np.arange(8, 18, 1)
    ax.set_xticks(hours * 60)  # Convert hours to minutes
    ax.set_xticklabels([f"{h}:00" for h in hours])
    # Set x-axis label
    ax.set_xlabel('Time')

    # Set title
    #ax.set_title('Procedure Timeline')
    
    return ax    
    
    
def plot_ward_bed_occupancy_data(ward_wise_beds_occupancy_data, plot_type, time_begin, time_stop, Initial_beds_avail, plot_name, y_label):
    fig, ax2 = plt.subplots(figsize=(6, 3))
    
    for i, (ward_name, ward_beds_occupancy_data) in enumerate(ward_wise_beds_occupancy_data.items()):
        
        df1 = convert_to_uniform_data(ward_beds_occupancy_data, '2H', time_begin, time_stop)
        df_accu = df1.copy()
        #print(df1)
        
        df_accu['data'] = df1['data'].cumsum() + Initial_beds_avail[i]
        #print(df_accu.iloc[0:20,:])
        #print(df_accu)
        #x, y = zip(*bed_occupancy)
        #y = [INITIAL_POST_WARD_BED_AVAILABILITY + _ for _ in list(accumulate(y))]
        if plot_type == 'line':
            ax2.plot(df_accu['timestamp'].tolist(), df_accu['data'].tolist() , label=f"{ward_name} occupancy")
        else:
            if i == 0:
                y_sum = list(0 for _ in range(len(df_accu)))
                # calculate the width of each bar based on the available width and the number of elements
                bar_width = 5 / len(y_sum)
                
            #ax.bar(df_accu['timestamp'], df_accu['data'],  width=bar_width, color='gray', label='Data Set 2')
            #print(df_accu['timestamp'].tolist())
            ax2.bar(df_accu['timestamp'].tolist(), df_accu['data'].tolist() ,bottom=y_sum, width=bar_width, label=f"{ward_name} occupancy")
            y_sum = [a+b for a,b in zip(y_sum, df_accu['data'].tolist())]

    # Convert the timestamps to datetime objects
    label_datetimes = pd.date_range(df_accu['timestamp'].min(), df_accu['timestamp'].max(), periods= 20).tolist()
    
    tick_labels = [dateTime.strftime('%B %d, %H:%M') for dateTime in label_datetimes]
    plt.xticks(label_datetimes, tick_labels, rotation=65, ha = 'right')
    ax2.set_title(plot_name)
    ax2.set_xlabel("Date and Time")
    #ax2.set_xlim(STARTING_DATE_TIME, ENDING_DATE_TIME)
    ax2.set_ylabel(y_label)
    ax2.legend()
    plt.show()
    fig.savefig(plot_name+'.png')   



def predict_the_time(model_type, dataset, feature_variables_names, model_sub_level, force_encoding_cols, target_data_type, trained_model_dir = None, patient_objects = [], surgery_objects = [] , model_retrain = False, categorised_modelling = False, similar_cols_for_encoding = None):
    
    model_name = create_model_name(model_type, target_data_type, feature_variables_names, model_sub_level)
    #print(model_name)
    if not trained_model_dir is None:
        model_folder = trained_model_dir
    else: 
        model_folder = 'Trained_models'
    if categorised_modelling:
        model_folder = os.path.join(model_folder, 'Category_basis')

    if os.path.exists(os.path.join(model_folder, model_name +'.pkl')) and not model_retrain:
        with open(os.path.join(model_folder, model_name +'.pkl'), 'rb') as file:
            model_train_results = pickle.load(file)
            
    else:
        raise ValueError(f"The corresponding trained model {model_name} is not available for prediction in the folder: {model_folder}")
     
    input_data_encoded = prepare_input_data(dataset, feature_variables_names, force_encoding_cols, model_train_results[-2], patient_objects, surgery_objects, not model_type == 'StochasticNormalDistribution', similar_cols_for_encoding)
    

    return predict_with_model(model_type, model_train_results[0], input_data_encoded)



def predict_with_model(model_type, trained_model, X_input_df, log_transformer = None):
    # Predict:
    
    if model_type == 'NeuralNet':
        X_tensor = torch.tensor(X_input_df.to_numpy(), dtype=torch.float32)
        trained_model.eval()
        with torch.no_grad():
            #categorical_data = X_tensor[:, :trained_model.num_categorical_features]
            #numerical_data = X_tensor[:, trained_model.num_categorical_features:]
            outputs = trained_model(X_tensor)
            y_pred = outputs.squeeze().numpy()
    elif 'Stochastic' in model_type:
        must_match_columns_prioritywise = trained_model[0][0]
        #print(must_match_columns_prioritywise)
        training_dataset = trained_model[1]
        #must_match_columns = ['Actual Procedure 1 Code 1', 'Consultant Code']
        y_pred = data_filter_based_prediction(model_type, training_dataset, X_input_df, must_match_columns_prioritywise, trained_model[0][1])
    
    elif 'BayesianRidge' in model_type:
        y_pred, y_std = trained_model.best_estimator_.predict(X_input_df, return_std=True)
        if not log_transformer is None:
            y_pred = log_transformer.inverse_transform(y_pred)
            y_std = log_transformer.inverse_transform(y_std)
        return y_pred, y_std
    # for 'RegressionPipeline' ,'DecisionTreeRegressor' , SupportVectorRegression , gradient_boosting
    else: 
        y_pred = trained_model.best_estimator_.predict(X_input_df)
        
    if not log_transformer is None:
        y_pred = log_transformer.inverse_transform(y_pred)
    
    return y_pred



def shift_feature_to_data_category(modelling_parameters, feature_name, feature_value):
        
        updated_para = copy.deepcopy(modelling_parameters)
        
        #removing the feature from input feature
        updated_para[2].remove(feature_name)
        
        #adding to data category
        updated_para[3][feature_name] = feature_value
        
        return updated_para
    

def prediction_with_categorised_models(input_data_or_surgery_obj, parameters_for_predictions_model, variable_to_predict, trained_model_dir = None, patient_objects = [], surgery_objects = [], model_retrain = False, similar_cols_for_encoding = None):
    
    further_categorising_feature = parameters_for_predictions_model[2][0]
    
    if isinstance(input_data_or_surgery_obj, list):
        feature_related_data = input_data_or_surgery_obj[0]
    elif isinstance(input_data_or_surgery_obj, pd.DataFrame):
        feature_related_data = input_data_or_surgery_obj.at[0,further_categorising_feature]
    elif isinstance(input_data_or_surgery_obj, pd.Series):
        feature_related_data = input_data_or_surgery_obj[further_categorising_feature]
    else:
        try:
            feature_related_data = input_data_or_surgery_obj.get_selected_information([further_categorising_feature])       
        except ValueError as e:
            feature_related_data = input_data_or_surgery_obj.patient.get_selected_information([further_categorising_feature])
        #print(feature_related_data)
        #extract value data from dict data as it is in the dict format
        feature_related_data = feature_related_data[further_categorising_feature]

    updated_parameters = shift_feature_to_data_category(parameters_for_predictions_model, further_categorising_feature, feature_related_data)
    #print(updated_parameters[-1])
    #updated_parameters = shift_feature_to_data_category(parameters_for_predictions_model, updated_parameters[2][0], surgery_obj.procedure_type)

    procedure_time_data = predict_the_time(*updated_parameters, variable_to_predict, trained_model_dir, patient_objects, surgery_objects, False, True, similar_cols_for_encoding)

    #print(f'predicted with categorised model for {further_categorising_feature} with value {feature_related_data}')
               
    return procedure_time_data



def prepare_input_data(dataset, feature_variables_names, force_encoding_cols, input_data_format, patient_objects = [], surgery_objects = [], econding_required = True, similar_cols_for_encoding = None):
    if len(dataset) > 0:
        input_data = dataset[feature_variables_names]
    else:
        input_data = []
        for patient_object, surgery_object in zip(patient_objects, surgery_objects):
            extracted_values = {}
            extracted_values.update(search_features_values_from_obj(patient_object, feature_variables_names))
            non_extracted_features = [element for element in feature_variables_names if element not in extracted_values]
            extracted_values.update(search_features_values_from_obj(surgery_object, non_extracted_features))

            non_extracted_features = [element for element in feature_variables_names if element not in extracted_values]
            if len(non_extracted_features)>0:
                #print(non_extracted_features)
                extracted_values.update(select_feature_values_from_dataset(dataset, non_extracted_features,'Patient ID', patient_object.ID))
                if len(extracted_values) < len(feature_variables_names): 
                    non_extracted_features = [element for element in feature_variables_names if element not in extracted_values]
                    raise ValueError(f"Following features value could not be extracted:{non_extracted_features}")
            input_data.append(extracted_values)
        input_data = pd.DataFrame(input_data).fillna(0)
    #print(input_data)
    if not econding_required:
        return input_data
    
    return encoding_input_data_and_reindexing_as_reference(input_data, force_encoding_cols, input_data_format, similar_cols_for_encoding)





def search_features_values_from_obj(obj, features):
    result = {}

    # Search for features in patient attributes
    for feature in features:
        attr = check_attribute_presence(obj, feature)
        if attr is not None:
            if getattr(obj, attr) is not None:
                result[feature] = getattr(obj, attr)

        elif feature in obj.additional_attributes:
            if obj.additional_attributes[feature] is not None:
                result[feature] = obj.additional_attributes[feature]
    #print(result)

    return result


def select_feature_values_from_dataset(df, features, feature, value):
    # Load CSV data into a DataFrame
    
    # Boolean indexing to select rows with matching feature value
    selected_rows = df[df[feature] == value]
    
    # Filter the selected rows to include only the specified features
    selected_features = selected_rows[features]
    #print(selected_features)
    
    # Convert the selected features DataFrame to a single dictionary
    selected_dict = selected_features.iloc[0].to_dict()
    
    return selected_dict



def update_patient_data(patients_df, patients_data_df, fixed_data_for_patient = []):
    
    if 'Age group at admit' not in patients_data_df and 'Age group at admit' not in patients_df.columns.tolist():
        patients_df['Age group at admit'] = patients_data_df['Age'].apply(lambda x: age_to_group(x))
    
    data_prev_cols = [col for col in patients_df.columns if col not in fixed_data_for_patient]
    
    patients_ID_columns = ['Patient ID']

    procedure_code_related_col = [col for col in patients_df.columns if 'procedure code' in col.lower()][0]

    if procedure_code_related_col in patients_df.columns and procedure_code_related_col in patients_data_df.columns:
        patients_ID_columns.append(procedure_code_related_col)

    cols_to_ignore = []
    for col in data_prev_cols:
        if col in patients_data_df.columns and col not in patients_ID_columns:
            cols_to_ignore.append(col)
            
    updated_patients_df = pd.merge(patients_df, patients_data_df[[col for col in patients_data_df.columns if not col in cols_to_ignore]], on=patients_ID_columns, how = 'inner')
    
    #updated_patients_df = updated_patients_df[data_prev_cols + fixed_data_for_patient]

    #updated_patients_df = []
    
    return updated_patients_df[list(dict.fromkeys(list(patients_df.columns)+fixed_data_for_patient))]
