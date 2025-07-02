
#import packages

import pandas as pd  # 2.0.3
import numpy as np  ## 1.24.3
import sys
#from sklearn.preprocessing import StandardScaler   #version used 1.3.0
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
#import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt  # 3.7.2
#from joblib import dump, load         #import if required - to extract model
import pickle
import os
import seaborn as sns
import re
import torch     #version used 2.1.2
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from scipy import stats   ## version used 1.11.1
from data_processing_nd_encoding_related_functions import prepare_IO_dataset, preprocess_dataset
"""------------------------------------------------------------------"""





def category_based_modelling(data_file, essential_categ, essential_features, additonal_features, 
                             target_data_types, min_categorical_count_threshold, Z_score_threshold_for_outlier_removal, extra_one_hot_encoding_cols, model_type, training_related_parameters, min_data_reqs, models_retrain, trained_model_dir = None, skip_modelling_if_no_categorisation = False, similar_cols_for_encoding = None, log_transform=False):
    
    output = []
    
     # Preprocess the dataset
    preprocessed_result = preprocess_dataset(data_file, essential_categ, essential_features, additonal_features,
                                             target_data_types, {}, min_categorical_count_threshold, Z_score_threshold_for_outlier_removal)
    preprocessed_datasets = [result[1] for result in preprocessed_result]
    
    # feature considered for further categorisation
    feature_for_catego_further = essential_features[0]
    
    #lets sort the preprocess dataset on the basis of next_feature count
    preprocessed_datasets = [df1.sort_values(by= feature_for_catego_further, key=lambda x: x.map(x.value_counts()), ascending=False) for df1 in preprocessed_datasets]
    
    #the next features count for the current feature categorised data
    if feature_for_catego_further in min_data_reqs:
        
        # get the counts of all category data that falls under the feature 'feature_for_catego_further'
        next_features_count = preprocessed_datasets[0][feature_for_catego_further].value_counts()
        
        #print(next_features_count[next_features_count >= min_data_reqs[feature_for_catego_further]])
        
        #Building model with the categorisation that satisfy the minimum data conditions
        for index, value in next_features_count[next_features_count >= min_data_reqs[feature_for_catego_further]].items():
            
            new_categor = essential_categ.copy()
            
            #adding the data to the data segmentation dictinary 
            new_categor[feature_for_catego_further] = index
            #print(f'Further categorisation happens with {feature_for_catego_further} for {index} as count is {value}')
            
            new_min_data_reqs = min_data_reqs.copy()
            
            new_min_categorical_count_threshold = min_categorical_count_threshold.copy()
            
            #removing the minimum data requirements for the categorised column as its redundant
            if feature_for_catego_further in new_min_categorical_count_threshold:
                del new_min_categorical_count_threshold[feature_for_catego_further]
                
            new_z_score_threshold = Z_score_threshold_for_outlier_removal.copy()
            
            if isinstance(Z_score_threshold_for_outlier_removal, dict) and feature_for_catego_further in Z_score_threshold_for_outlier_removal:
                
                new_z_score_threshold['overall'] = new_z_score_threshold[feature_for_catego_further]
                del new_z_score_threshold[feature_for_catego_further]

            output = output + category_based_modelling(data_file, new_categor, essential_features[1:], additonal_features, 
                                                       target_data_types, new_min_categorical_count_threshold, new_z_score_threshold, extra_one_hot_encoding_cols, model_type,  training_related_parameters, new_min_data_reqs, models_retrain, skip_modelling_if_no_categorisation = False, similar_cols_for_encoding = similar_cols_for_encoding, log_transform=log_transform)  

        # Filter data that doesn't meet the minimum requirements
        leftover_1st_feature_data = next_features_count[next_features_count < min_data_reqs[feature_for_catego_further]].index.tolist()
        
        #print(f'leftover features data are {leftover_1st_feature_data}')
        
        #The data not being able to get categorised will be dealt together and the data considered for categorical modelling will be removed
        if leftover_1st_feature_data:
            #print(preprocess_datasets[0][feature_for_catego_further])
            preprocessed_datasets = [df[df[feature_for_catego_further].isin(leftover_1st_feature_data)] for df in preprocessed_datasets]
    if not skip_modelling_if_no_categorisation:
        #print(preprocess_datasets[0])
        # filtering the leftover data
        
        #selecting additional features considering the proper match of available data set and features data 
        #addi_feat_count = min(max(int(len(preprocessed_datasets[0])/20) - len(essential_features), 0), len(additonal_features))
        
        #model_input_features = essential_features + additonal_features[0:addi_feat_count]
        model_input_features = essential_features + additonal_features
        
        preprocessed_datasets = [df[model_input_features+[df.columns[-1]]] for df in preprocessed_datasets]
        
        #print(preprocess_datasets[0])
        
        IO_datasets = prepare_IO_dataset(preprocessed_datasets, 'one-hot', [col for col in extra_one_hot_encoding_cols if col in model_input_features], similar_cols_for_encoding = similar_cols_for_encoding)
    
        if model_type == 'StochasticNormalDistribution':
            IO_datasets = [[df.iloc[:,:-1], df.iloc[:,-1]] for df in preprocessed_datasets]
        
        #training_related_parameters = get_model_training_related_parameters(model_type)
        
        print(f'model categorised for {essential_categ.values()} with input features {model_input_features}')
        
        Trained_res, Test_res = model_training_nd_testing_results(model_type, training_related_parameters,
                                                                  target_data_types, model_input_features, essential_categ, IO_datasets, models_retrain, False, True,  log_transform = log_transform, trained_model_dir = trained_model_dir)
        output = output + [[Trained_res, Test_res]]
    
    return output




def check_attribute_presence(obj, attribute):
    # Modify attribute name and object attribute names
    modified_attribute = attribute.replace(" ", "").replace("_", "").lower()
    modified_obj_attributes = [attr.replace(" ", "").replace("_", "").lower() for attr in dir(obj)]
    
    # Check if modified attribute name is present in modified object attribute names
    if modified_attribute in modified_obj_attributes:
        index = modified_obj_attributes.index(modified_attribute)
        return dir(obj)[index]
    else:
        return None
    

def create_range_filtered_dataset(train_data, input_data, range_percent, numerical_cols, columns_for_data_matching):
    filtered_data = train_data.copy()
    #print(input_data)
    #print(input_data)
    for column in columns_for_data_matching:
        if column in numerical_cols:  # Numeric variable
            min_val = filtered_data[column].min()
            max_val = filtered_data[column].max()
            covering_range = (max_val - min_val) * range_percent
            
            lower_bound = max(min_val,input_data[column] - covering_range)
            upper_bound = min(max_val,input_data[column] + covering_range)
            
            filtered_data = filtered_data[(filtered_data[column] >= lower_bound) & (filtered_data[column] <= upper_bound)]
        else:  # String variable
            #print(input_data[column])
            #print(filtered_data)
            filtered_data = filtered_data[filtered_data[column] == input_data[column]]
        
    return filtered_data

    

def data_filter_based_prediction(model_type, training_data_set, X_input_df, must_match_columns_prioritywise, scaling_factors):
    if model_type == 'StochasticNormalDistribution':
        return gaussan_based_prediction(model_type, training_data_set, X_input_df, must_match_columns_prioritywise, scaling_factors)



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





def gaussan_based_prediction(model_type, training_data_set, X_input_df, must_matched_columns_prioritywise, scaling_factors):
    y_predic = pd.Series(dtype=float)
    #print(f'orginal length of Y_out and x_out are:{len(y_predic)}, {len(X_input_df)}')
    train_data_merged = pd.concat([training_data_set[0],training_data_set[1]], axis=1)
    numerical_cols = training_data_set[0].select_dtypes(include=['int', 'float']).columns.tolist()
    #print(numerical_cols)
    for ind, row in X_input_df.iterrows():
        range_covering_percentange = 0.1
        while range_covering_percentange < 0.9:
            filtered_dataset = create_range_filtered_dataset(train_data_merged, row, range_covering_percentange, 
                                                             numerical_cols, must_matched_columns_prioritywise)
            if len(filtered_dataset) >= 1:
                break
            range_covering_percentange = range_covering_percentange * 1.8 
        
        index = len(must_matched_columns_prioritywise)
        while len(filtered_dataset) == 0:
            print(f'For {row} Except following columns data category, other are generalised due to lack of data {must_matched_columns_prioritywise[0:index]}')
            filtered_dataset = create_range_filtered_dataset(train_data_merged, row, range_covering_percentange, 
                                                                 numerical_cols, must_matched_columns_prioritywise[0:index])
                
            index = index-1
            #print(filtered_dataset)
        if len(filtered_dataset) == 0:
            print(f' no enough data found' )
            y_predic.at[ind] = 0
        else:
            y_train = filtered_dataset.iloc[:,-1]

            # Compute the mean and standard deviation of the target variable from the training data
            mean = y_train.mean()
            std_dev = y_train.std()
            if np.isnan(std_dev):
                std_dev = 0

            # Generate random predictions from the normal distribution with the mean and standard deviation
            predicted_value = np.random.normal(mean*scaling_factors[0], std_dev*scaling_factors[1], 1)
            if not np.isnan(predicted_value):
                y_predic.at[ind] = abs(predicted_value[0])
    #print(f'final length of Y_out and x_out are:{len(y_predic)}') 
    #print(y_predic)
    return y_predic
        

def get_model_training_related_parameters(model_type):
    parameters =  []

    if model_type == 'RegressionPipeline&ElNet':
        #settting variables for pipline model step and parameters for GridSearchCV
        el_net_pipeline= ('elasticnet', ElasticNet())
        pipeline_param = {'elasticnet__l1_ratio':[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1], 
               'elasticnet__alpha':[ 0.0125, 0.025, 0.05, .125, .25, .5, 1., 2., 4.]}
        parameters.extend([el_net_pipeline, pipeline_param])

    elif model_type == 'NeuralNet':
        #settting variables for nnet model
        hidden_dim = 128
        learning_rate = 0.01
        num_epochs = 40
        parameters.extend([hidden_dim, learning_rate, num_epochs])
        
    elif model_type == 'StochasticNormalDistribution':
        must_match_columns = ['Actual Procedure 1 Code 1', 'Consultant Code']
        parameters.append(must_match_columns)
        scaling_factors = [1.075, 0.5]
        parameters.append(scaling_factors)
    return parameters



def identify_corresponding_feature(model_name_str, feature_name, essen_modelling_data_categ, possible_extra_catego, encoded_data):
    def is_string_in_list(string, lst):
        for item in lst:
            if string in item:
                return True
        return False
    def reverse_one_hot_encoding(selected_feature, one_hot_encoded_data):
        reversed_data = []

        for index, row in one_hot_encoded_data.iterrows():
            for column_name, value in row.items():
                if value == 1 and column_name.startswith(selected_feature):
                    feature_value = column_name.split('_')[-1]  # Remove the underscore and keep only the relevant part
                    reversed_data.append(feature_value)
                    break

        return reversed_data

    if not is_string_in_list(feature_name, encoded_data.columns) and feature_name in possible_extra_catego:
        str_temp = model_name_str.split('_')[-2]
        return [str_temp for _ in range(len(encoded_data))]
    else:
        return reverse_one_hot_encoding(feature_name, encoded_data)



def model_training_nd_testing_results(model_type, training_related_parameters, target_data_types, model_input_features, dict_for_data_model_categorisation,IO_dataset, model_retrain, plot_display, categorised_modelling = False, stochasticity_on_prediction = False, log_transform = False, residual_plot = True, trained_model_dir = None):
    Trained_results = []
    Test_results = []
    if not trained_model_dir is None:
        model_folder = trained_model_dir
    else: 
        model_folder = 'Trained_models'
        
    if categorised_modelling:
        model_folder = os.path.join(model_folder, 'Category_basis')
        
    for i in range(len(target_data_types)):
        if log_transform:
            model_name = create_model_name(model_type, f'LOG{target_data_types[i]}', model_input_features,dict_for_data_model_categorisation)
            #print(model_name)
        else:
             model_name = create_model_name(model_type, target_data_types[i], model_input_features,dict_for_data_model_categorisation)

        if os.path.exists(os.path.join(model_folder,model_name +'.pkl')) and not model_retrain:
            with open(os.path.join(model_folder,model_name +'.pkl'), 'rb') as file:
                train_result = pickle.load(file)
                #Trained_results.append([model_name] + train_result)
        else:
            start_time = time.time()
            print(f'Starting to prepare for model Training: {model_name}')
            train_result = train_the_model(model_type, training_related_parameters, IO_dataset[i], log_transform = log_transform)
            with open(os.path.join(model_folder,model_name +'.pkl'), 'wb') as file:
                pickle.dump(train_result, file)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Training time: {:.2f} seconds".format(elapsed_time))
            
        #print(train_result)
        
        if log_transform:
            transformer_log = train_result[-3]
        else:
            transformer_log = None
            
        Trained_results.append([model_name] + train_result)
            
        Test_results.append(test_model(model_name, model_type,train_result[0], *train_result[-2:], None, None, residual_plot, plot_display, stochasticity_on_prediction, transformer_log))
        
    return Trained_results, Test_results


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





def plot_real_vs_prediction(y_real, y_pred):
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=y_real, y=np.abs(y_pred),
           color="orange",edgecolors='white', ax=ax1)
    if hasattr(y_pred, 'name'):
        ax1.set_ylabel(f"Predicted {y_pred.name}")
    else:
        ax1.set_ylabel(f"Real {y_real.name}")
    ax1.set_xlabel(f"Real {y_real.name}")
    sns.lineplot(x=y_real, y=y_real, color='blue', ax=ax1)
    ax1.set_xlim(0,max(y_real)+1)
    return ax1

def plot_residuals(y_test,  y_pred, color_data = None, max_legend_entries = None):
    errors = (y_test - np.abs(y_pred)) / y_test
    fig, ax2 = plt.subplots()
    
    if color_data is None:
        errors = [x for x in errors if -1.5 <= x <= 1.5]
        ax2.hist(errors, bins=30, edgecolor='k', color='red', alpha=0.7)
        # Calculate metrics
        mean = np.mean(errors)
        median = np.median(errors)
        modes = stats.mode(errors)
        #mode_values = modes.mode
        std_dev = np.std(errors)
        skew = stats.skew(errors)
        kurtosis = stats.kurtosis(errors)

        # Print the metrics
        metrics_str = f"Mean      : {mean:.2f}\nMedian   : {median:.2f}\nStd dev   : {std_dev:.2f}\nSkewness: {skew:.2f}\nKurtosis  : {kurtosis:.2f}"
        ax2.text(0.05, 0.72, metrics_str, transform=ax2.transAxes)
        ax2.set_ylabel('Frequency')
    else:
        data = pd.DataFrame({'Errors': errors, 'Category': color_data})
        data = data[(data['Errors'] >= -1.5) & (data['Errors'] <=1.5 )]
        sns.kdeplot(data=data, x='Errors', hue='Category', common_norm=False, ax=ax2, alpha=0.7, fill=False, legend=True,  palette='Set1')
        sns.kdeplot(data=errors, ax=ax2, label='Overall', color='black', linestyle='--')

        
        #sns.histplot(data=data, x='Errors', hue='Category', bins=60, element='step', common_norm=False, kde=True, ax=ax2, alpha=0.7)
        ax2.set_ylabel('Density')
        
    ax2.set_xlabel('Relative Error')
    ax2.set_title('Distribution of Errors')
    ax2.set_xlim([-1.5, 1.5])
    
    return ax2


def plot_with_performance_test_old(model_name,  y_test,  y_pred, color_data = None, colors_code = None, max_legend_entries = None, error_dist_plot = False, only_error_plot = True):

    # Find mean squared error, root and negate it - this should be very similar to best score above if model is not over/under fit
    test_rmse = round(np.sqrt(mean_squared_error(y_test, np.abs(y_pred))), 2)
    #normalised_rmse = round(test_rmse / (max(y_test) - min(y_test) + 1e-10), 2)
    mean_absolute_percentage_error = round(np.mean(np.abs((y_test - np.abs(y_pred)) / y_test)),2)
    # Find r2
    r2 = round(r2_score(y_test, np.abs(y_pred)), 2)

    print('Prediction Test Result \n RMSE: ' + str(test_rmse))
    print('MAPE: ' + str(mean_absolute_percentage_error))
    print('R2: ' + str(r2))
    
    if error_dist_plot:
        errors = (y_test - np.abs(y_pred)) / y_test
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    else:
        fig, ax1 = plt.subplots()
    
    if color_data is not None:
        sns.scatterplot(x=y_test, y=np.abs(y_pred), hue=color_data, palette='Set1', 
                        edgecolors='white', ax=ax1).set(title=f"{model_name}'s prediction \n vs Actual Time")
    else:
        sns.scatterplot(x=y_test, y=np.abs(y_pred), color="orange",edgecolors='white', ax=ax1).set(title=f"{model_name}'s prediction \n vs Actual Time")
    
    ax1.set_ylabel(f"Predicted {y_test.name}")
    ax1.set_xlabel(f"Real {y_test.name}")
    sns.lineplot(x=y_test, y=y_test, color='blue', ax=ax1)
    metrics_str = f"RMSE : {test_rmse:.2f}\nMAPE : {mean_absolute_percentage_error:.2f}\nR2   : {r2:.2f}"
    #ax1.text(0.05, 0.90 , f'RMSE   : {test_rmse}', transform=ax1.transAxes)
    #ax1.text(0.05, 0.84, f'MAPE   : {mean_absolute_percentage_error}', transform=ax1.transAxes)
    #ax1.text(0.05, 0.78, f'R2      : {r2}', transform=ax1.transAxes)
    ax1.text(0.05, 0.80, metrics_str, transform=ax1.transAxes)
    
    ax1.set_xlim(0,max(y_test)+1)
    
    if error_dist_plot:
        if color_data is None:
            errors = [x for x in errors if -1.5 <= x <= 1.5]
            ax2.hist(errors, bins=30, edgecolor='k', color='red', alpha=0.7)
            # Calculate metrics
            mean = np.mean(errors)
            median = np.median(errors)
            modes = stats.mode(errors)
            #mode_values = modes.mode
            std_dev = np.std(errors)
            skew = stats.skew(errors)
            kurtosis = stats.kurtosis(errors)
            
            # Convert mode_values to a string
            #mode_str = ', '.join(map(str, mode_values)) if len(modes)/2 > 1 else f"{mode_values:.2f}"

            
            # Print the metrics
            metrics_str = f"Mean      : {mean:.2f}\nMedian   : {median:.2f}\nStd dev   : {std_dev:.2f}\nSkewness: {skew:.2f}\nKurtosis  : {kurtosis:.2f}"
            ax2.text(0.05, 0.72, metrics_str, transform=ax2.transAxes)
            ax2.set_ylabel('Frequency')
        else:
            data = pd.DataFrame({'Errors': errors, 'Category': color_data})
            data = data[(data['Errors'] >= -1.5) & (data['Errors'] <=1.5 )]
            sns.kdeplot(data=data, x='Errors', hue='Category', common_norm=False, ax=ax2, alpha=0.7, fill=False, legend=True,  palette='Set1')
            sns.kdeplot(data=errors, ax=ax2, label='Overall', color='black', linestyle='--')

            
            #sns.histplot(data=data, x='Errors', hue='Category', bins=60, element='step', common_norm=False, kde=True, ax=ax2, alpha=0.7)
            ax2.set_ylabel('Density')
            
        ax2.set_xlabel('Relative Error')
        ax2.set_title('Distribution of Errors')
        ax2.set_xlim([-1.5, 1.5])
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
    
    if color_data is not None:
        # Get the unique data types and their counts
        unique_data_types, data_type_counts = np.unique(color_data, return_counts=True)

        # Limit the number of legend entries if there are more than max_legend_entries unique data types
        if len(unique_data_types) > max_legend_entries:
            ncol = 2  # Specify the number of columns for the legend
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
           # Sort the data types by count in descending order
            sorted_data_types = sorted(unique_data_types, key=lambda x: data_type_counts[np.where(unique_data_types == x)[0][0]], reverse=True)

            # Get the top max_legend_entries data types and their corresponding handles
            selected_labels = [label for label in sorted_data_types[:max_legend_entries] if label in by_label]
            selected_handles = [by_label[label] for label in selected_labels]

            ax1.legend(selected_handles, selected_labels, ncol=ncol, bbox_to_anchor=(1, 0), loc='lower right')
        else:
            ax1.legend(bbox_to_anchor=(1, 0), loc='lower right')
    
    if error_dist_plot:
        return [test_rmse, mean_absolute_percentage_error, r2, fig, ax1, ax2]
    else:
        return [test_rmse, mean_absolute_percentage_error, r2, fig, ax1]
         
            


def plot_with_performance_test(model_name,  y_test,  y_pred, color_data = None, category_colors = None, 
                                max_legend_entries = None, error_dist_plot = False, only_error_plot = False):

    def assign_color(label):
        if label not in category_colors:
            # Assign a new color for the label
            color = tuple(random.random() for _ in range(3))
            category_colors[label] = color
        
        return category_colors[label]

    
    # Find mean squared error, root and negate it - this should be very similar to best score above if model is not over/under fit
    test_rmse = round(np.sqrt(mean_squared_error(y_test, np.abs(y_pred))), 2)
    #normalised_rmse = round(test_rmse / (max(y_test) - min(y_test) + 1e-10), 2)
    mean_absolute_percentage_error = round(np.mean(np.abs((y_test - np.abs(y_pred)) / y_test)),2)
    # Find r2
    r2 = round(r2_score(y_test, np.abs(y_pred)), 2)

    print('Prediction Test Result \n RMSE: ' + str(test_rmse))
    print('MAPE: ' + str(mean_absolute_percentage_error))
    print('R2: ' + str(r2))
    
    if error_dist_plot:
        errors = (y_test - np.abs(y_pred)) / y_test
        
        if only_error_plot:
            fig, ax2 = plt.subplots()
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    else:
        fig, ax1 = plt.subplots()

    #setting palett setting
    if color_data is not None:
    
        if category_colors is not None:
            colors = [assign_color(category) for category in np.unique(color_data)]
            palett = sns.color_palette(colors)
        else:
            palett='Set1'
                        

    if not only_error_plot:
        if category_colors is not None:
            sns.scatterplot(x=y_test, y=np.abs(y_pred), hue=color_data, palette=palett, 
                            edgecolors='white', ax=ax1).set(title=f"{model_name}'s prediction \n vs Actual Time")
        else:
            sns.scatterplot(x=y_test, y=np.abs(y_pred), color="orange",edgecolors='white', ax=ax1).set(title=f"{model_name}'s prediction \n vs Actual Time")
        
        ax1.set_ylabel(f"Predicted {y_test.name}")
        ax1.set_xlabel(f"Real {y_test.name}")
        sns.lineplot(x=y_test, y=y_test, color='blue', ax=ax1)
        metrics_str = f"RMSE : {test_rmse:.2f}\nMAPE : {mean_absolute_percentage_error:.2f}\nR2   : {r2:.2f}"
        #ax1.text(0.05, 0.90 , f'RMSE   : {test_rmse}', transform=ax1.transAxes)
        #ax1.text(0.05, 0.84, f'MAPE   : {mean_absolute_percentage_error}', transform=ax1.transAxes)
        #ax1.text(0.05, 0.78, f'R2      : {r2}', transform=ax1.transAxes)
        ax1.text(0.05, 0.80, metrics_str, transform=ax1.transAxes)
        
        ax1.set_xlim(0,max(y_test)+1)
    
    if error_dist_plot:
        if color_data is None:
            errors = [x for x in errors if -1.5 <= x <= 1.5]
            ax2.hist(errors, bins=30, edgecolor='k', color='red', alpha=0.7)
            # Calculate metrics
            mean = np.mean(errors)
            median = np.median(errors)
            modes = stats.mode(errors)
            #mode_values = modes.mode
            std_dev = np.std(errors)
            skew = stats.skew(errors)
            kurtosis = stats.kurtosis(errors)
            
            # Convert mode_values to a string
            #mode_str = ', '.join(map(str, mode_values)) if len(modes)/2 > 1 else f"{mode_values:.2f}"

            
            # Print the metrics
            metrics_str = f"Mean      : {mean:.2f}\nMedian   : {median:.2f}\nStd dev   : {std_dev:.2f}\nSkewness: {skew:.2f}\nKurtosis  : {kurtosis:.2f}"
            ax2.text(0.05, 0.72, metrics_str, transform=ax2.transAxes)
            ax2.set_ylabel('Frequency')
        else:
            data = pd.DataFrame({'Errors': errors, 'Category': color_data})
            data = data[(data['Errors'] >= -1.5) & (data['Errors'] <=1.5 )]
            sns.kdeplot(data=data, x='Errors', hue='Category', common_norm=False, ax=ax2, alpha=0.7, fill=False, legend=True,  palette='Set1')
            sns.kdeplot(data=errors, ax=ax2, label='Overall', color='black', linestyle='--')

            
            #sns.histplot(data=data, x='Errors', hue='Category', bins=60, element='step', common_norm=False, kde=True, ax=ax2, alpha=0.7)
            ax2.set_ylabel('Density')
            
        ax2.set_xlabel('Relative Error')
        ax2.set_title('Distribution of Errors')
        ax2.set_xlim([-1.5, 1.5])
        
        # Adjust layout to prevent overlapping
        plt.tight_layout()
    
    if not only_error_plot and color_data is not None:
        # Get the unique data types and their counts
        unique_data_types, data_type_counts = np.unique(color_data, return_counts=True)

        # Limit the number of legend entries if there are more than max_legend_entries unique data types
        if len(unique_data_types) > max_legend_entries:
            ncol = 2  # Specify the number of columns for the legend
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
           # Sort the data types by count in descending order
            sorted_data_types = sorted(unique_data_types, key=lambda x: data_type_counts[np.where(unique_data_types == x)[0][0]], reverse=True)

            # Get the top max_legend_entries data types and their corresponding handles
            selected_labels = [label for label in sorted_data_types[:max_legend_entries] if label in by_label]
            selected_handles = [by_label[label] for label in selected_labels]

            ax1.legend(selected_handles, selected_labels, ncol=ncol, bbox_to_anchor=(1, 0), loc='lower right')
        else:
            ax1.legend(bbox_to_anchor=(1, 0), loc='lower right')
    
    if error_dist_plot:
        if only_error_plot:
            return [ fig, ax2]
        else:    
            return [test_rmse, mean_absolute_percentage_error, r2, fig, ax1, ax2]
    else:
        return [test_rmse, mean_absolute_percentage_error, r2, fig, ax1]

        
     



def test_model(model_name, model_type, trained_model, X_test, y_test, color_col, max_legend_entries, error_plot, plot_display = False, stochasticity_enable = False, log_transformer = None):
    # Predict:
    if 'BayesianRidge' in model_type:
        y_pred, y_std = predict_with_model(model_type, trained_model, X_test, log_transformer)
        
        if stochasticity_enable:
            y_pred = [np.random.normal(pred, std) for pred, std in zip(y_pred, y_std)]
    else:
        y_pred = predict_with_model(model_type, trained_model, X_test, log_transformer)

    if color_col is not None:
        color_data = X_test[color_col]
    else:
        color_data = None
        
    perf_nd_plot_res = plot_with_performance_test(model_name,  y_test,  y_pred, color_data, max_legend_entries, error_plot)
    
    if not plot_display:
        plt.close(perf_nd_plot_res[3])
    
    return perf_nd_plot_res


    
def train_the_model(model_type, model_training_related_parameters, IO_dataset, test_size=0.2, random_state=42, log_transform = False):
    
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(*IO_dataset, test_size = test_size, random_state=random_state)
    
    if log_transform:
        log_transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
        y_train_df = log_transformer.transform(y_train_df)
        #y_test_df = log_transformer.transform(y_test_df)
    else:
        log_transformer = None

    
    if model_type == 'RegressionPipeline&ElNet':
        #run train_RegressionPipeline_model function for regression, store results, model and coefficients in a list
        
        training_output = train_RegressionElNet_model( X_train_df,  y_train_df, model_training_related_parameters)
    
    elif model_type == 'NeuralNet':
        #train_result = train_nn_model(preprocess_dataset[i][1], model_input_features, target_data_types[i], string_columns, numerical_columns,*training_related_parameters)
        #return train_nn_model(*model_training_related_parameters, *IO_dataset)
        if log_transform:
            training_output =  train_and_select_best_nn_model(X_train_df, X_test_df, y_train_df, log_transformer.transform(y_test_df), model_training_related_parameters)
        else:
             training_output =  train_and_select_best_nn_model(X_train_df, X_test_df, y_train_df, y_test_df, model_training_related_parameters)
    
    elif model_type == 'StochasticNormalDistribution':
        #X_train, X_test, y_train, y_test = train_test_split(IO_dataset[0], IO_dataset[1], test_size=0.2, random_state=42)
        return [[model_training_related_parameters, IO_dataset], X_test_df, y_test_df]
    
    else:
        function_name = "train_" + model_type + "_model"
        # Call the function directly
        if hasattr(sys.modules[__name__], function_name):
            training_output = globals()[function_name]( X_train_df,  y_train_df, model_training_related_parameters)
            
    
    return training_output + [log_transformer, X_test_df, y_test_df]



def train_and_select_best_nn_model(X_train_df, X_test_df, y_train_df, y_test_df, hyperparameters):
    
    # Unpack hyperparameters
    #learning_rates, num_epochs_list, hidden_dims, optimizers, batch_sizes = hyperparameters
    
    #X_train, X_test_df, y_train, y_test_df = train_test_split(X, y, test_size = test_size, random_state=random_state)
    
    X_train, X_test, y_train, y_test = [data_df.to_numpy() for data_df in [X_train_df, X_test_df, y_train_df, y_test_df]]

    # Convert the data to PyTorch tensors
    #y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = [torch.tensor(np_data, dtype=torch.float32) for 
                                                                    np_data in [X_train, y_train, X_test, y_test]]
    
    # Create the custom datasets and data loaders
    train_dataset = TimeDataset(X_train_tensor, y_train_tensor)
    #test_dataset = SurgicalTimeDataset(X_test_tensor, y_test_tensor)
    
    best_model = None
    best_mse = float('inf')
    best_hyper_parameter_combination = []
    for batch_size in hyperparameters['batch_sizes']:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for learning_rate in hyperparameters['learning_rates']:
            for num_epochs in hyperparameters['num_epochs_list']:
                for hidden_dim in hyperparameters['hidden_dims']:
                    for optimizer_class in hyperparameters['optimizer_classes']:
                        model = TimePredictionModel(X_train.shape[1], hidden_dim)
                        #print(f'{batch_size, learning_rate, num_epochs, hidden_dim, optimizer_class}')
                        
                        #try:
                        train_nn_model(model, num_epochs, train_loader, optimizer_class(model.parameters(), lr=learning_rate), nn.MSELoss())
                        #except ValueError as e:
                         #   continue
                        
                        model.eval()
                        with torch.no_grad():
                            y_pred = model(X_test_tensor)
                            y_pred = y_pred.squeeze().numpy()
                            #mse = mean_squared_error(y_test, y_pred.squeeze().numpy())
                            try:
                                mse = mean_squared_error(y_test, y_pred)
                            except ValueError as e:
                                continue

                        if mse < best_mse:
                            best_mse = mse
                            best_model = model
                            best_hyper_parameter_combination = [learning_rate, num_epochs, hidden_dim, optimizer_class, batch_size]
    print(f'Best MSE reached {best_mse} with following combination of hyper_parameter {best_hyper_parameter_combination}')
    
    return [best_model,best_hyper_parameter_combination]
    
    
#Elastic Net model function with Pipeline for scaling
def train_BayesianRidge_model(X_train, y_train, hyper_parameters):
    
    model = BayesianRidge()
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    cv = GridSearchCV(model, hyper_parameters, cv=5)
    
    cv.fit(X_train, y_train)

    return [cv]
    
    
    
def train_DecisionTreeRegressor_model(X_train, y_train, hyper_parameters):
    # Split your data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create a Decision Tree regressor model
    decision_tree_model = DecisionTreeRegressor()

    # Create a GridSearchCV object to find the best hyperparameters
    grid_search = GridSearchCV(decision_tree_model, hyper_parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    return [grid_search]



def train_XGBoostRegressor_model(X_train, y_train, hyper_parameters):
    # Split your data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create regressor model
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # Create a GridSearchCV object to find the best hyperparameters
    grid_search = GridSearchCV(model, hyper_parameters, scoring='neg_mean_squared_error', cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    
    coeff = pd.DataFrame()
    coeff["Columns"] = X_train.columns
    coeff['Coefficient Estimate'] = grid_search.best_estimator_.feature_importances_

    return [grid_search, coeff]
    


    
def train_GradientBoostingRegressor_model(X_train, y_train, hyper_parameters):
    # Split your data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create regressor model
    model = GradientBoostingRegressor()

    # Create a GridSearchCV object to find the best hyperparameters
    grid_search = GridSearchCV(model, hyper_parameters, scoring='neg_mean_squared_error', cv=5)

    # Fit the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_
    
    coeff = pd.DataFrame()
    coeff["Columns"] = X_train.columns
    coeff['Coefficient Estimate'] = grid_search.best_estimator_.feature_importances_

    return [grid_search, coeff]
    

#Elastic Net model function with Pipeline for scaling
def train_RegressionElNet_model(X_train, y_train, hyper_parameters):
    
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    steps = [('standardscaler', StandardScaler()), ('elasticnet', ElasticNet())]
    el_net_pipeline = Pipeline(steps)
    
    cv = GridSearchCV(el_net_pipeline, hyper_parameters, scoring=['neg_root_mean_squared_error', 'r2'],
                      refit='neg_root_mean_squared_error', n_jobs=5, verbose=10, cv=5)
    cv.fit(X_train, y_train)
    
    # Check cv results including mean error:
    cv_rmse = round(cv.best_score_, 2)
    print('CV Best RMSE Score: ' + str(cv_rmse))
    # Dataframe with coefficients
    coeff = pd.DataFrame()
    coeff["Columns"] = X_train.columns
    coeff['Coefficient Estimate'] = pd.Series(cv.best_estimator_.steps[1][1].coef_)

    return [cv, coeff]



from sklearn.preprocessing import PolynomialFeatures
def train_RegressionElNetSecondOrder_model(X, y, hyper_parameters, test_size, random_state):
    def count_elements_in_string(my_list, my_string):
        element_counts = []
        for item in my_list:
            count = my_string.count(item)
            if count == 2:
                element_counts.append(item)
        return element_counts

    el_net_pipeline= ('elasticnet', ElasticNet())
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    
    X_poly = poly.fit_transform(X)

    # Get the names of the generated feature names
    feature_names = poly.get_feature_names_out(X.columns)

    # Convert the transformed array back to a DataFrame
    feature_prefixes = X.columns.str.split('_').str[0].unique()

    # Keep only the interaction terms involving different feature prefixes
    #filtered_feature_names = [name for name in feature_names if not len(count_elements_in_string(feature_prefixes, name)) ==1]

    filtered_feature_indices = [i for i, name in enumerate(feature_names) if not len(count_elements_in_string(feature_prefixes, name)) ==1]

    # Filter the transformed array to include only the desired features
    X_filtered = X_poly[:, filtered_feature_indices]

    # Convert the filtered array back to a DataFrame (optional)
    df_filtered = pd.DataFrame(X_filtered, columns=np.array(feature_names)[filtered_feature_indices])

    X_train, X_test, y_train, y_test = train_test_split(df_filtered, y, test_size=test_size, random_state=random_state)
    
    steps = [('standardscaler', StandardScaler()), el_net_pipeline]
    pipe = Pipeline(steps)
    
    cv = GridSearchCV(pipe, hyper_parameters, scoring=['neg_root_mean_squared_error', 'r2'],
                      refit='neg_root_mean_squared_error', n_jobs=5, verbose=10, cv=5)
    cv.fit(X_train, y_train)
    
    # Check cv results including mean error:
    cv_rmse = round(cv.best_score_, 2)
    print('CV Best RMSE Score: ' + str(cv_rmse))
    # Dataframe with coefficients
    coeff = pd.DataFrame()
    coeff["Columns"] = X_test.columns
    coeff['Coefficient Estimate'] = pd.Series(cv.best_estimator_.steps[1][1].coef_)

    return [cv,coeff, X_test, y_test]




def train_SupportVectorRegression_model(X_train, y_train, param_grid):

    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define a Support Vector Regressor (SVR)
    svr_regressor = SVR()

    # Initialize GridSearchCV with cross-validation
    grid_search = GridSearchCV(estimator=svr_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train, y_train.values.ravel())

    # Get the best hyperparameters from GridSearchCV
    return [grid_search]



# Custom Dataset for surgical procedure time prediction
class TimeDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    


class TimePredictionModel(nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(TimePredictionModel, self).__init__()
        
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        #self.n_layers = n_layers
        
        if not isinstance(hidden_dim, int):
            layers = [nn.Linear(num_features, hidden_dim[0]), nn.ReLU()]

            # Dynamically add hidden layers
            for i in range(len(hidden_dim) - 1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(nn.Linear(hidden_dim[-1], 1))
        else:
            layers = [nn.Linear(num_features, hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(hidden_dim, 1))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


    

def train_nn_model(model, num_epochs, train_loader, optimizer, criterion):
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            #train_loss += loss.item() * data.size(0)    
    
    
def train_nn_model_compre( hidden_dim, learning_rate, num_epochs, X_encoded, y):

    # Split the data into training and test sets
    X_train, X_test_df, y_train, y_test_df = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    X_train = X_train.to_numpy()
    X_test = X_test_df.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test_df.to_numpy()

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create the custom datasets and data loaders
    train_dataset = TimeDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = TimeDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate the model and define the loss function and optimizer
    model = TimePredictionModel(X_encoded.shape[1], hidden_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print('model training')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)

        # Evaluation on the test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, targets in test_loader:
                #categorical_data = data[:, :encoded_categorical_data.shape[1]]
                #numerical_data = data[:, encoded_categorical_data.shape[1]:]
                outputs = model(data)
                loss = criterion(outputs, targets.unsqueeze(1))
                test_loss += loss.item() * data.size(0)

        # Print the epoch statistics
        train_loss /= len(train_dataset)
        test_loss /= len(test_dataset)
        #print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    return [model, X_test_df, y_test_df]
    



def search_features_values_from_obj(obj, features):
    result = {}

    # Search for features in patient attributes
    for feature in features:
        attr = check_attribute_presence(obj, feature)
        if attr is not None:
            result[feature] = getattr(obj, attr)

        elif feature in obj.additional_attributes:
            result[feature] = obj.additional_attributes[feature]

    return result


def select_feature_values_from_data_file(csv_file, features, feature, value):
    # Load CSV data into a DataFrame
    df = pd.read_csv(csv_file)
    
    # Boolean indexing to select rows with matching feature value
    selected_rows = df[df[feature] == value]
    
    # Filter the selected rows to include only the specified features
    selected_features = selected_rows[features]
    
    # Convert the selected features DataFrame to a single dictionary
    selected_dict = selected_features.iloc[0].to_dict()
    
    return selected_dict


def identify_encoded_features(original_features, modified_features):
    encoded_features = []
    original_set = set(original_features)
    for feature in modified_features:
        if feature not in original_set:
            original_feature = feature.rsplit('_', 1)[0]
            if original_feature in original_features and original_feature not in encoded_features:
                encoded_features.append(original_feature)

    return encoded_features



def get_procedure_time(model_type,patient_data_file, feature_variables_names, model_sub_level, target_data_type, patient_objects, surgery_objects, model_retrain = False):
    model_name = create_model_name(model_type, target_data_type, feature_variables_names,model_sub_level)
    if os.path.exists(os.path.join('Trained_models',model_name +'.pkl')) and not model_retrain:
        with open(os.path.join('Trained_models',model_name +'.pkl'), 'rb') as file:
            model_train_results = pickle.load(file)
            
    else:
        raise ValueError("The corresponding trained model is not available for prediction in the folder: Trained_models")
     
    input_data_encoded = prepare_input_data(patient_data_file, feature_variables_names, model_train_results[1],patient_objects, surgery_objects)
    #encoded_features = identify_encoded_features(feature_variables_names, model_train_results[1].columns.tolist())
    #non_encoded_features = [feature for feature in feature_variables_names if feature not in encoded_features]
   
    return predict_with_model(model_type,model_train_results[0], input_data_encoded)
    


def plot_actual_vs_Planned(data_file, Speciality_str):
    
    df_all = Dataset(data_file)
    
    df_spec_selec = df_all.loc[df_all.Specialty == Speciality_str]
    
    df_data = df_spec_selec[['Hosp',
                         'planned duration',
                         'actual duration']]
    
    df_data = df_data.dropna()

    df_data = df_data.loc[(df_data['actual duration'] > 0) & (df_data['actual duration'] < 500)]

    df_data.rename(columns={"actual duration": "Actual Duration (min)", "planned duration": "Planned Duration (min)"}, inplace=True)

    within_planned_time = df_data.loc[df_data["Actual Duration (min)"] <= df_data["Planned Duration (min)"]]

    overrun = df_data.loc[df_data["Actual Duration (min)"] > df_data["Planned Duration (min)"]]

    df_data['30_min_before'] = df_data["Planned Duration (min)"] - 30

    underrun = df_data.loc[df_data["Actual Duration (min)"] < df_data["30_min_before"]]

    N_ops_outside_planned_time = underrun.shape[0] + overrun.shape[0]

    N_of_proc_within_30_min = df_data.shape[0] - underrun.shape[0] - overrun.shape[0]

    #Plot Values

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_data, x="Planned Duration (min)", y="Actual Duration (min)", hue="Hosp",
                ax=ax).set(title="T&O Actual time vs Planned (min)")
    ax.set_xlim(0,400)
    ax.plot(df_data["Planned Duration (min)"], df_data["Planned Duration (min)"], color='maroon', linewidth=2)

