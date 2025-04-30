
#import packages

import pandas as pd   
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler   ## scikit version 1.3.0, previously used 1.0.2

import matplotlib.pyplot as plt
#from joblib import dump, load         #import if required - to extract model
import pickle
import os
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import random
import time
import copy
#from predictive_model_related_functions import create_model_name
#from scipy import stats
"""------------------------------------------------------------------"""


def add_count_column(df, columns_to_count, new_col):
    # Calculate the count of non-missing and non-zero values for each row
    df[new_col] = df[columns_to_count].apply(lambda row: (row != 0) & (pd.notna(row))).sum(axis=1)
    return df


def age_to_group(age, cutoffs = None):
    if cutoffs is None:
        cutoffs = [5, 15, 30, 50, 60, 70, 80]
    for i, cutoff in enumerate(cutoffs):
        if age <= cutoff:
            if i == 0:
                return f"0to{cutoff}"
            elif i == len(cutoffs) - 1:
                return f"{cutoffs[i-1]+1}to{cutoff}"
            else:
                return f"{cutoffs[i-1]+1}to{cutoff}"
    return f"Above{cutoffs[-1]}"



def annonate_data_count_over_boxplot(ax_or_axes, df, y_col, x_col, hue_col = None, col_col= None):
    
    def annonate_data_count_for_given_axis(ax, facet_df, x_data, hue_order):
        
        for i, cat in enumerate(x_data):
            #print(cat)

            x_data_filtered = facet_df[facet_df[x_col] == cat]

            for j, subcat in enumerate(hue_order):
                #print(subcat)
                #postion = i + (j * 1/(len(hue_levels)+1)) - 0.25  # Adjust the offset as needed
                #sub_box_positions.append(position)
                x_offset = (j - (len(hue_order) - 1) / 2) * 1/(len(hue_order) +1)
                #x_offset = -0.25 + j* 1/(len(hue_levels))

                #unique_combinations.append(f"{cat}_{subcat}")

                num_points = len(x_data_filtered[x_data_filtered[hue_col] == subcat])
                #print(num_points)
                if num_points > 0:
                    ax.text(i+x_offset, np.mean(x_data_filtered[x_data_filtered[hue_col] == subcat][y_col]),
                            f'n={num_points}', ha='center', rotation=90, fontsize=12, weight='bold', color='black')


    if col_col is not None:
        
        #having the ordered facet_col, x_col and hue_order
        facet_columns= list(ax_or_axes.col_names)
        x_data = [label.get_text() for label in ax_or_axes.axes.flat[0].get_xticklabels()]
        #print(facet_columns) # Get the facet column names
        #print(x_data)
        hue_order = []
        
        for ax_id, ax in enumerate(ax_or_axes.axes.flat):
            if len(hue_order) == 0:
                handles, labels = ax.get_legend_handles_labels()
                hue_order = [int(label) if label.isdigit() else label for label in labels]
                #print(hue_order)
            
            facet_df = df[df[col_col]== facet_columns[ax_id]]
            #print(f'Size of overall data for {ax_id+1} th facet is: {facet_df.shape[0]}')
            annonate_data_count_for_given_axis(ax, facet_df, x_data, hue_order)
            
    if col_col is None and hue_col is not None:
        
        x_data =  [label.get_text() for label in ax_or_axes.get_xticklabels()]
        #print(x_data)
            
        handles, labels = ax_or_axes.get_legend_handles_labels()
        
        hue_order = [int(label) if label.isdigit() else label for label in labels]
        #print(hue_order)
        
        annonate_data_count_for_given_axis(ax_or_axes, df, x_data, hue_order)
    


    
    
def create_faceted_boxplot(df, y_col, x_col, hue_col,  col_col):
    
    # Create the boxplot with facets and styling
    g = sns.catplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        kind="box",
        #palette="viridis",
        aspect=0.8,  # Adjust aspect ratio for facet_grid-like appearance
        col=col_col,  # Facet by the specified column
    )

    # Customize plot appearance
    g.set_axis_labels(x_col, y_col)
    
    #g.set(ylim=(0, 110))  # Set the y-axis limits
    # Set additional customizations
    plt.subplots_adjust(top=0.85)  # Adjust the space for facet titles
    

    # Rotate x-axis labels
    g.set_xticklabels(rotation=45, ha="right")

            
     # Set facet column labels
    g.set_titles(col_template= col_col+": {col_name}")
    
    # Modify the legend position (middle of the plot)
    g._legend.set_bbox_to_anchor((0.8, 0.8))
    g._legend.set_title(f"{hue_col}")  # Customize the legend title
    
    return g
    


def categorise_IO_with_encoding(df, string_columns, encoding_type='label'):
    # Extract input and output data
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # Last column
    
    # Encode string variables
    encoders = {}
    for col in string_columns:
        if encoding_type == 'label':
            encoders[col] = LabelEncoder()
            X[col] = encoders[col].fit_transform(X[col])
        elif encoding_type == 'one-hot':
            encoder = OneHotEncoder(sparse=False, drop='first')
            encoded_cols = pd.DataFrame(encoder.fit_transform(X[[col]]))
            encoded_cols.columns = [col + '_' + str(value) for value in encoder.categories_[0][1:]]
            X = pd.concat([X, encoded_cols], axis=1)
            X = X.drop(col, axis=1)
        else:
            raise ValueError("Invalid encoding type. Please choose either 'label' or 'one-hot'.")
    
    # Return the categorized and encoded data
    return X, y


def categorize_columns(df):
    string_columns = []
    numerical_columns = []

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            string_columns.append(col)
        else:
            numerical_columns.append(col)

    return string_columns, numerical_columns



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




def Dataset(file_name):
    df = pd.read_csv(file_name)
    #df["planned start date/time"]= pd.to_datetime(df["planned start date/time"])
    return(df)



def data_prep_first_step_after_SQl_extraction(input_file_name, output_file_name, procedures_columns):
    
    def add_count_column(df1, columns_to_count, new_col):
        # Calculate the count of non-missing and non-zero values for each row
        #df1[new_col] = df1[columns_to_count].apply(lambda row: (row != 0) & (pd.notna(row))).sum(axis=1)
        # Normalize column names for case-insensitive lookup
        columns_mapping = {col.lower(): col for col in df1.columns}
        # Map the columns_to_count to their original case using the mapping
        columns_to_count_mapped = [columns_mapping[col.lower()] for col in columns_to_count]
        # Perform the operation
        df1[new_col] = df1[columns_to_count_mapped].apply(lambda row: (row != 0) & (pd.notna(row))).sum(axis=1)

    def categorise_covid_period(date):
        if date < pd.Timestamp('2020-03-01'):
            return 'pre-covid'
        elif date >= pd.Timestamp('2020-03-01') and date <= pd.Timestamp('2021-03-30'):
            return 'covid'
        else:
            return 'post-covid'
    
    # Check the file extension and read the file accordingly
    if input_file_name.endswith('.csv'):
        df = pd.read_csv(input_file_name)
        print(f"DataFrame loaded from CSV: {input_file_name}")
    elif input_file_name.endswith('.xlsx'):
        df = pd.read_excel(input_file_name, engine='openpyxl')  # Use openpyxl for .xlsx files
        print(f"DataFrame loaded from Excel file: {input_file_name}")
    else:
        raise ValueError("Unsupported file format. Please use a .csv or .xlsx file.")

    
    df = df.replace(['Unknown', '^Not Stated', 'Not Known'], np.nan, regex=True)

    #add_count_column(df, ['Actual Procedure 1 Code 1','Actual Procedure 2 Code 1','Actual Procedure 3 Code 1'], 'N_procedures')
    add_count_column(df, procedures_columns, 'N_procedures')

    anaesthetic_cols = ['anaesthetist 1 Surname, Forename',
                      'anaesthetist 2 Surname, Forename',
                      'anaesthetist 3 Surname, Forename']
    if not 'Anaesthetist Count' in df.columns:
        add_count_column(df, anaesthetic_cols, 'Anaesthetist Count')

    df = df.replace({'ETHNICITY': {'White - Any other White background' : 'Any other White background',
                          'Any Other Ethnic Group': 'Any other ethnic group',
                          'Black African or Black British African' : 'Black or Black British - African',
                         'Any other Black background': 'Black or Black British - Any other Black background',
                          'Black Caribbean or Black British Caribbean':'Black or Black British - Caribbean',
                          'White - British': 'White British',
                          'White - Irish' : 'White Irish',
                          'Mixed - White and Asian':'Mixed White and Asian',
                          'Mixed - White and Black Caribbean' : 'Mixed White and Black Caribbean',
                          'Mixed - Any other background': 'Any other mixed background',
                          'Mixed - White and Black African' : 'Mixed White and Black African',
                          'Pakistani or British Pakistani': 'Asian or Asian British - Pakistani',
                          'Other Ethnic Group - Chinese': 'Asian or Asian British - Chinese',
                          'Chinese': 'Asian or Asian British - Chinese',
                          'Bangladeshi or British Bangladeshi':'Asian or Asian British - Bangladeshi',
                          'Asian - other' : 'Asian or Asian British - Any other Asian background',
                          'Indian or British Indian':'Asian or Asian British - Indian'}})


    white = ['White British', 'White Irish', 'Any other White background']
    other_ethnic_group = ['Any other ethnic group',
                          'Black or Black British - African',
                          'Black or Black British - Any other Black background',
                          'Black or Black British - Caribbean',
                          'Mixed White and Asian',
                          'Mixed White and Black Caribbean',
                          'Any other mixed background',
                          'Mixed White and Black African',
                          'Asian or Asian British - Pakistani',
                          'Asian or Asian British - Chinese',
                          'Asian or Asian British - Chinese',
                          'Asian or Asian British - Bangladeshi',
                          'Asian or Asian British - Any other Asian background',
                          'Any other Asian background',
                          'Asian or Asian British - Indian']
    
    conditions = [df['ETHNICITY'].isin(white), df['ETHNICITY'].isin(other_ethnic_group),
                  df['ETHNICITY'].isna()]
    
    categories = ['Any white background', 'Any other ethnic background', "None"]
    
    df['Ethnicity_grouped'] = np.select(conditions, categories)

    Codes_for_conditions = {
        'IHD': ['I200', 'I201', 'I208', 'I209', 'I210', 'I211', 'I212', 'I213', 'I214', 'I219', 'I220', 'I221', 'I228', 'I229', 
                'I230', 'I231', 'I232', 'I233', 'I234', 'I235', 'I236', 'I238', 'I240', 'I241', 'I248', 'I249', 'I250', 'I251', 'I252', 'I253', 'I254', 'I255', 'I256', 'I258', 'I259'],
        'PAD': ['I7021', 'I7020', 'I7081', 'I7080', 'I7091', 'I7090', 'I72', 'I730', 'I731', 'I738', 
                'I739', 'I742', 'I743', 'I744', 'I745', 'I748', 'I749', 'I77', 'I792', 'I798'],
        'Myocardial infraction': ['I21', 'I22', 'I23', 'I252', 'I258'],
        'Cerebral vascular': ['G450', 'G451', 'G452', 'G454', 'G458', 'G459', 'G46', 'I60',
                              'I61', 'I62', 'I63', 'I64', 'I65', 'I66', 'I67', 'I68', 'I69'],
        'Congestive heart failure': ['I500'],
        'Connective tissue disorder': ['M05', 'M060', 'M063', 'M069', 'M32', 'M332', 'M34', 'M353'],
        'Dementia': ['F00', 'F01', 'F02', 'F03', 'F051'],
        'Hypertension': ['I10', 'I11', 'I12', 'I13', 'I15', 'I16'],
        'Diabetes': ['E101', 'E105', 'E106', 'E108', 'E109', 'E111', 'E115', 'E116', 'E118', 'E119', 'E131', 'E135', 'E136', 'E138', 'E139', 'E141', 'E145', 'E146', 'E148', 'E149'],
        'Liver disease': ['K702', 'K703', 'K717', 'K73', 'K74'],
        'Obesity': ['Z68', 'E66'],
        'Peptic ulcer': ['K25', 'K26', 'K27', 'K28'],
        'Periph vasc disease': ['I71', 'I739', 'I790', 'R02', 'Z958', 'Z959']
    }

    secondary_code_cols = df.columns[df.columns.str.startswith('Secondary_Diagnosis')][0:-1:2]

    for key, string_list in Codes_for_conditions.items():
        #print(string_list)
        #default value is 0
        diagnosis_result = np.zeros(len(df))
        for index, row in df.iterrows():
            for diag_code in row[secondary_code_cols]:  # Exclude the last column (NewColumn)
                if pd.isna(diag_code):
                    break
                if any(s in diag_code for s in string_list):
                    diagnosis_result[index]= 1
                    break  # Stop checking if a string is found in any column
        df[key] = diagnosis_result

    if 'AGE_ON_ADMISSION' in df.columns:
        df['Age group at admit'] = df['AGE_ON_ADMISSION'].apply(lambda x: age_to_group(x, [5, 15, 30, 50, 60, 70, 80]))
            
    # Check and proceed line by line
    if 'Arrived Date/Time' in df.columns:
        df['Arrived Date/Time'] = pd.to_datetime(df['Arrived Date/Time'], format="%d/%m/%Y %H:%M")
    
    if 'Out of Theatre Date/Time' in df.columns:
        df['Out of Theatre Date/Time'] = pd.to_datetime(df['Out of Theatre Date/Time'], format="%d/%m/%Y %H:%M")
    
    if 'Arrived Date/Time' in df.columns and 'Out of Theatre Date/Time' in df.columns and 'H4 Minutes' in df.columns:
        df['Out of Theatre Date/Time'] = df.apply(
            lambda row: row['Arrived Date/Time'] + pd.to_timedelta(row['H4 Minutes'], unit='minutes')
            if pd.isnull(row['Out of Theatre Date/Time']) else row['Out of Theatre Date/Time'], 
            axis=1
        )
    
    if 'Discharge Date/Time' in df.columns and 'Admission Date/Time' in df.columns:
        df['Actual LOS'] = (
            pd.to_datetime(df['Discharge Date/Time'], format="%d/%m/%Y %H:%M") - 
            pd.to_datetime(df['Admission Date/Time'], format="%d/%m/%Y %H:%M")
        ).dt.total_seconds() / (24 * 3600)
    
    if 'Discharge Date/Time' in df.columns and 'Out of Theatre Date/Time' in df.columns:
        df['Post-Op LOS'] = (
            pd.to_datetime(df['Discharge Date/Time'], format="%d/%m/%Y %H:%M") - 
            pd.to_datetime(df['Out of Theatre Date/Time'], format="%d/%m/%Y %H:%M")
        ).dt.total_seconds() / (24 * 3600)
    
    if 'Admission Date/Time' in df.columns:
        df['Admission Date/Time'] = pd.to_datetime(df['Admission Date/Time'], format="%d/%m/%Y %H:%M")
        df['Month'] = df['Admission Date/Time'].apply(lambda x: x.strftime('%B'))
        df['Covid Flag'] = df['Admission Date/Time'].apply(categorise_covid_period)
    
    if 'Planned Start Date/Time' in df.columns:
        df['Day of the week'] = pd.to_datetime(df["Planned Start Date/Time"], format="%d/%m/%Y %H:%M").dt.day_name()

    if not 'Patient Classification' in df.columns:
        if 'Intended Management' in df.columns and not df['Intended Management'].isna.any():
            df['Patient Classification'] = df['Intended Management'].apply(lambda x: 'Day case admission' if x == 'Daycase' else 'Ordinary admission')
        elif 'RTT WL Category' in df.columns:
            df['Patient Classification'] = df['RTT WL Category'].apply(lambda x: 'Day case admission' if x == 'Daycase' else 'Ordinary admission')
    
    if not 'anaesthetist Expected?' in df.columns:
        df['anaesthetist Expected?'] = 'Unknown'
    
    # Check the file extension and save accordingly
    if output_file_name.endswith('.csv'):
        df.to_csv(output_file_name, index=False)
        print(f"DataFrame saved as CSV: {output_file_name}")
    elif output_file_name.endswith('.xlsx'):
        df.to_excel(output_file_name, index=False, engine='openpyxl')  # Use openpyxl for .xlsx files
        print(f"DataFrame saved as Excel file: {output_file_name}")
    else:
        print("Unsupported file format. Please use a .csv or .xlsx file.")

    return df



#define functions
def unique_values(df):
    counts = []
    for col in df.columns:
        temp_name = df[col].value_counts().to_frame()
        temp_name.name = col
        counts.append(temp_name)
        
    return(counts)

def descriptive_stats(df):
    numeric = df.select_dtypes(include=np.number)

    descriptives = []
    for col in numeric.columns:
        temp = df[col].describe().round()
        temp.name = col
        descriptives.append(temp)
        d = pd.DataFrame(descriptives)
       
    return(d)


def encoding_and_normalising_data(raw_data, string_columns, numerical_columns, encoder, scaler):
    # Copy the raw data to avoid modifying it directly
    data = raw_data.copy()

    # Normalize numerical variables
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Perform one-hot encoding for categorical variables
    encoded_data = encoder.fit_transform(data[string_columns])

    # Concatenate encoded categorical data with numerical data
    preprocessed_data = np.concatenate((encoded_data, data[numerical_columns]), axis=1)

    return preprocessed_data, encoded_data


def filter_dataset_till_possible(df, filtered_level, external_categories, threshold):
    # Filter DataFrame based on external_categories
    filtered_df = df.copy()
    selected_indices = []
    
    for col in external_categories.columns:
        
        if col not in filtered_df.columns and filtered_level == 0:
            break
        
        if col not in filtered_df.columns and filtered_level != 0:
            continue
        
        current_df = filtered_df[filtered_df[col] == external_categories.loc[0,col]]
        
        if len(current_df) == 0 and filtered_level == 0:
            break
        
        if len(current_df) == 0 and filtered_level != 0:
            continue
        
        # Update selected_indices with the current_df indices
        selected_indices = list(set(current_df.index))
        
        # Check if the length falls below the threshold
        if len(current_df) < threshold and filtered_level ==0:
            filtered_level+=1
            break
        
        if len(current_df) < threshold and filtered_level !=0:
            #print(f"Filtered DataFrame size ({len(current_df)}) is below the threshold ({threshold}).")
            # Fill up the remaining data count from previously selected indices
            remaining_count = threshold - len(current_df)
            remaining_indices = list(set(filtered_df.index) - set(selected_indices))
            
            # Randomly sample remaining_count indices
            #print(remaining_indices)
            if len(remaining_indices) > 0:
                random_remaining_indices = random.sample(remaining_indices, remaining_count)
                selected_indices += random_remaining_indices
            
            break
        
        filtered_df = current_df.copy()
        #print(f'selected indices are: {selected_indices}')
        filtered_level+=1
        #print(filtered_level)
    
    '''
    if filtered_level == 0 and len(selected_indices) == 0:
        # Randomly sample remaining_count indices
        remaining_indices = list(filtered_df.index)
        random_remaining_indices = random.sample(remaining_indices, threshold)
        selected_indices += random_remaining_indices
    '''

    return filtered_df.loc[selected_indices], filtered_level


def generate_box_plot_old(df, category_columns_with_limits, value_column,  figsize=(10, 6), print_data_count = False):
    output_folder = 'Pre_analysis'
    """
    Generate a box plot based on the number of category columns and limited unique values.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        category_columns (list of str): List of category column names. If there's only one column, it's single column box plotting; otherwise, mixed plotting is performed.
        value_column (str, optional): The name of the value column for mixed plotting. If None, mixed plotting without values is generated.
        figsize (tuple): Figure size for the generated plot. Default is (10, 6).

    Returns:
        None (displays and saves the plot).
    """
    dataframe = df.copy()
    category_columns = list(category_columns_with_limits.keys())
    plt.figure(figsize=figsize)
    
    if len(category_columns) == 1:
        # Single column box plotting
        category_column = category_columns[0]

        if category_columns_with_limits[category_column] is not None:
            top_categories = dataframe[category_column].value_counts().nlargest(category_columns_with_limits[category_column]).index
            dataframe = dataframe[dataframe[category_column].isin(top_categories)]

        if value_column is not None:
            ax= sns.boxplot(data=dataframe, x=category_column, y=value_column)
            plt.ylabel(value_column)
        else:
            ax= sns.boxplot(data=dataframe, x=category_column)
        
    elif len(category_columns) == 2:
        
        #filtering the dataframe to the top most value counts under given category
        for catego in category_columns:
            max_data  = category_columns_with_limits.get(catego, None)
            if not max_data is None:
                top_count_data = dataframe[catego].value_counts().nlargest(max_data).index
                dataframe = dataframe[dataframe[catego].isin(top_count_data)]
        
        category1_name, category2_name = category_columns[0:2]
        
        if value_column is not None:
            
            #dataframe[category1_name] = pd.Categorical(dataframe[category1_name], categories=dataframe[category1_name].unique(), ordered=True)
            
            dataframe = dataframe.sort_values(by = category1_name)

            ax = sns.boxplot(data=dataframe, x=category1_name, y=value_column, hue=category2_name)
            
            handles, labels = ax.get_legend_handles_labels()
            
            hue_order = [int(label) if label.isdigit() else label for label in labels]
            
            #print(hue_order)
            #hue_order = list(dataframe[category2_name].unique())
            
        plt.title(f'Box Plot for Unique {category1_name} and {category2_name} \n Combined Data against {value_column}')
        
    elif len(category_columns) == 3:
        
        #filtering the dataframe to the top most value counts under given category
        for catego in category_columns:
            max_data  = category_columns_with_limits.get(catego, None)
            if not max_data is None:
                top_count_data = dataframe[catego].value_counts().nlargest(max_data).index
                dataframe = dataframe[dataframe[catego].isin(top_count_data)]
        
        category1_name, category2_name = category_columns[0:2]
        
        '''dataframe[category1_name] = pd.Categorical(dataframe[category1_name], categories=dataframe[category1_name].unique(), ordered=True)   
        dataframe = dataframe.sort_values(by = category1_name)
        dataframe.reset_index(drop=True, inplace=True)'''
        dataframe =  dataframe.sort_values(by=[category1_name, category2_name])
        
        dataframe['CombinedCategory'] = dataframe[category1_name] + '_' + dataframe[category2_name]
        #dataframe['CombinedCategory'] = pd.Categorical(dataframe['CombinedCategory'], categories=dataframe['CombinedCategory'].unique(), ordered=True)

        #dataframe = dataframe.sort_values(by = 'CombinedCategory')
        
        dataframe.reset_index(drop=True, inplace=True)

        ax = sns.boxplot(x='CombinedCategory', y=value_column, hue=category_columns[-1], data=dataframe, palette="Set2")
        
        handles, labels = ax.get_legend_handles_labels()
        hue_order = [int(label) if label.isdigit() else label for label in labels]
        #print(hue_order)
        
             
        #stripplot_i.set_xticks(range(i, i+1))
        #print(box_positions)
        #print(default_xtick_positions)
        #ax.set_xticks(box_positions)
        #ax.set_xticklabels(unique_combinations, rotation=90)
        #plt.xticks(box_positions, unique_combinations, rotation=45, ha='right')
        
        plt.title(f'Box Plot for combined {category1_name} and {category2_name} distributed for {category_columns[-1]} \n Combined Data against {value_column}')
        
    if print_data_count:
        
        box_positions = []
        unique_combinations = []
        default_xtick_positions = ax.get_xticks()
        root_category = category1_name if len(category_columns) <=2 else 'CombinedCategory'
        hue_category = category_columns[-1]
        hue_levels = list(dataframe[category_columns[-1]].unique())
        for i, cat in enumerate(list(dataframe[root_category].unique())):
            #print(cat)
            sub_box_positions = []
            frst_step_data_filtered = dataframe[dataframe[root_category] == cat]
            for j, subcat in enumerate(hue_order):
                #print(subcat)
                #postion = i + (j * 1/(len(hue_levels)+1)) - 0.25  # Adjust the offset as needed
                #sub_box_positions.append(position)
                x_offset = (j - (len(hue_levels) - 1) / 2) * 1/(len(hue_levels) +1)
                #x_offset = -0.25 + j* 1/(len(hue_levels))

                #unique_combinations.append(f"{cat}_{subcat}")

                num_points = len(frst_step_data_filtered[frst_step_data_filtered[hue_category] == subcat])
                #print(num_points)
                if num_points > 0:
                    ax.text(i+x_offset, np.mean(frst_step_data_filtered[frst_step_data_filtered[hue_category] == subcat][value_column]),
                            f'n={num_points}', ha='center', rotation=90, fontsize=10, weight='bold', color='black')

            #box_positions.extend(sub_box_positions)

    
    plt.ylabel(value_column)
        
    
    #plt.xlabel(category1_name)
    #dataframe['CombinedCategory'].unique()
    plt.legend(title=category_columns[-1])
    plt.xticks(rotation=65, ha='right')
    #plt.savefig(f'{output_folder}/Box_plot_combined_{category1_name}_{category2_name}.png')  # Save the plot
    plt.show()
    
    return dataframe



def generate_box_plot(df, category_columns_with_limits, value_column,  figsize=(10, 6), legend_location = (0.8,0.8), print_data_count = False):
    output_folder = 'Pre_analysis'
    """
    Generate a box plot based on the number of category columns and limited unique values.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        category_columns (list of str): List of category column names. If there's only one column, it's single column box plotting; otherwise, mixed plotting is performed.
        value_column (str, optional): The name of the value column for mixed plotting. If None, mixed plotting without values is generated.
        figsize (tuple): Figure size for the generated plot. Default is (10, 6).

    Returns:
        None (displays and saves the plot).
    """
    dataframe = df.copy()
    
    category_columns = list(category_columns_with_limits.keys())
    
    for catego in category_columns:
        
        max_data  = category_columns_with_limits.get(catego, None)
        if not max_data is None:
            if isinstance(max_data, (list, tuple)):
                top_count_data = [a for a in max_data]
            else:
                top_count_data = dataframe[catego].value_counts().nlargest(max_data).index
            
            dataframe = dataframe[dataframe[catego].isin(top_count_data)]
    
    plt.figure(figsize=figsize)
    
    if len(category_columns) == 1:
        # Single column box plotting
        category_column = category_columns[0]
        
        if value_column is not None:
            ax= sns.boxplot(data=dataframe, x=category_column, y=value_column)
            plt.ylabel(value_column)
        else:
            ax= sns.boxplot(data=dataframe, x=category_column)
        
    elif len(category_columns) == 2:
        
        #filtering the dataframe to the top most value counts under given category
            
        category1_name, category2_name = category_columns[0:2]
            
        #dataframe[category1_name] = pd.Categorical(dataframe[category1_name], categories=dataframe[category1_name].unique(), ordered=True)
        dataframe = dataframe.sort_values(by = category1_name)

        ax = sns.boxplot(data=dataframe, x=category1_name, y=value_column, hue=category2_name)
        #ax.legend(bbox_to_anchor = legend_location)
        
        if print_data_count:
            
            annonate_data_count_over_boxplot(ax, dataframe, value_column, category1_name, hue_col = category2_name)
            
        plt.title(f'Box Plot for Unique {category1_name} and {category2_name} \n Combined Data against {value_column}')
        
        
    elif len(category_columns) == 3:
        
        g = create_faceted_boxplot(dataframe, value_column, category_columns[1], category_columns[2], category_columns[0])
        g._legend.set_bbox_to_anchor(legend_location)
        
        #g.set_titles(f'Box Plot on {value_column} for {category_columns[1]} distributed \n for {category_columns[2]} with facet grid for {category_columns[0]} ')
        
        if print_data_count:
            annonate_data_count_over_boxplot(g, dataframe, value_column,category_columns[1] ,category_columns[2], category_columns[0])
        
    #plt.show()

    return dataframe, ax, plt.gcf()



def get_similar_data_from_datasets(min_dataset_count, predic_model_type, model_input_features, time_variable_name,
                                                model_nd_data_categ, categorised_modelling, reference_data, data_type = 'test',
                                   preprocessed_dataset = None, force_encoding_col = [],  similar_cols_for_encoding = None):   
    
    #tenta_plan2.rename(columns={'Age': 'AGE_ON_ADMISSION'}, inplace=True)
    
    model_infeed_parameters = [predic_model_type, None, model_input_features, model_nd_data_categ]
    #string_cols_updated = string_columns.copy()
    
    if categorised_modelling:
        model_folder = os.path.join('Trained_models', 'Category_basis')
    else:
        model_folder = 'Trained_models'
    
    #testing_dataset = None
    categorisation_based_model_available = False
    categ_features_matching_count = 0
    
    if categorised_modelling:
        first_feature_data_val = reference_data.loc[0, model_input_features[0]]
        #first_feature_data_val = reference_data[model_input_features[0]]
        updated_parameters = shift_feature_to_data_category(model_infeed_parameters, model_input_features[0], 
                                                           first_feature_data_val)
        #print(updated_parameters)
        model_name = create_model_name(predic_model_type, time_variable_name, *updated_parameters[2:])
        #print(model_name)
        if os.path.exists(os.path.join(model_folder, model_name +'.pkl')):
            categorisation_based_model_available = True
            categ_features_matching_count+=1
            #model_input_value_data = row.loc[model_input_features[1:]].to_frame().T
            model_input_value_data = reference_data[model_input_features[1:]]
            #string_cols_updated.remove(model_input_features[0])

    if not categorisation_based_model_available:
        model_name = create_model_name(predic_model_type, time_variable_name, model_input_features, model_nd_data_categ)
        model_input_value_data = reference_data.copy()
        

    with open(os.path.join(model_folder, model_name +'.pkl'), 'rb') as file:
        model_training_res = pickle.load(file)

    #select from the testing dataset first
    if 'test' in data_type.lower():
        related_dataset_input_encoded = model_training_res[-2].reset_index(drop=True)
        related_y_value = model_training_res[-1].reset_index(drop=True)
        
    else:
        if categorisation_based_model_available:
            preprocessed_dataset_updated = preprocessed_dataset.loc[preprocessed_dataset[model_input_features[0]] == first_feature_data_val]
            preprocessed_dataset_updated = preprocessed_dataset_updated.drop(model_input_features[0], axis=1)
        else:
            preprocessed_dataset_updated = preprocessed_dataset.copy()
            
        input_output_data = prepare_IO_dataset([preprocessed_dataset_updated], 'one-hot', force_encoding_col, similar_cols_for_encoding = similar_cols_for_encoding)[0]
        related_dataset_input_encoded = input_output_data[0].reset_index(drop=True)
        related_y_value = input_output_data[1].reset_index(drop=True)
        
        #we have to make the dataset format similar to model's related testing dataset by removing non-related data
        extra_columns = [col for col in related_dataset_input_encoded.columns if col not in model_training_res[-2].columns]
        #print(extra_columns)
        for col in extra_columns:
            
            #as it is encoded already we have to remove both the data and the cols
            related_dataset_input_encoded = related_dataset_input_encoded[related_dataset_input_encoded[col] != 1]

            # Remove the column
            related_dataset_input_encoded = related_dataset_input_encoded.drop(col, axis=1)
        #print(related_dataset_input_encoded
        related_dataset_input_encoded = related_dataset_input_encoded.reindex(columns = model_training_res[-2].columns, fill_value=0)
        
    reference_data_encoded = one_hot_encode_dataframe(model_input_value_data, force_encoding_col)
    #print(reference_data_encoded)
    
    similar_dataset_encoded, categ_features_matching_count = filter_dataset_till_possible(related_dataset_input_encoded, 
                                                                          categ_features_matching_count, reference_data_encoded,min_dataset_count)
    
    #if predic_model_type
    #if categorisation_based_model is available, the first of the features is already matched so adding 1 if
    #if categorisation_based_model_available:
    #    categ_features_matched_index_count+=1
    
    related_y_value = related_y_value[similar_dataset_encoded.index].reset_index(drop=True)

    similar_dataset_encoded = similar_dataset_encoded.reset_index(drop=True)
    
    prediction_result = predict_with_model(predic_model_type, model_training_res[0], similar_dataset_encoded)

    return [reference_data_encoded, categ_features_matching_count, similar_dataset_encoded, related_y_value, prediction_result]





def identify_corresponding_feature(model_name_str, feature_name, essen_modelling_data_categ, possible_extra_catego, encoded_data):
    def is_string_in_list(string, lst):
        for item in lst:
            if string in item:
                return True
        return False

    if not is_string_in_list(feature_name, encoded_data.columns) and feature_name in possible_extra_catego:
        str_temp = model_name_str.split('_')[-2]
        return [str_temp for _ in range(len(encoded_data))]
    else:
        return reverse_one_hot_encoding(feature_name, encoded_data)




def one_hot_encode_dataframe(df, num_cols_for_encoding, categorical_cols = None):
    
    from sklearn.compose import ColumnTransformer

    if len(df) == 1:
        df1 = pd.DataFrame(df.to_dict())
    else:
        df1 = df.copy()

    #df1 = pd.DataFrame(df.to_dict())
    #df1.index = df.index
    
    index = df.index
    
    # Identify categorical (string) and numerical columns
    df1 = df1.infer_objects()  # Convert object columns to more specific types
    if categorical_cols is None:
        categorical_cols = df1.select_dtypes(include='object').columns.tolist() + num_cols_for_encoding

    #numerical_cols = df1.select_dtypes(include=['int', 'float']).columns.tolist()
    numerical_cols = df1.columns.difference(categorical_cols)

    # ColumnTransformer to handle different column types
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output = False), categorical_cols)
        ], remainder='passthrough')

    # Fit and transform the data
    
    encoded_data = preprocessor.fit_transform(df1)

    # Get the encoded column names
    encoded_column_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)

    # Combine encoded column names with numerical column names
    new_column_names = list(encoded_column_names) + list(numerical_cols)

    # Create a new dataframe with the encoded data and extended column names
    encoded_df = pd.DataFrame(encoded_data, columns=new_column_names)

    encoded_df.index = index

    return encoded_df



def one_hot_encoding_for_similar_columns(df1, similar_cols):
    df = df1.copy()
    
    for key, similar_col in similar_cols.items():

        # Creating a new DataFrame with Customer_ID and fruit/side dish columns
        #df_fruit_side = pd.concat([df['Customer_ID'], df[fruit_columns], df[side_dish_columns]], axis=1)
        df_col = df[similar_col].replace('Unknown', None)
        # Grouping fruits and side dishes by Customer_ID
        df_grouped = df_col.groupby(df.index).agg(lambda x: x.dropna().tolist())
        #print(df_grouped)
        #df_grouped = df_col.agg(lambda x: x.dropna().tolist())

        df_grouped = df_grouped.apply(lambda x: sum(x, []), axis=1)
        #print(df_grouped)

        # Initializing the MultiLabelBinarizer
        mlb = MultiLabelBinarizer()

        # Transforming the grouped DataFrame using one-hot encoding for each row
        #encoded_data = mlb.fit_transform(df_grouped.apply(lambda x: sum(x, []), axis=1))
        encoded_data = mlb.fit_transform(df_grouped)
        #encoded_data = mlb.fit_transform(df_grouped.apply(lambda x: x.split(', '), axis=1))

        # Creating a DataFrame with one-hot encoded columns
        encoded_df = pd.DataFrame(encoded_data, columns=mlb.classes_, index=df_grouped.index)
        # Creating a mapping of original prefixes

        #prefix_mapping = longest_common_substring(similar_col[0], similar_col[1])

        #print([similar_col[0], similar_col[1]])
        #print(prefix_mapping)
        new_column_names = {}
        for col in encoded_df.columns:
            #print(col)
            #if col != 'Customer_ID':
            new_column_names[col] = f'{key}_{col}'
        
        encoded_df.rename(columns=new_column_names, inplace=True)
        # Adding prefixes back to the column names
        # Creating new columns with correct prefixes

        # Merging the original DataFrame with the encoded DataFrame
        df = pd.concat([df, encoded_df], axis=1)
        #df = df.merge(encoded_df, how='left', left_on=df.index, right_index=True)

        # Dropping the original fruit and side dish columns
        df = df.drop(columns=similar_col)
    
    return df



# data exploration with box plots
def plot_data(axes, x, y, color, xlabel):

  # Plot the inputs x,y in the provided color
  axes.bar(x, y, color=color)

  # Set the x-axis label
  axes.set_xlabel(xlabel)
    
    
    
    
def plot_processed_data(preprocessed):
    cleaned_vals = unique_values(preprocessed)
    
    fig, ax = plt.subplots(7,3, figsize=(24,24))

    axe = ax.ravel()
    
    for i, dataframe in enumerate(cleaned_vals):
        plot_data(axe[i], cleaned_vals[i].index.values, 
                  cleaned_vals[i].iloc[:,0],
                  'blue', 
                  cleaned_vals[i].columns.values)
    plt.show()
    
    return fig



def preprocess_dataset(data_file, data_categ_dict, input_features_ess, input_features_addi, output_data_types, filtering_data, min_count_thresholds, z_scores_threshold, infrequent_data_replacement = None):

    
    df_selected = pd.read_csv(data_file, low_memory=False)
    for key in data_categ_dict.keys():
        df_selected = df_selected.loc[df_selected[key] == data_categ_dict[key]]
    
    for key, value in filtering_data.items():
        #if key in input_features:
        df_selected = df_selected[df_selected[key].isin(value)]  

    preprocessed_set = []
    for i in range(len(output_data_types)):
        filtered_df, removed_df = select_features_nd_target(df_selected, input_features_ess, input_features_addi, output_data_types[i], min_count_thresholds, z_scores_threshold, infrequent_data_replacement = infrequent_data_replacement)
        
        preprocessed_set.append([removed_df,filtered_df])
         
    return preprocessed_set


def prepare_IO_dataset(preprocessed_dataset, encoding_type, num_cols_for_encoding, similar_cols_for_encoding = None, string_cols = None):
    IO_sets = []
    for i, dataset in enumerate(preprocessed_dataset):
        if encoding_type == 'one-hot':
            i_o_data = []
            if similar_cols_for_encoding is not None:
                x_data = one_hot_encoding_for_similar_columns(dataset.iloc[:, :-1], similar_cols_for_encoding)
            else:
                x_data = dataset.iloc[:, :-1]
            
            i_o_data = [one_hot_encode_dataframe(x_data, num_cols_for_encoding, string_cols), dataset.iloc[:, -1]]
            
        IO_sets.append(i_o_data) 
        
    return IO_sets



def identify_encoded_features(original_features, modified_features):
    encoded_features = []
    original_set = set(original_features)
    for feature in modified_features:
        if feature not in original_set:
            original_feature = feature.rsplit('_', 1)[0]
            if original_feature in original_features and original_feature not in encoded_features:
                encoded_features.append(original_feature)

    return encoded_features


def reverse_one_hot_encoding(selected_feature, one_hot_encoded_data):
    reversed_data = []

    for index, row in one_hot_encoded_data.iterrows():
        for column_name, value in row.items():
            if value == 1 and column_name.startswith(selected_feature):
                feature_value = column_name.split('_')[-1]  # Remove the underscore and keep only the relevant part
                reversed_data.append(feature_value)
                break

    return reversed_data

def prepare_input_data(patient_data_file, feature_variables_names, input_data_format, patient_objects, surgery_objects):
    input_data = []
    for patient_object, surgery_object in zip(patient_objects, surgery_objects):
        extracted_values = {}
        extracted_values.update(search_features_values_from_obj(patient_object, feature_variables_names))
        non_extracted_features = [element for element in feature_variables_names if element not in extracted_values]
        extracted_values.update(search_features_values_from_obj(surgery_object, non_extracted_features))
        non_extracted_features = [element for element in feature_variables_names if element not in extracted_values]
        if len(non_extracted_features)>0:
            extracted_values.update(select_feature_values_from_data_file(patient_data_file, non_extracted_features,'ID', patient_object.ID))
            if len(extracted_values) < len(feature_variables_names): 
                non_extracted_features = [element for element in feature_variables_names if element not in extracted_values]
                raise ValueError(f"Following features value could not be extracted:{non_extracted_features}")
        input_data.append(extracted_values)
    input_data = pd.DataFrame(input_data).fillna(0)
    print(input_data)
    input_data_encoded = pd.get_dummies(input_data)
    
    not_found = [element for element in input_data_encoded.columns if element not in input_data_format.columns]
    
    if len(not_found) > 0:
        raise ValueError(f"Model was not trained for following features:{not_found}")
    
    return input_data_encoded.reindex(columns = input_data_format.columns, fill_value=0)


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





def prepare_IO_data(data_file, data_categ_dict, input_features, output_data_types, grouping_column, encoding_type):

    #use Dataset function to read your dataset 
    df_selected = Dataset(data_file)
    for key in data_categ_dict.keys():
        df_selected = df_selected.loc[df_selected[key] == data_categ_dict[key]]
        
    #t_o_col = trauma_ortho.loc[trauma_ortho.Hosp == "COL"]
    preprocessed_set = []
    IO_sets = []
    for i in range(len(output_data_types)):
        preprocessed_data = Preprocess(select_features_nd_target(df_selected,input_features, output_data_types[i], grouping_column))
        #print(i)
        #print(preprocessed_data[1])
        preprocessed_set.append(preprocessed_data)
        if encoding_type == 'one-hot':
            i_o_data = [one_hot_encode_dataframe(preprocessed_data[1].iloc[:, :-1]), preprocessed_data[1].iloc[:, -1]]
        #print(i_o_data)
        IO_sets.append(i_o_data) 
        
    return preprocessed_set, IO_sets


def read_z_table(file_path):
    """
    Reads an Excel file containing a Z-distribution probability table and returns
    a dictionary with Z-values as keys and probabilities as values.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    dict: A dictionary with Z-values as keys and probabilities as values.
    """
    # Load the Excel file
    df = pd.read_excel(file_path, index_col=0)

    # Initialize an empty dictionary to store the Z-values and probabilities
    z_dict = {}

    # Iterate through the DataFrame to construct the dictionary
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            # Construct the Z-value key
            z_value = round(df.index[i] + df.columns[j], 2)
            # Get the probability value
            probability = df.iloc[i, j]
            # Store in dictionary
            z_dict[z_value] = probability

    return z_dict


#function to preprocess dataset for the model
#returns 5 variables
def remove_neg(df_1):

    #explore missing values
    missing = df_1.isna().sum()
    
    #Basic cleaning steps :
    
    #drop negative values for time
    no_neg = df_1.loc[df_1.iloc[:,-1] > 0]

    #dropping all missing values
    preprocessed = no_neg.dropna()

    #Explore features in the cleaned dataset

    #stats = descriptive_stats(preprocessed)

    return([missing, preprocessed])



def remove_outliers_by_category(df, category_col_nd_thres_dict, target_col):
    
    #retrieving the original df columns types
    
    modified_df = df.copy()
    outliers =  pd.DataFrame(columns = df.columns)
    
    for category_col, thres in category_col_nd_thres_dict.items():
        
        
        unique_categories = modified_df[category_col].unique() 
        
        #create empty cleaned_df
        cleaned_df = pd.DataFrame(columns = modified_df.columns)

        for category in unique_categories:
            
            category_df = modified_df[modified_df[category_col] == category].copy()
            
            #z_scores = np.abs(stats.zscore(category_df[target_col]))
            category_df.loc[:,'z_scores'] = (category_df[target_col] - category_df[target_col].mean()) / category_df[target_col].std()

            if 'N_procedures' in df.columns:
                
                category_df.loc[:,'is_outlier'] = ((category_df['N_procedures'] <=1) & (np.abs(category_df['z_scores']) > thres)) | ((category_df['N_procedures'] > 1) & ((category_df['z_scores'] > 1.5 * thres) | (category_df['z_scores'] < thres*-1)))
                
            else:
                category_df.loc[:,'is_outlier'] = (np.abs(category_df['z_scores']) > thres)

            cleaned_category_df = category_df[~category_df['is_outlier']].drop(columns = ['is_outlier', 'z_scores'])
            
            outliers = pd.concat([outliers, category_df[category_df['is_outlier']].drop(columns = ['is_outlier', 'z_scores'])])
            
            cleaned_df = pd.concat([cleaned_df, cleaned_category_df])
            
        modified_df = cleaned_df.copy()
    
    #retrieving the previous datatype incase has changed during the concating process
    for col in modified_df.columns:
        modified_df[col] = modified_df[col].astype(df[col].dtype)
        outliers[col] = outliers[col].astype(df[col].dtype)

    return modified_df.sort_index().reset_index(drop=True), outliers.sort_index().reset_index(drop=True)



def replace_or_remove_infrequent_data(df, column_name, count_threshold, replacement):
    # Count the occurrences of each value in the specified column
    value_counts = df[column_name].value_counts()
    
    if replacement is None:
        infrequent_mask = df[column_name].map(value_counts) < count_threshold
        # Remove rows with infrequent values
        removed_df = df[infrequent_mask].copy()
        df = df[~infrequent_mask].copy()
    else:
        # Replace infrequent values with the replacement value
        infrequent_mask = (df[column_name].map(value_counts) < count_threshold) | \
                      (df[column_name].isin(['Unknown', 'unknown', 'nan'])) | \
                      (df[column_name].isna())
        df.loc[infrequent_mask, column_name] = replacement
        #removed_df = df[df[column_name] == replacement].copy()
        removed_df = pd.DataFrame(columns = df.columns)
    
    return df, removed_df


    
def select_features_nd_target(df, features_esses, features_addi, target, min_count_thresholds, z_scores_threshold, infrequent_data_replacement = None):

    IO_variables = [str1 for str1 in features_esses+features_addi]
    IO_variables.append(target)
    #print(IO_variables)
    result_df = df[IO_variables]
    
    removed_df = pd.DataFrame(columns = result_df.columns)
    
    #negative data removing
    
    removed_df = pd.concat([removed_df, result_df[result_df[target] <= 0 ]])
    
    result_df = result_df.loc[result_df[target] > 0].copy()
    
    #filling 'Unknown' or '0' data for addi features data to protect the data from being removed
    string_columns = result_df[features_addi].select_dtypes(include=['object']).columns
    numeric_columns = result_df[features_addi].select_dtypes(include=[np.number]).columns
    
    result_df[string_columns] = result_df[string_columns].fillna("Unknown")
    result_df[numeric_columns] = result_df[numeric_columns].fillna(0)
   
    #result_df[features_addi] = result_df[features_addi].fillna('Unknown')
    
    result_df = result_df.dropna()
    
    #result_df = result_df.reset_index()
    
    if isinstance(z_scores_threshold, (int, float)) or (isinstance(z_scores_threshold, dict) and 'overall' in z_scores_threshold):
        
        if isinstance(z_scores_threshold, dict):
            z_score_threshold = z_scores_threshold['overall']
            del z_scores_threshold['overall']
        else:
            z_score_threshold = z_scores_threshold
        
        z_scores = np.abs((result_df[target] - result_df[target].mean()) / result_df[target].std())
        #print(z_scores)
        # Define a threshold for identifying outliers (e.g., Z-score > 3)
        #z_scores_threshold = 1.5

        # Filter out rows with Z-scores greater than the threshold
        
        removed_df = pd.concat([removed_df, result_df[z_scores > z_score_threshold].reset_index(drop=True)])
        
        result_df = result_df[z_scores <= z_score_threshold].reset_index(drop=True)
        
    if isinstance(z_scores_threshold, dict) and len(z_scores_threshold)>0:
        
        result_df, outlier_df = remove_outliers_by_category(result_df, z_scores_threshold, target)
        
        removed_df = pd.concat([removed_df, outlier_df])

    if not min_count_thresholds is None:
        for grouping_column, min_count_threshold in min_count_thresholds.items():
            if infrequent_data_replacement is not None and grouping_column in infrequent_data_replacement:
                result_df, removed_temp = replace_or_remove_infrequent_data(result_df, grouping_column, min_count_threshold, infrequent_data_replacement[grouping_column])
            else:
                result_df, removed_temp = replace_or_remove_infrequent_data(result_df, grouping_column, min_count_threshold, None)
            #grouped = result_df.groupby(grouping_column)
            #result_df = grouped.filter(lambda x: len(x) >= min_count_threshold)
            #removed_df = pd.concat([removed_df, grouped.filter(lambda x: len(x) < min_count_threshold)])
            removed_df = pd.concat([removed_df, removed_temp])

    return result_df.reset_index(drop=True), removed_df.reset_index(drop=True)


def select_features_nd_target_old(df, features, target, grouping_column):

    IO_variables = [str1 for str1 in features]
    IO_variables.append(target)
    no_neg = df[IO_variables]
    no_neg = no_neg.loc[no_neg[target] >= 0].copy()
    no_missing = no_neg.dropna()
    
    #remove outliers 
    def is_outlier(s):
        lower_limit = s.mean() - (s.std() * 3)
        upper_limit = s.mean() + (s.std() * 3)
        return ~s.between(lower_limit, upper_limit)

    return no_missing[~no_missing.groupby(grouping_column)[target].apply(is_outlier)]




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




def shift_feature_to_data_category(modelling_parameters, feature_name, feature_value):
        
        updated_para = copy.deepcopy(modelling_parameters)
        
        #removing the feature from input feature
        updated_para[2].remove(feature_name)
        
        #adding to data category
        updated_para[3][feature_name] = feature_value
        
        return updated_para
    



#split data into predictor and target variables, encode categorical variables
def X_y_data(df):

    #predictor variables
    X = df.iloc[:,:-1]

    #target variable
    y = df.iloc[:,-1]

    #ecnode categorical variables
    X_encoded = pd.get_dummies(X)
    
    return(X_encoded, y)
