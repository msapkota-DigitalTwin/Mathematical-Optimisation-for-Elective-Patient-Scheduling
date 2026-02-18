import random
#import simpy
import pandas as pd  # version 2.0.3
import datetime
import numpy as np  ##  version 1.24.3
from itertools import product

from procedure_time_and_variabiltiy_related_functions import update_patient_data, generate_procedure_time_chances_for_consultants, compute_TASKS_average_surgical_time, load_timeslot_file, create_schedule_and_upate_dataset, convert_schedule_to_binary_matrix

from optimisation_related_functions import create_optimum_schedule, convert_binary_matrix_to_schedule


class Scheduling_Model:
    def __init__(self, df_cases, df_sessions):
        self.CASES = tuple(df_cases["Patient ID"])
        self.cases_count = len(tuple(df_cases["Patient ID"]))
        self.CASES_RTT_WAIT_WEEKS = {}
        self.CASES_RTT_PRIORITY = {}
        self.CASES_SURGERY_PRIORITY = {}
        self.SESSIONS = tuple(df_sessions["Session ID"])
        self.SESSION_DURATION = pd.Series(df_sessions["Total Slot Minutes"].values, index=df_sessions["Session ID"]).to_dict()
        self.SUM_ALL_SESSIONS_DURATION = sum(df_sessions["Total Slot Minutes"])
        self.SESSION_START_TIME = {}
        self.SESSION_DATES = {}
        self.DISJUNCTIONS = set()
        self.TASKS = tuple(product(df_cases['Patient ID'], df_sessions['Session ID']))
        self.TASKS_DURATION = None
        self.TASKS_DURATION_DEVIATION_RATIO_CHANCES = {}
        self.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD = {}
        self.patients_df = None
        self.sessions_df = df_sessions
        self.M = None
        self.lb = None
        self.ub = None
        self.session_max_util = 0.9
        self.theatre_prpn_min = 7.5
        self.TASKS_ASSIGNED = {}
        self.CASE_START_TIME = {}
        self.schedule_new = None
        self.sessions_updated = None
        self.consultant_procedures_template_dict = {}
        self.SUM_CASES_SURGERY_PRIORITISATION_REWARD = 0
        
        #self.consultant_procedure_historical_count = None
        
    def generate_session_start_times(self, df_sessions):
        sessions_temp = df_sessions.copy()
        sessions_temp["Start"] = pd.to_datetime(sessions_temp["Session Planned Start Date/Time"])
        midnight = pd.to_datetime(sessions_temp['Start'].dt.date)
        sessions_temp['Start'] = (sessions_temp['Start'] - midnight).dt.total_seconds() / 60
        self.SESSION_START_TIME = dict(zip(sessions_temp["Session ID"], sessions_temp["Start"]))
        
    def add_cases_RTT_waiting_time(self, df_cases):
        wait_weeks_cols = [col for col in df_cases.columns if 'wait' in col.lower() and 'week' in col.lower()]
        
        if len(wait_weeks_cols) > 0:
        #self.CASES_RTT_WAIT_WEEKS = pd.Series(df_cases[wait_weeks_col].values, index=df_cases["Patient ID"]).to_dict()
            self.CASES_RTT_WAIT_WEEKS = df_cases.dropna(subset=["Patient ID"]).set_index("Patient ID")[wait_weeks_cols[0]].fillna(0).to_dict()
            
        elif 'RTT clock start' in df_cases.columns:
            # Compute wait weeks from RTT clock start to today
            today = pd.to_datetime(datetime.datetime.today().date())
            df_cases["RTT clock start"] = pd.to_datetime(df_cases["RTT clock start"], errors='coerce')
            
            df_cases["Computed Wait Weeks"] = (
                ((today - df_cases["RTT clock start"]).dt.days / 7)
                .clip(lower=0)  # Make negative durations zero
                .fillna(0)
            )
            
            self.CASES_RTT_WAIT_WEEKS = (
                df_cases.dropna(subset=["Patient ID"])
                .set_index("Patient ID")["Computed Wait Weeks"]
                .to_dict()
            )
        else:
            self.CASES_RTT_WAIT_WEEKS = {}
            
        self.SUM_CASES_RTT_WAIT_WEEKS = sum(value for key, value in self.CASES_RTT_WAIT_WEEKS.items())
    
    def add_cases_RTT_Priority(self, df_cases):
        if 'RTT WL Priority' in df_cases.columns:
            self.CASES_RTT_PRIORITY = pd.Series(df_cases["RTT WL Priority"].values, index=df_cases["Patient ID"]).to_dict()

    def add_cases_surgery_Priority(self, df_cases):
        if 'Procedure Priority' in df_cases.columns:
            self.CASES_SURGERY_PRIORITY = pd.Series(df_cases["Procedure Priority"].values, index=df_cases["Patient ID"]).to_dict()
            self.CASES_SURGERY_PRIORITISATION_REWARD = {key: 0 if pd.isna(value) else 1.25 - int(value[-1])*0.25 for key, value in self.CASES_SURGERY_PRIORITY.items()}
            self.SUM_CASES_SURGERY_PRIORITISATION_REWARD = sum(self.CASES_SURGERY_PRIORITISATION_REWARD.values())
    
    def get_ordinal_session_dates(self, df_sessions):
        sessions_temp = df_sessions.copy()
        sessions_temp["Date"] = pd.to_datetime(df_sessions["Session Planned Start Date/Time"], format="%d/%m/%Y").apply(lambda date: date.toordinal())
        self.SESSION_DATES = dict(zip(sessions_temp["Session ID"], sessions_temp["Date"]))
    
    def generate_disjunctions(self, df_cases, df_sessions):
        cases = df_cases["Patient ID"].tolist()
        sessions = df_sessions["Session ID"].tolist()
        self.DISJUNCTIONS = {(case1, case2, session) for case1, case2, session in product(cases, cases, sessions) if (case1 != case2)}

    def generate_tasks_durations(self, df_cases, df_sessions, patient_data, procedure_time_pred_parameters):

        if self.patients_df is None:

            proce_related_col = [col for col in df_cases.columns if 'procedure' in col.lower()][0]

            all_features_with_identifying_name = [proce_related_col if x == 'Actual Procedure 1 Code 1' else x for x in procedure_time_pred_parameters[4]]
            possible_varying_data_for_patient = procedure_time_pred_parameters[5]
            
            fixed_data_for_patient = [col for col in all_features_with_identifying_name if col not in possible_varying_data_for_patient]
            
            self.add_patient_data(df_cases, patient_data, fixed_data_for_patient)
        
        self.TASKS_DURATION = compute_TASKS_average_surgical_time(self.patients_df, df_sessions, *procedure_time_pred_parameters)
        
    
    def generate_chances_for_tasks_durations(self, df_cases, df_sessions, variability_counts, patient_data, procedure_time_pred_parameters):

        proce_related_col = [col for col in df_cases.columns if 'procedure' in col.lower()][0]
            
        if self.patients_df is None:
            all_features_with_identifying_name = [proce_related_col if x == 'Actual Procedure 1 Code 1' else x for x in procedure_time_pred_parameters[4]]
            possible_varying_data_for_patient = procedure_time_pred_parameters[5]
            fixed_data_for_patient = [col for col in all_features_with_identifying_name if col not in possible_varying_data_for_patient]
            
            self.add_patient_data(df_cases, patient_data, fixed_data_for_patient)
        
        #temp_data = generate_chances_for_tasks_durations(self.patients_df, self.sessions_df, variability_counts,  *procedure_time_pred_parameters)
        temp_data = generate_procedure_time_chances_for_consultants(self.patients_df, self.sessions_df, variability_counts,  *procedure_time_pred_parameters)
        
        #proce_related_col = [col for col in df_cases.columns if 'procedure' in col.lower()][0]

        for task in self.TASKS:
            procedure_code = df_cases.loc[df_cases['Patient ID'] == task[0],proce_related_col].values[0]
            consultant_ID = df_sessions.loc[df_sessions['Session ID'] == task[1],'Consultant Code'].values[0]
            
            if (procedure_code, consultant_ID) in temp_data and len(temp_data[(procedure_code, consultant_ID)]) > 0:
                self.TASKS_DURATION_DEVIATION_RATIO_CHANCES[task] = [_ for _ in temp_data[(procedure_code, consultant_ID)]]
            #explore data from removed data during pre-processing
            elif procedure_code in list(procedure_time_pred_parameters[3]['Actual Procedure 1 Code 1']):
                time_temp = list(procedure_time_pred_parameters[3][procedure_time_pred_parameters[3]['Actual Procedure 1 Code 1'] == procedure_code]['H4 Minutes'])
                self.TASKS_DURATION_DEVIATION_RATIO_CHANCES[task] = time_temp
            
            else:
                self.TASKS_DURATION_DEVIATION_RATIO_CHANCES[task] = []
            
            '''
            for model_type in models_types:
                if task+(model_type,) in temp_data:
                    self.TASKS_DURATION_DEVIATION_RATIO_CHANCES[task+(model_type,)] = temp_data[task+(model_type,)]
                else:
                    self.TASKS_DURATION_DEVIATION_RATIO_CHANCES[task+(model_type,)] = 'nan'
            '''

    def obtain_mean_sd_for_tasks_durations_chances(self, df_cases, df_sessions, variability_counts, patient_data, procedure_time_pred_parameters):
        if len(self.TASKS_DURATION_DEVIATION_RATIO_CHANCES.keys()) > 0:
            temp_data = self.TASKS_DURATION_DEVIATION_RATIO_CHANCES
        else:
            temp_data = self.generate_chances_for_tasks_durations(self.patients_df, self.sessions_df, variability_counts,  patient_data, procedure_time_pred_parameters)
            #models_types = procedure_time_pred_parameters[0]
        for task in self.TASKS:
            if task in temp_data and len(temp_data[task]) > 0:
                if len(temp_data[task]) >= 5:
                    self.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[task] = [np.mean([chance for chance in temp_data[task]]), np.std([chance for chance in temp_data[task]])]
                else:
                    self.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[task] = [1, 0.25]
                
            else:
                self.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[task] = np.nan
            '''
            for model_type in models_types:
                if task+(model_type,) in temp_data:
                    self.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[task+(model_type,)] = [np.mean([chance for chance in temp_data[task+(model_type,)]]), np.std([chance for chance in temp_data[task+(model_type,)]])]
                else:
                    self.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[task+(model_type,)] = 'nan'
            '''

    def set_additional_params(self, lb = None, ub = None, session_max_util = None):
        self.lb = lb
        self.ub = ub
        self.M = 1e3 * ub
        self.session_max_util = session_max_util

    def add_TASKS_ASSIGNED_variable(self, task, value = None):
        self.TASKS_ASSIGNED[task] = value

    def add_case_start_time_variable(self, task):
        self.CASE_START_TIME[task] = (self.lb, self.ub)

    def update_sessions_df(self):
        # Add columns and values for predictive model required features  
        #sessions_timeslot_df = pd.read_excel(timeslot_file_path)
        self.sessions_df['Day of the week'] = self.sessions_df['Session Planned Start Date/Time'].apply(lambda x: x.day_name())
        self.sessions_df['Covid Flag'] = 'post-covid'
        if 'Theatre Suite Name' not in self.sessions_df.columns:
            self.sessions_df['Theatre Suite Name'] = self.sessions_df['Theatre Name'].apply(lambda x: 'Elmstead Theatres' if x in ['Theatre 05', 'Theatre 5'] else 'Constable Theatres')
        
        self.sessions_df['Total Slot Minutes'] = (self.sessions_df['Session Planned End Date/Time'] - self.sessions_df['Session Planned Start Date/Time']).dt.total_seconds() // 60
        self.sessions_df['Remaining Slot Minutes'] = (self.sessions_df['Session Planned End Date/Time'] - self.sessions_df['Session Planned Start Date/Time']).dt.total_seconds() // 60
        self.sessions_df['H4 Minutes Booked'] = 0
        
        #return sessions_timeslot_df

    def add_patient_data(self,patients_df, patients_data_df, fixed_data_for_patient):
        
        self.patients_df = update_patient_data(patients_df, patients_data_df, fixed_data_for_patient)

    def add_consultant_procedures_template(self, template_dataframe):
        
        self.consultant_procedures_template_dict = {
            col: tuple(sorted(template_dataframe[col].dropna()))
            for col in template_dataframe.columns
        }

    
    def create_schedule_with_continuous_filling(self, procedure_time_prediction_parameters = None):
        self.schedule_new, self.sessions_updated, self.patients_not_scheduled, self.sessions_all_booked =  create_schedule_and_upate_dataset(self.patients_df, self.sessions_df, predicted_times = self.TASKS_DURATION, 
                                                                                                                                             procedure_time_prediction_parameters = procedure_time_prediction_parameters, 
                                                                                                                                                prep_min = self.theatre_prpn_min, max_sess_uti = self.session_max_util) 

        temp_matrix = convert_schedule_to_binary_matrix(self.schedule_new, 'Session ID', 'Patient ID', all_sessions = list(self.sessions_df['Session ID']), all_cases = list(self.patients_df['Patient ID']))
        
        self.schedule_new, self.sessions_updated = convert_binary_matrix_to_schedule(temp_matrix, self, list(self.sessions_df['Session ID']), list(self.patients_df['Patient ID']), self.theatre_prpn_min)
        
                                                                                                                                             
    def create_schedule_with_optimisation(self, 
                          optimisation_algorithm, 
                          objectives = (), 
                          weightage = None, 
                          best_solution_weightage = None,
                          hyper_param = {}, 
                          solutions_only =False, 
                          sessions_selection_prioritise = False,
                          consult_procedure_restriction = False,
                          consult_patient_restriction = False
                                         ):
        
        self.schedule_new, self.sessions_updated, self.cases_not_considered, self.best_solution, _ , __ = create_optimum_schedule(self.patients_df,
                            self.sessions_df, 
                            self.TASKS_DURATION,
                            self, 
                            optimisation_algorithm, 
                            objectives = objectives,                                                                
                            weightage = weightage, 
                            best_solution_weightage = best_solution_weightage,
                            hyper_param = hyper_param, 
                            solutions_only = solutions_only, 
                            sessions_selection_prioritise = sessions_selection_prioritise,
                            consult_procedure_restriction = consult_procedure_restriction,
                            consult_patient_restriction = consult_patient_restriction                                             )
        
        if optimisation_algorithm == 'Simulated Annealing':
            self.pareto_archive = _
            self.iterations_pareto_added = __
        else:
            self.scip_solver = _
        
        #if solutions_only or not best_solution_weightage:
        #    self.patients_not_scheduled = None
        #else:
        if self.schedule_new is not None:
            self.patients_not_scheduled = self.patients_df.drop(index=self.schedule_new['Patient ID'].tolist())
