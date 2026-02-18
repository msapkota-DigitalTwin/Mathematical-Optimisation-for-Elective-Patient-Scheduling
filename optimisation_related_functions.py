import pandas as pd
import datetime
import numpy as np
from itertools import product
#from scheduling_related_functions import combined_mean_sd
from pyscipopt import Model, Eventhdlr, SCIP_EVENTTYPE, quicksum    #Version: 5.1.1
import random
import math
from data_processing_nd_encoding_related_functions import read_z_table


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



def convert_binary_matrix_to_schedule(bin_matrix, data_obj, sessions_considered, cases_considered, prep_min = None, cases_not_considered = []):
    
    if bin_matrix is not None:
        schedule = pd.DataFrame([(session , case,  data_obj.TASKS_DURATION.loc[case, session] * round(bin_matrix[i, j])) for i, session in enumerate(sessions_considered) for j, case in enumerate(cases_considered) if round(bin_matrix[i, j])==1], columns = [ 'Session ID', 'Patient ID','Planned H4 Minutes'])
    else:
        schedule = pd.DataFrame([(session , case,  data_obj.TASKS_DURATION.loc[case, session] * data_obj.TASKS_ASSIGNED[(case, session)]) for i, session in enumerate(sessions_considered) for j, case in enumerate(cases_considered) if data_obj.TASKS_ASSIGNED[(case, session)]==1], columns = [ 'Session ID', 'Patient ID','Planned H4 Minutes'])

    proce_related_col = [col for col in data_obj.patients_df.columns if 'procedure' in col.lower() ][0]
    
    schedule = pd.merge(schedule, data_obj.sessions_df[['Session ID', 'Session Planned Start Date/Time', 'Consultant Code', 'Theatre Name']], on = 'Session ID')
    #schedule = data_obj.patients_df
    schedule = pd.merge( data_obj.patients_df[[proce_related_col ]], schedule, left_index=True, right_on = 'Patient ID', how = 'right')
    
    # Calculate the timedelta for each row and add to the 'Scheduled Start Date/Time' column
    if prep_min is not None:
        schedule['Prepn Min'] = prep_min
        schedule['H4 Minutes Cumulative'] = schedule.groupby('Session ID')['Planned H4 Minutes'].cumsum()
        schedule['Prepn Min Cumulative'] = schedule.groupby('Session ID')['Prepn Min'].cumsum()
        # Convert 'Session Planned Start Date/Time' to datetime
        schedule['Session Planned Start Date/Time'] = pd.to_datetime(schedule['Session Planned Start Date/Time'])
        schedule['Scheduled Start Date/Time'] = schedule['Session Planned Start Date/Time'] + \
                                                     schedule.apply(lambda row: pd.Timedelta(minutes=row['H4 Minutes Cumulative'] +
                                                                                                           row['Prepn Min Cumulative'] -
                                                                                                           row['Planned H4 Minutes']), axis=1)
        schedule = schedule.drop(columns = ['Prepn Min', 'Prepn Min Cumulative', 'H4 Minutes Cumulative'])

    sessions_updated = pd.merge( data_obj.sessions_df[['Session ID', 'Session Planned Start Date/Time', 'Consultant Code', 'Theatre Name', 'Total Slot Minutes']],schedule.groupby('Session ID')['Planned H4 Minutes'].sum().rename('Total Booked Minutes'), on = 'Session ID')
    sessions_updated = sessions_updated[sessions_updated['Session ID'].isin(sessions_considered)]
    sessions_updated['Session Utilisation'] = sessions_updated['Total Booked Minutes']/sessions_updated['Total Slot Minutes']
    sessions_updated['Session Utilisation'] = sessions_updated['Session Utilisation'].round(3)
    if len(data_obj.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD.keys()) >= len(data_obj.CASES):
        
        #from data_processing_nd_encoding_related_functions import read_z_table
        #z_cdf_data = read_z_table('Z_table_positive.xlsx') | read_z_table('Z_table_negative.xlsx')
        z_cdf_data = read_z_table()
        #from optimisation_related_functions import session_over_running_chances
        if bin_matrix is not None:
            session_overrnning_chances = session_over_running_chances(data_obj, pred_model_types= None, task_repeating_count = None, analytics_based = True, prep_min = prep_min, cdf_table_data = z_cdf_data, individual = bin_matrix.T.flatten())
        else:
            session_overrnning_chances = session_over_running_chances(data_obj, pred_model_types= None, task_repeating_count = None, analytics_based = True, prep_min = prep_min, cdf_table_data = z_cdf_data, individual = None)
        
        sessions_updated['Session Overrunning Chances'] = sessions_updated['Session ID'].map(session_overrnning_chances)
        sessions_updated['Session Overrunning Chances'] = sessions_updated['Session Overrunning Chances'].round(3)
    
    return schedule, sessions_updated



def convert_cases_length_arr_to_schedule(sol_arr, data_obj, prep_min = None, cases_not_considered = []):
    if sol_arr is not None:
        schedule = pd.DataFrame([(session , data_obj.CASES[case_ind],  data_obj.TASKS_DURATION.loc[data_obj.CASES[case_ind], session]) for case_ind, session in enumerate(sol_arr) if session!=-1], columns = [ 'Session ID', 'Patient ID','Planned H4 Minutes'])
    
    schedule = pd.merge(schedule, data_obj.sessions_df[['Session ID', 'Session Planned Start Date/Time', 'Consultant Code', 'Theatre Name']], on = 'Session ID')
    proce_related_col = [col for col in data_obj.patients_df.columns if 'procedure' in col.lower() ][0]
    schedule = pd.merge( data_obj.patients_df[[proce_related_col ]], schedule, left_index=True, right_on = 'Patient ID', how = 'right')
    
    # Calculate the timedelta for each row and add to the 'Scheduled Start Date/Time' column
    if prep_min is not None:
        schedule['Prepn Min'] = prep_min
        schedule['H4 Minutes Cumulative'] = schedule.groupby('Session ID')['Planned H4 Minutes'].cumsum()
        schedule['Prepn Min Cumulative'] = schedule.groupby('Session ID')['Prepn Min'].cumsum()
        # Convert 'Session Planned Start Date/Time' to datetime
        schedule['Session Planned Start Date/Time'] = pd.to_datetime(schedule['Session Planned Start Date/Time'])
        schedule['Scheduled Start Date/Time'] = schedule['Session Planned Start Date/Time'] + \
                                                     schedule.apply(lambda row: pd.Timedelta(minutes=row['H4 Minutes Cumulative'] +
                                                                                                           row['Prepn Min Cumulative'] -
                                                                                                           row['Planned H4 Minutes']), axis=1)
        schedule = schedule.drop(columns = ['Prepn Min', 'Prepn Min Cumulative', 'H4 Minutes Cumulative'])

    sessions_updated = pd.merge( data_obj.sessions_df[['Session ID', 'Session Planned Start Date/Time', 'Consultant Code', 'Theatre Name', 'Total Slot Minutes']],schedule.groupby('Session ID')['Planned H4 Minutes'].sum().rename('Total Booked Minutes'), on = 'Session ID')
    #sessions_updated = sessions_updated[sessions_updated['Session ID'].isin(sessions_considered)]
    sessions_updated['Session Utilisation'] = sessions_updated['Total Booked Minutes']/sessions_updated['Total Slot Minutes']
    sessions_updated['Session Utilisation'] = sessions_updated['Session Utilisation'].round(2)
    if len(data_obj.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD.keys()) >= len(data_obj.CASES):
        #from data_processing_nd_encoding_related_functions_old import read_z_table
        #z_cdf_data = read_z_table('Z_table_positive.xlsx') | read_z_table('Z_table_negative.xlsx')
        z_cdf_data = read_z_table()
        if sol_arr is not None:
            binary_array = np.zeros((len(sol_arr), len(data_obj.SESSIONS)), dtype=int)
            # Fill the binary array
            for i, elem in enumerate(sol_arr):
                if elem != -1:
                    col_index = data_obj.SESSIONS.index(elem)  # Find the column index of the element
                    binary_array[i, col_index] = 1          
            session_overrnning_chances = session_over_running_chances(data_obj, pred_model_types= None, task_repeating_count = None, analytics_based = True, prep_min = prep_min, cdf_table_data = z_cdf_data, individual = binary_array.flatten())

        else:
            session_overrnning_chances = session_over_running_chances(data_obj, pred_model_types= None, task_repeating_count = None, analytics_based = True, prep_min = prep_min, cdf_table_data = z_cdf_data, individual = None)
        sessions_updated['Session Overrunning Chances'] = sessions_updated['Session ID'].map(session_overrnning_chances)
        sessions_updated['Session Overrunning Chances'] = sessions_updated['Session Overrunning Chances'].round(2)
    
    return schedule, sessions_updated



def create_optimum_schedule(patients_df, sessions_timeslot_df, 
                            possible_task_durations_df,
                            data_obj, 
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
    """
    Generate an optimum schedule based on the provided inputs and constraints.

    Parameters:
    - data_obj: Data object containing scheduling data.
    - theatre_max_util: Maximum utilization for the theatre.
    - theatre_prpn_min: Minimum preparation time for the theatre.
    - possible_task_durations_df: DataFrame of possible task durations.
    - patients_df: DataFrame of patients.
    - sessions_timeslot_df: DataFrame of session time slots.

    Returns:
    - schedule_new: DataFrame of the generated schedule.
    - sessions_updated: Updated sessions DataFrame.
    - patients_not_scheduled: DataFrame of patients not scheduled.
    """

    try:
        # Update data object constraints
        #data_obj.session_max_util = theatre_max_util
        #data_obj.theatre_prep_min = theatre_prpn_min

        if 'objective3' in objectives:
            data_obj.SUM_CASES_RTT_WAIT_WEEKS = sum(data_obj.CASES_RTT_WAIT_WEEKS.values())

        # Solve the optimization problem
        if 'mip' in optimisation_algorithm.lower() or 'scip' in optimisation_algorithm.lower():
            optimisation_solutions = solve_optimisation_problem_with_MIP_solver(
                data_obj, 
                possible_task_durations_df, 
                objectives = objectives,
                obj_weightage = weightage,
                hyper_param = hyper_param,
                consider_consult_procedure_combination = consult_procedure_restriction,
                consider_consult_patient_combination = consult_patient_restriction
            )

            if optimisation_solutions[0] is not None:
                '''
                schedule_new = pd.DataFrame(
                    [list(task) + [possible_task_durations_df.loc[*task]] 
                     for task in data_obj.TASKS if data_obj.TASKS_ASSIGNED[task] == 1],
                    columns=['Patient ID', 'Session ID', 'Planned H4 Minutes']
                ).sort_values(by='Session ID')
                '''
                if solutions_only:
                    schedule_new = None
                    sessions_updated = None
                else:
                    schedule_new, sessions_updated = convert_binary_matrix_to_schedule(None, 
                                            data_obj, 
                                            data_obj.SESSIONS, 
                                            data_obj.CASES, 
                                            prep_min = data_obj.theatre_prpn_min,
                                            cases_not_considered = optimisation_solutions[-2]
                                            )
    
                return schedule_new, sessions_updated, optimisation_solutions[1],  optimisation_solutions[0], optimisation_solutions[-1], None
            else:
                data_obj.data_label.config(text="Failed Schedule generation")
                return None, None, None, None, None, None

        elif 'simulated' in optimisation_algorithm.lower():
            
            solution_init = np.zeros((len(data_obj.SESSIONS), len(data_obj.CASES)), dtype=int)

            # removing non-applicable cases from the list

            for task in data_obj.TASKS:
                data_obj.TASKS_ASSIGNED[task] = 0
                
            case_index_not_considered = []
            cases_not_considered = []
            
            cases = [case for case in data_obj.CASES]
            sessions = [_ for _ in data_obj.SESSIONS]
            
            n_init = len(cases)  # number of events
            s = len(sessions)   # number of slots
            #d = [30, 45, 60, 90, 30, 45, 60, 90, 30, 45, 60, 90, 30, 45, 60, 90, 30, 45, 60, 90]  # durations of events
            
            for i, session in enumerate(sessions):
                for j, case in enumerate(cases):
                    if pd.isna(data_obj.TASKS_DURATION.loc[case, session]) and j not in case_index_not_considered:
                        case_index_not_considered.append(j)
                        cases_not_considered.append(case)
                        break
                    
            #waiting_weeks = waiting_weeks[~case_index_not_considered]
            
            # Sort indices in descending order
            case_index_not_considered.sort(reverse=True)
            
            # Remove elements by index
            '''
            for index in case_index_not_considered:
                del waiting_weeks[index]
                del cases[index]
            
            #n = n - len(case_index_not_considered)
            
            T_session = [_ for key, _ in data_obj.SESSION_DURATION.items()]  # total available time in each slot
            total_durations = sum(T_session)
            '''
            #data_obj.session_max_util = theatre_max_util
            
            pareto_archive, iterations_aaded_paretos = simulated_annealing_with_paretos(solution_init, data_obj,
                           case_index_not_considered, 
                           objectives = objectives, 
                           Obj_weightage = weightage, 
                           INITIAL_TEMPERATURE = hyper_param['INITIAL_TEMPERATURE'], 
                           ALPHA = hyper_param['ALPHA'], 
                           FINAL_TEMPERATURE = hyper_param['FINAL_TEMPERATURE'],
                           MAX_ITERATIONS = hyper_param['MAX_ITERATIONS'], 
                           theatre_prp_min = data_obj.theatre_prpn_min,
                           consider_session_filled_ratio = sessions_selection_prioritise,
                           consider_consult_procedure_combination = consult_procedure_restriction,
                           consider_consult_patient_combination = consult_patient_restriction
                            )
 
            #print(f'Solution with Simulated annealing is reached after {iterations} iterations')
            #print(f'Objective value for reached solution: {best_kpis}')

            best_solution = None
            best_schedule= None
            sessions_updated = None

            #if solutions_only or not best_solution_weightage:
                
            if not solutions_only and best_solution_weightage:
                best_weighted_KPI = 0
                for solution in pareto_archive:
                    if best_weighted_KPI < sum([a*b for a, b in zip(solution['f'], best_solution_weightage)]) :
                        best_solution = solution['x']
            
                for task, value in zip(data_obj.TASKS, best_solution.T.flatten()):
                    data_obj.TASKS_ASSIGNED[task] = value
                
                best_schedule, sessions_updated = convert_binary_matrix_to_schedule(best_solution, data_obj, data_obj.SESSIONS, data_obj.CASES, prep_min= data_obj.theatre_prpn_min)

            return best_schedule, sessions_updated, cases_not_considered, best_solution, pareto_archive, iterations_aaded_paretos

    except Exception as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None



def multi_objectives(model, pred_model_types = [], task_repeating_count = 20, cases_to_skip = [], objectives_list = None, weightage = None, analytics_based = True, prep_min = 7.5, cdf_table_data= None, individual = None):
    #return pe.summation(model.CASES_IN_SESSION)
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value
            #model.CASE_START_TIME[task] = value[1]
    
    result = []
    for objective_type in objectives_list:
        if objective_type == 'objective1':
            result.append(objective1_utilisation(model, cases_to_skip= cases_to_skip))
        elif objective_type == 'objective2':
            result.append(objective2_overrunning_counterchances(model, pred_model_types, task_repeating_count, analytics_based = analytics_based, prep_min=prep_min, cdf_table_data = cdf_table_data))
        elif objective_type == 'objective3':
            result.append(objective3_RTT_waiting_time_average(model)) 
        elif objective_type == 'objective4':
            result.append(objective4_surgical_prioritising_score(model)) 
    if weightage is None:
        return tuple(result)
    elif len(objectives_list) > len(weightage):
        merged = 0
        unmerged = []
        for key, temp_a in zip(objectives_list, result):
            if key in weightage:
                merged+= temp_a*weightage[key]
            else:
                unmerged.append(temp_a)
        return tuple([merged] + unmerged)
    else:
        if isinstance(weightage, dict):
            return tuple([sum(res*weight for res, weight in zip(result, weightage.values()))])
        else:
            return tuple([sum(res*weight for res, weight in zip(result, weightage))])


def multi_objectives_cases_length_variable(individual, data_obj, task_repeating_count = 30, cases_to_skip = [], objectives_list = None, weightage = None, analytics_based = True, prep_min = 7.5, cdf_table_data= None):
    result = []
    for objective_type in objectives_list:
        if objective_type == 'objective1':
            result.append(objective1_cases_length_variable(individual, data_obj.CASES, data_obj.SESSION_DURATION, data_obj.TASKS_DURATION, data_obj.session_max_util))
        elif objective_type == 'objective2':
            # Initialize an empty 2D binary array
            binary_array = np.zeros((len(individual), len(data_obj.SESSIONS)), dtype=int)
            # Fill the binary array
            for i, elem in enumerate(individual):
                if elem != -1:
                    col_index = data_obj.SESSIONS.index(elem)  # Find the column index of the element
                    binary_array[i, col_index] = 1          
            result.append(objective2_overrunning_counterchances(data_obj, None, task_repeating_count, analytics_based = analytics_based, prep_min=prep_min, cdf_table_data = cdf_table_data, individual=binary_array.flatten()))
        elif objective_type == 'objective3':
            result.append(objective3_cases_length_variable(individual, data_obj))   
    if weightage is None:
        return tuple(result)
    elif len(objectives_list) > len(weightage):
        merged = 0
        unmerged = []
        for key, temp_a in zip(objectives_list, result):
            if key in weightage:
                merged+= temp_a*weightage[key]
            else:
                unmerged.append(temp_a)
        return tuple([merged] + unmerged)
    else:
        if isinstance(weightage, dict):
            return tuple([sum(res*weight for res, weight in zip(result, weightage.values()))])
        else:
            return tuple([sum(res*weight for res, weight in zip(result, weightage))])




def objective1_utilisation(model,individual = None, cases_to_skip = []):
    #return pe.summation(model.CASES_IN_SESSION)
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value
            #model.CASE_START_TIME[task] = value[1]
    if len(cases_to_skip) == 0:
        return 1/model.SUM_ALL_SESSIONS_DURATION * sum([model.TASKS_DURATION.loc[*task] * model.TASKS_ASSIGNED[task] for task in model.TASKS])
    else:
        return 1/model.SUM_ALL_SESSIONS_DURATION * sum([model.TASKS_DURATION.loc[*task] * model.TASKS_ASSIGNED[task]
                                                                                        for task in model.TASKS if not task[0] in cases_to_skip])



def objective2_overrunning_counterchances(model,  pred_model_types, task_repeating_count, analytics_based = True, prep_min = 7.5, cdf_table_data = None, individual = None):
    #return pe.summation(model.CASES_IN_SESSION)
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value
            #model.CASE_START_TIME[task] = value[1]
    
    overrun_chances = session_over_running_chances(model, pred_model_types, task_repeating_count = task_repeating_count, analytics_based = analytics_based, prep_min = prep_min, cdf_table_data = cdf_table_data, individual = None)
    
    return 1- np.mean([item for _, item in overrun_chances.items() if item >=0])


def objective3_RTT_waiting_time_average(model, individual = None):
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value
            #model.CASE_START_TIME[task] = value[1]
    return np.sum([model.CASES_RTT_WAIT_WEEKS[case] for case in model.CASES if any(model.TASKS_ASSIGNED[(case, session)] == 1 for session in model.SESSIONS)])/model.SUM_CASES_RTT_WAIT_WEEKS



def objective4_surgical_prioritising_score(model, individual = None):
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value
            #model.CASE_START_TIME[task] = value[1]
    return np.sum([model.CASES_SURGERY_PRIORITISATION_REWARD[case] for case in model.CASES if any(model.TASKS_ASSIGNED[(case, session)] == 1 for session in model.SESSIONS)])/model.SUM_CASES_SURGERY_PRIORITISATION_REWARD



def objective1_cases_length_variable(solution, cases, session_times, case_to_session_times, max_sess_util):

    session_used_time = {key:0 for key in session_times.keys()}  # Track remaining time in each session

    for case_ind, sol_i in enumerate(zip(cases,solution)):
        case, session = sol_i[0], sol_i[1]
        if session == -1:
            continue
        #available_session_times[session] -= case_to_session_times.loc[case,session]  # Update session time
        session_used_time[session] += case_to_session_times.loc[case,session]
        if session_used_time[session] > session_times[session]*max_sess_util:
            return 0

    return np.mean([session_used_time[session]/session_times[session] for session in session_times.keys()])


def objective3_cases_length_variable(individual, data_obj):
    return np.sum([data_obj.CASES_RTT_WAIT_WEEKS[case] for ind, case in enumerate(data_obj.CASES) if individual[ind] != -1])/data_obj.SUM_CASES_RTT_WAIT_WEEKS




def satisfies_constraints(model, individual = None, cases_to_skip = [], specific_sessions = [], cases_repetition_test = True):
    # Implement constraint checking logic here
        #Evaluate SESSION_ASSIGNMENT constraint
    if len(specific_sessions) == 0:
        specific_sessions = model.SESSIONS
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value

    if cases_repetition_test:
        for case in model.CASES:
            if case in cases_to_skip:
                continue
            constraint_check = session_assignment_constraint(model, case)
            if not constraint_check[0]:
                return False  
                
    for session in specific_sessions:
        constraint_check = cases_total_duration_for_session(model, session, cases_to_skip)
        if not constraint_check[0]:
            return False 
                
    return True



def satisfies_soft_constraints(model, solution_2d_arr, sessions_index_for_checking, proce_col):
    # Implement constraint checking logic here
        #Evaluate SESSION_ASSIGNMENT constraint
    for i in sessions_index_for_checking:
        consultant = model.sessions_df.iloc[i]['Consultant Code']
        for j, procedure in enumerate(model.patients_df[proce_col]):
            if solution_2d_arr[i,j] == 0:
                continue
            if not consultant in model.consultant_procedures_template_dict or not procedure in model.consultant_procedures_template_dict[consultant]:
                return False 
                
    return True


def satisfies_patient_consultant_restriction(model, solution_2d_arr, sessions_index_for_checking, proce_col):
    for i in sessions_index_for_checking:
        consultant_ref = model.sessions_df.iloc[i]['Consultant Code']
        for j,(procedure, cons_check) in enumerate(zip(model.patients_df[proce_col], model.patients_df['Consultant Code'])):
            if solution_2d_arr[i,j] == 0:
                continue
                
            elif pd.isna(cons_check):
                if cons_check in model.consultant_procedures_template_dict and procedure in model.consultant_procedures_template_dict[cons_check]:
                    continue
                else:
                    return False
            
            elif consultant_ref != cons_check:
                return False 
    return True


    

def is_sum_less_or_equal_to_requirement(lst, threshold):
    total = 0
    index = 0
    
    while index < len(lst) and total <= threshold:
        total += lst[index]
        index += 1
    # return if threshold is crossed, crossing index and crossing elements
    return total <= threshold, index-1

# Cases can be assigned to a maximum of one session
def session_assignment_constraint(model, case):
    #return is_sum_less_or_equal_to_requirement([model.TASKS_ASSIGNED[(case, session)] for session in model.SESSIONS],1)
    sessions_shuffled = [session for session in model.SESSIONS if model.TASKS_ASSIGNED[(case, session)] == 1]
    if len(sessions_shuffled) ==0:
        return True, 0, []
    #random.shuffle(sessions_shuffled)
    constraint_result = is_sum_less_or_equal_to_requirement([model.TASKS_ASSIGNED[(case, session)] for session in sessions_shuffled],1)
    return constraint_result[0], sessions_shuffled[constraint_result[1]:]


def cases_total_duration_for_session(model, session, cases_to_skip = []):
    
    cases_shuffled = [case for case in model.CASES if model.TASKS_ASSIGNED[(case, session)]==1 and case not in cases_to_skip]
    if len(cases_shuffled) == 0:
        return True, 0, []
    #random.shuffle(cases_shuffled)
    constraint_result = is_sum_less_or_equal_to_requirement([model.TASKS_DURATION.loc[case, session] for case in cases_shuffled], model.SESSION_DURATION[session]*model.session_max_util)    
    return constraint_result[0], cases_shuffled[constraint_result[1]:]


def session_over_running_chances_old(model, pred_model_types = None, task_repeating_count = None, analytics_based = True, prep_min = 7.5, cdf_table_data = None, individual = None):
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value
            #model.CASE_START_TIME[task] = value[1]
    overrun_chances = {}
    #underrun_chances = {}

    if pred_model_types is None:
        pred_model_types = list(set(ele[-1] for ele in list(model.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD.keys())[0:10]))
    
    for session in model.SESSIONS:
        #session_duration = sum(model.TASKS_DURATION[task].value for task in model.TASKS if model.TASKS_ASSIGNED[task].value == 1 and model.TASKS_ASSIGNED[task].value == session)
        session_target_duration = model.SESSION_DURATION[session]
        
        # Fill arrays with data
        #for ind1, model_type in enumerate(pred_model_types):
        chances_means = []
        chances_stds = []
        task_process_time = []
        
        for case in model.CASES:
            if model.TASKS_ASSIGNED[(case, session)] == 1:
                chances_means.append(np.mean([model.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[(case, session, model_type)][0] for model_type in pred_model_types]))
                chances_stds.append(np.mean([model.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[(case, session, model_type)][1] for model_type in pred_model_types]))
                task_process_time.append(model.TASKS_DURATION.loc[case, session])
        
        #print(f'Session {session} cases durations: {task_process_time}' )
        # Filter out zero values from task_process_time
        valid_task_process_time_mask = [ind for ind, _ in enumerate(task_process_time) if _ != 0]
        if len(valid_task_process_time_mask) != len(task_process_time):
            task_process_time = [ele for ind, ele in enumerate(task_process_time) if ind in valid_task_process_time_mask]
            chances_means = [ele for ind, ele in enumerate(chances_means) if ind in valid_task_process_time_mask]
            chances_stds = [ele for ind, ele in enumerate(chances_stds) if ind in valid_task_process_time_mask]
        
        if len(task_process_time) == 0:
            overrun_chances[session] = 0
            continue

        # Calculate residuals_weights only for valid elements
        sum_task_process_time = sum(task_process_time)
        chances_weights = [t/sum_task_process_time for t in task_process_time ]

        #for ind1 in range(len(pred_model_types)):
        comb_mean, comb_std = combined_mean_sd(chances_means, chances_stds, chances_weights)

        combined_Z_normalised = (session_target_duration * 1.0625 / (sum_task_process_time + (len(task_process_time)+1)*prep_min)  - comb_mean)/comb_std
        combined_Z_normalised = min(round(combined_Z_normalised, 2), 3.33)
        
        overrun_chances[session] = 1- cdf_table_data[combined_Z_normalised]
        
    return overrun_chances


def session_over_running_chances(model, pred_model_types = None, task_repeating_count = None, analytics_based = True, prep_min = 7.5, cdf_table_data = None, individual = None):
    if individual is not None:
        for task, value in zip(model.TASKS, individual):
            model.TASKS_ASSIGNED[task] = value
            #model.CASE_START_TIME[task] = value[1]
    overrun_chances = {}
    #underrun_chances = {}

    #if pred_model_types is None:
    #    pred_model_types = list(set(ele[-1] for ele in list(model.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD.keys())[0:10]))
    
    for session in model.SESSIONS:
        #session_duration = sum(model.TASKS_DURATION[task].value for task in model.TASKS if model.TASKS_ASSIGNED[task].value == 1 and model.TASKS_ASSIGNED[task].value == session)
        session_target_duration = model.SESSION_DURATION[session]
        
        # Fill arrays with data
        #for ind1, model_type in enumerate(pred_model_types):
        chances_means = []
        chances_stds = []
        task_process_time = []
        
        for case in model.CASES:
            if model.TASKS_ASSIGNED[(case, session)] == 1:
                chances_means.append(model.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[(case, session)][0])
                chances_stds.append(model.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[(case, session)][1])
                task_process_time.append(model.TASKS_DURATION.loc[case, session])
        
        #print(f'Session {session} cases durations: {task_process_time}' )
        # Filter out zero values from task_process_time
        valid_task_process_time_mask = [ind for ind, _ in enumerate(task_process_time) if _ != 0]
        if len(valid_task_process_time_mask) != len(task_process_time):
            task_process_time = [ele for ind, ele in enumerate(task_process_time) if ind in valid_task_process_time_mask]
            chances_means = [ele for ind, ele in enumerate(chances_means) if ind in valid_task_process_time_mask]
            chances_stds = [ele for ind, ele in enumerate(chances_stds) if ind in valid_task_process_time_mask]
        
        if len(task_process_time) == 0:
            overrun_chances[session] = 0
            continue

        # Calculate residuals_weights only for valid elements
        sum_task_process_time = sum(task_process_time)
        chances_weights = [t/sum_task_process_time for t in task_process_time ]

        #for ind1 in range(len(pred_model_types)):
        comb_mean, comb_std = combined_mean_sd(chances_means, chances_stds, chances_weights)

        combined_Z_normalised = (session_target_duration * 1.0625 / (sum_task_process_time + (len(task_process_time)+1)*prep_min)  - comb_mean)/comb_std
        combined_Z_normalised = min(round(combined_Z_normalised, 2), 3.33)
        
        overrun_chances[session] = 1- cdf_table_data[combined_Z_normalised]
        
    return overrun_chances


# Neighbor Solution: Generate a neighboring solution by either swapping cases between sessions or adding an unassigned case to a session
def neighbor_solution(solution, 
                      unassigned_cases_indices, 
                      num_sessions, 
                      num_cases, 
                      cases_indices_to_skip, 
                      sessions_fill_ratio = None
                     ):
    new_solution = solution.copy()  # Deep copy the current solution
    # only adding once case in any random session
    #adding_criteria = 1-np.mean(sessions_fill_ratio) if sessions_fill_ratio is not None else 0.7
    adding_prob = random.random() < 0.7
    changed_session_indices = []
    if sessions_fill_ratio is None:
        weightage_session_selec = None
    else:
        b = num_sessions-sum(sessions_fill_ratio)
        weightage_session_selec = [(1-a)/b for a in sessions_fill_ratio]
    
    #unassigned_cases_indices_new = [_ for _ in unass]
    if adding_prob and unassigned_cases_indices:
        # Try adding an unassigned case to a session
        case_ind = random.choice(unassigned_cases_indices)
        #session_ind = random.randint(0, num_sessions - 1)
        #if weightage_session_selec is not None:
        session_ind = np.random.choice(range(num_sessions), p=weightage_session_selec)
        new_solution[session_ind, case_ind] = 1
        unassigned_cases_indices.remove(case_ind)
        changed_session_indices.append(session_ind)
        
        # Try remove one case to balance addition
        if random.random() < 0.5 * (0.6 if sessions_fill_ratio is None else sessions_fill_ratio[session_ind]):
            session_cases_indices = [i for i in range(num_cases) if new_solution[session_ind,i] == 1 and i!=case_ind]
            if len(session_cases_indices) > 0:
                case_ind = random.choice(session_cases_indices)
                #session_ind = random.randint(0, num_sessions - 1)
                new_solution[session_ind, case_ind] = 0
                unassigned_cases_indices.append(case_ind)

    #swapping_prob = random.random()
    if not adding_prob or random.random() < 0.3:
        # Swap a case between sessions
        session_from, session_to = random.sample(range(num_sessions), 2)
        
        session_cases_indices = [i for i in range(num_cases) if new_solution[session_from,i] == 1]
        
        if len(session_cases_indices) > 0:
            case_1 = random.choice(session_cases_indices)
            #while case_ind in cases_indices_to_skip:
            #    case_ind = random.randint(0, num_cases - 1)
        else:
            case_1 = random.randint(0, num_cases - 1)
            while case_1 in cases_indices_to_skip:
                case_1 = random.randint(0, num_cases - 1)
        #if new_solution[session_from, case_from] == 1:
        # moving case_1 from session_from to session_to
        new_solution[session_from, case_1] = 0
        new_solution[session_to, case_1] = 1

        if not session_from in changed_session_indices:
            changed_session_indices.append(session_from)
        if not session_to in changed_session_indices:
            changed_session_indices.append(session_to)
        
        # moving case_2 from session_to to session_from with 0.7 probability
        if random.random() < 0.7:
            session_cases_indices = [i for i in range(num_cases) if new_solution[session_to,i] == 1 and i != case_1]
            if len(session_cases_indices) > 0:
                #case_1 = random.choice(session_cases_indices)
                case_2 = random.choice(session_cases_indices)
                #while case_ind in cases_indices_to_skip:
                #    case_ind = random.randint(0, num_cases - 1)
                #if new_solution[session_from, case_from] == 1:
                new_solution[session_from, case_2] = 1
                new_solution[session_to, case_2] = 0
    
    return new_solution, changed_session_indices



def pareto_dominates(kpis1, kpis2):
    """Returns True if KPIs1 Pareto dominates KPIs2, False otherwise."""
    return all(x >= y for x, y in zip(kpis1, kpis2)) and any(x > y for x, y in zip(kpis1, kpis2))





def simulated_annealing(solution_init, data_obj, case_index_not_considered = [], objectives = ('objective1',), Obj_weightage = None, INITIAL_TEMPERATURE = 1000, ALPHA = 0.99, FINAL_TEMPERATURE = 0.001, MAX_ITERATIONS = 100000, theatre_prp_min = 7.5, consider_session_filled_ratio = False):
    # Ensure initial solution satisfies constraints
    
    print('Running Optimisation Problem with Simulated Annealing Algorithm')
    
    #z_cdf_data = read_z_table('Z_table_positive.xlsx') | read_z_table('Z_table_negative.xlsx')
    z_cdf_data = read_z_table()
    solution = solution_init.copy()

    cases_not_considered = [data_obj.CASES[i] for i in case_index_not_considered]
    
    if not satisfies_constraints(data_obj, individual = list(solution.T.flatten()), cases_to_skip=cases_not_considered):
        raise ValueError("Initial solution is invalid.")

    best_solution = solution
    
    #each_session_obj1 = {session:1/data_obj.SESSION_DURATION[session] * sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[case_ind], session]]* solution[sess_ind, case_ind] ) for sess_ind, session in enumerate(data_obj.SESSIONS)}
    #return 1/data_obj.SESSION_DURATION[session] * sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[], session] ])
    if consider_session_filled_ratio:
        each_session_obj1 = [1/data_obj.SESSION_DURATION[session] * sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[case_ind], session]* solution_init[sess_ind, case_ind] for case_ind in range(len(data_obj.CASES)) ]) for sess_ind, session in enumerate(data_obj.SESSIONS)]
    else:
        each_session_obj1 = None
    
    #print(each_session_obj1)
    temperature = INITIAL_TEMPERATURE
    iterations = 0
    n_cases = len(data_obj.CASES)
    n_sessions = len(data_obj.SESSIONS)
    
    #multi_objectives_solving = Obj_weightage is None or len(objectives) > len(Obj_weightage)
    #multi_objectives_solving = True
    kpis_length = len(objectives) if Obj_weightage is None else len(objectives)-len(Obj_weightage.keys()) + 1
    
    #if multi_objectives_solving:
    #current_value = multi_objectives_balanced(data_obj, None, 0, cases_to_skip = cases_not_considered, objectives_list = objectives, weightage = Obj_weightage, individual = solution.T.flatten())
    current_kpis = multi_objectives(data_obj, None, 0, cases_to_skip = cases_not_considered, objectives_list = objectives, weightage = Obj_weightage, prep_min= theatre_prp_min, cdf_table_data = z_cdf_data, individual = solution_init.T.flatten())
    #current_value = objective1_utilisation(data_obj, individual = solution.T.flatten(), cases_to_skip=cases_not_considered)
    
    best_kpis = current_kpis
    pareto_archive = [current_kpis]
    best_KPIS_list = [current_kpis]

    prev_unassigned_cases_indices = [i for i in range(n_cases) if i not in case_index_not_considered and np.sum(solution[:,i]) == 0]
    
    while temperature > FINAL_TEMPERATURE and iterations < MAX_ITERATIONS:
        iterations += 1
        
        # Generate a neighboring solution by swapping a case between sessions
        new_unassigned_cases_indices = [_ for _ in prev_unassigned_cases_indices]
        
        new_solution, changed_sessions_indices = neighbor_solution(solution, new_unassigned_cases_indices, n_sessions, n_cases, cases_indices_to_skip = case_index_not_considered, sessions_fill_ratio = each_session_obj1)
        #new_solution, changed_sessions_indices = neighbor_solution(solution, new_unassigned_cases_indices, n_sessions, n_cases, cases_indices_to_skip = case_index_not_considered, sessions_fill_ratio = None)
        
        # Check if the solution is feasible
        if satisfies_constraints(data_obj, individual = new_solution.T.flatten(), cases_to_skip=cases_not_considered):
            #new_value = objective1_utilisation(data_obj, individual = new_solution.T.flatten(), cases_to_skip=cases_not_considered)
            new_kpis = multi_objectives(data_obj, None, 0, cases_to_skip = cases_not_considered, objectives_list = objectives, weightage = Obj_weightage, prep_min= theatre_prp_min, cdf_table_data = z_cdf_data, individual = new_solution.T.flatten())
        
        else:
            new_kpis = [float('-inf')]* kpis_length  # Invalid solutions should have a very low value
        
        #if pareto_basis:
        if pareto_dominates(new_kpis, current_kpis):
            acceptance_probability = [1.0]*kpis_length
            
            # Update the best solution found so far
            
            if pareto_dominates(new_kpis, best_kpis):
                best_solution = new_solution.copy()
                best_kpis = new_kpis
                pareto_archive.append(current_kpis)
        else:
            # Calculate an acceptance probability based on a combination of objectives
            acceptance_probability = [math.exp((new_kpi - current_kpi) / temperature) for new_kpi, current_kpi in zip(new_kpis, current_kpis)]
        
        # Decide whether to accept the new solution
        if all(random.random() < prob for prob in acceptance_probability):
            solution = new_solution.copy()
            current_kpis = new_kpis
            prev_unassigned_cases_indices = new_unassigned_cases_indices
            if consider_session_filled_ratio:
                for sess_ind in changed_sessions_indices:
                    session = data_obj.SESSIONS[sess_ind]
                    each_session_obj1[sess_ind] = min(0.9, sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[case_ind], session]* solution[sess_ind, case_ind] for case_ind in range(len(data_obj.CASES)) ]))

        # Cool down
        temperature *= ALPHA

        if iterations%100 == 0:
            print(f'{iterations}th iterations is completed with best solution value reached {best_kpis}')
        
        best_KPIS_list.append(best_kpis)
    
    return best_solution, best_kpis, best_KPIS_list, pareto_archive, iterations
    


def simulated_annealing_with_paretos(solution_init, 
                                     data_obj, 
                                     case_index_not_considered = [], 
                                     objectives = ('objective1',), 
                                     Obj_weightage = None,
                                     INITIAL_TEMPERATURE = 100000, 
                                     ALPHA = 0.999, 
                                     FINAL_TEMPERATURE = 0.001, 
                                     MAX_ITERATIONS = 100000, 
                                     theatre_prp_min = 7.5, 
                                     consider_session_filled_ratio = False,
                                     consider_consult_procedure_combination = False,
                                     consider_consult_patient_combination = False
                                     ):
    # Ensure initial solution satisfies constraints
    
    print('Running Optimisation Problem with Simulated Annealing Algorithm')
    
    #z_cdf_data = read_z_table('Z_table_positive.xlsx') | read_z_table('Z_table_negative.xlsx')
    z_cdf_data = read_z_table()
    solution = solution_init.copy()

    cases_not_considered = [data_obj.CASES[i] for i in case_index_not_considered]
    
    if not satisfies_constraints(data_obj, individual = list(solution.T.flatten()), cases_to_skip=cases_not_considered):
        raise ValueError("Initial solution is invalid.")

    #best_solution = solution
    
    #each_session_obj1 = {session:1/data_obj.SESSION_DURATION[session] * sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[case_ind], session]]* solution[sess_ind, case_ind] ) for sess_ind, session in enumerate(data_obj.SESSIONS)}
    #return 1/data_obj.SESSION_DURATION[session] * sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[], session] ])
    if consider_session_filled_ratio:
        each_session_obj1 = [1/data_obj.SESSION_DURATION[session] * sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[case_ind], session]* solution_init[sess_ind, case_ind] for case_ind in range(len(data_obj.CASES)) ]) for sess_ind, session in enumerate(data_obj.SESSIONS)]
        each_session_obj1 = [max(0.05, val) for val in each_session_obj1]
    else:
        each_session_obj1 = None
    
    #print(each_session_obj1)
    temperature = INITIAL_TEMPERATURE
    iterations = 0
    n_cases = len(data_obj.CASES)
    n_sessions = len(data_obj.SESSIONS)
    
    #multi_objectives_solving = Obj_weightage is None or len(objectives) > len(Obj_weightage)
    #multi_objectives_solving = True
    kpis_length = len(objectives) if Obj_weightage is None else len(objectives)-len(Obj_weightage.keys()) + 1
    
    #if multi_objectives_solving:
    #current_value = multi_objectives_balanced(data_obj, None, 0, cases_to_skip = cases_not_considered, objectives_list = objectives, weightage = Obj_weightage, individual = solution.T.flatten())
    current_kpis = multi_objectives(data_obj, None, 0, 
                                    cases_to_skip = cases_not_considered, 
                                    objectives_list = objectives, 
                                    weightage = Obj_weightage, 
                                    prep_min= theatre_prp_min, 
                                    cdf_table_data = z_cdf_data, 
                                    individual = solution_init.T.flatten()
                                   )
    #current_value = objective1_utilisation(data_obj, individual = solution.T.flatten(), cases_to_skip=cases_not_considered)
    current_kpis = [round(_, 4) for _ in current_kpis]
    
    #best_kpis = current_kpis
    pareto_archive = [{'x': solution , 'f':current_kpis}]
    #best_KPIS_list = [current_kpis]

    prev_unassigned_cases_indices = [i for i in range(n_cases) if i not in case_index_not_considered and np.sum(solution[:,i]) == 0]

    iterations_candidate_pareto_KPIs = {}
    soft_constraints_pass = True

    proce_related_col = [col for col in data_obj.patients_df.columns if 'procedure' in col.lower()][0]

    while temperature > FINAL_TEMPERATURE and iterations < MAX_ITERATIONS:
        iterations += 1
        
        # Generate a neighboring solution by swapping a case between sessions
        new_unassigned_cases_indices = [_ for _ in prev_unassigned_cases_indices]
        
        new_solution, changed_sessions_indices = neighbor_solution(
                   solution,
                   new_unassigned_cases_indices, 
                   n_sessions, 
                   n_cases, 
                   cases_indices_to_skip = case_index_not_considered, 
                   sessions_fill_ratio = each_session_obj1
                      )
        #new_solution, changed_sessions_indices = neighbor_solution(solution, new_unassigned_cases_indices, n_sessions, n_cases, cases_indices_to_skip = case_index_not_considered, sessions_fill_ratio = None)
        

        if consider_consult_patient_combination:
            soft_constraints_pass = satisfies_patient_consultant_restriction(data_obj, new_solution, changed_sessions_indices, proce_related_col)

        elif consider_consult_procedure_combination:
            soft_constraints_pass = satisfies_soft_constraints(
                data_obj,
                new_solution,
                changed_sessions_indices,
                proce_related_col
            )
            
        # Check if the solution is feasible
        if soft_constraints_pass and satisfies_constraints(data_obj, 
                                           individual = new_solution.T.flatten(),
                                           cases_to_skip=cases_not_considered,
                                           specific_sessions = [data_obj.SESSIONS[_] for _ in changed_sessions_indices],
                                           cases_repetition_test = False
                                            ):
            #new_value = objective1_utilisation(data_obj, individual = new_solution.T.flatten(), cases_to_skip=cases_not_considered)
            new_kpis = multi_objectives(data_obj, None, 0, 
                                        cases_to_skip = cases_not_considered, 
                                        objectives_list = objectives, 
                                        weightage = Obj_weightage, 
                                        prep_min= theatre_prp_min, 
                                        cdf_table_data = z_cdf_data, 
                                        #individual = new_solution.T.flatten()
                                       )
            new_kpis = [round(_, 4) for _ in new_kpis]
        else:
            new_kpis = [float('-inf')]* kpis_length  # Invalid solutions should have a very low value
        
        #if pareto_basis:
        #if pareto_dominates(new_kpis, current_kpis):
        #    acceptance_probability = [1.0]*kpis_length
            
            # Update the best solution found so far
            
            #if pareto_dominates(new_kpis, best_kpis):
            #    best_solution = new_solution.copy()
            #    best_kpis = new_kpis
            #pareto_archive.append(current_kpis)
        #else:
            # Calculate an acceptance probability based on a combination of objectives
            #acceptance_probability = [math.exp((new_kpi - current_kpi) / temperature) for new_kpi, current_kpi in zip(new_kpis, current_kpis)]
        acceptance_probability = [1.0 if new >= current else math.exp((new - current) / temperature) for new, current in zip(new_kpis, current_kpis)]
        
        # Decide whether to accept the new solution
        if any(random.random() < prob for prob in acceptance_probability):
            
            solution = new_solution.copy()
            current_kpis = new_kpis
            prev_unassigned_cases_indices = new_unassigned_cases_indices
            if consider_session_filled_ratio:
                for sess_ind in changed_sessions_indices:
                    session = data_obj.SESSIONS[sess_ind]
                    each_session_obj1[sess_ind] = max(0.05, sum([data_obj.TASKS_DURATION.loc[data_obj.CASES[case_ind], session]* solution[sess_ind, case_ind] for case_ind in range(len(data_obj.CASES)) ]))

            pareto_archive, pareto_changed = update_pareto_archive(pareto_archive, {'x': solution, 'f':current_kpis})

            if pareto_changed:
                iterations_candidate_pareto_KPIs[iterations] = current_kpis
        # Cool down
        temperature *= ALPHA

        if iterations%100 == 0:
            print(f'{iterations}th iterations is completed with current solution value reached {current_kpis}')
        
        #best_KPIS_list.append(best_kpis)
    
    return pareto_archive, iterations_candidate_pareto_KPIs



def solve_optimisation_problem_with_MIP_solver(data_obj,
                    possible_task_durations_df = None, 
                    objectives = (), 
                    obj_weightage = {'objective1':1, 'objective2':0.5, 'objective3':1, 'objective4':1}, 
                    hyper_param = {},
                    consider_consult_procedure_combination = False,
                    consider_consult_patient_combination = False
                    ):
    
    for task in data_obj.TASKS:
        data_obj.add_TASKS_ASSIGNED_variable(task,0)
    
    case_index_not_considered = []
    n = len(data_obj.CASES)  # number of events
    s = len(data_obj.SESSIONS)   # number of slots
    #d = [30, 45, 60, 90, 30, 45, 60, 90, 30, 45, 60, 90, 30, 45, 60, 90, 30, 45, 60, 90]  # durations of events
    durations = np.zeros((s, n), dtype=int)
     
    case_index_not_considered = []

    if possible_task_durations_df is not None:
        for i, session in enumerate(data_obj.SESSIONS):
            for j, case in enumerate(data_obj.CASES):
                if pd.notna(possible_task_durations_df.loc[case, session]):
                    durations[i, j] = possible_task_durations_df.loc[case, session]
        
                else:
                    if not j in case_index_not_considered:
                        case_index_not_considered.append(j)
    else:
        for i, session in enumerate(data_obj.SESSIONS):
            for j, case in enumerate(data_obj.CASES):
                if pd.notna(data_obj.TASKS_DURATION.loc[case, session]):
                    durations[i, j] = data_obj.TASKS_DURATION.loc[case, session]

                else:
                    if not j in case_index_not_considered:
                        case_index_not_considered.append(j)
                #durations[i, j] = math.ceil(durations[i, j] / 5) * 5
    cases_with_no_consultant_provided = []
    
    if consider_consult_patient_combination and 'Consultant Code' in data_obj.patients_df.columns:
        
        consultant_patient_possibility = np.zeros((s, n), dtype=int)
        
        for j, case in enumerate(data_obj.CASES):
            #related_procedure = data_obj.patients_df.loc[case, proce_related_col]
            consultant_required = data_obj.patients_df.loc[case, 'Consultant Code']
            if pd.isna(consultant_required):
                if not j in case_index_not_considered:
                    cases_with_no_consultant_provided.append(j)
                continue
            
            for i, session_consultant in enumerate(data_obj.sessions_df['Consultant Code']):
                if consultant_required == session_consultant:
                    consultant_patient_possibility[i, j] = 1

        cases_with_no_consultant_provided = [a-len([_ for _ in case_index_not_considered if a>_]) for a in cases_with_no_consultant_provided]
        
        consultant_patient_possibility = np.delete(consultant_patient_possibility, case_index_not_considered, axis=1)

    
    if consider_consult_procedure_combination or len(cases_with_no_consultant_provided) > 0:
        proce_related_col = [col for col in data_obj.patients_df.columns if 'procedure' in col.lower()][0]
        consultant_procedure_possibility = np.zeros((s, n), dtype=int)
        for i, session_consultant in enumerate(data_obj.sessions_df['Consultant Code']):
            #session = session_row['Session ID']
            if not session_consultant in data_obj.consultant_procedures_template_dict:
                continue
            for j, case in enumerate(data_obj.CASES):
                related_procedure = data_obj.patients_df.loc[case, proce_related_col]
                if related_procedure in data_obj.consultant_procedures_template_dict[session_consultant]:
                    consultant_procedure_possibility[i, j] = 1

        consultant_procedure_possibility = np.delete(consultant_procedure_possibility, case_index_not_considered, axis=1)

    
        
            
    durations = np.delete(durations, case_index_not_considered, axis=1)
    #duration_ratio_chances_mean = np.delete(duration_ratio_chances_mean, case_index_not_considered, axis=1)
    #duration_ratio_chances_sd = np.delete(duration_ratio_chances_sd, case_index_not_considered, axis=1)
    #waiting_weeks = waiting_weeks[~case_index_not_considered]
    
    # Sort indices in descending order
    case_index_not_considered.sort(reverse=True)
    
    # Remove elements by index
    #for index in case_index_not_considered:
    #    del waiting_weeks[index]
    
    n = n - len(case_index_not_considered)
    
    T_session = [_ for key, _ in data_obj.SESSION_DURATION.items()]  # total available time in each slot
    total_durations = sum(T_session)
    
    # Redirect standard output
    #log_output = io.StringIO()
    #sys.stdout = log_output
    
    opti_model = Model("scheduling")

    # Include the custom event handler
    log_handler = LogEventHandler()
    opti_model.includeEventhdlr(log_handler, "LogEventHandler", "Handler for capturing log data")
    
    
    # Set verbosity level (e.g., 4 for detailed information)
    opti_model.setIntParam('display/verblevel', 4)
    
    # You can also set other parameters to customize the logging output
    opti_model.setIntParam('display/freq', 1)  # Display output at every node
    
    # Decision variables
    x = {}
    for i in range(s):
        for j in range(n):
            x[i, j] = opti_model.addVar(vtype="B", name=f"x({i},{j})")
    
    #if len(objectives) ==1 and ('objective1' in objectives or 'util' in objectives[0]):
    #   #Objective function (maximize total duration of events assigned to slots)
    #    opti_model.setObjective(
    #        sum(durations[i][j] * x[i, j] / total_durations for i in range(s) for j in range(n)),
    #        "maximize"
    #    )

    #else:
    combined_objective = None
    # Linear part of the objective: resource utilization
    if 'objective1' in objectives:
        resource_utilisation_objective = quicksum(durations[i, j] * x[i, j] for j in range(n) for i in range(s)) / total_durations
    
        combined_objective = obj_weightage['objective1']*resource_utilisation_objective
        #not_overrunning_objective = quicksum(1.0625 - quicksum(duration_ratio_chances_mean[i, j] * x[i, j] * durations[i, j] for j in range(n))/T_session[i] for i in range(s)) / s
    
    if 'objective2' in objectives:

        duration_ratio_chances_mean = np.zeros((len(data_obj.SESSIONS), len(data_obj.CASES)), dtype=float)
        duration_ratio_chances_sd = np.zeros((len(data_obj.SESSIONS),len(data_obj.CASES)), dtype=float)

        for i, session in enumerate(data_obj.SESSIONS):
            for j, case in enumerate(data_obj.CASES):
                if not j in case_index_not_considered:
                    duration_ratio_chances_mean[i, j] = data_obj.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[(case, session)][0]
                    duration_ratio_chances_sd[i, j] = data_obj.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD[(case, session)][1]

        duration_ratio_chances_mean = np.delete(duration_ratio_chances_mean, case_index_not_considered, axis=1)
        duration_ratio_chances_sd = np.delete(duration_ratio_chances_sd, case_index_not_considered, axis=1)      
        #print(duration_ratio_chances_mean)
        
        not_overrunning_objective = quicksum(1.0625 - quicksum(duration_ratio_chances_mean[i, j] * x[i, j] * durations[i, j] for j in range(n))/T_session[i] for i in range(s)) / s

        combined_objective = obj_weightage['objective2']* not_overrunning_objective if combined_objective is None else combined_objective + obj_weightage['objective2']* not_overrunning_objective
        
    
    if 'objective3' in objectives:
        waiting_weeks = [value for key, value in data_obj.CASES_RTT_WAIT_WEEKS.items()]
        
        # Remove elements by index
        for index in case_index_not_considered:
            del waiting_weeks[index]
        sum_waiting_weeks = sum(waiting_weeks)
        #Waiting weeks related objectives
        waiting_weeks_objective = quicksum(waiting_weeks[j] * x[i, j] for j in range(n) for i in range(s)) / sum_waiting_weeks

        combined_objective = obj_weightage['objective3']* waiting_weeks_objective if combined_objective is None else combined_objective + obj_weightage['objective3']* waiting_weeks_objective
        # Combine the objectives with the given weights using quicksum
        #combined_objective = weight_resources * resource_utilisation_objective + weight_probability * not_overrunning_objective
        # + obj_weightage['objective3']* waiting_weeks_objective

    if 'objective4' in objectives:
        priority_reward = [value for key, value in data_obj.CASES_SURGERY_PRIORITISATION_REWARD.items()]
        
        # Remove elements by index
        for index in case_index_not_considered:
            del priority_reward[index]
        sum_priority_reward = sum(priority_reward)
        priority_reward_objective = quicksum(priority_reward[j] * x[i, j] for j in range(n) for i in range(s)) / sum_priority_reward

        combined_objective = obj_weightage['objective4']* priority_reward_objective if combined_objective is None else combined_objective + obj_weightage['objective4']* priority_reward_objective

    opti_model.setObjective(combined_objective, "maximize")

    
    # Constraints
    if consider_consult_patient_combination and 'Consultant Code' in data_obj.patients_df.columns:
        for j in range(n):
            # if consultant provided implement consultant constraints else go for next one
            if not j in cases_with_no_consultant_provided:
                for i in range(s):
                    if consultant_patient_possibility[i,j] == 0:
                        opti_model.addCons(x[i, j] == 0)
            else:
                for i in range(s):
                    if consultant_procedure_possibility[i,j] == 0:
                        opti_model.addCons(x[i, j] == 0)
                                        
        #consultant procedure constraints
    elif consider_consult_procedure_combination:
        for j in range(n):
            for i in range(s):
                if consultant_procedure_possibility[i,j] == 0:
                    opti_model.addCons(x[i, j] == 0)
    
    # Each event is assigned to at most one slot
    for j in range(n):
        opti_model.addCons(
            sum(x[i, j] for i in range(s)) <= 1)
        #opti_model.addCons(n_x <= T_session[i]/95)
    
    # The total duration of the events assigned to each slot does not exceed the available time
    for i in range(s):
        #opti_model.addCons(
        #   sum(durations[i][j] * x[i, j] for j in range(n)) <= int(T[i] * data_obj.session_max_util / 5) * 5,
        #    name=f"Slot_{i+1}_Time")
        
        t_x = quicksum(durations[i, j] * x[i, j] for j in range(n))
        n_x = quicksum(x[i, j] for j in range(n))
        #sum(durations[i][j] * x[i, j] for j in range(n)) <= int(T[i] * data_obj.session_max_util / 5) * 5,
        opti_model.addCons(t_x <= int(T_session[i]* data_obj.session_max_util / 5) * 5)
        #opti_model.addCons(quicksum(duration_ratio_chances_mean[i, j] * x[i, j] * durations[i, j] for j in range(n)) <= int(T_session[i]* data_obj.session_max_util / 5) * 5)
        #opti_model.addCons(n_x * 1 >= 1)
        #opti_model.addCons(n_x <= T_session[i]/95)   
        opti_model.addCons(n_x <=max(3,min(round(T_session[i]/max(80, np.mean(durations[i,:])),0),7))) # to avoid overpopulating the session with lesser time requiring procedures
        #opti_model.addCons(t_x + (n_x + 1.5) * data_obj.theatre_prep_min /5)*5 <= T_session[i])   # limiting total time to prepn + surgical time
        #opti_model.addCons(t_x + n_x * data_obj.theatre_prep_min <= T_session[i] - np.ceil(1.5 * data_obj.theatre_prep_min/5)*5 ) # limiting total time to prepn + surgical time
        
        #opti_model.addCons(n_x <= + 1.5) * int(T_session[i]* (1-data_obj.session_max_util) / 5) * 5 data_obj.theatre_prep_min /5)*5 <= T_session[i])   # limiting total time to prepn + surgical time
    
    # Set various stopping criteria
    if 'time limit' in hyper_param:
        opti_model.setRealParam('limits/time', hyper_param['time limit']) 
    else:
        opti_model.setRealParam('limits/time', 100)
        # Time limit: 100 sec
    if 'nodes limit' in hyper_param:
        opti_model.setRealParam('limits/nodes', hyper_param['nodes limit']) 
    
    opti_model.optimize()
    
    status = opti_model.getStatus()
    print(f"Optimization status: {status}")
    
    # Extract the best solution found
    if status != 'optimal':
        print("Optimal solution not reached, extracting the best solution found.")
    
    if opti_model.getNSols() > 0:
        best_sol = opti_model.getBestSol()
        #best_solution_values = {(i, j): opti_model.getSolVal(best_sol, x[i, j]) for i in range(s) for j in range(n)}
        print(f"Total Value: {opti_model.getSolObjVal(best_sol)}")
        for i, session in enumerate(data_obj.SESSIONS):
            #print(f"Session {session}:")
            session_list = []
            session_time = 0
            for j, case in enumerate([element for idx, element in enumerate(data_obj.CASES) if idx not in case_index_not_considered]):
                if round(best_sol[x[i, j]])  == 1:
                    session_list.append(case)
                    session_time+= durations[i,j]
                    data_obj.TASKS_ASSIGNED[(case, session)] = 1
            #print(f"  Cases {session_list} and (Total Duration: {session_time})")

        return best_sol, [data_obj.CASES[i] for i in case_index_not_considered], opti_model
    
    else:
        print("No optimal solution found.")
        return None, None, None


class LogEventHandler(Eventhdlr):
    def __init__(self):
        self.progress_data = []

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)

    def eventexec(self, event):
        node = event.getNode()
        if node:
            node_id = node.getNumber()
            obj_val = round(self.model.getObjVal(),3)
            time_elapsed = self.model.getSolvingTime()
            
            '''
            if self.model.getNSols() > 0:
                best_sol = round(self.model.getBestSol())
                self.progress_data.append((node_id, obj_val, time_elapsed, best_sol))
                 # Print progress information
                print(f"Node {node_id}, Objective Value {obj_val}, Time Elapsed {time_elapsed}, Best sol: {best_sol}")
            else:
            '''
            self.progress_data.append((node_id, obj_val, time_elapsed))
             # Print progress information
            print(f"Node {node_id}, Objective Value {obj_val}, Time Elapsed {time_elapsed}")
                

from typing import Tuple


#def update_pareto_archive(archive: list[dict], candidate: dict) -> Tuple[list[dict], bool]:
def update_pareto_archive(archive, candidate ):
    new_archive = []
    updated = False
    for sol in archive:
        if pareto_dominates(candidate["f"], sol["f"]):
            continue  # candidate dominates → remove sol
        if pareto_dominates(sol["f"], candidate["f"]):
            return archive, False  # candidate is dominated → discard
        new_archive.append(sol)
    new_archive.append(candidate)
    return new_archive, True