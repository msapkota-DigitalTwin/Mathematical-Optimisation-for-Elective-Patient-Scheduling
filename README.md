# Optimal Surgical Scheduling - Implementation Manual (Optimal_Schedule_Generator)

**Document Version:** 2.0  
**Date:** February 2026  
**System:** Surgical Elective Patient Scheduling Model  
**Module:** Optimal_Schedule_generator.ipynb  
**Specialty:** Trauma & Orthopedics (T&O) - Theatre Planning

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements & Dependencies](#system-requirements--dependencies)
3. [Data Requirements & Preparation](#data-requirements--preparation)
4. [Implementation Steps](#implementation-steps)
5. [Configuration Parameters](#configuration-parameters)
6. [Optimization Algorithm](#optimization-algorithm)
7. [Output & Results Analysis](#output--results-analysis)
8. [Troubleshooting & Validation](#troubleshooting--validation)

---

## Overview

The Optimal_Schedule_Generator is an advanced surgical scheduling system designed for NHS trusts to optimally allocate surgical cases to available theatre sessions. The system integrates:

- **Surgical time prediction models** (trained on historical data)
- **Multi-objective optimization** using MIP (Mixed Integer Programming) solver
- **Real-time patient wait time tracking** (RTT - Referral To Treatment)
- **Consultant procedure restrictions** based on historical practice
- **Theatre capacity optimization** with variability handling

### Key Features

✓ Predictive task duration estimation with confidence intervals  
✓ 4-objective optimization framework (scheduling efficiency, duration variability, RTT wait times, procedure priority)  
✓ Consultant-patient compatibility enforcement  
✓ Theatre prep time management (default 7.5 minutes between procedures)  
✓ Visual schedule output with occupancy analysis  
✓ Comprehensive performance metrics reporting  

---

## System Requirements & Dependencies

### Python Version
- Python 3.7+
- Jupyter Notebook or VS Code with Jupyter extension

### Core Libraries
```
pandas         >= 2.0.3
numpy          >= 1.24.3
matplotlib     >= 3.5.0
scikit-learn   >= 1.0.0
scipy          >= 1.9.0
```

### Optimization Solver
- **SCIP** (Solving Constraint Integer Programs) - for MIP solver
- Installation: `pip install pyscipopt`

### Custom Modules (in same directory)
1. **Scheduling_Model.py** - Core scheduling model class
2. **procedure_time_and_variabiltiy_related_functions.py** - Surgical time prediction
3. **data_processing_nd_encoding_related_functions.py** - Data preprocessing
4. **optimisation_related_functions.py** - Optimization and objective functions
5. **predictive_model_related_functions.py** - ML model integration

### Supporting Files
- `label_mapping.pickle` - Procedure code to procedure name mapping
- `Consultant_procedure_template.xlsx` - Historical consultant-procedure associations
- Trained predictive models (Feb 2025 models directory)

---

## Data Requirements & Preparation

### 1. Session Planning Data
**File:** `Future_TandO_Sessions_filtered.xlsx`  
**Location:** `IO_data/July_Aug_2025/`

**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Session ID | String | Unique session identifier |
| Consultant Code | String | Consultant unique code (format: C + digits) |
| Session Planned Start Date/Time | DateTime | Session start time |
| Total Slot Minutes | Integer | Available theatre minutes |
| Day of the week | String | Auto-calculated from date |

**Example:**
```
Session ID: S_TAO_20250703_AM_001
Consultant Code: C6077137
Total Slot Minutes: 180
Session Planned Start Date/Time: 03/07/2025 08:00
```

### 2. Patient (Cases) Data (patient_df)
**File:** `Patient_list.xlsx`  
**Location:** `IO_data/July_Aug_2025/`

**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| Patient ID | String | Unique patient identifier |
| Intended Procedure Code | String | OPCS procedure code |
| Procedure Priority | String | Urgency level (urgent, soon, soon) |
| RTT Waiting Weeks | Numeric | Weeks on waiting list |
| Consultant Code | String | Associated consultant(if specified) |
| Allocated Time (Optional) | Numeric | Planned procedure duration (minutes) |

### 3. Patients' Further Data
**File:** `Patient_data.xlsx` (Optional - only if separate (not available) from patient_df)  
**Location:** `IO_data/July_Aug_2025/`

**Purpose:** Data for surgical time prediction and additional patient attributes  
**Note:** If all required patient attributes are contained within patient_df, this file is not required. The system will use patient_df when available.

**Key Columns (if needed):**
- Patient ID
- Gender
- Age group at Admit
- Patient Classification (Daycase or Ordniary admission)
- Others as per ML model features directory

### 4. Consultant-Procedure Template
**File:** `Consultant_procedure_template.xlsx`  
**Location:** `IO_data/`

**Purpose:** Define which consultants can perform which procedures (historical capability matrix)  
**Structure:**
```
|C6077137  | C6077138  | ... |
| M151     | M171      | ... |
| M152     | M152      | ... |
```

### 5. Trained Predictive Models
**Location:** `Theatre_Procedure_Time_Predictive_Modelling/.........../`

**Contents:**
- Trained RandomForest/XGBoost models for surgical time prediction
- Model categorization data
- Pre-processing parameters
- Feature importance mappings
- Removed/filtered training data for fallback estimates

---

## Data Validation Requirements

### Uniqueness Constraints (Critical)

All input data must satisfy the following uniqueness requirements:

#### Patient ID Uniqueness
- **Requirement:** Each Patient ID in `patient_df` must be unique
- **Consequence of Violation:** Duplicate patients will create multiple tasks and invalid scheduling decisions also gives error
- **Validation Code:**
```python
# Check for duplicate patient IDs
duplicate_patients = patient_df[patient_df.duplicated(subset=['Patient ID'], keep=False)]
assert len(duplicate_patients) == 0, f"Duplicate Patient IDs found:\n{duplicate_patients['Patient ID'].unique()}"
print(f"✓ Patient ID uniqueness: {len(patient_df)} unique patients")
```

#### Session ID Uniqueness
- **Requirement:** Each Session ID in `sessions_plan` must be unique
- **Consequence of Violation:** Duplicate sessions will cause conflicting resource allocation
- **Validation Code:**
```python
# Check for duplicate session IDs
duplicate_sessions = sessions_plan[sessions_plan.duplicated(subset=['Session ID'], keep=False)]
assert len(duplicate_sessions) == 0, f"Duplicate Session IDs found:\n{duplicate_sessions['Session ID'].unique()}"
print(f"✓ Session ID uniqueness: {len(sessions_plan)} unique sessions")
```

### Data Completeness Requirements

#### Required Patient Columns
- Patient ID (must not be null, must be unique)
- Intended Procedure Code (must not be null)
- Procedure Priority (could be be one of: P1, P2, P3, P4 or '')
- RTT Waiting Weeks (numeric, can be 0 or NULL)
- Consultant Code (optional, but if present must reference valid consultant in sessions_plan)

#### Required Session Columns
- Session ID (must not be null, must be unique)
- Consultant Code (must not be null, must be valid format: starts with 'C', not 'C9999')
- Total Slot Minutes (must be numeric, > 0)
- Session Planned Start Date/Time (must be valid datetime)

### Data Validation Workflow

```python
def validate_input_data(patient_df, sessions_plan):
    """Comprehensive data validation before scheduling"""
    
    # 1. Patient ID Uniqueness
    assert patient_df['Patient ID'].nunique() == len(patient_df), \
        "Patient IDs are not unique"
    
    # 2. Session ID Uniqueness
    assert sessions_plan['Session ID'].nunique() == len(sessions_plan), \
        "Session IDs are not unique"
    
    # 3. No null Patient IDs
    assert patient_df['Patient ID'].notna().all(), \
        "Found null Patient IDs"
    
    # 4. No null Session IDs
    assert sessions_plan['Session ID'].notna().all(), \
        "Found null Session IDs"
    
    # 5. Required patient columns present
    required_patient_cols = ['Patient ID', 'Intended Procedure Code', 'Procedure Priority']
    for col in required_patient_cols:
        assert col in patient_df.columns, f"Missing required column: {col}"
    
    # 6. Required session columns present
    required_session_cols = ['Session ID', 'Consultant Code', 'Session Planned Start Date/Time']
    for col in required_session_cols:
        assert col in sessions_plan.columns, f"Missing required column: {col}"
    
# Run validation
validate_input_data(patient_df, sessions_plan)
```

---



### Step 1: Load Dependencies

```python
import random
import matplotlib.pyplot as plt
from itertools import accumulate, product
import pandas as pd
import datetime
import numpy as np
import pickle
import os
import sys
from matplotlib.figure import Figure
import math
import string
import time
```

### Step 2: Define Data Directories

```python
work_direc = r"\\esneft.nhs.uk\share\Finance\Capacity & Analytics\Business Informatics\Analytics Team\Surgical Elective Patient Scheduling Model\Optimal_Scheduling_Model\IO_data"

trained_models_direc = r"\\esneft.nhs.uk\share\Finance\Capacity & Analytics\Business Informatics\Analytics Team\Surgical Elective Patient Scheduling Model\Theatre_Procedure_Time_Predictive_Modelling\Data_and_model_Feb_2025"
```

**Note:** Update paths based on your network environment. For local testing:
```python
work_direc = r"./IO_data"
trained_models_direc = r"./trained_models"
```

### Step 3: Load Session Planning Data

```python
from procedure_time_and_variabiltiy_related_functions import load_timeslot_file

sessions_plan = load_timeslot_file(os.path.join(work_direc, 'July_Aug_2025/Future_TandO_Sessions_filtered.xlsx'))

# Filter active consultants only (Consultant Code must start with 'C', exclude C9999 codes)
sessions_plan = sessions_plan[
    sessions_plan['Consultant Code'].apply(
        lambda x: False if type(x) == 'int' or not str(x).startswith('C') or str(x).startswith('C9999') else True
    )
]

# Optional: Limit to first N sessions for testing
# sessions_plan = sessions_plan.iloc[0:17, :]
```

**Output Validation:**
```python
print(f"Total sessions loaded: {len(sessions_plan)}")
print(f"Unique consultants: {sessions_plan['Consultant Code'].nunique()}")
print(f"Total available slot minutes: {sessions_plan['Total Slot Minutes'].sum()}")
```

### Step 4: Load Patient Data

```python
patient_df = pd.read_excel(os.path.join(work_direc, 'July_Aug_2025/Patient_list.xlsx'))

# Optional: Load additional patient data if attributes are not in patient_df
# patient_data = pd.read_excel(os.path.join(work_direc, 'July_Aug_2025/Patient_data.xlsx'))

print(f"Total patients to schedule: {len(patient_df)}")
```

**Note:** If all patient attributes are already in `patient_df` and separate patient_data file is not required.

### Step 5: Obtain Prediction Model Parameters

```python
from procedure_time_and_variabiltiy_related_functions import get_prediction_required_parameters

procedure_time_prediction_parameters = get_prediction_required_parameters(work_direc, trained_models_direc)

# Parameters tuple contains:
# [0] models_types: List of model types (e.g., ['RandomForest', 'XGBoost'])
# [1] data_model_categorisation: Model categorization metadata
# [2] pred_model_related_preprocessed_data: Preprocessed training data
# [3] removed_data_from_processing: Data filtered during preprocessing
# [4] model_input_features_all: All input features required by models
# [5] possible_varying_data_for_patient: Non-fixed patient attributes
# [6] force_encoding_cols: Categorical columns needing encoding
# [7] trained_models_direc: Path to trained models
```

### Step 6: Initialize Scheduling Model

```python
from Scheduling_Model import Scheduling_Model

sched_model = Scheduling_Model(patient_df, sessions_plan)

print(f"Model initialized with {len(sched_model.CASES)} cases and {len(sched_model.SESSIONS)} sessions")
```

**Scheduling_Model Class Structure:**

The Scheduling_Model contains critical attributes:

| Attribute | Type | Purpose |
|-----------|------|---------|
| CASES | tuple | Patient IDs to schedule |
| SESSIONS | tuple | Available theatre sessions |
| SESSION_DURATION | dict | Minutes per session |
| TASKS | tuple | (Patient ID, Session ID) pairs |
| TASKS_DURATION | DataFrame | Predicted surgical durations |
| TASKS_DURATION_DEVIATION_RATIO_CHANCES | dict | Duration probability distributions |
| CASES_RTT_WAIT_WEEKS | dict | Patient RTT waiting weeks |
| CASES_SURGERY_PRIORITISATION_REWARD | dict | Priority weightage rewards |
| session_max_util | float | Max utilization (default 0.9 = 90%) |
| theatre_prpn_min | float | Prep time in minutes (default 7.5) |

### Step 7: Generate Task Durations (Surgical Time Prediction)

```python
# If patient_data is available (separate from patient_df)
sched_model.generate_tasks_durations(patient_df, sessions_plan, patient_data, procedure_time_prediction_parameters)

# OR if all patient attributes are in patient_df (EPIC data scenario)
sched_model.generate_tasks_durations(patient_df, sessions_plan, patient_df, procedure_time_prediction_parameters)
```

**Note:** Pass `patient_df` twice if patient data attributes are already contained within the patient dataframe. This is common in EPIC-sourced data where all patient information is consolidated.

**This step:**
- Uses trained ML models to predict surgical duration for each patient-session pair
- Returns probabilities of completing within different time frames
- Falls back to planned duration if prediction fails

### Step 8: Add RTT Waiting Time Data (Objective 3)

```python
sched_model.add_cases_RTT_waiting_time(patient_df)
sched_model.SUM_CASES_RTT_WAIT_WEEKS = sum(sched_model.CASES_RTT_WAIT_WEEKS.values())

print(f"RTT tracking: {len(sched_model.CASES_RTT_WAIT_WEEKS)} patients")
print(f"Total RTT weeks across waitlist: {sched_model.SUM_CASES_RTT_WAIT_WEEKS}")
```

**RTT Objective:** Minimize total waiting weeks for patients already on the list

### Step 9: Add Surgery Priority Data (Objective 4)

```python
sched_model.add_cases_surgery_Priority(patient_df)

print(f"Sum of priority rewards: {sched_model.SUM_CASES_SURGERY_PRIORITISATION_REWARD}")
```

**Priority Mapping:**
| Priority Level | Reward Value |
|--------|------|
| P1 | Higher |
| P2 | Medium |
| P3, P4 (routine) | Standard |

### Step 10: Generate Duration Variability Data (Objective 2)

```python
# Generate probability distributions for procedure duration deviations
sched_model.generate_chances_for_tasks_durations(
    patient_df, sessions_plan, 
    variability_counts=30,  # Generate 30 duration scenarios per task
    patient_data=patient_data,
    procedure_time_pred_parameters=procedure_time_prediction_parameters
)

# Calculate mean and standard deviation for each task's duration distribution
sched_model.obtain_mean_sd_for_tasks_durations_chances(
    patient_df, sessions_plan, 30, patient_data, procedure_time_prediction_parameters
)

print(f"Duration variability data: {len(sched_model.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD)} tasks with stats")
```

**This step creates:**
- Duration confidence intervals (68%, 95%, 99% probability bands)
- Mean and standard deviation for schedule robustness calculation

### Step 11: Configure Model Parameters

```python
# Theatre utilization and prep time settings
sched_model.session_max_util = 0.9   # 90% max utilization
sched_model.theatre_prpn_min = 7.5   # 7.5 minutes prep between procedures

# Load consultant procedure template
sched_model.add_consultant_procedures_template(
    pd.read_excel(os.path.join(work_direc, 'Consultant_procedure_template.xlsx'))
)

print(f"Consultant procedures template loaded with {len(sched_model.consultant_procedures_template_dict)} consultants")
```

---

## Configuration Parameters

### Optimization Hyperparameters

```python
hyper_param_SCIP = {
    'time limit': len(patient_df) * len(sessions_plan) / 20
    # Time limit = (num_patients × num_sessions) / 20
    # Example: 250 patients × 17 sessions = 212.5 seconds (~3.5 minutes)
}
```

### Objective Weightage Configuration

The system optimizes 4 objectives with configurable weightages:

```python
obj_weightage = {
    'objective1': 0.4,                                          # Scheduling rate
    'objective2': 0.15,                                         # Duration variability
    'objective3': 0.2 * len(patient_df) / len(sessions_plan) / 5,  # RTT waiting time
    'objective4': 0.45 * len(patient_df) / len(sessions_plan) / 5  # Priority
}

# Normalize weights to sum to 1.0
objectives = ('objective1', 'objective2', 'objective3', 'objective4')
sum_value = sum(obj_weightage[key] for key in objectives)
obj_weightage = {key: value/sum_value for key, value in obj_weightage.items()}
```

#### Objective Definitions

| Objective | Weight | Formula | Description |
|-----------|--------|---------|-------------|
| objective1 | 0.4 | # Scheduled Patients | Maximize patients scheduled |
| objective2 | 0.15 | Σ(std dev) | Minimize duration variance (reliability) |
| objective3 | Scaled | Σ(RTT weeks) | Minimize total waiting weeks |
| objective4 | Scaled | Σ(priority rewards) | Maximize priority satisfaction |

**Scaling Logic:**
- Objectives 3 & 4 scale with patient/session ratio
- Prevents objectives from dominating due to scale differences
- Default scaling: 0.2/5 = 0.04 and 0.45/5 = 0.09 (baseline)

---

## Optimization Algorithm

### Create Optimal Schedule with MIP Solver

```python
sched_model.create_schedule_with_optimisation(
    optimisation_algorithm='MIP Solver',
    objectives=('objective1', 'objective2', 'objective3', 'objective4'),
    weightage=obj_weightage,
    hyper_param=hyper_param_SCIP,
    consult_patient_restriction=True,      # Enforce consultant-patient compatibility
    consult_procedure_restriction=True     # Enforce consultant-procedure template
)
```

### Solver Configuration

**Algorithm:** SCIP (Solving Constraint Integer Programs)

**Decision Variables:**
- **x[task]** ∈ {0,1}: Whether task (patient-session) is assigned
- **s[task]** ∈ [0, ub]: Start time for scheduled task

**Constraints:**
1. Each patient scheduled ≤ 1 session
2. Session capacity: Σ(task_duration) ≤ session_duration × session_max_util
3. Consultant restrictions: Only if consultant-procedure template allows
4. Theatre prep time: ≥ 7.5 min between procedures
5. Temporal ordering: Tasks within session don't overlap

**Objective Function:**

```
Maximize: Σ λ_i × f_i(solution)

Where:
λ_1 = 0.40 → f_1 = # scheduled patients  
λ_2 = 0.15 → f_2 = -Σ(std_dev[task])    (minimize variance)
λ_3 = 0.04 → f_3 = -Σ(RTT_weeks)        (minimize wait)  
λ_4 = 0.09 → f_4 = Σ(priority_reward)   (maximize priority)
```

### Solver Output

After optimization completes:

```python
print(f"Scheduled patients: {sched_model.schedule_new.shape[0]}")
print(f"Not scheduled: {sched_model.patients_not_scheduled.shape[0]}")
print(f"Objective value: {sched_model.objective_value}")
```

---

## Output & Results Analysis

### Step 1: Test Results with Multi-Objective Analysis

```python
from optimisation_related_functions import multi_objectives
from data_processing_nd_encoding_related_functions import read_z_table

multi_objectives(
    sched_model,
    None,
    0,
    cases_to_skip=sched_model.cases_not_considered,
    objectives_list=('objective1', 'objective2', 'objective3', 'objective4'),
    prep_min=sched_model.theatre_prpn_min,
    cdf_table_data=read_z_table()
)
```

**Outputs:**
- Objective values for final schedule
- Performance metrics vs baseline
- Statistical significance tests

### Step 2: Enrich Schedule with Patient Attributes

```python
# Add priority information
sched_model.schedule_new['Procedure Priority'] = sched_model.schedule_new['Patient ID'].map(
    sched_model.patients_df['Procedure Priority']
)

# Add priority reward values
sched_model.schedule_new['Procedure Priority Reward'] = sched_model.schedule_new['Patient ID'].map(
    sched_model.CASES_SURGERY_PRIORITISATION_REWARD
)

# Add RTT waiting data
sched_model.schedule_new['RTT WAIT WEEKS'] = sched_model.schedule_new['Patient ID'].map(
    sched_model.CASES_RTT_WAIT_WEEKS
)

# Do same for unscheduled patients
sched_model.patients_not_scheduled['Procedure Priority'] = sched_model.patients_not_scheduled['Patient ID'].map(
    sched_model.patients_df['Procedure Priority']
)
sched_model.patients_not_scheduled['Procedure Priority Reward'] = sched_model.patients_not_scheduled['Patient ID'].map(
    sched_model.CASES_SURGERY_PRIORITISATION_REWARD
)
sched_model.patients_not_scheduled['RTT WAIT WEEKS'] = sched_model.patients_not_scheduled['Patient ID'].map(
    sched_model.CASES_RTT_WAIT_WEEKS
)
```

### Step 3: Analyze Scheduling Performance

```python
# Procedure priority distribution
print("Scheduled by priority:")
print(sched_model.schedule_new['Procedure Priority'].value_counts())

print("\nNot scheduled by priority:")
print(sched_model.patients_not_scheduled['Procedure Priority'].value_counts())

# Average RTT waiting time comparison
scheduled_rtt = sched_model.schedule_new['RTT WAIT WEEKS'].mean()
not_scheduled_rtt = sched_model.patients_not_scheduled['RTT WAIT WEEKS'].mean()

print(f"\nAverage RTT weeks:")
print(f"Scheduled: {scheduled_rtt:.2f} weeks")
print(f"Not scheduled: {not_scheduled_rtt:.2f} weeks")
```

### Step 4: Visualize Theatre Schedule

```python
from procedure_time_and_variabiltiy_related_functions import plot_theatre_occupancy_data

# Load procedure name mappings
with open(os.path.join(work_direc, 'label_mapping.pickle'), 'rb') as file:
    label_mapping = pickle.load(file)

# Create theatre occupancy visualization
ax = plot_theatre_occupancy_data(
    sched_model.schedule_new[[
        'Scheduled Start Date/Time',
        'Planned H4 Minutes',
        'Patient ID',
        'Intended Procedure Code',
        'Session ID',
        'Consultant Code'
    ]],
    figsize=(5.5, len(sessions_plan)/1.9),
    label_mapping=label_mapping,
    color_mapping_col='Intended Procedure Code',
    grouping_col='Session ID'
)

# plt.savefig(os.path.join(work_direc, 'Output_files/schedule_output.png'), dpi=300, bbox_inches='tight')
```

**Visualization Shows:**
- Each theatre session as a horizontal bar
- Patient procedures as colored blocks
- Time utilization and gaps
- Consultant assignments

### Step 5: Compare with Consultant Planning

```python
# Load actual planned patient data
df_actual = patient_data[patient_data['Session ID'].isin(sessions_plan['Session ID'].values)]

# Visualize what was actually planned
ax_planned = plot_theatre_occupancy_data(
    df_actual[[
        'Planned Start Date/Time',
        'Planned Duration',
        'Patient ID',
        'Intended Procedure Code',
        'Session ID',
        'Consultant Code'
    ]],
    figsize=(5.5, len(sessions_plan)/1.9),
    label_mapping=label_mapping,
    color_mapping_col='Intended Procedure Code',
    grouping_col='Session ID',
    extra_y_label='Consultant Code'
)
```

---

## Output Files & Formats

### Primary Outputs

#### 1. Scheduled Patients (DataFrame)
**Variable:** `sched_model.schedule_new`

**Columns:**
- Patient ID
- Session ID
- Consultant Code
- Intended Procedure Code
- Planned H4 Minutes (predicted duration)
- Scheduled Start Date/Time
- Procedure Priority
- RTT WAIT WEEKS
- Procedure Priority Reward

**Export:**
```python
sched_model.schedule_new.to_excel(
    os.path.join(work_direc, 'Output_files/scheduled_patients.xlsx'),
    index=False
)
```

#### 2. Unscheduled Patients (DataFrame)
**Variable:** `sched_model.patients_not_scheduled`

**Includes same columns as scheduled, indicates capacity/constraint reasons**

```python
sched_model.patients_not_scheduled.to_excel(
    os.path.join(work_direc, 'Output_files/unscheduled_patients.xlsx'),
    index=False
)
```

#### 3. Updated Session Information
**Variable:** `sched_model.sessions_updated`

**Contains:**
- Session ID
- Actual assignments per session
- Utilization rates
- Remaining capacity

#### 4. Schedule Visualization
**Format:** PNG (high-res for printing)
**Resolution:** 300 DPI
**Shows:** Theatre occupancy, time utilization, procedure mix

---

## Troubleshooting & Validation

### Issue 1: Missing Task Durations

**Symptom:** Many NaN values in TASKS_DURATION after generation

**Causes:**
- Trained models don't cover specific procedure-consultant combination
- Missing patient attributes in prediction features

**Solution:**
```python
# Check coverage
missing_tasks = sched_model.TASKS_DURATION[sched_model.TASKS_DURATION.isna().all(axis=1)]
print(f"Missing task durations: {len(missing_tasks)}")

# Fallback to planned duration
for id in missing_tasks.index:
    patient_id = id[0] if isinstance(id, tuple) else id
    fallback = patient_df[patient_df['Patient ID'] == patient_id]['Planned Duration'].values
    if len(fallback) > 0:
        sched_model.TASKS_DURATION.loc[id, :] = fallback[0]
```

### Issue 2: Consultant not in Procedure Template

**Symptom:** Consultant appears in schedule but not in template

**Causes:**
- Template missing consultant entry
- New consultants added after template creation

**Solution:**
```python
# Identify missing consultants
template_consultants = set(sched_model.consultant_procedures_template_dict.keys())
session_consultants = set(sessions_plan['Consultant Code'].unique())
missing = session_consultants - template_consultants

print(f"Consultants in sessions but not in template: {missing}")

# Add consultant with all procedures they attempt in patient_data
for consultant in missing:
    procedures = set(
        patient_data[patient_data['Consultant Code'] == consultant]
        ['Intended Procedure Code'].unique()
    )
    sched_model.consultant_procedures_template_dict[consultant] = list(procedures)
```

### Issue 3: Poor Scheduling Rate (< 80%)

**Symptom:** Only 60-70% of patients scheduled

**Causes:**
1. Consultant procedure restrictions too strict
2. Session capacity insufficient
3. Objective weightage favoring other criteria

**Solutions:**

```python
# A) Relax consultant restriction
sched_model.create_schedule_with_optimisation(
    optimisation_algorithm='MIP Solver',
    objectives=objectives,
    weightage=obj_weightage,
    hyper_param=hyper_param_SCIP,
    consult_patient_restriction=False,  # Allow patient override
    consult_procedure_restriction=False # Allow procedure override
)

# B) Adjust objective weights to prioritize scheduling
obj_weightage_aggressive = {
    'objective1': 0.7,  # Increase from 0.4
    'objective2': 0.05,
    'objective3': 0.05,
    'objective4': 0.2
}
# Normalize and retry...

# C) Check session capacity
print("Session capacity analysis:")
for session_id in sessions_plan['Session ID']:
    total_duration = sessions_plan[sessions_plan['Session ID'] == session_id]['Total Slot Minutes'].values[0]
    scheduled_duration = sched_model.schedule_new[sched_model.schedule_new['Session ID'] == session_id]['Planned H4 Minutes'].sum()
    utilization = scheduled_duration / total_duration * 100
    print(f"{session_id}: {scheduled_duration:.0f}/{total_duration:.0f} min ({utilization:.1f}%)")
```

### Issue 4: Validation & Constraint Checking

```python
# Verify no patient assigned twice
double_bookings = sched_model.schedule_new.groupby('Patient ID').size()[
    sched_model.schedule_new.groupby('Patient ID').size() > 1
]
assert len(double_bookings) == 0, f"Double-booked patients: {double_bookings.index.tolist()}"

# Verify session capacity not exceeded
for session_id in sched_model.SESSIONS:
    assigned = sched_model.schedule_new[sched_model.schedule_new['Session ID'] == session_id]
    total_assigned = assigned['Planned H4 Minutes'].sum()
    session_capacity = sched_model.SESSION_DURATION[session_id] * sched_model.session_max_util
    assert total_assigned <= session_capacity, \
        f"Session {session_id} over capacity: {total_assigned} > {session_capacity}"

# Verify consultant-procedure restrictions (if enforced)
if consult_procedure_restriction:
    for _, row in sched_model.schedule_new.iterrows():
        consultant = row['Consultant Code']
        procedure = row['Intended Procedure Code']
        allowed = procedure in sched_model.consultant_procedures_template_dict[consultant]
        assert allowed, f"Consultant {consultant} cannot do procedure {procedure}"

print("✓ All constraints validated successfully")
```

---

## Data Validation Requirements

### Critical Uniqueness Constraints

Before running the scheduling model, **ALL input data must pass rigorous validation checks**, particularly around uniqueness constraints. These are not optional - violations will cause the model to fail or produce incorrect results.

#### Patient ID Uniqueness (CRITICAL)

Each Patient ID must be **unique** with **NO repetition allowed**. Every patient should appear exactly once in the patient_df.

```python
# Validate patient ID uniqueness
duplicate_patient_ids = patient_df[patient_df.duplicated(subset=['Patient ID'], keep=False)]
assert len(duplicate_patient_ids) == 0, \
    f"ERROR: Patient IDs must be unique. Found {len(duplicate_patient_ids)//2} duplicate Patient IDs: {duplicate_patient_ids['Patient ID'].unique().tolist()}"

print(f"✓ Patient ID uniqueness verified: {len(patient_df)} unique patients")
```

#### Session ID Uniqueness (CRITICAL)

Each Session ID must be **unique** with **NO repetition allowed**. Every scheduled session should appear exactly once in the sessions_plan.

```python
# Validate session ID uniqueness  
duplicate_session_ids = sessions_plan[sessions_plan.duplicated(subset=['Session ID'], keep=False)]
assert len(duplicate_session_ids) == 0, \
    f"ERROR: Session IDs must be unique. Found {len(duplicate_session_ids)//2} duplicate Session IDs: {duplicate_session_ids['Session ID'].unique().tolist()}"

print(f"✓ Session ID uniqueness verified: {len(sessions_plan)} unique sessions")
```

### Required Data Columns

#### Patient DataFrame (patient_df) - Required Columns

| Column Name | Data Type | Description | Validation |
|---|---|---|---|
| Patient ID | String/Int | Unique patient identifier | Must be unique, no duplicates |
| Age | Float | Patient age | Positive value |
| RTT WAIT WEEKS | Float | Referral-to-treatment weeks waited | Non-negative |
| Intended Procedure Code | String | Primary procedure code | Non-empty |
| Consultant Code | String | Assigned consultant identifier | Format: C followed by digits |
| Planned H4 Minutes | Float | Estimated procedure duration | Positive value |
| Procedure Priority | Int | Priority level (1=urgent, 4=routine) | Values in [1, 2, 3, 4] |

#### Session DataFrame (sessions_plan) - Required Columns

| Column Name | Data Type | Description | Validation |
|---|---|---|---|
| Session ID | String | Unique session identifier | Must be unique, no duplicates |
| Consultant Code | String | Session consultant | Format: C followed by digits |
| Total Slot Minutes | Float | Available session duration | Positive value |
| Session Date | Date | Scheduled session date | Valid date format |
| Theatre Name | String | Theatre location | Non-empty |

### Comprehensive Data Validation Workflow

Before proceeding with the scheduling model, execute the following comprehensive validation function:

```python
def validate_input_data(patient_df, sessions_plan, patient_data=None):
    """
    Comprehensive validation of input data for scheduling model.
    
    Parameters:
    -----------
    patient_df : pd.DataFrame
        Patient data including age, procedures, priorities
    sessions_plan : pd.DataFrame
        Available sessions with consultant and duration info
    patient_data : pd.DataFrame, optional
        Additional patient details (if separate from patient_df)
    
    Returns:
    --------
    bool : True if all validations pass, raises AssertionError otherwise
    """
    
    print("\n" + "="*60)
    print("DATA VALIDATION REPORT")
    print("="*60 + "\n")
    
    # ==== PATIENT DATA VALIDATION ====
    print("1. PATIENT DATA VALIDATION")
    print("-" * 60)
    
    # Check for missing data
    assert not patient_df.isnull().any().any(), \
        f"ERROR: Found null values in patient_df:\n{patient_df.isnull().sum()}"
    print("✓ No null values in patient_df")
    
    # CRITICAL: Patient ID Uniqueness
    assert patient_df['Patient ID'].nunique() == len(patient_df), \
        f"ERROR: CRITICAL - Duplicate Patient IDs found. Expected {len(patient_df)} unique IDs, found {patient_df['Patient ID'].nunique()}"
    print(f"✓ Patient ID Uniqueness: {len(patient_df)} unique patients (NO DUPLICATES)")
    
    # Check required patient columns exist
    required_patient_cols = [
        'Patient ID', 'Age', 'RTT WAIT WEEKS', 'Intended Procedure Code',
        'Consultant Code', 'Planned H4 Minutes', 'Procedure Priority'
    ]
    missing_cols = [col for col in required_patient_cols if col not in patient_df.columns]
    assert len(missing_cols) == 0, \
        f"ERROR: Missing required patient columns: {missing_cols}"
    print(f"✓ All required patient columns present: {len(required_patient_cols)} columns")
    
    # Validate Procedure Priority values
    assert patient_df['Procedure Priority'].isin([1, 2, 3, 4]).all(), \
        f"ERROR: Procedure Priority must be in [1, 2, 3, 4]. Found: {patient_df['Procedure Priority'].unique()}"
    print(f"✓ Procedure Priority validation: All values in [1, 2, 3, 4]")
    
    # Validate Planned Duration positive
    assert (patient_df['Planned H4 Minutes'] > 0).all(), \
        f"ERROR: Planned H4 Minutes must be positive. Found {(patient_df['Planned H4 Minutes'] <= 0).sum()} invalid values"
    print(f"✓ Planned H4 Minutes: All values positive (min={patient_df['Planned H4 Minutes'].min():.1f}, max={patient_df['Planned H4 Minutes'].max():.1f})")
    
    # ==== SESSION DATA VALIDATION ====
    print("\n2. SESSION DATA VALIDATION")
    print("-" * 60)
    
    # Check for missing data
    assert not sessions_plan.isnull().any().any(), \
        f"ERROR: Found null values in sessions_plan:\n{sessions_plan.isnull().sum()}"
    print("✓ No null values in sessions_plan")
    
    # CRITICAL: Session ID Uniqueness
    assert sessions_plan['Session ID'].nunique() == len(sessions_plan), \
        f"ERROR: CRITICAL - Duplicate Session IDs found. Expected {len(sessions_plan)} unique IDs, found {sessions_plan['Session ID'].nunique()}"
    print(f"✓ Session ID Uniqueness: {len(sessions_plan)} unique sessions (NO DUPLICATES)")
    
    # Check required session columns exist
    required_session_cols = [
        'Session ID', 'Consultant Code', 'Total Slot Minutes', 'Session Date', 'Theatre Name'
    ]
    missing_cols = [col for col in required_session_cols if col not in sessions_plan.columns]
    assert len(missing_cols) == 0, \
        f"ERROR: Missing required session columns: {missing_cols}"
    print(f"✓ All required session columns present: {len(required_session_cols)} columns")
    
    # Validate session duration positive
    assert (sessions_plan['Total Slot Minutes'] > 0).all(), \
        f"ERROR: Total Slot Minutes must be positive. Found {(sessions_plan['Total Slot Minutes'] <= 0).sum()} invalid values"
    print(f"✓ Session Duration: All values positive (min={sessions_plan['Total Slot Minutes'].min():.1f}, max={sessions_plan['Total Slot Minutes'].max():.1f})")
    
    # ==== CROSS-VALIDATION ====
    print("\n3. CROSS-DATA VALIDATION")
    print("-" * 60)
    
    # Verify all patient consultants exist in session data
    patient_consultants = set(patient_df['Consultant Code'].unique())
    session_consultants = set(sessions_plan['Consultant Code'].unique())
    missing_consultants = patient_consultants - session_consultants
    assert len(missing_consultants) == 0, \
        f"ERROR: {len(missing_consultants)} consultants in patient data but NO sessions available: {missing_consultants}"
    print(f"✓ All {len(patient_consultants)} patient consultants have sessions available")
    
    # Optional: Validate patient_data if provided separately
    if patient_data is not None:
        print("\n4. ADDITIONAL PATIENT DATA VALIDATION (EPIC SOURCE)")
        print("-" * 60)
        
        # Check alignment with patient_df
        assert not patient_data.isnull().any().any(), \
            f"ERROR: Found null values in patient_data:\n{patient_data.isnull().sum()}"
        print("✓ No null values in patient_data")
        
        # Verify patient_data Patient ID matches patient_df
        patient_data_ids = set(patient_data['Patient ID'].unique())
        patient_df_ids = set(patient_df['Patient ID'].unique())
        extra_in_data = patient_data_ids - patient_df_ids
        extra_in_df = patient_df_ids - patient_data_ids
        
        assert len(extra_in_data) == 0 and len(extra_in_df) == 0, \
            f"ERROR: Patient ID mismatch between DataFrames. Extra in data file: {len(extra_in_data)}, Extra in main df: {len(extra_in_df)}"
        print(f"✓ Patient ID alignment: {len(patient_data_ids)} records match between sources")
        
        # Check patient_data doesn't have duplicate IDs
        assert patient_data['Patient ID'].nunique() == len(patient_data), \
            f"ERROR: Duplicate Patient IDs in patient_data file. Expected {len(patient_data)} unique, found {patient_data['Patient ID'].nunique()}"
        print(f"✓ Patient data ID Uniqueness: No duplicates")
    
    print("\n" + "="*60)
    print("✓ ALL VALIDATION CHECKS PASSED")
    print("="*60 + "\n")
    
    return True


# Execute validation before model initialization
validate_input_data(patient_df, sessions_plan, patient_data=None)  # Set to patient_data if separate file exists

# Then proceed with model initialization
sched_model = Scheduling_Model(patient_df, sessions_plan)
```

### Validation Checklist

Before calling the scheduling model, verify:

- [ ] **Patient ID Uniqueness**: Every patient appears exactly once (no duplicates)
- [ ] **Session ID Uniqueness**: Every session appears exactly once (no duplicates)  
- [ ] **No null values**: patient_df and sessions_plan have complete data
- [ ] **Required columns present**: All mandatory columns listed above exist
- [ ] **Data types correct**: Numeric fields are float/int, string fields are text
- [ ] **Procedure Priority valid**: All values in [1, 2, 3, 4]
- [ ] **Positive durations**: Planned H4 Minutes > 0, Total Slot Minutes > 0
- [ ] **Consultant availability**: All patient consultants have available sessions
- [ ] **No future dates**: All session dates are future dates from current date
- [ ] **Consultant format**: Consultant codes follow expected format (e.g., C001, C002)

---

## Performance Metrics Summary

### Key Performance Indicators (KPIs)

```python
def calculate_kpis(sched_model, patient_df):
    """Calculate and display key performance indicators"""
    
    total_patients = len(patient_df)
    scheduled = len(sched_model.schedule_new)
    not_scheduled = len(sched_model.patients_not_scheduled)
    
    kpis = {
        'Scheduling Rate': f"{scheduled/total_patients*100:.1f}%",
        'Patients scheduled': scheduled,
        'Patients not scheduled': not_scheduled,
        'Avg RTT weeks (scheduled)': f"{sched_model.schedule_new['RTT WAIT WEEKS'].mean():.2f}",
        'Avg RTT weeks (not scheduled)': f"{sched_model.patients_not_scheduled['RTT WAIT WEEKS'].mean():.2f}",
        'Avg session utilization': f"{sched_model.schedule_new.groupby('Session ID')['Planned H4 Minutes'].sum().mean() / sched_model.session_max_util:.1f}%",
        'Total objective value': f"{sched_model.objective_value:.2f}"
    }
    
    for metric, value in kpis.items():
        print(f"{metric:.<40} {value}")
    
    return kpis
```

---

## Quality Assurance

### Pre-Optimization Checklist

- [ ] All required input files present and accessible
- [ ] Session data loaded with no null Consultant Codes
- [ ] Patient list contains all required columns
- [ ] Procedure codes consistent between patient and session data
- [ ] Trained models directory accessible
- [ ] Label mapping pickle file exists
- [ ] Consultant procedure template populated

### Post-Optimization Checklist

- [ ] No patient assigned to multiple sessions
- [ ] No session exceeds capacity
- [ ] All assigned procedures in consultant template (if restricted)
- [ ] All scheduled start times valid and non-overlapping
- [ ] Theatre prep time maintained (≥7.5 min gaps)
- [ ] Priority distribution reasonable
- [ ] RTT wait time properly calculated

---

## References & Further Reading

- SCIP Solver Documentation: https://scipopt.org/
- Pandas Data Processing: https://pandas.pydata.org/docs/
- Surgical Time Prediction Models: See `predictive_model_related_functions.py`
- Multi-objective Optimization: See `optimisation_related_functions.py`

---

**Document Last Updated:** February 2026  
**Maintainer:** Analytics Team, Business Informatics  
**System:** East Suffolk & North Essex Foundation Trust (ESNEFT)
