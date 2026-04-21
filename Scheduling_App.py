import pandas as pd  # version 2.0.3 # 2.2.0
import datetime
import numpy as np  ##  version 1.24.3  #1.26.4
import pickle
import os
import tkinter as tk   # version 8.6
from tkinter import filedialog, ttk
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from procedure_time_and_variabiltiy_related_functions import update_patient_data, load_timeslot_file, plot_theatre_occupancy_data, get_prediction_required_parameters

from data_processing_nd_encoding_related_functions import age_to_group

from Scheduling_Model import Scheduling_Model

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Scroll to the end of the text widget



class SurgicalSchedulerApp(tk.Frame):
    def __init__(self, master):
        self.master = master
        self.master.title("Surgical Scheduler")
        self.patient_file_path = None
        self.timeslot_file_path = None
        self.schedule = None
        self.schedule_new = None
        self.filtered_schedule = None
        self.patients_df = None
        #self.TASKS_DURATION = None
        #self.TASKS_DURATION = None
        self.procedure_time_prediction_parameters = None
        #self.fixed_data_for_patient = 
        self.theatre_prpn_min = 7.5
        self.session_max_util = 0.9
        #self= None
        self.ML_dir = r"\\esneft.nhs.uk\share\Finance\Capacity & Analytics\Business Informatics\Analytics Team\Surgical Elective Patient Scheduling Model\Theatre_Procedure_Time_Predictive_Modelling\Latest_data_and_model"

        self.create_widgets()

        # Snapshot of attributes that are allowed to exist after init
        self._init_attrs = set(self.__dict__.keys())

    def create_widgets(self):

        # Create entry widgets for inputs
        self.label_ML_dir = tk.Label(self.master, text="ML dir")
        #self.label_prpn_time.place(relx=0.55, rely=0.03, relwidth=0.18, relheight=0.06)
        self.label_ML_dir.place(relx=0.05, rely=0.015, relwidth=0.1, relheight=0.04)
        
        self.entry_ML_dir = ttk.Entry(self.master)
        self.entry_ML_dir.place(relx=0.125, rely=0.015, relwidth=0.44, relheight=0.04)
        self.entry_ML_dir.bind("<Return>", self.store_ML_dir)  # Bind Enter key to store_valu

        self.IO_dir_button = tk.Button(self.master, text="Select IO direc", command=self.store_IO_dir, bg = 'lightgreen')
        #self.patient_file_button.pack(pady=7, padx=10)
        #self.patient_file_button.place(relx=0.05, rely=0.025, relwidth=0.20, relheight=0.05
        self.IO_dir_button.place(relx=0.05, rely=0.08, relwidth=0.15, relheight=0.05)
        
        self.patient_file_button = tk.Button(self.master, text="Select Patient List", command=self.load_patient_file, state = tk.DISABLED)
        #self.patient_file_button.pack(pady=7, padx=10)
        #self.patient_file_button.place(relx=0.05, rely=0.025, relwidth=0.20, relheight=0.05
        self.patient_file_button.place(relx=0.215, rely=0.08, relwidth=0.165, relheight=0.05)
        
        self.timeslot_file_button = tk.Button(self.master, text="Select Session File", command=self.load_timeslot_file, state = tk.DISABLED)
        #self.timeslot_file_button.pack(pady=7, padx=10)
        #self.timeslot_file_button.place(relx=0.28, rely=0.025, relwidth=0.25, relheight=0.05)
        self.timeslot_file_button.place(relx=0.40, rely=0.08, relwidth=0.155, relheight=0.05)
        
        #self.schedule_button = tk.Button(self.master, text="Compute Procedure Times", command=self.compute_all_possible_tasks_duration, state=tk.DISABLED, bg="lightcoral")
        self.schedule_button = tk.Button(self.master, text="Create Schedule", command=self.selection_based_commands, state=tk.DISABLED, bg="lightcoral")
        #self.schedule_button.pack(pady=10)
        self.schedule_button.place(relx=0.76, rely=0.14, relwidth=0.19, relheight=0.05)

        #self.possibility_button = tk.Button(self.master, text="Other Possibilities", command=self.obtain_possibilities_for_plan, state=tk.DISABLED, bg="lightcoral")
        #self.schedule_button.pack(pady=10)
        #self.possibility_button.pack_forget()

        # Create a Combobox for selecting Scheduling Algorithm
        self.algorithm_combobox_var = tk.StringVar(self.master, "Scheduling Options")
        #self.algorithm_combobox = ttk.Combobox(self.master, textvariable=self.algorithm_combobox_var, state="disabled")
        #self.update_combobox(self.algorithm_combobox, ['Sessions Continuous Fillling', 'Optimum Utilisation', 'Multi-Objective'], append = True)
        # Define the combobox with options directly
        self.algorithm_combobox = ttk.Combobox(
            self.master, 
            textvariable=self.algorithm_combobox_var, 
            state="readonly", 
            values=['Algorithm-1', 'Algorithm-2']
        )
        #self.algorithm_combobox.place(relx=0.20, rely=0.08, relwidth=0.25, relheight=0.05)
        self.algorithm_combobox.place(relx=0.58, rely=0.025, relwidth=0.23, relheight=0.05)
        self.algorithm_combobox.configure(state=tk.DISABLED)

        # Bind the combobox selection event to the update_button_text method
        self.algorithm_combobox.bind("<<ComboboxSelected>>", self.update_command_button_text)
        
        self.data_label = tk.Label(self.master, text="Task Visualisation", bg= 'white')
        #self.data_label.pack(pady=10)
        self.data_label.place(relx=0.05, rely=0.15, relwidth=0.52, relheight=0.05)

        # Create labels for inputs
        self.label_prpn_time = tk.Label(self.master, text="Input Prpn Time \n (in min)")
        #self.label_prpn_time.place(relx=0.55, rely=0.03, relwidth=0.18, relheight=0.06)
        self.label_prpn_time.place(relx=0.82, rely=0.025, relwidth=0.13, relheight=0.06)

        # Create entry widgets for inputs
        self.entry_prpn_time = ttk.Entry(self.master, state=tk.DISABLED)
        self.entry_prpn_time.place(relx=0.82, rely=0.085, relwidth=0.13, relheight=0.04)
        self.entry_prpn_time.bind("<Return>", self.store_theatre_prpn_time)  # Bind Enter key to store_valu

        # Create labels for inputs
        self.label_max_sess_util = tk.Label(self.master, text="Input Max Sess Utilis \n [0-1]")
        self.label_max_sess_util.place(relx=0.58, rely=0.10, relwidth=0.16, relheight=0.06)
        
        # Create entry widgets for inputs
        self.entry_max_sess_util = ttk.Entry(self.master, state=tk.DISABLED)
        #self.entry_max_sess_util.place(relx=0.75, rely=0.09, relwidth=0.18, relheight=0.04)
        self.entry_max_sess_util.place(relx=0.58, rely=0.16, relwidth=0.16, relheight=0.04)
        self.entry_max_sess_util.bind("<Return>", self.store_max_sess_util)  # Bind Enter key to store_value
        
        self.text_display = tk.Text(self.master)
        self.text_display.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.6)
        #self.text_display.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        # Create an empty OptionMenu initially
        self.results_var = tk.StringVar()
        self.results_var.set("Select Result to Export")
        self.results_menu = tk.OptionMenu(self.master, self.results_var, '')
        #self.results_menu.pack(pady=10)
        self.results_menu.place(relx=0.05, rely=0.825, relwidth=0.35, relheight=0.05)
        
        # Create a button to trigger displaying results
        self.data_display_button = tk.Button(self.master, text="Display Result", command=self.display_result)
        #self.save_button.pack(pady=10)
        self.data_display_button.place(relx=0.50, rely=0.825, relwidth=0.20, relheight=0.05)

        # Create a button to trigger saving results
        self.save_button = tk.Button(self.master, text="Save Result", command=self.save_result)
        #self.save_button.pack(pady=10)
        self.save_button.place(relx=0.75, rely=0.825, relwidth=0.20, relheight=0.05)

        # Create a Combobox for selecting columns
        self.columns_combobox_var = tk.StringVar(self.master, "Select Filtering Column")
        self.columns_combobox = ttk.Combobox(self.master, textvariable=self.columns_combobox_var, state="disabled")
        self.columns_combobox.pack_forget()
        #columns_combobox.place(relx=0.10, rely=0.825, relwidth=0.35, relheight=0.05)

        self.col_data_var = tk.StringVar(self.master, "Select Filtering data")
        #self.col_data_menu = tk.OptionMenu(self.master, self.col_data_var, '')
        self.col_data_menu = ttk.Combobox(self.master, textvariable = self.col_data_var, state="disabled")
        self.col_data_menu.pack_forget()

        #self.close_button = tk.Button(self.master, text="Close App", command=self.close_app, bg = 'red')
        #self.close_button.pack(pady=10)
        #self.close_button.place(relx=0.80, rely=0.9, relwidth=0.15, relheight=0.07)
        # Replace this:
        # self.close_button = tk.Button(self.master, text="Close App", command=self.close_app, bg='red')

        # With this:
        self.refresh_button = tk.Button(
            self.master,
            text="ReStart",
            command=self.refresh_app,
            bg="#FFBF00",     # amber
            fg="black",
            activebackground="#FFB000"
        )
        self.refresh_button.place(relx=0.80, rely=0.9, relwidth=0.15, relheight=0.07)

    def refresh_app(self):
        # --- A) Remove runtime-added self.* attributes (keep only init snapshot) ---
        for name in list(self.__dict__.keys()):
            if name.startswith("_"):
                continue
            if hasattr(self, "_init_attrs") and name not in self._init_attrs:
                delattr(self, name)

        # --- B) Reset init-time variables ---
        self.patient_file_path = None
        self.timeslot_file_path = None
        self.schedule = None
        self.schedule_new = None
        self.filtered_schedule = None
        self.patients_df = None
        self.procedure_time_prediction_parameters = None
        self.theatre_prpn_min = 7.5
        self.session_max_util = 0.9
        # keep your default ML_dir as-is (or set it explicitly if you prefer)

        # --- C) Reset widget values ---
        # ML dir entry: ENABLE for typing after refresh
        self.entry_ML_dir.configure(state=tk.NORMAL)  # ttk.Entry supports normal/disabled [web:32]
        self.entry_ML_dir.delete(0, tk.END)

        self.entry_prpn_time.configure(state=tk.NORMAL)
        self.entry_prpn_time.delete(0, tk.END)
        self.entry_prpn_time.configure(state=tk.DISABLED)

        self.entry_max_sess_util.configure(state=tk.NORMAL)
        self.entry_max_sess_util.delete(0, tk.END)
        self.entry_max_sess_util.configure(state=tk.DISABLED)

        self.text_display.delete("1.0", tk.END)

        self.algorithm_combobox_var.set("Scheduling Options")
        self.algorithm_combobox.configure(state=tk.DISABLED)

        self.results_var.set("Select Result to Export")
        menu = self.results_menu["menu"]
        menu.delete(0, "end")

        # --- D) Restore ALL buttons to their initial state + colors/text ---
        # IO dir button: enabled + lightgreen
        self.IO_dir_button.configure(state=tk.NORMAL, bg="lightgreen")  # Button bg works on tk.Button [web:37]

        # initial: patient/timeslot disabled
        self.patient_file_button.configure(state=tk.DISABLED, bg="SystemButtonFace")
        self.timeslot_file_button.configure(state=tk.DISABLED, bg="SystemButtonFace")


        # initial: schedule disabled + lightcoral + same label
        self.schedule_button.configure(state=tk.DISABLED, bg="lightcoral", text="Create Schedule")

        # these two appear initially enabled in your create_widgets
        self.data_display_button.configure(state=tk.NORMAL, text="Display Result")
        self.save_button.configure(state=tk.NORMAL, text="Save Result")


    
    def update_text(self, message):
        
        self.text_display.insert(tk.END, message)
        #self.text_display.see(tk.END)  # Scroll to the end of the text widget

    
    def store_IO_dir(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.IO_dir = folder_selected
            self.data_label.config(text=f"Following path set as I\O dir: \n {self.IO_dir}")
            
            self.patient_file_button.config(state=tk.NORMAL, bg = 'lightgreen')
            self.IO_dir_button.config(state=tk.DISABLED, bg="SystemButtonFace")
            os.chdir(self.IO_dir)
            #self.patient_file_button.config(state=tk.DISABLED, text="Select Patient List", command=self.load_patient_file, bg="SystemButtonFace")
        else:
            self.data_label.config(text=f"The selected directory doesn't exist")
        
        if self.procedure_time_prediction_parameters is None:
            self.procedure_time_prediction_parameters = get_prediction_required_parameters(self.IO_dir, self.ML_dir)
        

    def store_ML_dir(self, event):
        if os.path.exists(self.entry_ML_dir.get()):
            self.ML_dir = self.entry_ML_dir.get()
            self.data_label.config(text=f"Following path set as ML dir: \n {self.ML_dir}")
            self.entry_ML_dir.config(state=tk.DISABLED)
            #self.patient_file_button.config(state=tk.DISABLED, text="Select Patient List", command=self.load_patient_file, bg="SystemButtonFace")
        else:
            self.data_label.config(text=f"The selected directory doesn't exist")
        #state=tk.DISABLED, text="Select Patient List", command=self.load_patient_file, 
    
    
    def load_patient_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.patient_list_path = file_path
            #print(f"Patient list loaded: {self.patient_file_path}")

            self.data_label.config(text=f"Selected Patients' data is loaded")
            self.text_display.insert(tk.END, f"Selected Patients' is loaded\n")

            self.patients_df = pd.read_excel(self.patient_list_path)

            proce_related_col = [col for col in self.patients_df.columns if 'procedure' in col.lower()][0]

            #checking if the intended procedure data exist in past procedure data
            #if 'Primary Procedure Code' in self.procedure_time_prediction_parameters[2].columns:
            set_past_theatre_procedures = set(self.procedure_time_prediction_parameters[2]['Actual Procedure 1 Code 1']).union(self.procedure_time_prediction_parameters[3]['Actual Procedure 1 Code 1'])
            
            proced_found_mask = self.patients_df[proce_related_col].isin(set_past_theatre_procedures)

            if not proced_found_mask.all() and 'Primary Procedure Code' in self.procedure_time_prediction_parameters[2].columns:
                set_past_spell_procedures = set(self.procedure_time_prediction_parameters[2]['Primary Procedure Code']).union(self.procedure_time_prediction_parameters[3]['Primary Procedure Code'])
                proced_found_mask |= self.patients_df[proce_related_col].isin(set_past_spell_procedures)

            if not proced_found_mask.all():
                
                time_col = next((col for col in self.patients_df.columns if 'time' in col.lower()), None)
                
                time_pre_allocated_mask = self.patients_df[time_col].notna() if time_col else pd.Series(False, index=self.patients_df.index)

                final_mask = proced_found_mask | time_pre_allocated_mask  

                if not final_mask.all():
                    
                    df_unfound = self.patients_df[~final_mask].copy() 
                    if time_col:
                        self.data_label.config(text=f"Add procedure time for following patient-procedure under column {time_col}, \n will be removed till then:  {[(pat, pro) for pat, pro in zip(df_unfound['Patient ID'], df_unfound[proce_related_col])]}")
                    
                    else:
                        self.data_label.config(text=f"Add procedure time for following patient-procedure under column 'Allocated Time', will be removed till then \n {[(pat, pro) for pat, pro in zip(df_unfound['Patient ID'], df_unfound[proce_related_col])]}")

                    self.patients_df = self.patients_df[final_mask].copy().reset_index(drop = True)
            #patients_df = pd.read_excel(patient_file_path)
            #self.patients_df = self.patients_df[self.patients_df['Session Planned Start Date/Time'] < pd.to_datetime('2023-10-15')]
            
            
            self.create_data_frame_viewer(self.patients_df)

            #checking if all prediction required data present
            proce_related_col = [col for col in self.patients_df.columns if 'procedure' in col.lower()][0]
                
            all_features_with_identifying_name = [proce_related_col if x == 'Actual Procedure 1 Code 1' else x for x in self.procedure_time_prediction_parameters[4] ]

            fixed_data_for_patient = [col for col in all_features_with_identifying_name if col not in self.procedure_time_prediction_parameters[5]]
            #prediction_cols = self.procedure_time_prediction_paarameters[4]
            age_related_col = [col for col in self.patients_df.columns if 'age' in col.lower()][0]
            if age_related_col in self.patients_df.columns and age_related_col !='Age group at admit':
                self.patients_df['Age group at admit'] = self.patients_df[age_related_col].apply(lambda x: age_to_group(x, [5, 15, 30, 50, 60, 70, 80]))
    
            
            
            
            if all(col in self.patients_df.columns for col in fixed_data_for_patient):
                self.patient_file_path = file_path
                self.patients_df = self.patients_df.copy().set_index('Patient ID', drop=False)
                self.patient_file_button.config(state=tk.DISABLED, text="Select Patient List", command=self.load_patient_file, bg="SystemButtonFace")
                self.timeslot_file_button.configure(state=tk.NORMAL, bg="lightgreen")
            else:
                self.patient_file_button.config(text="Select Patient data File", command=self.update_patient_data, bg = 'lightgreen')

                self.data_label.config(text=f"Patient data required for \n {[col for col in fixed_data_for_patient if col not in self.patients_df.columns ]}")
                self.text_display.insert(tk.END, f"Patient data required for \n {[col for col in fixed_data_for_patient if col not in self.patients_df.columns]} \n")
            #self.patient_file_button.pack(pady=7, padx=10)
            #self.patient_file_button.place(relx=0.05, rely=0.05, relwidth=0.25, relheight=0.05)

    def update_patient_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.patient_file_path = file_path

            if self.patients_df is None:
                self.data_label.config(text=f"Load Patients' selection First")
            else:
                
                #patients_data_df = pd.read_excel(self.patient_file_path)
                #print(f"Patient list loaded: {self.patient_file_path}")
                proce_related_col = [col for col in self.patients_df.columns if 'procedure' in col.lower()][0]
                
                all_features_with_identifying_name = [proce_related_col if x == 'Actual Procedure 1 Code 1' else x for x in self.procedure_time_prediction_parameters[4] ]
    
                fixed_data_for_patient = [col for col in all_features_with_identifying_name if col not in self.procedure_time_prediction_parameters[5]]
                #pred

                self.patients_df = update_patient_data(self.patients_df, pd.read_excel(self.patient_file_path), fixed_data_for_patient)
                self.data_label.config(text=f"Selected Patients' data is updated")
                self.text_display.insert(tk.END, f"Selected Patients' is updated\n")
                self.create_data_frame_viewer(self.patients_df)
                
            self.patient_file_button.config(state=tk.DISABLED, text="Select Patient List", command=self.load_patient_file, bg="SystemButtonFace")

            self.timeslot_file_button.configure(state=tk.NORMAL, bg="lightgreen")

    
    def load_timeslot_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if file_path:
            self.timeslot_file_path = file_path
            #print(f"Timeslot file loaded: {self.timeslot_file_path}")
            self.data_label.config(text=f"Session data is loaded")
            self.text_display.insert(tk.END, f"Session data is loaded\n")
            self.sessions_df = load_timeslot_file(self.timeslot_file_path)
            #self.sessions_df 
            #self.check_enable_schedule_button(self.patients_df)
            self.create_data_frame_viewer(self.sessions_df)

            #self.algorithm_combobox.configure(state=tk.NORMAL)
            self.algorithm_combobox.configure(state='readonly')
            self.timeslot_file_button.configure(state = tk.DISABLED, bg="SystemButtonFace")

            if self.patients_df is not None:
                schedul_model_temp = Scheduling_Model(self.patients_df, self.sessions_df)

                # Copy attributes from ClassA but exclude Tkinter attributes
                for key, value in schedul_model_temp.__dict__.items():
                    if not hasattr(self, key):  # Prevent overwriting existing attributes
                        setattr(self, key, value)
                #Copy methods dynamically
                for name in dir(schedul_model_temp):
                    if callable(getattr(schedul_model_temp, name)) and not name.startswith("__"):  # Skip dunder methods
                        method = getattr(schedul_model_temp, name)
                        setattr(self, name, method.__get__(self, self.__class__))  # Bind method to self            
    
    
    def check_enable_schedule_button(self, data1):
        # Check if both data1 and data2 are loaded, then enable the schedule_button
        if all([data is not None for data in data1]):
            self.schedule_button.configure(state=tk.NORMAL, bg="lightgreen")
            
    
    def close_app(self):
        # Close the application by destroying the main window
        self.master.destroy()

    def change_bg_color(self, widget, color):
        widget.configure(bg=color)

    def create_data_frame_viewer(self, dataframe):
        self.clear_data_frame_viewer()
        # Create a Treeview widget to display the selected DataFrame
        columns = ['index'] + list(dataframe.columns) 
        self.tree = ttk.Treeview(self.master, columns=columns, show="headings")

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", width = 100)

        for index, row in dataframe.iterrows():
            self.tree.insert("", "end", values= tuple([index] + list(row)))

        # Add a vertical scrollbar to the Treeview
        self.vsb = ttk.Scrollbar(self.master, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand= self.vsb.set)

        # Add a horizontal scrollbar to the Treeview
        self.hsb = ttk.Scrollbar(self.master, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand= self.hsb.set)

        # Add the Treeview and scrollbars to the grid
        self.tree.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.57)
        #vsb.grid(row=2, column=2, sticky="ns")
        self.hsb.place(relx=0.05, rely=0.77, relwidth=0.9, relheight=0.025)
        self.vsb.place(relx=0.05, rely=0.25, relwidth=0.02, relheight=0.55)

    def clear_data_frame_viewer(self):
        # Destroy the tree and scrollbars
        if hasattr(self, 'tree'):
            self.vsb.destroy()
            self.hsb.destroy()
            self.tree.destroy()

    
    def is_numeric(self, value):
        """
        Check if the value can be converted to a float.
        """
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def store_theatre_prpn_time(self, event):
        if self.is_numeric(self.entry_prpn_time.get()):
            if float(self.entry_prpn_time.get()) > 0 and float(self.entry_prpn_time.get()) < 30:
                self.theatre_prpn_min = float(self.entry_prpn_time.get())
                #print(self.theatre_prpn_min)
                self.data_label.config(text=f"Theatre prpn min set as: {self.theatre_prpn_min}")
            else:
                self.data_label.config(text=f"Theatre prpn min is too large or negative: {self.entry_prpn_time.get()}")
            #self.possibility_button.configure(bg="yellow")
            #self.master.update_idletasks()
        else:
            self.data_label.config(text=f"Enter numeric for Theatre prpn min")
    
    def store_max_sess_util(self, event):
    #print(self.entry_max_sess_util.get())
        if self.is_numeric(self.entry_max_sess_util.get()) and float(self.entry_max_sess_util.get()) >0.5 and float(self.entry_max_sess_util.get())<=1:
            self.session_max_util = float(self.entry_max_sess_util.get())
            self.data_label.config(text=f"Session max utilisation set as: {self.session_max_util}")
            #print(self.session_max_util)
        else: 
            self.data_label.config(text=f"Session max utilisation not in appro range: {self.entry_max_sess_util.get()}")    
            
    def on_canvas_configure(self, event):
        # Update scroll region to allow scrolling
        self.plot_canvas.configure(scrollregion=self.plot_canvas.bbox("all"))
        #self.plot_canvas.itemconfig(self.canvas_widget, width=event.width, height=event.height)
        if hasattr(self, 'canvas_widget'):
            self.canvas_widget.config(width=event.width, height=event.height)    
            
    def create_plot_viewer(self, plt_dataframe = None):

        #self.clear_data_frame_viewer()
        
        if not hasattr(self, 'plot_frame'):  # Check if plot frame exists
            # Create a frame to hold the plot and scrollbar
            self.plot_frame = tk.Frame(self.master)
            self.plot_frame.place(relx=0.05, rely=0.2, relwidth=0.9, relheight=0.6)
    
            # Create a canvas for the plot
            self.plot_canvas = tk.Canvas(self.plot_frame)
            self.plot_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        #if not hasattr(self, 'y_scrollbar'):  # Check if y-scrollbar exists
            # Configure scrollbar for y-axis
        self.y_scrollbar = ttk.Scrollbar(self.plot_frame, orient="vertical", command=self.plot_canvas.yview)
        #self.y_scrollbar.pack(side=tk.RIGHT, fill="y")
        self.y_scrollbar.place(relx=0.95, rely=0.05, relwidth=0.05, relheight=0.8)
    
        #if not hasattr(self, 'x_scrollbar'):  # Check if x-scrollbar exists
            # Configure scrollbar for x-axis
        self.x_scrollbar = ttk.Scrollbar(self.plot_frame, orient="horizontal", command=self.plot_canvas.xview)
        #self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X, expand=True)
        self.x_scrollbar.place(relx=0.05, rely=1, relwidth=0.9, relheight=0.05)
        
        if 'Schedule Generated' in self.results_var.get():
            proce_related_col = [col for col in self.schedule_new.columns if 'procedure' in col.lower() ][0]

            plot_data_cols = ['Scheduled Start Date/Time','Planned H4 Minutes','Patient ID', proce_related_col, 'Session ID']
            if 'Scheduled Start Date/Time' not in self.schedule_new.columns:
                plot_data_cols[0] = 'Session Planned Start Date/Time'
            
            if plt_dataframe is None:
                plt_dataframe = self.schedule_new[plot_data_cols].copy()
            else:
                plt_dataframe = plt_dataframe[plot_data_cols].copy()
            
            plt_dataframe[plot_data_cols[0]] = pd.to_datetime(plt_dataframe[plot_data_cols[0]])
            self.fig_size = self.compute_figsize(len(plt_dataframe['Session ID'].unique()))
            
        elif 'Schedule Possibilities' in self.results_var.get():
            if plt_dataframe is None:
                plt_dataframe = self.schedule_possibilities[self.schedule_possibilities['Patient ID'].isin(self.filtered_schedule['Patient ID'])].copy()
            self.fig_size = self.compute_figsize(len(plt_dataframe)*1.2)
        elif 'Sessions Possibilities' in self.results_var.get():
            if plt_dataframe is None:
                plt_dataframe = self.sessions_possibilities[self.sessions_possibilities['Session ID'].isin(self.filtered_schedule['Session ID'])].copy()
            self.fig_size = self.compute_figsize(len(plt_dataframe)*1.2)
        
        #print(self.fig_size)
        # Clear previous plot
        if hasattr(self, 'ax'):
            self.ax.clear()
        
        # Clear previous canvas
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()
            #self.ax.destroy()
        # Create the figure and axes
        fig = Figure(figsize=self.fig_size, dpi=100)
        self.ax = fig.add_subplot(111)
        #fig.tight_layout()
        fig.subplots_adjust(left=0.125, right=0.85)  # Adjust horizontal margins only

        if hasattr(self, 'canvas_widget'):
            self.canvas_widget.pack_forget()  # Remove canvas widget from parent
            self.canvas_widget.destroy()

        if 'Schedule Generated' in self.results_var.get():
            if os.path.exists('label_mapping.pickle'):
                with open('label_mapping.pickle', 'rb') as file:
                    label_mapping = pickle.load(file)
            else:
                label_mapping = {}

            plot_theatre_occupancy_data(plt_dataframe, self.fig_size, label_mapping, ax = self.ax, color_mapping_col = proce_related_col, grouping_col='Session ID')
            # Create a separate canvas for the matplotlib plot
        elif 'Schedule Possibilities' in self.results_var.get():
            schedule_generated  = self.filtered_schedule.reset_index(drop=True)
            mapping = dict(zip(schedule_generated['Patient ID'], schedule_generated.index))
            # Sort df2 based on the order of 'A' in df1
            plt_dataframe = plt_dataframe.sort_values(by='Patient ID', key=lambda x: x.map(mapping)).reset_index(drop=True)
            continuous_data_plot(plt_dataframe, 'Patient ID', 'Planned H4 Minutes', discrete_columns= [schedule_generated['Planned H4 Minutes'].rename("Determin Models' Suggestion")], ax1 = self.ax, extra_label = schedule_generated['Procedure Code']) 
        
        elif 'Sessions Possibilities' in self.results_var.get():
            sessions_timeslot  = self.sessions_updated.reset_index(drop=True)
            mapping = dict(zip(sessions_timeslot['Session ID'], sessions_timeslot.index))
            # Sort df2 based on the order of 'A' in df1
            plt_dataframe = plt_dataframe.sort_values(by='Session ID', key=lambda x: x.map(mapping)).reset_index(drop=True)
            #continuous_data_plot(plt_dataframe, 'Session ID', 'H4 Minutes Booked', discrete_columns= [sessions_timeslot['H4 Minutes Booked'].rename("Determin Models' Suggestion")], ax1 = self.ax) 
            continuous_data_plot(plt_dataframe, 'Session ID', 'H4 Minutes Booked', discrete_columns = [], ax1 = self.ax) 
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.draw()

        
        # Configure canvas scrolling behavior
        self.plot_canvas.configure(yscrollcommand=self.y_scrollbar.set, xscrollcommand=self.x_scrollbar.set)
        self.plot_canvas.create_window((0, 0), window=self.canvas_widget, anchor="nw")
    
    def selection_based_commands(self):
        
        selection = self.algorithm_combobox_var.get()
        
        self.schedule_button.configure(state=tk.DISABLED)

        if self.TASKS_DURATION is None:
                
            if self.procedure_time_prediction_parameters is not None:
                self.compute_all_possible_tasks_duration()
        
        #self.schedule = None
        self.schedule_new = None
        self.filtered_schedule = None
        
        if 'first come basis' in str(selection).lower():
            #print(selection)
            self.create_schedule()
            
            #self.entry_max_sess_util.configure(state = tk.DISABLED)
            #self.entry_prpn_time.configure(state = tk.DISABLED)
            
        elif 'algorithm' in str(selection).lower():
            #if self.TASKS_DURATION == None:
                #self.compute_all_possible_tasks_duration()
            #self = DataObj(self.patients_df, self.sessions_df)
            #self.patients_df = self.patients_df
            #self.TASKS_DURATION = self.TASKS_DURATION
            
            #self.generate_tasks_durations(self.patients_df, self.sessions_df, self.procedure_time_prediction_parameters)
            #if os.file.exists(os.path.join(self.IO_dir, 'Optimisation Objectives')):
            objectives = ('objective1', 'objective2', 'objective3','objective4')
            
            if '1' in str(selection):
                solver_name = 'MIP Solver'
                obj_weightage =  {'objective1':0.45, 'objective2':0.15, 'objective3':0.2*len(self.CASES)/len(self.SESSIONS)/5,  'objective4':0.45*len(self.CASES)/len(self.SESSIONS)/5}
            
            else:
                solver_name = 'Simulated Annealing'
                obj_weightage = {'objective1':0.75, 'objective2':0.25}
            self.create_optimum_schedule(solver_name, objectives, obj_weightage)
            
        #self.schedule_button.configure(text = 'Create Schedule', state=tk.NORMAL, bg="lightgreen")
        #self.algorithm_combobox_var.set("Scheduling Algorithm Options")
        #, 'Optimum Utilisation', 'Multi-Objective'' elif
        #self.possibility_button.place(relx=0.60, rely=0.9, relwidth=0.15, relheight=0.05)
            
        #self.possibility_button.configure(state=tk.NORMAL, bg="lightgreen")

    
    def create_schedule(self):
        self.data_label.config(text="Schedule generation in progress")
        self.schedule_button.configure( bg="orange", state = tk.DISABLED)
        self.update_text("Scheduling Process started...\n")
        self.master.update_idletasks()
        
        try:
            if not self.patient_file_path or not self.timeslot_file_path:
                self.data_label.config(text="Please select both patient list and timeslot files.")
                
                return

            self.create_schedule_with_continuous_filling(self.procedure_time_prediction_parameters)

            self.data_label.config(text="Schedule generation Process is Complete")
            self.text_display.insert(tk.END, f"Schedule generation Process is Complete \n")
            
            new_dataset_names = ['Schedule Generated', 'Sessions Updated', 'Patients not Scheduled' , 'Schedule Generated Plot']
            
            self.update_dropdown_menu(new_dataset_names, self.results_var, self.results_menu, append=False)

            self.create_data_frame_viewer(self.schedule_new)
            #self.possibility_button.place(relx=0.7, rely=0.085, relwidth=0.15, relheight=0.05)

            
        except Exception as e:
            print(f"Error: {e}")

    def create_optimum_schedule(self, solver_name, optimisation_objectives, obj_weightage):

        if self.TASKS_DURATION_DEVIATION_RATIO_CHANCES_MEAN_SD == {} and 'objective2' in optimisation_objectives:
            self.data_label.config(text="Compiling chances related data for 2nd Objective")
            
            self.generate_chances_for_tasks_durations(self.patients_df, self.sessions_df, 30, None, self.procedure_time_prediction_parameters)
            self.obtain_mean_sd_for_tasks_durations_chances(self.patients_df, self.sessions_df, 30, None, self.procedure_time_prediction_parameters)
            self.update_sessions_df()
            self.update_text("Chances related data for 2nd Objective complied\n")

        if self.CASES_RTT_WAIT_WEEKS == {} and 'objective3' in optimisation_objectives:
            self.add_cases_RTT_waiting_time(self.patients_df)
            self.SUM_CASES_RTT_WAIT_WEEKS = sum(self.CASES_RTT_WAIT_WEEKS.values())
            
        if self.CASES_SURGERY_PRIORITY == {} and 'objective4' in optimisation_objectives:
            self.add_cases_surgery_Priority(self.patients_df)
            #self.SUM_CASES_RTT_WAIT_WEEKS = sum(self.CASES_RTT_WAIT_WEEKS.values())
        self.add_consultant_procedures_template(pd.read_excel(os.path.join(self.IO_dir,'Consultant_procedure_template.xlsx')))
        self.data_label.config(text="Schedule generation in progress")
        self.schedule_button.configure( bg="yellow")
        self.update_text("Scheduling Process started...\n")
        self.master.update_idletasks()
        #from optimisation_related_functions import create_optimum_schedule

        sum_value = sum(_ for key, _ in obj_weightage.items() if key in optimisation_objectives)

        obj_weightage_normalised = {key: value/sum_value for key, value in obj_weightage.items()}	

        try:
            if not self.patient_file_path or not self.timeslot_file_path:
                print("Please select first both the patient list and timeslot files.")
                return

            #self.text_display.insert(tk.END, f"Schedule generation in progress\n")
            #self.update_text("Scheduling Process started...\n")
            
            #self.schedule_new = generate_schedule_with_optimisation(self, solver_name, optimisation_objectives)
            
            #self.session_max_util = self.theatre_max_util
            #self.theatre_prep_min = self.theatre_prpn_min


            if 'mip' in solver_name.lower():
                hyper_param = {'time limit': len(self.patients_df)*len(self.sessions_df)/20} 
                #obj_weightage =  {'objective1':0.7, 'objective2':0.2, 'objective3':0.1}
                sol_weightage = None
            else:
                hyper_param = {'INITIAL_TEMPERATURE': 100000 , 'FINAL_TEMPERATURE' : 0.00001, 
                  'ALPHA' : 0.999, 'MAX_ITERATIONS' : 100000}
                #obj_weightage = {'objective1':0.75, 'objective2':0.25}
                
                sol_weightage = (0.4, 0.3, 0.4)   

            self.create_schedule_with_optimisation(optimisation_algorithm =solver_name,
                                                   objectives = optimisation_objectives,
                                                   weightage= obj_weightage_normalised,
                                                   best_solution_weightage= sol_weightage,
                                                   hyper_param= hyper_param,
                                                  consult_patient_restriction = 'Consultant Code' in self.patients_df.columns,
                                                  consult_procedure_restriction= True)

            #self.patients_not_scheduled = self.patients_df[~self.patients_df['Patient ID'].isin(self.schedule_new['Patient ID'])]
            self.data_label.config(text="Schedule generation Process is Complete")
            self.text_display.insert(tk.END, f"Schedule generation Process is Complete \n")
            
            new_dataset_names = ['Schedule Generated', 'Sessions Updated', 'Patients not Scheduled', 'Schedule Generated Plot']
                
            self.update_dropdown_menu(new_dataset_names, self.results_var, self.results_menu, append=False)

            self.create_data_frame_viewer(self.schedule_new)
            
        except Exception as e:
            print(f"Error during optimal scheduling: {e}")

    
    def compute_all_possible_tasks_duration(self):
        
        self.data_label.config(text="Computing Procedure-time for all \n Patient-Session Combinations")
        self.schedule_button.configure( bg="orange", state = tk.DISABLED)
        self.update_text("Computing Procedure-time...\n")
        self.master.update_idletasks()
        
        #self.possible_task_durations_dict = compute_TASKS_average_surgical_time(self.patients_df, self.sessions_df, *self.procedure_time_prediction_parameters)
        
        self.generate_tasks_durations(self.patients_df, self.sessions_df, None, self.procedure_time_prediction_parameters)

        #'''
        #self.TASKS_DURATION = pd.read_excel('Procedure-times_combinations_june.xlsx', index_col=0)
        self.create_data_frame_viewer(self.TASKS_DURATION)
        
        new_dataset_names = ['Procedure-times']
            
        self.update_dropdown_menu(new_dataset_names, self.results_var, self.results_menu, append=True)

        #self.schedule_button.config(state = tk.NORMAL, text="Create Schedule", command=self.create_schedule, bg = 'lightgreen')

        self.data_label.config(text="Procedure times are Computed")
        
    
    def obtain_possibilities_for_plan(self):
        self.data_label.config(text="Multiple Possibilities generation in progress")
        self.possibility_button.configure(bg="yellow")
        self.master.update_idletasks()
        
        try:
            if self.schedule_new is None:
                print("Please create tenta schedule or pass the plan.")
                return

            self.schedule_possibilities, self.sessions_possibilities = obtain_residual_based_possibilities_for_plan(30, self.schedule_new, self.sessions_df, self.patients_df, time_predicting_parameters = self.procedure_time_prediction_parameters,  prep_min = self.theatre_prpn_min)

            self.data_label.config(text="Multiple Possibilities generation is Complete")
            self.text_display.insert(tk.END, f"Schedule generation Process is Complete \n")
            
            new_dataset_names = ['Schedule Possibilities', 'Sessions Possibilities', 'Schedule Possibilities Plot', 'Sessions Possibilities Plot']
            
            self.update_dropdown_menu(new_dataset_names, self.results_var, self.results_menu, append=True)

            self.create_data_frame_viewer(self.schedule_possibilities)

            self.sessions_updated = self.add_over_under_run_chances(self.sessions_updated, self.sessions_possibilities, 'Session ID', 'H4 Minutes Booked', 30)
            
        except Exception as e:
            print(f"Error: {e}")


    def add_over_under_run_chances(self, single_time_dataset, multiple_time_dataset, row_matching_ID, reference_data_col, time_window):

        percentage_df = multiple_time_dataset.apply(lambda row: calculate_over_under_run_percentage(single_time_dataset, row, row_matching_ID, reference_data_col, reference_data_col, time_window), axis =1)
        
        # Remove existing columns in df_discrete before concatenating
        percentage_df_columns = list(percentage_df.columns)
        single_time_dataset = single_time_dataset.drop(columns=[col for col in percentage_df_columns if col in single_time_dataset.columns])

        return pd.concat([single_time_dataset, percentage_df], axis = 1)
        
    
    def update_dropdown_menu(self, dataset_names, results_var, results_menu, append=False):
        if not append:
            results_var.set(dataset_names[0])  # Set the default selected dataset
            results_menu['menu'].delete(0, 'end')  # Clear existing options

        for name in dataset_names:
            #results_var.set(results_menu['menu'][0])
            #results_menu['menu'].add_command(label=name, command=tk._setit(results_var, name))
            results_menu['menu'].add_command(label=name, command=lambda value=name: results_var.set(value))

    def prepare_plot_data(self):    

        self.columns_combobox.pack()
        self.columns_combobox.place(relx=0.05, rely=0.915, relwidth=0.20, relheight=0.05)
        #self.columns_combobox.configure(state=tk.NORMAL)
        #self.col_data_menu.configure(state=tk.NORMAL)
        #self.update_columns_combobox(list(self.schedule_new.columns), append = False)
        self.update_combobox(self.columns_combobox, list(self.schedule_new.columns), append = False)
        #self.columns_combobox.configure(state="disabled")
        #option_selected = 
        #self.columns_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_dropdown_menu(list(self.schedule_new[self.columns_combobox_var.get()].unique()), self.col_data_var, self.col_data_menu, append=False))
        #self.columns_combobox.bind("<ButtonRelease-1>", lambda event: self.update_dropdown_menu(list(self.schedule_new[self.columns_combobox_var.get()].unique()), self.col_data_var, self.col_data_menu, append=False))
        
        self.columns_combobox.bind("<<ComboboxSelected>>", lambda event: self.update_combobox(self.col_data_menu, list(self.schedule_new[self.columns_combobox_var.get()].unique()),  append=False))
        self.columns_combobox.bind("<ButtonRelease-1>", lambda event: self.update_combobox(self.col_data_menu, list(self.schedule_new[self.columns_combobox_var.get()].unique()), append=False))

        self.col_data_menu.pack()
        self.col_data_menu.place(relx=0.30, rely=0.915, relwidth=0.25, relheight=0.05)
         
        self.col_data_menu.bind("<<ComboboxSelected>>", lambda event : self.filter_schedule(self.columns_combobox_var.get(), self.col_data_var.get()))
        self.col_data_menu.bind("<ButtonRelease-1>", lambda event : self.filter_schedule(self.columns_combobox_var.get(), self.col_data_var.get()))


    def compute_figsize(self, data_size):
        # Compute figsize based on the size of the dataset
        plot_width = min(self.master.winfo_width() * 0.8, 800)  # Limiting width to 800
        plot_height = min(data_size * 0.7, self.master.winfo_height() * 0.6)  # Minimum height of 300
        return (6, plot_height)

    
    def display_result(self):
        # Get the selected DataFrame
        selected_result_name = self.results_var.get()
        
        self.clear_data_frame_viewer()
        
        if not 'plot' in selected_result_name.lower():
            
            self.columns_combobox_var.set('')
            self.columns_combobox.configure(state="disabled")
            
            self.col_data_var.set('')
            #self.col_data_menu.pack_forget()
            self.col_data_menu.configure(state="disabled")
            self.data_label.config(text=f"Data below: {selected_result_name}", bg = 'yellow')

           # Based on the selected DataFrame, get the corresponding DataFrame object
            if selected_result_name == "Procedure-times":
                if self.TASKS_DURATION is not None:
                    self.create_data_frame_viewer(self.TASKS_DURATION)
            elif selected_result_name == "Schedule Generated":
                self.create_data_frame_viewer(self.schedule_new)
            elif selected_result_name == "Sessions Updated":
                self.create_data_frame_viewer(self.sessions_updated)
            elif selected_result_name == "Patients not Scheduled":
                self.create_data_frame_viewer(self.patients_not_scheduled)
            elif selected_result_name == "Sessions Fully Booked":
                self.create_data_frame_viewer(self.sessions_fully_booked)
            elif selected_result_name == "Schedule Possibilities":
                self.create_data_frame_viewer(self.schedule_possibilities)
            elif selected_result_name == 'Sessions Possibilities':
                 self.create_data_frame_viewer(self.sessions_possibilities)
        
            else:
                print("Please select a result.")
                return
        else:
            if hasattr(self, 'tree'):
                self.tree.pack_forget() 
                
            if self.filtered_schedule is None:
                self.columns_combobox_var.set('Filtering Column')
                self.columns_combobox.configure(state="disabled")
                
                self.col_data_var.set('')
                #self.col_data_menu.pack_forget()
                self.col_data_menu.configure(state="disabled")
                self.data_label.config(text=f"Below is {selected_result_name} for all \n Filter the data for better view", bg = 'yellow')
                if hasattr(self, 'tree'):
                    self.tree.pack_forget() 
                #print(self.filtered_schedule)
                self.create_plot_viewer(self.schedule_new)
            
            self.prepare_plot_data()
            #if self.filtered_schedule is None:
            #    self.filtered_schedule = self.schedule_new
            
            if self.filtered_schedule is not None:
                self.data_label.config(text=f"Below is {selected_result_name} for {self.columns_combobox_var.get()}:{self.col_data_var.get()}", bg = 'yellow')
                #self.create_data_frame_viewer(self.filtered_schedule)
                if hasattr(self, 'tree'):
                    self.tree.pack_forget() 
                #print(self.filtered_schedule)
                self.create_plot_viewer(self.filtered_schedule)
                self.filtered_schedule = None
                #self.filtered_schedule = None
                

    def update_combobox(self, combobox, new_options, append=False):
        # Enable the columns Combobox and populate it with column names
        #columns_combobox = self.master.children["!combobox2"]  # Assuming the ID is consistent
        combobox["state"] = "readonly"
        
        if append:
            # Fetch the current values and ensure they're in a set to avoid duplicates
            current_options = set(combobox["values"])
            # Create a set of new options to be added
            new_options_set = set(new_options)
            # Update the combobox values, avoiding duplicates
            updated_options = list(current_options.union(new_options_set))
            combobox["values"] = updated_options
        else:
            # If append is False, replace the values with new_options
            combobox["values"] = new_options

        #self.columns_combobox_var.set(self.columns_combobox["values"][0])
        #columns_combobox.bind("<<ComboboxSelected>>", lambda event, widget=self.combobox: self.on_combobox_selected(event, widget))


    def update_command_button_text(self, event):
        selection = self.algorithm_combobox_var.get()
        #print(selection)
        self.check_enable_schedule_button([self.patients_df, self.sessions_df])
        '''
        if selection == 'Sessions Continuous Filling':
            self.entry_max_sess_util.configure(state = tk.NORMAL)
            self.entry_prpn_time.configure(state = tk.NORMAL)
            self.schedule_button.configure(text="Create Schedule")
        elif selection == 'Optimum Utilisation':
            #self.schedule_button.configure(text="Compute Tasks Duration")
            self.entry_max_sess_util.configure(state = tk.NORMAL)
            self.entry_prpn_time.configure(state = tk.NORMAL)
            self.schedule_button.configure(text="Create Schedule")
        '''
        if selection != '':
            self.entry_max_sess_util.configure(state = tk.NORMAL)
            self.entry_prpn_time.configure(state = tk.NORMAL)
            self.schedule_button.configure(text="Create Schedule")
            
    

    def filter_schedule(self, selected_column, data_selected):
        #print(f'Column and data selected are: {selected_column} and {data_selected}')
        #print(f'Column data selected is: ')
        # Filter data based on the selected column
        if selected_column:
            self.filtered_schedule = self.schedule_new[self.schedule_new[selected_column] == data_selected]
            #self.create_data_frame_viewer(self.filtered_schedule)
            #print(self.filtered_schedule)

    def save_result(self):
         # Get the selected DataFrame
        
        selected_result_name = self.results_var.get()
        #self.display_result()
        df = None
        fig = None
        # Based on the selected DataFrame, get the corresponding DataFrame object
        if selected_result_name == "Procedure-times":
            df = self.TASKS_DURATION
        elif selected_result_name == "Schedule Generated":
            df = self.schedule_new
        elif selected_result_name == "Sessions Updated":
            df = self.sessions_updated 
        elif selected_result_name == "Patients not Scheduled":
            df = self.patients_not_scheduled 
        elif selected_result_name == "Sessions Fully Booked":
            df = self.sessions_fully_booked
        elif selected_result_name == "Schedule Possibilities":
            df= self.schedule_possibilities
        elif selected_result_name == 'Sessions Possibilities':
            df = self.sessions_possibilities
        elif 'plot' in selected_result_name.lower():
            fig = self.ax.get_figure()
        else:
            print("Please select a result.")
            return

        # Ask the user to choose a file for saving
        #file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("PNG files", "*.png")])
        default_filename = selected_result_name + ".xlsx" if df is not None else selected_result_name + '_'+ self.col_data_var.get() +  ".png"
        file_path = filedialog.asksaveasfilename(defaultextension=default_filename, initialfile=default_filename,
                                                 filetypes=[("Excel files", "*.xlsx"), ("PNG files", "*.png")])

        if file_path:
            if df is not None:
                if file_path.endswith('.xlsx'):
                    # Save the DataFrame to the chosen file
                    df.to_excel(file_path, index=True)
            elif fig is not None:
                if not file_path.endswith('.xlsx'):
                    fig.savefig(file_path)
            self.data_label.config(text=f"{selected_result_name} is Saved as {os.path.basename(file_path)}")
            self.text_display.insert(tk.END, f"{selected_result_name} is Saved as {os.path.basename(file_path)}\n")
                
            #print(f"{selected_result} saved to {file_path}")


# Run the app
if __name__ == "__main__":
    #root = tk.Tk()
    #app = SurgicalSchedulerApp(root)
    #root.mainloop()
    root = tk.Tk()
    root.geometry("720x600") 
    app = SurgicalSchedulerApp(root)
    app_bg_color = "lightblue"
    app.change_bg_color(root, app_bg_color)
    
    root.mainloop()
