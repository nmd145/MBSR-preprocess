import mne, time
from datetime import datetime
from mne.io import read_raw_edf
from utils import group_consecutives


class EEG():
    def __init__(self,path_edf, path_stage):
        self.raw = None
        self.baseline = None
        self.bandpass = None
        self.corrected = None
        self.eog_channels = None
         
        # Raw data & meta data
        self.path_edf, self.path_stage = (path_edf, path_stage)
        raw = read_raw_edf(self.path_edf, preload=True, verbose=0)
        self.eog_channels = raw.copy().pick_channels(['EOG Left', 'EOG Right'])

        raw.pick_channels(['EEG Fp1-A2','EEG F7-A2','EEG F3-A2','EEG T5-A2', \
                           'EEG O1-A2','EEG Fp2-A1','EEG F4-A1','EEG F8-A1','EEG T6-A1','EEG O2-A1'])
        
        # Rename channel name to standard1005 
        
        raw.rename_channels({'EEG Fp1-A2': 'Fp1','EEG F7-A2': 'F7',
        'EEG F3-A2': 'F3', 'EEG T5-A2': 'T5','EEG O1-A2': 'O1',
        'EEG Fp2-A1': 'Fp2', 'EEG F4-A1': 'F4', 'EEG F8-A1': 'F8',
        'EEG T6-A1': 'T6', 'EEG O2-A1': 'O2'})

        raw.set_montage('standard_1005', raise_if_subset=False)

        self.raw = raw.copy()
        # import pdb; pdb.set_trace()
        self.meas_date, _ = self.raw.info['meas_date']
        # print(self.meas_date)
        # exit()
        self.ch_names = self.raw.info['ch_names']
        
        # Stage file: 
        with open(self.path_stage, 'r') as f: 
            stages = f.read().splitlines()
        
            fname, lname, subject, start_date, start_time, \
             end_date, end_time = stages[0].split(',')
        
            stages = stages[1:]
            
        # Stages and stages indices
        self.stages = stages
        self.subject = subject
        
        task_indices = [idx for idx, _ in enumerate(self.stages) if _ == '13']
        self.tasks = group_consecutives(task_indices)

        rest_indices = [idx for idx, _ in enumerate(self.stages) if _ == '12']
        self.rests = group_consecutives(rest_indices)


        self.start_timestamp = time.mktime(
            datetime.strptime(f"{start_date} {start_time}", \
                              "%m/%d/%Y %I:%M:%S %p").timetuple()) \
                                  + 7 * 3600 #Adjust timezone
        
        self.DIFFTIME = self.start_timestamp - self.meas_date

