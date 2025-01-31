import numpy as np
from kivy.event import EventDispatcher
from kivy.properties import StringProperty
from copy import deepcopy

class AnnotatorBase(EventDispatcher):
    instructions = StringProperty('')
    
    def __init__(self, visualizer=None) -> None:
        super().__init__()
        self.visualizer = visualizer
        self.data_dict = visualizer.get_data()
        self.data_type = visualizer.data_type
        self.current_time = None
        self.annotating = False
        self.label = 0
        self.last_added_annotation_idx = None
        self.initial_mouse_pos = None
        self.previous_data_dicts = []
        self.history_idx = 0
        self.update_instructions()

    def get_data_type(self):
        return self.data_type

    def __len__(self):
        return len(self.data_dict)

    def create_new_data_entry(self, current_time, mouse_pos):
        raise NotImplementedError

    def start_annotation(self, current_time, mouse_pos, time_window):
        # self.update_previous_dicts() #TODO restore history update 
        self.current_time = current_time
        self.initial_mouse_pos = mouse_pos
        updated_data = self.data_dict.get_data_at_time(current_time, time_window)
        if updated_data is not None:
            self.updated_data = updated_data
        else:
            self.updated_data = self.create_new_data_entry(current_time, mouse_pos)
        self.initial_data = deepcopy(self.updated_data)
        self.annotating = True

    def update_previous_dicts(self):
        self.previous_data_dicts = self.previous_data_dicts[:self.history_idx]
        self.previous_data_dicts.append(self.data_dict.copy())
        self.history_idx += 1

    def get_time_of_next_annotated_frame(self, current_time, backward=False):
        return float(self.data_dict.get_time_of_next_data_point(current_time, backward))

    def undo(self):
        if self.history_idx == len(self.previous_data_dicts):
            self.previous_data_dicts.append(self.data_dict.copy())
        if self.history_idx < 1:
            return
        self.history_idx -= 1
        self.visualizer.set_data(self.previous_data_dicts[self.history_idx])
        self.data_dict = self.visualizer.get_data()
        
    def redo(self):
        if self.history_idx >= (len(self.previous_data_dicts) - 1):
            return
        self.history_idx += 1
        self.visualizer.set_data(self.previous_data_dicts[self.history_idx])
        self.data_dict = self.visualizer.get_data()

    def delete_by_time(self, time):
        data_dict = self.data_dict
        idx = np.searchsorted(data_dict['ts'], time)
        if abs(data_dict['ts'][idx] - time) > 0.03:
            return
        self.delete_by_index(idx)

    def delete_by_index(self, idx):
        data_dict = self.data_dict
        self.update_previous_dicts()
        for d in data_dict:
            if not hasattr(data_dict[d], '__len__'):
                continue
            try:
                data_dict[d] = np.delete(data_dict[d], idx)
            except IndexError:
                pass
    
    def save(self, path, **kwargs):
        raise NotImplementedError

    def update(self, mouse_position, modifiers):
        raise NotImplementedError

    def stop_annotation(self):
        self.annotating = False

    def update_instructions(self):
        self.instructions = ''