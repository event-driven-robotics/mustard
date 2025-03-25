import numpy as np
from kivy.event import EventDispatcher
from kivy.properties import StringProperty
from copy import deepcopy
import os
from pathlib import Path

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
        self.initial_mouse_pos = None
        self.auto_save_path = home = os.path.join(Path.home(), '.mustard')
        if not os.path.exists(self.auto_save_path):
            os.makedirs(self.auto_save_path)
        self.update_instructions()

    def get_data_type(self):
        return self.data_type

    def __len__(self):
        return len(self.data_dict)

    def create_new_data_entry(self, current_time, mouse_pos):
        raise NotImplementedError

    def start_annotation(self, current_time, mouse_pos, time_window):
        self.current_time = current_time
        self.initial_mouse_pos = mouse_pos
        updated_data = self.data_dict.get_data_at_time(current_time, time_window)
        if updated_data is not None:
            self.updated_data = updated_data
        else:
            self.updated_data = self.create_new_data_entry(current_time, mouse_pos)
        self.initial_data = deepcopy(self.updated_data)
        self.annotating = True

    def get_time_of_next_annotated_frame(self, current_time, backward=False):
        return float(self.data_dict.get_time_of_next_data_point(current_time, backward))

    def undo(self):
        self.data_dict.undo()
        self.update_instructions()

    def redo(self):
        self.data_dict.redo()
        self.update_instructions()

    def delete_by_time(self, time):
        self.data_dict.delete_by_time(time)
        self.update_instructions()
        
    def save(self, path, **kwargs):
        raise NotImplementedError

    def update(self, mouse_position, modifiers):
        raise NotImplementedError

    def stop_annotation(self):
        self.data_dict.update_history()
        self.update_instructions()
        self.annotating = False
        self.save(os.path.join(self.auto_save_path, 'tmp'))

    def update_instructions(self):
        self.instructions = ''