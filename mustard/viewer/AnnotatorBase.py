import numpy as np
from kivy.event import EventDispatcher
from kivy.properties import StringProperty

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

    def get_data_type(self):
        return self.data_type

    def __len__(self):
        return len(self.data_dict['ts'])

    def create_new_data_entry(self, current_time, mouse_pos):
        raise NotImplementedError

    def start_annotation(self, current_time, mouse_pos):
        data_dict = self.data_dict
        self.update_previous_dicts()
        self.current_time = current_time
        self.initial_mouse_pos = mouse_pos
        self.annotation_idx = np.searchsorted(data_dict['ts'], current_time)
        try:
            if abs(data_dict['ts'][self.annotation_idx] - current_time) > 0.03: #TODO Specialize initial data selection based on time or mosue position
                raise IndexError
            self.initial_data = {x: data_dict[x][self.annotation_idx]
                                 for x in data_dict if hasattr(data_dict[x], '__len__')}
        except IndexError:
            self.create_new_data_entry(current_time, mouse_pos)
            self.initial_data = {x: data_dict[x][-1] for x in data_dict if hasattr(data_dict[x], '__len__')}

        self.sort_by_ts(data_dict)
        self.annotating = True

    def update_previous_dicts(self):
        self.previous_data_dicts = self.previous_data_dicts[:self.history_idx]
        self.previous_data_dicts.append(self.data_dict.copy())
        self.history_idx += 1

    def get_time_of_next_annotated_frame(self, current_time, backward=False):
        try:
            if backward:
                next_idx = np.searchsorted(self.data_dict['ts'], current_time - 0.0001) - 1
            else:
                next_idx = np.searchsorted(self.data_dict['ts'], current_time + 0.0001) % len(self)
            return float(self.data_dict['ts'][next_idx])
        except IndexError:
            return current_time

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

    @staticmethod
    def sort_by_ts(data_dict):
        argsort = np.argsort(data_dict['ts'])
        for d in data_dict:
            if hasattr(data_dict[d], '__len__'):
                data_dict[d] = data_dict[d][argsort]