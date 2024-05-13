import numpy as np
from kivy.event import EventDispatcher
from kivy.properties import StringProperty


class AnnotatorBase(EventDispatcher):
    instructions = StringProperty('')

    def __init__(self, visualizer=None) -> None:
        super().__init__()
        self.data_dict = visualizer.get_data()
        self.data_type = visualizer.data_type
        self.current_time = None
        self.annotating = False
        self.label = 0
        self.last_added_annotation_idx = None
        self.initial_mouse_pos = None

    def get_data_type(self):
        return self.data_type

    def __len__(self):
        return len(self.data_dict['ts'])

    def start_annotation(self, current_time, mouse_pos):
        raise NotImplementedError

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
        data_dict = self.data_dict
        if self.last_added_annotation_idx != -1 and data_dict['orderAdded'][self.last_added_annotation_idx] != -1:
            for d in data_dict:
                try:
                    data_dict[d] = np.delete(data_dict[d], self.last_added_annotation_idx)
                except IndexError:
                    pass
            try:
                self.last_added_annotation_idx = np.argmax(data_dict['orderAdded'])
            except ValueError:
                self.last_added_annotation_idx = -1
        else:
            return

    def save(self, path, **kwargs):
        raise NotImplementedError

    def update(self, mouse_position, modifiers):
        raise NotImplementedError

    def stop_annotation(self):
        raise NotImplementedError

    @staticmethod
    def sort_by_ts(data_dict):
        argsort = np.argsort(data_dict['ts'])
        for d in data_dict:
            if hasattr(data_dict[d], '__len__'):
                data_dict[d] = data_dict[d][argsort]
        return data_dict