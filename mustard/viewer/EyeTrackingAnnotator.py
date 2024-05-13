from .AnnotatorBase import AnnotatorBase
import numpy as np
import json


class EyeTrackingAnnotator(AnnotatorBase):

    def __init__(self, visualizer) -> None:
        super().__init__(visualizer)
        self.instructions = 'Annotating eyes. 1. click on eyeball center' + \
                            '2. match the iris center 3. adjust size with alt+mouse\n' +\
                            'Mouse: rotate, Ctrl+mouse: translate, Alt+mouse: resize'

    def start_annotation(self, current_time, mouse_pos):
        data_dict = self.data_dict
        self.current_time = current_time
        self.initial_mouse_pos = mouse_pos
        self.annotation_idx = np.searchsorted(data_dict['ts'], current_time)
        try:
            if abs(data_dict['ts'][self.annotation_idx] - current_time) > 0.03:
                raise IndexError
            self.initial_data = {x: data_dict[x][self.annotation_idx]
                                 for x in data_dict if hasattr(data_dict[x], '__len__')}
        except IndexError:
            data_dict['ts'] = np.append(data_dict['ts'], current_time)
            data_dict['eyeball_x'] = np.append(data_dict['eyeball_x'], mouse_pos[1])
            data_dict['eyeball_y'] = np.append(data_dict['eyeball_y'], mouse_pos[0])
            data_dict['eyeball_radius'] = np.append(data_dict['eyeball_radius'], np.mean(
                data_dict['eyeball_radius']) if len(data_dict['eyeball_radius']) else 100)
            data_dict['eyeball_phi'] = np.append(data_dict['eyeball_phi'], 0)
            data_dict['eyeball_theta'] = np.append(data_dict['eyeball_theta'], 0)
            try:
                last_added = np.max(data_dict['orderAdded']) + 1
            except ValueError:
                last_added = 0
            data_dict['orderAdded'] = np.append(data_dict['orderAdded'], last_added)
            self.initial_data = {x: data_dict[x][-1] for x in data_dict if hasattr(data_dict[x], '__len__')}

        self.last_added_annotation_idx = np.argmax(data_dict['orderAdded'])
        self.sort_by_ts(data_dict)
        self.annotating = True

    def save(self, path, **kwargs):
        data_dict = self.data_dict.copy()
        data_dict['ts'] -= data_dict['tsOffset']
        out_list = []
        for i in range(len(data_dict['ts'])):
            out_list.append({x: data_dict[x][i] for x in data_dict if hasattr(data_dict[x], '__len__')})
        with open(path, 'w') as f:
            json.dump(out_list, f)

    def update(self, mouse_position, modifiers):
        if self.annotating:
            data_dict = self.data_dict
            if 'ctrl' in modifiers:
                data_dict['eyeball_y'][self.annotation_idx] = self.initial_data['eyeball_y'] + \
                    (mouse_position[0] - self.initial_mouse_pos[0])
                data_dict['eyeball_x'][self.annotation_idx] = self.initial_data['eyeball_x'] + \
                    (mouse_position[1] - self.initial_mouse_pos[1])
            elif 'alt' in modifiers:
                data_dict['eyeball_radius'][self.annotation_idx] = self.initial_data['eyeball_radius'] - \
                    (mouse_position[1] - self.initial_mouse_pos[1])
            else:
                data_dict['eyeball_phi'][self.annotation_idx] = self.initial_data['eyeball_phi'] - \
                    np.deg2rad(mouse_position[1] - self.initial_mouse_pos[1])
                data_dict['eyeball_theta'][self.annotation_idx] = self.initial_data['eyeball_theta'] + \
                    np.deg2rad(mouse_position[0] - self.initial_mouse_pos[0])

    def stop_annotation(self):
        self.annotating = False
