from .AnnotatorBase import AnnotatorBase
import numpy as np
import json
from copy import deepcopy
from matplotlib import colormaps
from matplotlib.colors import rgb2hex
class EyeTrackingAnnotator(AnnotatorBase):

    def __init__(self, visualizer) -> None:
        super().__init__(visualizer)
        self.base_instructions = 'Annotating eyes. 1. click on eyeball center' + \
                            '2. match the iris center 3. adjust size with alt+mouse\n' +\
                            'Mouse: rotate, Ctrl+mouse: translate, Alt+mouse: resize'
        self.instructions = self.base_instructions
        self.cm = colormaps.get_cmap('RdYlGn')


    def create_new_data_entry(self, current_time, mouse_pos):
        data_dict = self.data_dict
        data_dict['ts'] = np.append(data_dict['ts'], current_time)
        data_dict['eyeball_x'] = np.append(data_dict['eyeball_x'], mouse_pos[1])
        data_dict['eyeball_y'] = np.append(data_dict['eyeball_y'], mouse_pos[0])
        data_dict['eyeball_radius'] = np.append(data_dict['eyeball_radius'], np.mean(
                data_dict['eyeball_radius']) if len(data_dict['eyeball_radius']) else 100)
        data_dict['eyeball_phi'] = np.append(data_dict['eyeball_phi'], 0)
        data_dict['eyeball_theta'] = np.append(data_dict['eyeball_theta'], 0)

    def save(self, path, **kwargs):
        data_dict = deepcopy(self.data_dict)
        print(data_dict['tsOffset'])
        data_dict['ts'] -= data_dict['tsOffset']
        out_list = []
        for i in range(len(data_dict['ts'])):
            out_list.append({x: data_dict[x][i] for x in data_dict if hasattr(data_dict[x], '__len__')})
        with open(path, 'w') as f:
            json.dump(out_list, f)

    def update(self, mouse_position, modifiers):
        if not self.annotating:
            return
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
        self.save('tmp_eyes.json')
        labeled_frames = len(self.data_dict['ts'])
        color = self.cm(labeled_frames * 20)
        hex = rgb2hex(color)
        self.instructions = self.base_instructions + f'\nFrames labeld: [color={hex}]{labeled_frames}[/color]'

        return super().stop_annotation()