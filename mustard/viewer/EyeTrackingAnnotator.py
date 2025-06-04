from .AnnotatorBase import AnnotatorBase
import numpy as np
import json
import os
from matplotlib import colormaps
from matplotlib.colors import rgb2hex


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class EyeTrackingAnnotator(AnnotatorBase):

    def __init__(self, visualizer) -> None:
        self.base_instructions = 'Annotating eyes. 1. click on eyeball center' + \
            '2. match the iris center 3. adjust size with alt+mouse\n' +\
            'Mouse: rotate, Ctrl+mouse: translate, Alt+mouse: resize, Right_click to toggle eye closed flag'
        self.instructions = self.base_instructions
        self.cm = colormaps.get_cmap('RdYlGn')
        self.fixed_radius = None
        self.fixed_x = None
        self.fixed_y = None
        super().__init__(visualizer)

    def create_new_data_entry(self, current_time, mouse_pos):
        if self.fixed_x is None and self.fixed_y is None:
            new_x = mouse_pos[1]
            new_y = mouse_pos[0]
            self.fixed_x = new_x
            self.fixed_y = new_y
        else:
            new_x = self.fixed_x
            new_y = self.fixed_y
        new_entry = {'eyeball_x': new_x,
                    'eyeball_y': new_y,
                    'eyeball_phi': 0,
                    'eyeball_theta': 0,
                    'eyeball_radius': self.fixed_radius if self.fixed_radius is not None else 100,
                    'eye_closed' : False,
                    'ts': current_time
                    }

        self.data_dict.insert_sorted(new_entry, current_time)
        return new_entry

    def save(self, path, **kwargs):
        if not os.path.splitext(path)[-1] == 'json':
            path = os.path.splitext(path)[0] + '.json'
        data_dict = self.data_dict.get_full_data_as_dict()
        data_dict['ts'] -= self.data_dict.ts_offset
        out_list = []
        for i in range(len(data_dict['ts'])):
            out_dict = {}
            for x in data_dict:
                if not hasattr(data_dict[x], '__len__'):
                    continue
                val = data_dict[x][i]
                out_dict.update({x: val})
            out_list.append(out_dict)
        with open(path, 'w') as f:
            json.dump(out_list, f,cls=NpEncoder)

    def update(self, mouse_position, modifiers):
        if not self.annotating or len(self) == 0:
            return
        if 'right_click' in modifiers:
            try:
                self.updated_data['eye_closed'] = not self.updated_data['eye_closed']
            except KeyError:
                self.updated_data['eye_closed'] = np.full(len(self.data_dict), False)
                self.updated_data['eye_closed'] = not self.updated_data['eye_closed']
        if 'ctrl' in modifiers:
            new_y = self.initial_data['eyeball_y'] + (mouse_position[0] - self.initial_mouse_pos[0])
            new_x = self.initial_data['eyeball_x'] + (mouse_position[1] - self.initial_mouse_pos[1])

            self.updated_data['eyeball_y'] = new_y
            self.updated_data['eyeball_x'] = new_x
            self.fixed_x = new_x
            self.fixed_y = new_y
            
        elif 'alt' in modifiers:
            radius = self.initial_data['eyeball_radius'] - \
                (mouse_position[1] - self.initial_mouse_pos[1])
            self.updated_data['eyeball_radius'] = radius
            if self.fixed_radius is not None:
                self.fixed_radius = radius
        else:
            self.updated_data['eyeball_phi'] = self.initial_data['eyeball_phi'] - \
                np.deg2rad(mouse_position[1] - self.initial_mouse_pos[1])
            self.updated_data['eyeball_theta'] = self.initial_data['eyeball_theta'] + \
                np.deg2rad(mouse_position[0] - self.initial_mouse_pos[0])

    def update_instructions(self):
        labeled_frames = len(self.data_dict)
        color = self.cm(labeled_frames * 20)
        hex = rgb2hex(color)
        self.instructions = self.base_instructions + f'\nFrames labeld: [color={hex}]{labeled_frames}[/color]'
        if self.fixed_radius is not None:
            self.instructions += f', radius = {self.fixed_radius}'
