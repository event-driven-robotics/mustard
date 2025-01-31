from .AnnotatorBase import AnnotatorBase
import numpy as np


class BoundingBoxAnnotator(AnnotatorBase):

    def __init__(self, visualizer) -> None:
        super().__init__(visualizer)
        self.instructions = 'Use num keys to change tag'

    def create_new_data_entry(self, current_time, mouse_pos):
        new_entry = {'ts': current_time,
                        'minY': mouse_pos[1],
                        'minX': mouse_pos[0],
                        'maxY': mouse_pos[1],
                        'maxX': mouse_pos[0],
                        'label': self.label
                        }
        
        self.data_dict.insert_sorted(new_entry, current_time)
        return new_entry

    def save(self, path, **kwargs):
        data_dict = self.data_dict.get_full_data_as_dict()
        viz = self.visualizer
        if kwargs.get('interpolate', False):
            boxes = []
            # TODO parametrize sample rate when saving interpolated
            for t in np.arange(0, data_dict['ts'][-1] + 0.01, 0.01):
                boxes_at_time = viz.get_frame(t, 0.01, **kwargs)
                if boxes_at_time is not None and len(boxes_at_time):
                    for b in boxes_at_time:
                        boxes.append(np.concatenate(([t], b)))
        else:
            boxes = np.column_stack((data_dict['ts'], data_dict['minY'], data_dict['minX'], data_dict['maxY'],
                                     data_dict['maxX'], data_dict['label']))
        np.savetxt(path, boxes, fmt='%f')

    def update(self, mouse_position, modifiers):
        if not self.annotating:
            return
        self.updated_data['ts'] = self.current_time
        self.updated_data['minY'] = min(mouse_position[1], self.initial_mouse_pos[1])
        self.updated_data['maxY'] = max(mouse_position[1], self.initial_mouse_pos[1])
        self.updated_data['minX'] = min(mouse_position[0], self.initial_mouse_pos[0])
        self.updated_data['maxX'] = max(mouse_position[0], self.initial_mouse_pos[0])

    def stop_annotation(self):
        try:
            if self.updated_data['minY'] == self.updated_data['maxY'] or self.updated_data['minX'] == self.updated_data['maxX']:
                self.undo()
        except IndexError:
            pass
        self.save('tmp_boxes.csv')