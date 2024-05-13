from .AnnotatorBase import AnnotatorBase
import numpy as np


class BoundingBoxAnnotator(AnnotatorBase):

    def __init__(self, visualizer) -> None:
        super().__init__(visualizer)
        self.instructions = 'Use num keys to change tag'

    def start_annotation(self, current_time, mouse_pos):
        data_dict = self.data_dict
        self.current_time = current_time
        self.initial_mouse_pos = mouse_pos
        data_dict['ts'] = np.append(data_dict['ts'], current_time)
        data_dict['minY'] = np.append(data_dict['minY'], mouse_pos[1])
        data_dict['minX'] = np.append(data_dict['minX'], mouse_pos[0])
        data_dict['maxY'] = np.append(data_dict['maxY'], mouse_pos[1])
        data_dict['maxX'] = np.append(data_dict['maxX'], mouse_pos[0])
        data_dict['label'] = np.append(data_dict['label'], self.label)
        try:
            added_annotation = data_dict['orderAdded'].max() + 1
        except ValueError:
            added_annotation = 0
        except KeyError:
            data_dict['orderAdded'] = np.full(len(data_dict['ts']) - 1, -1)
            added_annotation = 0

        data_dict['orderAdded'] = np.append(data_dict['orderAdded'], added_annotation)
        self.last_added_annotation_idx = np.argmax(data_dict['orderAdded'])

        self.sort_by_ts(data_dict)
        self.annotating = True

    def save(self, path, **kwargs):
        data_dict = self.data_dict
        viz = self.visualizer
        if kwargs.get('interpolate'):
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
        if self.annotating:
            data_dict = self.data_dict
            data_dict['ts'][self.last_added_annotation_idx] = self.current_time
            data_dict['minY'][self.last_added_annotation_idx] = min(mouse_position[1], self.initial_mouse_pos[1])
            data_dict['maxY'][self.last_added_annotation_idx] = max(mouse_position[1], self.initial_mouse_pos[1])
            data_dict['minX'][self.last_added_annotation_idx] = min(mouse_position[0], self.initial_mouse_pos[0])
            data_dict['maxX'][self.last_added_annotation_idx] = max(mouse_position[0], self.initial_mouse_pos[0])

    def stop_annotation(self):
        data_dict = self.data_dict
        try:
            if data_dict['minY'][-1] == data_dict['maxY'][-1] or data_dict['minX'][-1] == data_dict['maxX'][-1]:
                self.undo()
        except IndexError:
            pass
        self.annotating = False
