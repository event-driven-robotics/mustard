# -*- coding: utf-8 -*-
"""
Copyright (C) 2020 Event-driven Perception for Robotics
Authors: Massimiliano Iacono
         Sim Bamford

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

Collection of possible Viewers that can be spawned in the main app window. Each one will have
a visualiser which is responsible for the data management and retrieval.
"""

import numpy as np
import re
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.scatter import Scatter
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, StringProperty, ListProperty, DictProperty, NumericProperty
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window, Clock
from kivy.graphics.transformation import Matrix
from kivy.graphics.vertex_instructions import Rectangle

from bimvee.visualisers.visualiserBoundingBoxes import VisualiserBoundingBoxes
import os


class BoundingBox(Widget):
    mouse_over = BooleanProperty(False)

    def __init__(self, bb_color, x, y, width, height, **kwargs):
        super(BoundingBox, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.bb_color = bb_color
        Window.bind(mouse_pos=self.on_mouse_pos)

    def on_mouse_pos(self, window, pos):
        win_coords = self.to_window(*self.pos)
        self.mouse_over = win_coords[0] < pos[0] < win_coords[0] + self.size[0] and\
                          win_coords[1] < pos[1] < win_coords[1] + self.size[1]


class LabeledBoundingBox(BoundingBox):
    obj_label = StringProperty('')

    def __init__(self, bb_color, x, y, width, height, label, **kwargs):
        super(LabeledBoundingBox, self).__init__(bb_color, x, y, width, height, **kwargs)
        try:
            self.obj_label = '{:d}'.format(int(label))
        except ValueError:
            self.obj_label = label


class ZoomableImage(Scatter):

    def __init__(self, **kwargs):
        super(ZoomableImage, self).__init__(**kwargs)
        Clock.schedule_once(self.reset_zoom)

    def reset_zoom(self, _):
        self.apply_transform(Matrix().scale(1, 1, 1),
                             anchor=self.center)

    def on_touch_down(self, touch):
        if not self.transform_allowed:
            return False
        if not (self.clickable_area[0] < touch.x < self.clickable_area[0] + self.clickable_area[2] and
                self.clickable_area[1] < touch.y < self.clickable_area[1] + self.clickable_area[3]):
            return False
        if touch.is_mouse_scrolling:
            factor = None
            if touch.button == 'scrolldown':
                if self.scale < self.scale_max:
                    factor = 1.1
            elif touch.button == 'scrollup':
                if self.scale > self.scale_min:
                    factor = 1 / 1.1
            if factor is not None:
                self.apply_transform(Matrix().scale(factor, factor, factor),
                                     anchor=touch.pos)
            return False
        else:
            return super(ZoomableImage, self).on_touch_down(touch)


class Viewer(BoxLayout):
    data = DictProperty(force_dispatch=True)
    visualisers = ListProperty([], allownone=True)
    flipHoriz = BooleanProperty(False)
    flipVert = BooleanProperty(False)
    labeling = BooleanProperty(False)
    transform_allowed = BooleanProperty(False)
    mouse_on_image = BooleanProperty(False)
    settings = DictProperty({}, allownone=True)
    settings_values = DictProperty({}, allownone=True)
    title = StringProperty('Title')
    colorfmt = StringProperty('luminance')
    orientation = 'vertical'
    mouse_position = ListProperty([0, 0])
    label = NumericProperty(0)
    len_gt = NumericProperty(0)

    def __init__(self, **kwargs):
        super(Viewer, self).__init__(**kwargs)
        self.settings_box = None
        from matplotlib.pyplot import get_cmap
        self.cm = get_cmap('tab20')
        self.current_time = 0
        self.current_time_window = 0
        Window.bind(mouse_pos=self.on_mouse_pos)
        self.clicked_mouse_pos = None
        self.last_added_box_idx = -1
        self.cropped_region = [0, 0, 0, 0]

    def window_to_image_coords(self, x, y, flip=True):
        scale = self.image.parent.scale
        scaled_image_width = self.image.norm_image_size[0] * scale
        scaled_image_height = self.image.norm_image_size[1] * scale
        w_ratio = scaled_image_width / self.image.texture.width
        h_ratio = scaled_image_height / self.image.texture.height
        window_img_x, window_img_y = self.image.to_window(self.image.center_x, self.image.center_y)
        image_x = ((x - (window_img_x - scaled_image_width / 2)) / w_ratio)
        image_y = ((y - (window_img_y - scaled_image_height / 2)) / h_ratio)

        if flip:
            if self.flipHoriz:
                image_x = self.image.texture.width - image_x
            if self.flipVert:
                image_y = self.image.texture.height - image_y
        return image_x, image_y

    def on_mouse_pos(self, window, pos):
        image_x, image_y = self.window_to_image_coords(pos[0], pos[1])
        self.mouse_on_image = True
        if self.cropped_region[0] <= image_x <= self.cropped_region[0] + self.cropped_region[2]:
            self.mouse_position[0] = int(image_x)
        elif image_x < self.cropped_region[0]:
            self.mouse_position[0] = int(self.cropped_region[0])
            self.mouse_on_image = False
        elif image_x > self.cropped_region[2]:
            self.mouse_position[0] = int(self.cropped_region[2])
            self.mouse_on_image = False
        if self.cropped_region[1] <= image_y <= self.cropped_region[1] + self.cropped_region[3]:
            self.mouse_position[1] = int(self.image.texture.height - image_y)
        elif image_y < self.cropped_region[1]:
            self.mouse_position[1] = int(self.cropped_region[3])
            self.mouse_on_image = False
        elif image_y > self.cropped_region[3]:
            self.mouse_position[1] = int(self.cropped_region[1])
            self.mouse_on_image = False

    def remove_last_bbox(self):
        for v in self.visualisers:
            if isinstance(v, VisualiserBoundingBoxes):
                data_dict = v.get_data()
                if self.last_added_box_idx != -1 and data_dict['orderAdded'][self.last_added_box_idx] != -1:
                    for d in data_dict:
                        try:
                            data_dict[d] = np.delete(data_dict[d], self.last_added_box_idx)
                        except IndexError:
                            return
                    try:
                        self.last_added_box_idx = np.argmax(data_dict['orderAdded'])
                    except ValueError:
                        self.last_added_box_idx = -1
                else:
                    return
                v.set_data(data_dict)
                self.len_gt = len(data_dict['ts'])
                self.get_frame(self.current_time, self.current_time_window)

    def save_bboxes(self, path, ending_time):
        if self.labeling:
            data_dict = None
            for v in self.visualisers:
                if isinstance(v, VisualiserBoundingBoxes):
                    data_dict = v.get_data()
                    viz = v
                    break
            if data_dict is None:
                return
            if self.settings_values[viz.data_type]['interpolate']:
                boxes = []
                for t in np.arange(0, ending_time, 0.01):  # TODO parametrize sample rate when saving interpolated
                    boxes_at_time = viz.get_frame(t, self.current_time_window, **self.settings_values[viz.data_type])
                    if boxes_at_time != [[0, 0, 0, 0]] and len(boxes_at_time):
                        for b in boxes_at_time:
                            boxes.append(np.concatenate(([t], b)))
            else:
                boxes = np.column_stack((data_dict['ts'], data_dict['minY'], data_dict['minX'], data_dict['maxY'],
                                         data_dict['maxX'], data_dict['label']))
            np.savetxt(os.path.join(path, 'ground_truth.csv'), boxes, fmt='%f')

    def on_touch_move(self, touch):
        if self.clicked_mouse_pos is not None:
            for v in self.visualisers:
                if isinstance(v, VisualiserBoundingBoxes):
                    data_dict = v.get_data()
                    viz = v
                    break

            data_dict['ts'][self.last_added_box_idx] = self.current_time
            data_dict['minY'][self.last_added_box_idx] = min(self.mouse_position[1], self.clicked_mouse_pos[1])
            data_dict['maxY'][self.last_added_box_idx] = max(self.mouse_position[1], self.clicked_mouse_pos[1])
            data_dict['minX'][self.last_added_box_idx] = min(self.mouse_position[0], self.clicked_mouse_pos[0])
            data_dict['maxX'][self.last_added_box_idx] = max(self.mouse_position[0], self.clicked_mouse_pos[0])
            viz.set_data(data_dict)
            self.get_frame(self.current_time, self.current_time_window)
        return False

    def on_touch_up(self, touch):
        self.clicked_mouse_pos = None
        for v in self.visualisers:
            if isinstance(v, VisualiserBoundingBoxes):
                data_dict = v.get_data()
                try:
                    if data_dict['minY'][-1] == data_dict['maxY'][-1] or data_dict['minX'][-1] == data_dict['maxX'][-1]:
                        self.remove_last_bbox()
                except IndexError:
                    pass
        return False

    def on_touch_down(self, touch):
        if super(Viewer, self).on_touch_down(touch):
            return True
        if not self.mouse_on_image:
            return False
        if self.transform_allowed:
            return False
        if self.labeling:
            data_dict = None
            viz = None
            for v in self.visualisers:
                if isinstance(v, VisualiserBoundingBoxes):
                    data_dict = v.get_data()
                    viz = v
                    break
            if data_dict is None:
                data_dict = {
                    'ts': np.array([self.current_time]),
                    'minY': np.array([self.mouse_position[1]]),
                    'minX': np.array([self.mouse_position[0]]),
                    'maxY': np.array([self.mouse_position[1]]),
                    'maxX': np.array([self.mouse_position[0]]),
                    'label': np.array([self.label]),
                    'orderAdded': np.array([0])
                }
                viz = VisualiserBoundingBoxes(data_dict)
                self.settings['boundingBoxes'] = viz.get_settings()
                self.visualisers.append(viz)
            else:
                data_dict['ts'] = np.append(data_dict['ts'], self.current_time)
                data_dict['minY'] = np.append(data_dict['minY'], self.mouse_position[1])
                data_dict['minX'] = np.append(data_dict['minX'], self.mouse_position[0])
                data_dict['maxY'] = np.append(data_dict['maxY'], self.mouse_position[1])
                data_dict['maxX'] = np.append(data_dict['maxX'], self.mouse_position[0])
                data_dict['label'] = np.append(data_dict['label'], self.label)
                try:
                    added_box = data_dict['orderAdded'].max() + 1
                except ValueError:
                    added_box = 0
                except KeyError:
                    data_dict['orderAdded'] = np.full(len(data_dict['ts']) - 1, -1)
                    added_box = 0

                data_dict['orderAdded'] = np.append(data_dict['orderAdded'], added_box)
            # Sorting wrt timestamps
            argsort = np.argsort(data_dict['ts'])
            for d in data_dict:
                if hasattr(data_dict[d], '__len__'):
                    data_dict[d] = data_dict[d][argsort]

            self.last_added_box_idx = np.argmax(data_dict['orderAdded'])
            self.len_gt = len(data_dict['ts'])
            viz.set_data(data_dict)
            self.get_frame(self.current_time, self.current_time_window)
            self.clicked_mouse_pos = self.mouse_position[0], self.mouse_position[1]
        return False

    def on_visualisers(self, instance, value):
        self.init_visualisers()

    def init_visualisers(self):
        if self.visualisers is not None and self.visualisers:
            for v in self.visualisers:
                if v.data_type in ['dvs', 'frame', 'pose6q', 'point3', 'flowMap', 'imu']:
                    self.colorfmt = v.get_colorfmt()
                    self.data_shape = v.get_dims()
                    buf_shape = (dp(self.data_shape[0]), dp(self.data_shape[1]))
                    self.image.texture = Texture.create(size=buf_shape, colorfmt=self.colorfmt)
                if v.data_type == 'boundingBoxes':
                    self.len_gt = len(v.get_data()['ts'])

    def on_settings(self, instance, settings_dict):
        if self.settings_box is not None:
            self.settings_box.clear_widgets()
        self.settings_box = BoxLayout(size_hint=(1, 0.2), spacing=5)
        self.add_widget(self.settings_box)
        self.update_settings(self.settings_box, settings_dict, self.settings_values)

    def on_settings_change(self, instance, value):
        self.settings_values[instance.parent.id][instance.id] = value
        self.get_frame(self.current_time, self.current_time_window)

    def update_settings(self, parent_widget, settings_dict, settings_values):
        for key in settings_dict:
            if 'type' not in settings_dict[key]:
                if settings_dict[key]:
                    box = BoxLayout(orientation='vertical', spacing=5)
                    splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', key)).title()
                    box.add_widget(Label(text='Settings for {}'.format(splitted), size_hint=(1, 0.1)))
                    settings_grid = GridLayout(cols=2)
                    settings_grid.id = key
                    box.add_widget(settings_grid)
                    parent_widget.add_widget(box)
                    settings_values[key] = {}
                    self.update_settings(settings_grid, settings_dict[key], settings_values[key])
            elif settings_dict[key]['type'] == 'boolean':
                parent_widget.add_widget(Label(text=key))
                check_box = CheckBox(active=settings_dict[key]['default'])
                check_box.id = key
                parent_widget.add_widget(check_box)
                settings_values[key] = check_box.active
                check_box.bind(active=self.on_settings_change)
            elif settings_dict[key]['type'] == 'range':
                parent_widget.add_widget(Label(text=key))
                slider = Slider(value=settings_dict[key]['default'],
                                min=settings_dict[key]['min'],
                                max=settings_dict[key]['max'],
                                step=settings_dict[key]['step'])
                slider.id = key
                parent_widget.add_widget(slider)
                settings_values[key] = slider.value
                slider.bind(value=self.on_settings_change)
            elif settings_dict[key]['type'] == 'value_list':
                parent_widget.add_widget(Label(text=key))
                from kivy.uix.spinner import Spinner
                spinner = Spinner(text=settings_dict[key]['default'],
                                  values=settings_dict[key]['values'])
                spinner.id = key
                parent_widget.add_widget(spinner)
                settings_values[key] = spinner.text
                spinner.bind(text=self.on_settings_change)

    def on_data(self, instance, value):
        self.image.clear_widgets()
        for data_type in self.data.keys():
            if data_type in ['dvs', 'frame', 'pose6q', 'point3', 'flowMap', 'imu']:
                self.update_image(self.data[data_type])
            elif data_type in ['boundingBoxes']:
                self.update_b_boxes(self.data[data_type])
            elif data_type in ['skeleton']:
                self.update_skeleton(self.data[data_type])

    def update_image(self, data):
        if self.image.texture is not None:
            size_required = self.data_shape[0] * self.data_shape[1] * (1 + (self.colorfmt == 'rgb') * 2)
            if not isinstance(data, np.ndarray):
                data = np.zeros((self.data_shape[0], self.data_shape[1], 3), dtype=np.uint8)
            if data.size >= size_required:
                try:
                    if self.flipHoriz:
                        data = np.flip(data, axis=1)
                    if not self.flipVert:  # Not, because by default, y should increase downwards, following https://arxiv.org/pdf/1610.08336.pdf
                        data = np.flip(data, axis=0)
                except AttributeError:
                    pass  # It's not a class that allows flipping
                self.image.texture.blit_buffer(data.tostring(), bufferfmt="ubyte", colorfmt=self.colorfmt)

    def update_b_boxes(self, b_boxes):
        if b_boxes is None:
            return

        bb_copy = b_boxes.copy()
        texture_width = self.image.texture.width
        texture_height = self.image.texture.height
        x_img, y_img, image_width, image_height = self.get_image_bounding_box()

        w_ratio = image_width / texture_width
        h_ratio = image_height / texture_height
        for n, b in enumerate(bb_copy):
            for i in range(4):
                b[i] = dp(b[i])
            if self.flipHoriz:
                min_x = texture_width - b[3]
                max_x = texture_width - b[1]
                b[1] = min_x
                b[3] = max_x
            if self.flipVert:
                min_y = texture_height - b[2]
                max_y = texture_height - b[0]
                b[0] = min_y
                b[2] = max_y

            if not (self.cropped_region[0] < (b[3] + b[1]) / 2 < self.cropped_region[0] + self.cropped_region[2] and
                    self.cropped_region[1] < texture_height - ((b[2] + b[0]) / 2) < self.cropped_region[1] + self.cropped_region[3]):
                continue

            width = w_ratio * float(b[3] - b[1])
            height = h_ratio * float(b[2] - b[0])
            if width == 0 and height == 0:
                break

            x = x_img + w_ratio * float(b[1])
            y = y_img + h_ratio * (texture_height - float(b[2]))

            try:
                bb_color = self.cm.colors[b[4] % len(self.cm.colors)] + (1,)
                label = b[4]
                if label == 0:  # Label = 0 is considered as unlabeled
                    raise IndexError
                box_item = LabeledBoundingBox(bb_color=bb_color,
                                              x=x, y=y,
                                              width=width, height=height,
                                              label=label)
            except IndexError:
                box_item = BoundingBox(bb_color=self.cm.colors[0],
                                       x=x, y=y,
                                       width=width, height=height)

            box_item.id = 'box_{}'.format(n),

            self.image.add_widget(box_item)

    def update_skeleton(self, skeleton):
        if skeleton is None:
            return

        texture_width = self.image.texture.width
        texture_height = self.image.texture.height
        x_img, y_img, image_width, image_height = self.get_image_bounding_box()

        w_ratio = image_width / texture_width
        h_ratio = image_height / texture_height
        for i, joint in enumerate(skeleton):
            x = skeleton[joint][0]
            y = texture_height - skeleton[joint][1]

            if self.flipHoriz:
                x = texture_width - x
            if self.flipVert:
                y = texture_height - y

            if not (self.cropped_region[0] < x < self.cropped_region[0] + self.cropped_region[2] and
                    self.cropped_region[1] < y < self.cropped_region[1] + self.cropped_region[3]):
                continue

            x = int(x_img + w_ratio * x)
            y = int(y_img + h_ratio * y)

            if self.settings_values['skeleton']['show_labels']:
                box_item = LabeledBoundingBox(bb_color=self.cm.colors[i % len(self.cm.colors)] + (1,),
                                              x=x, y=y,
                                              width=2, height=2,
                                              label=joint)
            else:
                box_item = BoundingBox(bb_color=self.cm.colors[i % len(self.cm.colors)] + (1,),
                                       x=x, y=y,
                                       width=2, height=2)
            box_item.id = 'box_{}'.format(i),

            self.image.add_widget(box_item)

    def get_image_bounding_box(self):
        image_width = self.image.norm_image_size[0]
        image_height = self.image.norm_image_size[1]
        x_img = self.image.center_x - image_width / 2
        y_img = self.image.center_y - image_height / 2
        return x_img, y_img, image_width, image_height

    def get_frame(self, time_value, time_window):
        data_dict = {}
        self.current_time = time_value
        self.current_time_window = time_window
        for v in self.visualisers:
            data_dict[v.data_type] = {}
            data_dict[v.data_type] = v.get_frame(time_value, time_window, **self.settings_values[v.data_type])
            self.colorfmt = v.get_colorfmt()
        self.data.update(data_dict)

    def crop_image(self, x, y, width, height):
        _, _, image_width, image_height = self.get_image_bounding_box()

        crop_bl_x, crop_bl_y = self.window_to_image_coords(x, y, flip=False)
        crop_tr_x, crop_tr_y = self.window_to_image_coords(x + width, y + height, flip=False)

        texture_width = self.image.texture.width
        texture_height = self.image.texture.height
        crop_bl_x = max(0, crop_bl_x)
        crop_bl_x = min(texture_width, crop_bl_x)
        crop_bl_y = max(0, crop_bl_y)
        crop_bl_y = min(texture_height, crop_bl_y)
        crop_width = min(crop_tr_x - crop_bl_x, texture_width - crop_bl_x)
        crop_height = min(crop_tr_y - crop_bl_y, texture_height - crop_bl_y)

        w_ratio = image_width / texture_width
        h_ratio = image_height / texture_height

        subtexture = self.image.texture.get_region(crop_bl_x, crop_bl_y, crop_width, crop_height)

        self.cropped_region = [crop_bl_x, crop_bl_y, crop_width, crop_height]

        with self.image.canvas:
            self.image.canvas.clear()
            Rectangle(texture=subtexture,
                      pos=(crop_bl_x * w_ratio + self.image.center_x - image_width / 2,
                           crop_bl_y * h_ratio + self.image.center_y - image_height / 2),
                      size=(crop_width * w_ratio, crop_height * h_ratio))
        self.on_data(None, None)
