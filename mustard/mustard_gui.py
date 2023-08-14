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

Use kivy to create an app which can receive data dicts as imported by bimvee
importAe, and allow synchronised playback for each of the contained channels and datatypes. 
"""
# standard imports
from mustard.viewer import Viewer
from kivy.core.window import Window
from kivy.properties import DictProperty
from kivy.properties import StringProperty, NumericProperty
from kivy.properties import ObjectProperty
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.uix.slider import Slider
from kivy.app import App
import numpy as np
import sys
import os
import json
from textwrap import wrap

os.environ['KIVY_NO_ARGS'] = 'T'

# Optional import of tkinter allows setting of app size wrt screen size
try:
    import tkinter as tk
    from kivy.config import Config

    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    Config.set('graphics', 'position', 'custom')
    Config.set('graphics', 'left', int(screen_width / 8))
    Config.set('graphics', 'top', int(screen_width / 8))
    Config.set('graphics', 'width', int(screen_width / 4 * 3))
    Config.set('graphics', 'height', int(screen_height / 4 * 3))
    # Config.set('graphics', 'fullscreen', 1)
except ModuleNotFoundError:
    pass

# kivy imports

# To get the graphics, set this as the current working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# local imports (from bimvee)
try:
    from visualiser import VisualiserDvs
    from visualiser import VisualiserFrame
    from visualiser import VisualiserPoint3
    from visualiser import VisualiserPose6q
    from visualiser import VisualiserBoundingBoxes
    from visualiser import VisualiserOpticFlow
    from visualiser import VisualiserImu
    from timestamps import getLastTimestamp
except ModuleNotFoundError:
    if __package__ is None or __package__ == '':
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bimvee.visualiser import VisualiserDvs
    from bimvee.visualiser import VisualiserFrame
    from bimvee.visualiser import VisualiserPoint3
    from bimvee.visualiser import VisualiserPose6q
    from bimvee.visualiser import VisualiserBoundingBoxes
    from bimvee.visualiser import VisualiserOpticFlow
    from bimvee.visualiser import VisualiserImu
    from bimvee.timestamps import getLastTimestamp
    from bimvee.visualiser import VisualiserSkeleton


class ErrorPopup(Popup):
    label_text = StringProperty(None)


class WarningPopup(Popup):
    label_text = StringProperty(None)


class TextInputPopup(Popup):
    label_text = StringProperty(None)


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    load_path = StringProperty(None)


class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)
    save_path = StringProperty(None)

class DictEditor(GridLayout):
    dict = DictProperty(None)

    def on_dict(self, instance, value):
        # 2020_03_10 Sim: Why only import Spinner here? at the top of the file
        # it was causing a crash when starting in a thread - no idea why
        from kivy.uix.spinner import Spinner
        for n, topic in enumerate(sorted(value)):
            check_box = CheckBox()
            self.add_widget(check_box)
            self.add_widget(TextInput(text=str(n)))
            spinner = Spinner(values=['dvs', 'frame', 'pose6q', 'cam', 'imu', 'flowMap', 'skeleton'])
            if 'events' in topic:
                spinner.text = 'dvs'
                check_box.active = True
            elif 'image' in topic or 'depthmap' in topic:
                spinner.text = 'frame'
                check_box.active = True
            elif 'pose' in topic:
                spinner.text = 'pose6q'
                check_box.active = True
            elif 'flow' in topic:
                spinner.text = 'flowMap'
                check_box.active = True
            elif 'imu' in topic:
                spinner.text = 'imu'
                check_box.active = True
            elif 'skeleton' in topic:
                spinner.text = 'skeleton'
                check_box.active = True

            self.add_widget(spinner)
            self.add_widget(TextInput(text=topic))

    def get_dict(self):
        from collections import defaultdict
        out_dict = defaultdict(dict)
        for dict_row in np.array(self.children[::-1]).reshape((-1, self.cols))[1:]:
            if not dict_row[0].active:
                continue
            ch = dict_row[1].text
            type = dict_row[2].text
            data = dict_row[3].text
            out_dict[ch][type] = data
        return out_dict


class TemplateDialog(FloatLayout):
    template = DictProperty(None)
    cancel = ObjectProperty(None)
    load = ObjectProperty(None)


class DataController(GridLayout):
    ending_time = NumericProperty(.0)
    filePathOrName = StringProperty('')
    data_dict = ObjectProperty({})  # A bimvee-style container of channels

    def __init__(self, **kwargs):
        super(DataController, self).__init__(**kwargs)
        self.tmp_cache_path = os.path.join(os.path.dirname(__file__), 'tmp.json')
        try:
            with open(self.tmp_cache_path, 'r') as f:
                self.cache_json = json.load(f)
        except FileNotFoundError:
            with open(self.tmp_cache_path, 'w+') as f:
                self.cache_json = {"LastLoadedPath" : "~"}
                json.dump(self.cache_json, f)

    def update_children(self):
        for child in self.children:
            child.get_frame(self.time_value, self.time_window)

    def add_viewer_and_resize(self, data_dict, channel_name=''):
        visualisers = []
        settings = {}
        for data_type in data_dict.keys():
            if data_type == 'dvs':
                visualiser = VisualiserDvs(data_dict[data_type])
            elif data_type == 'frame':
                visualiser = VisualiserFrame(data_dict[data_type])
            elif data_type == 'pose6q':
                visualiser = VisualiserPose6q(data_dict[data_type])
                channel_name = channel_name + '\nred=x green=y, blue=z'
            elif data_type == 'point3':
                visualiser = VisualiserPoint3(data_dict[data_type])
            elif data_type == 'boundingBoxes':
                visualiser = VisualiserBoundingBoxes(data_dict[data_type])
            elif data_type == 'flowMap':
                visualiser = VisualiserOpticFlow(data_dict[data_type])
            elif data_type == 'imu':
                visualiser = VisualiserImu(data_dict[data_type])
                channel_name = channel_name + '\nred=x green=y, blue=z'
            elif data_type == 'skeleton':
                visualiser = VisualiserSkeleton(data_dict[data_type])
            else:
                print("Warning! {} is not a recognized data type. Ignoring.".format(data_type))
                continue
            
            settings[data_type] = visualiser.get_settings()
            visualisers.append(visualiser)
        if visualisers:
            new_viewer = Viewer()
            new_viewer.title = channel_name
            new_viewer.visualisers = visualisers
            new_viewer.settings = settings
            self.add_widget(new_viewer)

            self.cols = int(np.ceil(np.sqrt(len(self.children))))

    def add_viewer_for_each_channel_and_data_type(self, in_dict, seen_keys=[], recursionDepth=0):
        if isinstance(in_dict, list):
            print('    ' * recursionDepth + 'Received a list - looking through the list for containers...')
            for num, in_dict_element in enumerate(in_dict):
                seen_keys.append(num)
                self.add_viewer_for_each_channel_and_data_type(in_dict_element,
                                                               seen_keys=seen_keys,
                                                               recursionDepth=recursionDepth + 1)
        elif isinstance(in_dict, dict):
            print('    ' * recursionDepth + 'Received a dict - looking through its keys ...')
            for key_name in in_dict.keys():
                print('    ' * recursionDepth + 'Dict contains a key "' + key_name + '" ...')
                if isinstance(in_dict[key_name], dict):
                    seen_keys.append(key_name)
                    if 'ts' in in_dict[key_name]:
                        print('    ' * recursionDepth + 'Creating a new viewer, of type: ' + key_name)
                        self.add_viewer_and_resize(in_dict,
                                                   channel_name=seen_keys[-2])
                        break  # We suppose that all timestamped data are at the same level
                    else:  # recurse through the sub-dict
                        self.add_viewer_for_each_channel_and_data_type(in_dict[key_name],
                                                                       seen_keys=seen_keys,
                                                                       recursionDepth=recursionDepth + 1)
                elif isinstance(in_dict[key_name], list):
                    self.add_viewer_for_each_channel_and_data_type(in_dict[key_name],
                                                                   seen_keys=seen_keys,
                                                                   recursionDepth=recursionDepth + 1)
                else:
                    print('    ' * recursionDepth + 'Ignoring that key ...')

    def on_data_dict(self, instance, value):
        while len(self.children) > 0:
            self.remove_widget(self.children[0])
            print('Removed an old viewer; num remaining viewers: ' + str(len(self.children)))
        if (self.data_dict is None) or not self.data_dict:
            # When using ntupleviz programmatically, pass an empty dict or None
            # to allow the container to be passed again once updated
            return
        self.ending_time = float(getLastTimestamp(self.data_dict))  # timer is watching this
        self.add_viewer_for_each_channel_and_data_type(self.data_dict)

    def dismiss_popup(self):
        if hasattr(self, '_popup'):
            self._popup.dismiss()

    def show_template_dialog(self, topics):
        self.dismiss_popup()
        content = TemplateDialog(template=topics,
                                 cancel=self.dismiss_popup,
                                 load=self.load)
        self._popup = Popup(title="Define template", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_warning_popup(self, message):
        self.dismiss_popup()
        self._popup = WarningPopup(label_text=message,
                                   size_hint=(0.5, 0.5))
        self._popup.open()

    def show_load(self):
        self.dismiss_popup()
        content = LoadDialog(load=self.load,
                             cancel=self.dismiss_popup,
                             load_path=self.cache_json['LastLoadedPath'])
        self._popup = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self, save_fun):
        self.dismiss_popup()
        content = SaveDialog(save=save_fun,
                             cancel=self.dismiss_popup,
                             save_path=self.cache_json['LastLoadedPath'])
        self._popup = Popup(title="Save file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, selection, template=None):
        self.dismiss_popup()
        try:
            from importAe import importAe
        except ModuleNotFoundError:
            from bimvee.importAe import importAe

        # If both path and selection are None than it will try to reload previously given path
        if path is not None or selection is not None:
            if selection:
                self.filePathOrName = os.path.join(path, selection[0])
            else:
                self.filePathOrName = path

        self.update_cache_element("LastLoadedPath", self.filePathOrName)
        try:
            self.data_dict = importAe(filePathOrName=self.filePathOrName, template=template) 
            # TODO Handle rosbag case with template dialog
        except Exception as e:
            self.show_warning_popup('\n'.join(wrap(str(e), width=40)))
        self.update_children()

    def update_cache_element(self, key, val):
        self.cache_json[key] = val
        with open(self.tmp_cache_path, 'w') as f:
            json.dump(self.cache_json, f)


class TimeSlider(Slider):
    def __init__(self, **kwargs):
        super(TimeSlider, self).__init__(**kwargs)
        self.clock = None
        self.speed = 1

    def increase_slider(self, dt):
        self.value = min(self.value + dt / self.speed, self.max)
        if self.value >= self.max:
            if self.clock is not None:
                self.clock.cancel()

    def decrease_slider(self, dt):
        self.value = max(self.value - dt / self.speed, 0.0)
        if self.value <= 0.0:
            if self.clock is not None:
                self.clock.cancel()

    def play_pause(self):
        if self.clock is None:
            self.clock = Clock.schedule_interval(self.increase_slider, 0.001)
        else:
            if self.clock.is_triggered:
                self.clock.cancel()
            else:
                self.clock.cancel()
                self.clock = Clock.schedule_interval(self.increase_slider, 0.001)

    def pause(self):
        if self.clock is not None:
            self.clock.cancel()

    def play_forward(self):
        if self.clock is not None:
            self.clock.cancel()
        self.clock = Clock.schedule_interval(self.increase_slider, 0.001)

    def play_backward(self):
        if self.clock is not None:
            self.clock.cancel()
        self.clock = Clock.schedule_interval(self.decrease_slider, 0.001)

    def stop(self):
        if self.clock is not None:
            self.clock.cancel()
            self.set_norm_value(0)

    def reset(self):
        self.value = 0

    def step_forward(self):
        # self.increase_slider(self.time_window)
        self.increase_slider(0.016)

    def step_backward(self):
        # self.decrease_slider(self.time_window)
        self.decrease_slider(0.016)


class RootWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(RootWidget, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down, on_key_up=self._on_keyboard_up)

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'right':
            self.ids['time_slider'].step_forward()
        if keycode[1] == 'left':
            self.ids['time_slider'].step_backward()
        if keycode[1] == 'spacebar':
            self.ids['time_slider'].play_pause()
        for viewer in self.data_controller.children:
            viewer.transform_allowed = 'shift' in modifiers
            try:
                viewer.label = int(keycode[1][-1])
            except ValueError:
                continue
            # Return True to accept the key. Otherwise, it will be used by the system.
        return True

    def _on_keyboard_up(self, keyboard, keycode):
        for viewer in self.data_controller.children:
            viewer.transform_allowed = False
            # Return True to accept the key. Otherwise, it will be used by the system.
        return True

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        del self._keyboard


class Mustard(App):
    def build(self):
        return RootWidget()

    def setData(self, newDataDict):
        self.root.data_controller.data_dict = {}
        self.root.data_controller.data_dict = newDataDict
