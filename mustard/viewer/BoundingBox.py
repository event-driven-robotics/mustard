from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, StringProperty


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
