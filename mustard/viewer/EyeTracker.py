import numpy as np
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.widget import Widget
from kivy.properties import BooleanProperty, NumericProperty, ListProperty


class EyeTracker(Widget):
    mouse_over = BooleanProperty(False)
    iris_x_center = NumericProperty(None)
    iris_y_center = NumericProperty(None)
    gaze_line_x = NumericProperty(None)
    gaze_line_y = NumericProperty(None)
    center_x = NumericProperty(None)
    center_y = NumericProperty(None)
    ellipse_points = ListProperty()

    def __init__(self,
                 phi=None,
                 theta=None,
                 center_x=None,
                 center_y=None,
                 radius=None,
                 pointcloud=None,
                 **kwargs):
        super(EyeTracker, self).__init__(**kwargs)

        if phi is not None:
            r = 0.5
            c = np.sqrt(1 - r ** 2)

            self.ellipse_points = []
            for alpha in np.arange(-np.pi, np.pi, np.pi/180):
                xhat = r*np.cos(alpha)
                yhat = r*np.sin(alpha)
                x = radius*(xhat*np.cos(phi)+c*np.sin(phi)) + center_x
                y = radius*((xhat*np.sin(phi)-c*np.cos(phi))*np.sin(theta)+yhat*np.cos(theta)) + center_y
                self.ellipse_points.append(x)
                self.ellipse_points.append(y)

            self.iris_x_center = int(radius * (c*np.sin(phi)) + center_x)
            self.iris_y_center = int(radius * (c*(-np.sin(theta)*np.cos(phi))) + center_y)
            self.gaze_line_x = int(radius * 1.5 * (c*np.sin(phi)) + center_x)
            self.gaze_line_y = int(radius * 1.5 * (c*(-np.sin(theta)*np.cos(phi))) + center_y)
            self.center_x = int(center_x)
            self.center_y = int(center_y)
            with self.canvas:
                Color(1, 0, 0, 1)
                Line(points=self.ellipse_points)
                Color(0, 1, 0, 1)
                Ellipse(pos=(self.iris_x_center - 3, self.iris_y_center - 3), size=(6, 6))
                Ellipse(pos=(self.center_x - 3, self.center_y - 3), size=(6, 6))
                Color(0, 0, 1, 1)
                Line(points=(self.center_x, self.center_y, self.gaze_line_x, self.gaze_line_y))
        if pointcloud is not None:
            with self.canvas:
                Color(0.57, 0.47, 0.91, 0.5)
                for x, y in pointcloud:
                    Ellipse(pos=(x - 3, y - 3), size=(6, 6))
