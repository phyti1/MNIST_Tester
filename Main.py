

from random import random
import numpy as np

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.uix.progressbar import ProgressBar
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.core.window import Window
from kivy.graphics import *

import tensorflow as tf


class MyPaintWidget(Widget):

    # widget initialized
    def on_kv_post(self, args):
        with self.canvas:
            # self.canvas.add(Color(1., 0, 0))
            # Rectangle(pos = self.pos, size = self.size)
            Line(points=(self.pos[0], self.pos[1], self.pos[0] + self.size[0]/3, self.pos[1]))
            Line(points=(self.pos[0], self.pos[1], self.pos[0], self.pos[1] + self.size[1]/3))
            Line(points=(self.pos[0], self.pos[1] + self.size[1], self.pos[0], self.pos[1] + self.size[1]/3*2))
            Line(points=(self.pos[0], self.pos[1] + self.size[1], self.pos[0] + self.size[0]/3, self.pos[1] + self.size[1]))
            Line(points=(self.pos[0] + self.size[0], self.pos[1] + self.size[1], self.pos[0] + self.size[0], self.pos[1] + self.size[1]/3*2))
            Line(points=(self.pos[0] + self.size[0], self.pos[1] + self.size[1], self.pos[0] + self.size[0]/3*2, self.pos[1] + self.size[1]))
            Line(points=(self.pos[0] + self.size[0], self.pos[1], self.pos[0] + self.size[0]/3*2, self.pos[1]))
            Line(points=(self.pos[0] + self.size[0], self.pos[1], self.pos[0] + self.size[0], self.pos[1] + self.size[1]/3))


    def on_touch_down(self, touch):
        if Widget.on_touch_down(self, touch):
            return True
        # print(touch.x, touch.y)
        with self.canvas:
            # Color(*color, mode='hsv')
            d = 30.
            # Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), size=(d, d))
            # Line(circle=(touch.x, touch.y, 2), width=2)
            touch.ud['current_line'] = Line(points=(touch.x, touch.y), width=25)
        
    def split_list(self, alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts]
            for i in range(wanted_parts) ]

    def on_touch_move(self, touch):
        if 'current_line' in touch.ud:
            touch.ud['current_line'].points += (touch.x, touch.y)
            _image = self.export_as_image()
            sizeX, sizeY = self.size
            _count = len(_image.texture.pixels)
            _columnIndex = 0
            _rowIndex = 0
            _smallPic=np.zeros((1, 28, 28))
            # take every 4th element because pixel consists of (R,B,G,A) elements, take 28 elements per row (self.width is row width in pixel)
            sliceIndex = int(4 * (self.width / 28))
            _reducedRowsBluePixels = _image.texture.pixels[0::sliceIndex]
            # print(len(_image.texture.pixels))
            # print(len(_reducedRowsBluePixels))
            _rows = self.split_list(_reducedRowsBluePixels, wanted_parts=self.height)
            # take every 28th row
            _rows = _rows[0::int(self.height / 28)]
            for rowIndex, row in enumerate(_rows):
                for columnIndex, px in enumerate(row):
                    _smallPic[0][rowIndex][columnIndex] = px/255
            # print(_smallPic)
            applyInput(smallImage=_smallPic)

ValuePredictions = []
BestMatchLabel = Label(font_size='20sp')
ProgressBars = [ProgressBar(max=1) for i in range(10)]
class MyPaintApp(App):

    def clear_canvas(self):
        self.painter.canvas.clear()
        self.painter.on_kv_post(None)

    def build(self):
        root = FloatLayout()
        # MUSS durck 28 dividierbar sein und quadratisch
        self.painter = MyPaintWidget(width = 476, height = 476, x=100, y=100)

        _paintArea = Widget()
        _paintArea.x = 100
        _paintArea.add_widget(self.painter)
        # clear
        clearbtn = Button(text='Clear')
        clearbtn.bind(on_release = lambda a:self.clear_canvas())
        _paintArea.add_widget(clearbtn)
        # train
        _trainBtn = Button(text='train', pos=(100,0))
        _trainBtn.bind(on_release = lambda a:TrainModel())
        _paintArea.add_widget(_trainBtn)

        # top->bottom, left->right
        _settingsStack = BoxLayout(orientation='vertical')
        for i in range(10):
            _boxLayout = BoxLayout(orientation='vertical')
            _boxLayout.add_widget(Label(text='Number ' + str(i) + ':'))
            _boxLayout.add_widget(ProgressBars[i])
            _settingsStack.add_widget(_boxLayout)
        _settingsStack.add_widget(Label(text="Best match: "))

        _settingsStack.add_widget(BestMatchLabel)
        _settingsStack.pos=(600,0)
        _settingsStack.size_hint=(0.2,1)
        root.add_widget(_paintArea)
        root.add_widget(_settingsStack)
        root.spacing = 5
        return root


model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
])
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
def TrainModel():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
    predictions = model(x_train[:1]).numpy()
    print(predictions)
    # The tf.nn.softmax function converts these logits to "probabilities" for each class:
    tf.nn.softmax(predictions).numpy()
    # The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    _newLoss = loss_fn(y_train[:1], predictions).numpy()
    print(_newLoss)
    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
    model.compile(optimizer='Adam', loss=loss_fn, metrics=['accuracy'])
    # The Model.fit method adjusts the model parameters to minimize the loss:
    model.fit(x_train, y_train, epochs=5)
    # The Model.evaluate method checks the models performance, usually on a "Validation-set".
    model.evaluate(x_test,  y_test, verbose=2)
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the TensorFlow tutorials.
    # If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
    probability_model(x_test[:5])

def applyInput(smallImage):
    # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    # array3d = _npSmallImage[..., np.newaxis]
    _predicitons = probability_model.predict(smallImage)
    _valuePredictions = _predicitons[0]
    _bestMatch = np.argmax(_valuePredictions)
    for i, value in enumerate(_valuePredictions):
        ProgressBars[i].value = value
    BestMatchLabel.text = str(_bestMatch)
    print(_valuePredictions)
    # print(_bestMatch)


if __name__ == '__main__':
    # Window.size = (600, 300)
    print("")
    print("")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("")
    print("")
    print("Is GPU avaliable: ", tf.config.list_physical_devices('GPU'))
    #TrainModel()

    MyPaintApp().run()