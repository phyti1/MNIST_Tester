
import os
from random import random
import numpy as np
from enum import Enum
import torchvision
from torchvision import datasets, transforms

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
from kivy.uix.dropdown import DropDown
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.core.window import Window
from kivy.graphics import *

import matplotlib
# matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
# from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds

class DatasetsE(Enum):
    Zero = 0
    MNSIT = 1
    EMNIST = 2

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
            _myPaintApp.applyInput(smallImage=_smallPic)

class DatasetDropDown(DropDown):
    def on_kv_post(self, args):
        _myPaintApp.ActiveDataset = args
        btn = Button(text='MNIST', size_hint_y=None, height=44)
        def apply_MNIST(Button):
            self.select(Button.text)
            _myPaintApp.ActiveDataset = DatasetsE.MNSIT
        btn.bind(on_release=apply_MNIST)
        self.add_widget(btn)
        btn = Button(text='EMNIST', size_hint_y=None, height=44)
        def apply_EMNIST(Button):
            self.select(Button.text)
            _myPaintApp.ActiveDataset = DatasetsE.EMNIST
        btn.bind(on_release=apply_EMNIST)            
        self.add_widget(btn)

ValuePredictions = []
BestMatchLabel = Label(font_size='20sp')
ProgressBars = [ProgressBar(max=1) for i in range(10)]

class MyPaintApp(App):

    ActiveDataset = DatasetsE.Zero
    _epochs = 5
    _datasetPath = "C:\\tensorflow_datasets\\"
    _smallImage = []

    def SetDatabasePath(self, instance, value):
        self._datasetPath = value

    def SetEpochs(self, instance, value):
        self._epochs = int(value)

    def clear_canvas(self):
        self.painter.canvas.clear()
        self.painter.on_kv_post(None)
    
    def build_plot(self):
            # display output
            _smallPic = np.array(self._smallImage, dtype='float')
            if(len(_smallPic) == 0):
                return
            pixels = _smallPic.reshape((28, 28))
            _myPaintApp.plot = plt.imshow(pixels, cmap='gray')
            plt.show()
    
    def build(self):
        root = FloatLayout()
        # MUSS durch 28 dividierbar und quadratisch sein
        self.painter = MyPaintWidget(width = 476, height = 476, x=120, y=10)

        _paintArea = Widget()
        _paintArea.x = 110
        _paintArea.add_widget(self.painter)
        # path
        _datasetPathInput = TextInput(text=self._datasetPath, multiline=False, pos=(10,500), height=28, width=500)
        _datasetPathInput.bind(text=self.SetDatabasePath)
        _paintArea.add_widget(_datasetPathInput)
        # clear
        _clearbtn = Button(text='Clear', pos=(10, 10))
        _clearbtn.bind(on_release = lambda a:self.clear_canvas())
        _paintArea.add_widget(_clearbtn)
        # train
        _trainBtn = Button(text='train', pos=(10,400), height=44)
        _trainBtn.bind(on_release = lambda a:self.TrainModel())
        _paintArea.add_widget(_trainBtn)
        # epochs
        _paintArea.add_widget(Label(text='Epochs/training\niterations:', pos=(10,350), height=55))
        _epochInput = TextInput(text=str(self._epochs), multiline=False, pos=(80,350), height=28, width=28)
        _epochInput.bind(text=self.SetEpochs)
        _paintArea.add_widget(_epochInput)
        # dataset selector
        _dropdown = DatasetDropDown()
        _selectButton = Button(text='Select Dataset', size_hint=(None, None), pos=(10,300), height=44)
        _selectButton.bind(on_release=_dropdown.open)
        _dropdown.bind(on_select=lambda instance, x: setattr(_selectButton, 'text', x))
        _paintArea.add_widget(_selectButton)
        _plotButton = Button(text='show plot', pos=(10,110), height=44)
        _plotButton.bind(on_release=lambda a: self.build_plot())
        _paintArea.add_widget(_plotButton)
        # _paintArea.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        # top->bottom, left->right
        _settingsStack = BoxLayout(orientation='vertical')
        for i in range(10):
            _boxLayout = BoxLayout(orientation='vertical')
            _boxLayout.add_widget(Label(text='Number ' + str(i) + ':'))
            _boxLayout.add_widget(ProgressBars[i])
            _settingsStack.add_widget(_boxLayout)
        _settingsStack.add_widget(Label(text="Best match: "))

        _settingsStack.add_widget(BestMatchLabel)
        _settingsStack.pos=(620,0)
        _settingsStack.size_hint=(0.2,1)
        root.add_widget(_paintArea)
        root.add_widget(_settingsStack)
        root.spacing = 5
        return root


    model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(0)
    ])
    def set_model(self, denisty):
        self.model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(denisty)
        ])
        # Save the entire model as a SavedModel.
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model.save(model_dir)

    def flipRotate(self, input):
        # rot90(x, 3) -> counter clock wise
        return np.array([np.fliplr(np.rot90(val, 3)) for val in input])

    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    def TrainModel(self):
        #region mnist
        if(self.ActiveDataset == DatasetsE.MNSIT):
            mnist = tf.keras.datasets.mnist
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
        #endregion

        #region emnist
        #region old, slow
        # batch_size=-1 to get the full dataset in NumPy arrays from the returned tf.Tensor object
        # https://stackoverflow.com/questions/48532761/letters-in-emnist-training-dataset-are-rotated-and-little-vague
        # emnistTrain, emnistTest = tfds.load(name="emnist", split=[tfds.Split.TRAIN, tfds.Split.TEST], data_dir="D:\\tensorflow_datasets\\", batch_size=-1)
        # emnistTrain = tfds.as_numpy(emnistTrain)
        # emnistTest = tfds.as_numpy(emnistTest)
        # # seperate the x and y and removes unneccessary 4th dimension
        # x_train, y_train = emnistTrain["image"], emnistTrain["label"] 
        # x_test, y_test = emnistTest["image"], emnistTest["label"]
        # # https://stackoverflow.com/questions/37152031/numpy-remove-a-dimension-from-np-array
        # x_train = x_train[:, :, :, 0]
        # x_test = x_test[:, :, :, 0]
        #endregion
        elif(self.ActiveDataset == DatasetsE.EMNIST):
            _transformFix = torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    torchvision.transforms.ToTensor()
                ])
            emnistTrain = torchvision.datasets.EMNIST(self._datasetPath, download=True, split='byclass', train=True)
            emnistTest = torchvision.datasets.EMNIST(self._datasetPath, download=True, split='byclass', train=False)
            x_train, y_train, x_test, y_test = self.flipRotate(emnistTrain.train_data.numpy()), emnistTrain.train_labels.numpy(), self.flipRotate(emnistTest.test_data.numpy()), emnistTest.test_labels.numpy()
        #endregion
        else:
            print("Error: no database was selected")
            return
        # get values from 0 to 1
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # apply dense (result/label range) to model, reapply probablilty_model
        _maxLabelValue = int(y_train.max())
        self.set_model(denisty=_maxLabelValue + 1)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        # For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.
        predictions = self.model(x_train[:1]).numpy()
        print(predictions)
        # The tf.nn.softmax function converts these logits to "probabilities" for each class:
        tf.nn.softmax(predictions).numpy()
        # The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        _newLoss = loss_fn(y_train[:1], predictions).numpy()
        print(_newLoss)
        # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        self.model.compile(optimizer='Adam', loss=loss_fn, metrics=['accuracy'])
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                        save_weights_only=True,
                                                        verbose=1)
        
        # The Model.fit method adjusts the model parameters to minimize the loss:
        self.model.fit(x_train, y_train, epochs=self._epochs, callbacks=[cp_callback])
        # The Model.evaluate method checks the models performance, usually on a "Validation-set".
        self.model.evaluate(x_test,  y_test, verbose=2)
        self.probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        # The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the TensorFlow tutorials.
        # If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
        self.probability_model(x_test[:5])

    def applyInput(self, smallImage):
        # probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        # array3d = _npSmallImage[..., np.newaxis]
        self._smallImage = smallImage
        _predicitons = self.probability_model.predict(smallImage)
        _valuePredictions = _predicitons[0]
        try:
            _bestMatch = np.argmax(_valuePredictions)
        except:
            _bestMatch = "---"
            return
        for i, value in enumerate(_valuePredictions):
            if(i < 10):
                ProgressBars[i].value = value
        if(_bestMatch < 10):
            # number 0-9
            BestMatchLabel.text = str(_bestMatch)
        elif(_bestMatch < 36):
            # capital letter (-10 to get index 0-25, +65 for ascii position)
            BestMatchLabel.text = chr(_bestMatch - 10 + 65)
        else:
            # small letter (-10 - 26 to get index 0-25, + 97 to get ascii position)
            BestMatchLabel.text = chr(_bestMatch - 10 - 26 + 97)

        # print(_valuePredictions)
        # print(_bestMatch)

_myPaintApp = MyPaintApp()
checkpoint_dir = "./saved/cp-{epoch:04d}.ckpt"
model_dir = "./saved/saved_model/"
if __name__ == '__main__':
    # Window.size = (600, 300)
    print("")
    print("")
    # Note: Use tf.config.experimental.list_physical_devices('GPU') to confirm that TensorFlow is using the GPU.
    print("Is GPU avaliable: ", tf.config.list_physical_devices('GPU'))
    try:
        _myPaintApp.model = tf.keras.models.load_model(model_dir)
        print("Restored model: " + model_dir)
        latest = tf.train.latest_checkpoint(os.path.dirname(checkpoint_dir))
        # Load the previously saved weights
        _myPaintApp.model.load_weights(latest)
        print("Restored checkpoint: " + latest)
        _myPaintApp.probability_model = tf.keras.Sequential([_myPaintApp.model, tf.keras.layers.Softmax()])
        print("applied probability model")
    except:
        pass
    #TrainModel()

    _myPaintApp.run()