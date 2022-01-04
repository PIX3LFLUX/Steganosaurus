from kivy.app import App 
from kivy.uix.widget import Widget 
from kivy.lang import Builder 
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from FFT_stego import *
from kivy.config import Config
from kivy.properties import ObjectProperty
import os
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.uix.checkbox import CheckBox
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.bubble import Bubble, BubbleButton
from android.permissions import request_permissions, Permission
request_permissions([Permission.WRITE_EXTERNAL_STORAGE])
request_permissions([Permission.READ_EXTERNAL_STORAGE])

__version__ = "1.0.3"

#from kivy.core.window import Window

#Config.set('graphics', 'resizable', '0') #0 being off 1 being on as in true/false
#Config.set('graphics', 'height', '720')
#Config.set('graphics', 'width', '330')


#define our different Screens
class MainMenu(Screen):
    pass


        

#Options for Encoding
class EncodeStegoOptions(Screen):
    def checkbox_click(self, instance, value):
        global optCutGlobal
        if value is True:
            print("Checkbox Checked")
            optCutGlobal=value
        else:
            print("Checkbox Unchecked")
            optCutGlobal=value
    def pushRecursiveCounter(self):
        global recursive_cntGlobal 
        recursive_cntGlobal=int(self.recursiveCounter.text)
        print("Recursive Counter: ", recursive_cntGlobal)

#Verschlüsseln
class EncodeStego(Screen):

    message =ObjectProperty(None)

    def pushMessage(self):
        global secretText
        secretText=self.message.text

        print("Message: ", secretText)

class ShowCut(Screen):
    pass


#Entschlüsseln
class DecodeStego(Screen):

    cut = ObjectProperty(None)

    def pushCut(self):
        global cutGlobal
        cutGlobal=self.cut.text
        cutGlobal= float(cutGlobal)
        print("Cut:",cutGlobal)

        #return cutGlobal
#Load Decoded File and show message
class LoadDialog(Screen):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    text_input = ObjectProperty(None)

    def load(self, path, filename):
        with open(os.path.join(path, filename[0])) as stream:
            #self.text_input.text = steg_decode(filename[0])
            print("Filename:",filename[0])
            print("Cut Global:",cutGlobal)
            print(steg_decode_simple(filename[0],cutGlobal))
            
            
            

#Create and Save Encoded File
class SaveDialog(Screen):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

    def save(self, path, filename):
        global cutStego
        with open(os.path.join(path, filename[0]))as stream:
            

            

            print("TEST")
            print("TEST")
            print("Filename:",filename[0])
            print("Text:",secretText)
            print("Optcut enabled?:",optCutGlobal)
            print("Recursive Counter:",recursive_cntGlobal)
            cut=steg_encode_simple(filename[0],secretText,optCutGlobal,recursive_cntGlobal)
            print("Cut:",cut)
            screen = self.manager.get_screen('showcut')
            screen.ids['calculatedCut'].text = str(cut)

            
            


class Root(Screen):
    pass
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    
#define our WindowManager class which is responsible in switching screens
class WindowManager(ScreenManager):
    pass      

#designate our .kv design file
kv= Builder.load_file('StegoApp.kv')


class AwesomeApp(App):
    def build(self):
        #Window.clearcolor =(.5, .5 , .5 ,1)
        return kv


Factory.register('Root', cls=Root)
Factory.register('LoadDialog', cls=LoadDialog)
Factory.register('SaveDialog', cls=SaveDialog)

if __name__ =='__main__':
    AwesomeApp().run()
