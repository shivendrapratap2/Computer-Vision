import tkinter
from tkinter import *
from tkinter import ttk
import time
import threading

import threading

class MyThread(threading.Thread):
    
    def __init__(self, root):
        super(MyThread, self).__init__()
        self.root= root
        
    def run(self):
        i = 0
        while i < 50000000:
            i+=1
        print ('Done')
        self.root.destroy()


#Define your Progress Bar function, 
def startProcess():
    root2 = Tk()
    ft = ttk.Frame(root2)  
    ft.pack(expand=True, fill=tkinter.BOTH, side=tkinter.TOP)
    pb_hD = ttk.Progressbar(ft, orient='horizontal', mode='indeterminate')
    pb_hD.pack(expand=True, fill=tkinter.BOTH, side=tkinter.TOP)
    pb_hD.start(50)
    
    # This will block while the mainloop runs
    #t1.join()

# Define the process of unknown duration with root as one of the input And once done, add root.quit() at the end.


# Now define our Main Functions, which will first define root, then call for call for "task(root)" --- that's your progressbar, and then call for thread1 simultaneously which will  execute your process_of_unknown_duration and at the end destroy/quit the root.


root = tkinter.Tk()
Button(root, text= "  start process  ", command= lambda : startProcess()).pack()
t1=MyThread(root2)
t1.start()
root.mainloop()
