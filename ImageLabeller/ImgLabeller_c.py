# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 18:09:03 2020

@author: Shivendra
"""

from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import os, cv2
from tkinter.filedialog import askopenfilename
from get_inference import Predict, load_model
import xml.etree.cElementTree as ET
 
class Imglabeller:
        
    def __init__(self, root):
        
        # Variable declaration        
        self.root= root
        self.models= ['frcnn_inception_v2', 'mobilenet_v1', 'ssd_inception_v2']
        self.labels= ['person', 'cat', 'dog', 'bottle', 'chair']
        self.labels2show = []
        self.MODEL_FOLDER = ''
        self.Imgdir= ''
        self.curr_img_idx= -1
        self.imgFiles= []
        self.thresh= 0.7
        self.coords= []
        self.items= []
        self.detection_graph = None
        self.sess = None 
        self.model_tracker = 0
        self.model_var= IntVar()
        self.label_var= [IntVar() for i in range(len(self.labels))]
    
        #Grid.rowconfigure(root, 4, weight=1)
        #Grid.columnconfigure(root, 0, weight=1)
        #Grid.columnconfigure(root, 1, weight=1)
        #Grid.columnconfigure(root, 2, weight=1)
        
        ## UI design
        
        UInamelabel= Label(root, text= " ** ImgLabeller ** ", font= ('courier',15,'bold'), fg= 'white', bg= 'Black')
        thresh_label= Label(root, text= "Threshold", font= ('courier',10,'bold'), fg= 'Black', bg= 'white')
        self.thresh_entry= Entry(root)
        
        Path_button= Button(root, text= "  Open folder  ", command= self.Open_Folder)
        Next_button= Button(root, text= "  Next image  ", command= self.Next)
        Prev_button= Button(root, text= "  Prev image  ", command= self.Prev)
        Save_button= Button(root, text= "Save Annotation", command= self.Save_Annotations)
        Detect_button= Button(root, text= "   Detect   ", command= self.Detect)
        
        self.img_frame= Frame(root, width= 400, height= 300, bg= 'Grey', relief= SUNKEN)
        model_filter_frame= Frame(root, width= 160, height= 150, relief= SUNKEN)
        label_filter_frame= Frame(root, width= 160, height= 225, relief= SUNKEN);
        
        
        UInamelabel.grid(row= 0, column= 2, rowspan= 2, columnspan= 5, padx= (10,10), pady= (10,10))
        thresh_label.grid(row= 10, column= 4, rowspan= 2, columnspan= 2, padx= (10,10), pady= (10,20))
        self.thresh_entry.grid(row= 10, column= 6, rowspan= 2, padx= (10,10), pady= (10,20))
        
        Path_button.grid(row= 2, column= 0, rowspan= 2, columnspan= 2, padx= (10,10))
        Next_button.grid(row= 4, column= 0, rowspan= 2, columnspan= 2, padx= (10,10))
        Prev_button.grid(row= 6, column= 0, rowspan= 2, columnspan= 2, padx= (10,10))
        Save_button.grid(row= 8, column= 0, rowspan= 2, columnspan= 2, padx= (10,10))
        Detect_button.grid(row= 10, column= 2, rowspan= 2, columnspan= 2, padx= (10,10), pady= (10,20))
        
        self.img_frame.grid(row= 2, column= 2, rowspan= 8, columnspan= 5, padx= (10,10), pady= (10,10))
        model_filter_frame.grid(row= 0, column= 7, rowspan= 5, columnspan= 2, padx= (10,10), pady= (20,10))
        label_filter_frame.grid(row= 5, column= 7, rowspan= 7, columnspan= 2, padx= (10,10), pady= (10,20))
        
        Model_filter_label= Label(model_filter_frame, text= "Select Model", font= ('courier',12,'bold'), fg= 'Black')
        Model_filter_label.grid(row= 0, column= 0, rowspan= 2)
        
        for i in range(len(self.models)):
            Checkbutton(model_filter_frame, text= self.models[i], onvalue= i+1, variable= self.model_var,
                        command= self.Model_selection).grid(row= i+2, column= 0, sticky= W)
            
        Label_filter_label= Label(label_filter_frame, text= "Label Filter", font= ('courier',12,'bold'), fg= 'Black')
        Label_filter_label.grid(row= 0, column= 0, rowspan= 2)
        for i in range(len(self.labels)):
            Checkbutton(label_filter_frame, text= self.labels[i], onvalue= i+1, variable= self.label_var[i],
                        command= self.Labels2show).grid(row= i+2, column= 0, sticky= W)

    # open the folder conataining only images            
    def Open_Folder(self):

        for widget in self.img_frame.winfo_children():
            widget.destroy()
            
        path= askopenfilename(filetypes =(("PNG FILE", "*.png"),("JPG FILE", "*.jpg"),
                                          ("JPEG FILE", "*.jpeg")),title = "Choose a Directory")
        self.Imgdir= '/'.join(path.split('/')[:-1])
        curr_img= path.split('/')[-1]
        self.imgFiles= os.listdir(self.Imgdir)
        self.curr_img_idx= self.imgFiles.index(curr_img)
        sel_img= Image.open(self.Imgdir + '/' + self.imgFiles[self.curr_img_idx])
        self.Show_image(sel_img)
        return
    
    # select the model
    def Model_selection(self):
        i = self.model_var.get()
        self.MODEL_FOLDER = self.models[i-1]
        return
    
    # select the labels to detect
    def Labels2show(self):
        idx= [self.label_var[i].get() for i in range(len(self.label_var))]
        self.labels2show = [self.labels[i-1] for i in idx if i != 0]
        return
    
    # Next image selection
    def Next(self): 
        for widget in self.img_frame.winfo_children():
            widget.destroy()
        if self.curr_img_idx != len(self.imgFiles)-1:
            self.curr_img_idx+=1
        next_img= Image.open(self.Imgdir + '/' + self.imgFiles[self.curr_img_idx])
        self.Show_image(next_img)
        return
    
    # Prev image selection
    def Prev(self):
        for widget in self.img_frame.winfo_children():
            widget.destroy()
        if self.curr_img_idx > 0:
            self.curr_img_idx-=1
        prev_img= Image.open(self.Imgdir + '/' + self.imgFiles[self.curr_img_idx]) 
        self.Show_image(prev_img)
        return
    
    # Detection of labels with selected model and threshold
    def Detect(self):

        if self.thresh_entry.get() != '':
            self.thresh= float(self.thresh_entry.get())
            if self.thresh > 1.0:
                self.popupmsg('Enter a valid threshold')
                return
        else:
            self.popupmsg('Enter a threshold value')
            return
        
        if self.model_var.get() == 0:
            self.popupmsg('select a model')
            return
        
        assert self.imgFiles[self.curr_img_idx], 'Not an iamge'
        img= cv2.imread(self.Imgdir +'/' + self.imgFiles[self.curr_img_idx])
        img= cv2.resize(img, (600,1024))
        
        if len(self.labels2show) == 0:
            self.popupmsg('select labels to detect')
            return
    
        if self.model_tracker != self.model_var.get():
            self.detection_graph, self.sess = load_model(self.MODEL_FOLDER)
            self.coords, self.items, img= Predict(img, self.detection_graph, self.sess, self.MODEL_FOLDER,
                                                  self.labels2show, threshold= self.thresh)
            self.model_tracker= self.model_var.get()
        else:
            self.coords, self.items, img= Predict(img, self.detection_graph, self.sess, self.MODEL_FOLDER,
                                                  self.labels2show, threshold= self.thresh)  
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        self.Show_image(img_pil)
        return
    
    # save annotation of detected labels in xml file
    def Save_Annotations(self):      
        filename= self.imgFiles[self.curr_img_idx].split('.')[0] + '.xml'
        annot_path= self.Imgdir+'/annotations/'
        if not os.path.isdir(annot_path):
            os.mkdir(self.Imgdir+'/annotations/')
        
        root= ET.Element("root")
        doc= ET.SubElement(root, "doc")
        
        for i in range(len(self.coords)):
            ET.SubElement(doc, "field"+str(i), name="coordinates").text = str(self.coords[i])
            ET.SubElement(doc, "field"+str(i+1), name="class").text = self.items[i]
        
        tree = ET.ElementTree(root)
        tree.write(annot_path+filename)
        return
    
    # shows the selected image
    def Show_image(self, image):
        for widget in self.img_frame.winfo_children():
            widget.destroy()
        image = image.resize((400, 300), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(image)
        temp_label = Label(self.img_frame, image = img)
        temp_label.pack()
        temp_label.image = img
    
    # popup window in case of any missing parameter           
    def popupmsg(self, msg):
        popup= Tk()
        popup.configure()
        popup.wm_title("Error !")
        label= Label(popup, text=msg, font=("Arial", 12), fg = 'red')
        label.pack(side="top", fill="x", pady=20, padx = 20)
        B1= Button(popup, text="  Okay  ", command = popup.destroy)
        B1.pack(pady=(0,10))
        popup.mainloop()


root= Tk()
def main():
    GUI=Imglabeller(root)
    root.mainloop()

if __name__ == '__main__':
    main()