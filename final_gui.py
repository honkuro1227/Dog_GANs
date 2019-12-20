
import PIL.Image
from tkinter import *
from tkinter import ttk, colorchooser
from uwimg import*
import subprocess
import shutil
#import subprocess
#import os
#import io
class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 1
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)#drwaing the line 
        self.c.bind('<ButtonRelease-1>',self.reset)
        self.count=0

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):    #reseting or cleaning the canvas 
        self.old_x = None
        self.old_y = None      

    def changeW(self,e): #change Width of pen through slider
        self.penwidth = e
           

    def clear(self):
        self.c.delete(ALL)
        
    def save_as_png(self):
        self.count+=1
        count=self.count
        self.c.postscript(file = 'test' + '.eps') 
        # use PIL to convert to PNG 
        image1 = PIL.Image.open('test'+'.eps')
        filename='output.jpg'
        image1.save(filename, 'JPEG') 
        im = load_image(filename)
        res = colorize_sobel(im)
        save_image(res, 'testdata')
        save_image(im,'testdata_hed')
        source      = 'testdata.jpg'
        sourceh ='testdata_hed.jpg'
        destination = 'pytorch-CycleGAN-and-pix2pix/datasets/Demo/test/testdata_'+str(count)+'.jpg'
        output= 'pytorch-CycleGAN-and-pix2pix/datasets/Demo/val/'+filename
        shutil.move(source, destination)
        shutil.move(sourceh,'pytorch-hed/crop_image/testdata_hed_'+str(count)+'.jpg')
        shutil.move(filename,output)

    def change_fg(self):  #changing the pen color
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):  #changing the background color canvas
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg
    def run_model(self):
        subprocess.call(["ls", "-l"])
    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
#        Label(self.controls, text='Pen Width:',font=('arial 18')).grid(row=0,column=0)
#        self.slider = ttk.Scale(self.controls,from_= 5, to = 100,command=self.changeW,orient=VERTICAL)
#        self.slider.set(self.penwidth)
#        self.slider.grid(row=0,column=1,ipadx=30)
#        self.controls.pack(side=LEFT)
        
        self.c = Canvas(self.master,width=384,height=384,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)
        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors',menu=colormenu)
        colormenu.add_command(label='Brush Color',command=self.change_fg)
        colormenu.add_command(label='Background Color',command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy)
        optionmenu.add_command(label='Save',command=self.save_as_png)
        

        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('Application')
    root.mainloop()
