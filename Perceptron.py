import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np
import threading

X = []
W = []
d = []

colors = ['#FF0000','#5900FF','#FF8300','#FF00F3','#FBFF00',
            '#00BDFF','#9BFF00','#00FFFF','#0BB000','#9034A7',
            '#8C4D09','#B8B1A9','#FFBC95','#7B9363','#35665F',
            '#0D2F5D','#592795','#D984D1','#545555','#1F6388',]

def select_x_file():
    global X, d

    x_file = askopenfilename()
    X = []

    with open(x_file) as f:
        lines = f.readlines()

    ax.cla()
    print_axis()

    if len(d) == len(lines):
        for i in range(len(lines)):
            v = lines[i].split(' ')
            vector = list(map(float, v))
            ax.plot(vector[0],vector[1],'.', color=colors[int("".join(str(j) for j in d[i]),2)])
            X.append([1,vector[0],vector[1]])
        start_button.config(state=NORMAL)
        canvas.draw()
    else:
        print("The number of patterns does not match the number of desired values.")
    

def select_d_file():
    global  d, W

    d_file = askopenfilename()
    d = []
    W = []
    
    with open(d_file) as f:
        lines = f.readlines()

    if len(lines) > 0:
        for i in range(len(lines)):
            v = lines[i].split(' ')
            vector = list(map(int, v))
            d.append(vector)
        x_file_button.config(state=NORMAL)
        print("The 'd' file has been loaded.")

        for i in range(len(d[0])):
            W.append([random.random(), random.random(), random.random()])
    else:
        print("The 'd' file is empty.")
    

def print_axis():
    global ax
    #Imprimimos los ejes del plano cartesiano
    ejeX = [-5,5]
    ejeY = [-5,5]
    zeros = [0,0]
    ax.plot(ejeX, zeros, 'k')
    ax.plot(zeros, ejeY, 'k')
    plt.xlim(-5,5)
    plt.ylim(-5,5)

def print_line():
    global eta, epoch_inter, X, W, d

    epoch = int(epoch_inter.get())
    error = True

    while epoch and error:
        error = False
        for i in range(len(X)):
            for j in range(len(W)):
                Y = np.dot(X[i],W[j]) >= 0
                e = d[i][j]-Y
                if e != 0:
                    error = True
                    W[j][1] = W[j][1] + (float(eta.get())*e*X[i][1])
                    W[j][2] = W[j][2] + (float(eta.get())*e*X[i][2])
                    W[j][0] = W[j][0] + (float(eta.get())*e*X[i][0])

        ax.cla()

        
        #Imprimimos los puntos en la grafica
        for i in range(len(X)):
            Y = np.dot(W,X[i]) >= 0
            ax.plot(X[i][1],X[i][2],'.', color=colors[int("".join(str(j) for j in list(map(int, Y))),2)])

        for i in range(len(W)):
            m=-W[i][1]/W[i][2]
            b=-W[i][0]/W[i][2]

            plt.axline((X[0][1], (X[0][1]*m)+b), slope=m, color='k', linestyle='--')
        print_axis()
        
        epoch-=1

        canvas.draw()

def clean_screen():
    global X, W, d
    X = []
    d = []
    ax.cla()
    print_axis()
    canvas.draw()
    W = []
    x_file_button.config(state=DISABLED)
    start_button.config(state=DISABLED)
    

#Inizializamos la grafica de matplotlib
fig, ax= plt.subplots(facecolor='#8D96DA')
print_axis()

mainwindow = Tk()
mainwindow.geometry('750x600')
mainwindow.wm_title('Perceptron')
eta = StringVar(mainwindow, 0)
epoch_inter = StringVar(mainwindow, 0)
#Colocamos la grafica en la interfaz
canvas = FigureCanvasTkAgg(fig, master = mainwindow)
canvas.get_tk_widget().place(x=10, y=10, width=580, height=580)

#Colocamos las etiquetas, cuadros de entrada y boton

Eta_label = Label(mainwindow, text = "Eta: ")
Eta_label.place(x=600, y=110)

Eta_entry = Entry(mainwindow, textvariable=eta)
Eta_entry.place(x=600, y=130) 

Epoch_label = Label(mainwindow, text = "Num. Epoch: ")
Epoch_label.place(x=600, y=160)

Epoch_entry = Entry(mainwindow, textvariable=epoch_inter)
Epoch_entry.place(x=600, y=180)

start_button = Button(mainwindow, text="Go!", command=lambda:threading.Thread(target=print_line).start(), state=DISABLED)
start_button.place(x=600, y=230)

x_file_button = Button(mainwindow, text="Select X file", command=select_x_file, state=DISABLED)
x_file_button.place(x=600, y=260)

d_file_button = Button(mainwindow, text="Select D file", command=select_d_file)
d_file_button.place(x=600, y=290)

clean_button = Button(mainwindow, text="Clean", command=clean_screen)
clean_button.place(x=600, y=320)

#Mostramos la interfaz
mainwindow.mainloop()
