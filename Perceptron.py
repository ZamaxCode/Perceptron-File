import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np
import threading

X = []
d = []

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
            vector = lines[i].split(' ')
            vector[0] = float(vector[0])
            vector[1] = float(vector[1])
            if d[i]:
                ax.plot(vector[0],vector[1],'.g')
            else:
                ax.plot(vector[0],vector[1],'.r')
            X.append(vector)
        
        start_button.config(state=NORMAL)
        canvas.draw()
    else:
        print("The number of patterns does not match the number of desired values.")
    

def select_d_file():
    global  d

    d_file = askopenfilename()
    d = []
    
    with open(d_file) as f:
        lines = f.readlines()
    if len(lines) > 0:
        for line in lines:
            d.append(int(line))
        x_file_button.config(state=NORMAL)
        print("The 'd' file has been loaded.")
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

def ActivationFunc():
    global w1, w2, theta
    #Generamos el vector F(u) con true y false
    F_u = np.dot(X,[w1,w2])-theta >= 0
    #Retornamos f(u), los valores de X, m y b
    return F_u

def print_line():
    global w1, w2, theta, eta, epoch_inter, X, d

    epoch = int(epoch_inter.get())
    error = True

    while epoch and error:
        error = False
        for i in range(len(X)):
            Y = np.dot(X[i],[w1,w2])-theta >= 0
            e = d[i]-Y
            if e != 0:
                error = True
                w1 = w1 + (float(eta.get())*e*X[i][0])
                w2 = w2 + (float(eta.get())*e*X[i][1])
                theta = theta - (float(eta.get())*e)

        ax.cla()

        Y=[]
        m=-w1/w2
        b=theta/w2
        Y = ActivationFunc()
        #Imprimimos los puntos en la grafica
        for i in range(len(X)):
            #Si la funcion f(u) da 1, entonces el punto se imprime de color verde
            #En caso contrario será rojo
            if Y[i]:
                ax.plot(X[i][0],X[i][1],'.g')
            else:
                ax.plot(X[i][0],X[i][1],'.r')

        #Coloca una linea a partir de un punto dado y la pendiente
        plt.axline((X[0][0], (X[0][0]*m)+b), slope=m, color='b')
        print_axis()

        W1_label.config(text="W1: {:.4f}".format(w1))
        W2_label.config(text="W2: {:.4f}".format(w2))
        Theta_label.config(text="Theta: {:.4f}".format(theta))
          
        epoch-=1

        canvas.draw()

def clean_screen():
    global X, d
    X = []
    d = []
    ax.cla()
    print_axis()
    canvas.draw()
    w1 = random.random()
    w2 = random.random()
    theta = random.random()
    W1_label.config(text="W1: {:.4f}".format(w1))
    W2_label.config(text="W2: {:.4f}".format(w2))
    Theta_label.config(text="Theta: {:.4f}".format(theta))
    x_file_button.config(state=DISABLED)
    start_button.config(state=DISABLED)
    

#Inizializamos la grafica de matplotlib
fig, ax= plt.subplots(facecolor='#8D96DA')
plt.xlim(-2,2)
plt.ylim(-2,2)
print_axis()

mainwindow = Tk()
mainwindow.geometry('750x600')
mainwindow.wm_title('Perceptron')
#Creamos los valores de los pesos y humbral de activacion 
w1 = random.random()
w2 = random.random()
theta = random.random()
eta = StringVar(mainwindow, 0)
epoch_inter = StringVar(mainwindow, 0)
#Colocamos la grafica en la interfaz
canvas = FigureCanvasTkAgg(fig, master = mainwindow)
canvas.get_tk_widget().place(x=10, y=10, width=580, height=580)

#Colocamos las etiquetas, cuadros de entrada y boton
W1_label = Label(mainwindow, text = "W1: {:.4f}".format(w1))
W1_label.place(x=600, y=20) 

W2_label = Label(mainwindow, text = "W2: {:.4f}".format(w2))
W2_label.place(x=600, y=50) 

Theta_label = Label(mainwindow, text = "Theta: {:.4f}".format(theta))
Theta_label.place(x=600, y=80) 

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
