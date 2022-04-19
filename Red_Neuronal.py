import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter.filedialog import askopenfilename
import numpy as np
import threading

#---------------------------------------------------------------------------------------------------------
#Cargar entradadas X
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
            X.append([1,float(lines[i])])
        X = np.matrix(X)

        ax.plot(np.array(X[:,1]),d, 'o', color="black")
        
        start_button.config(state=NORMAL)
        canvas.draw()
    else:
        print("The number of patterns does not match the number of desired values.")
    
#---------------------------------------------------------------------------------------------------------
#Cargar valores deseados de las entradas
def select_d_file():
    global  d, W

    d_file = askopenfilename()
    d = []
    
    with open(d_file) as f:
        lines = f.readlines()

    if len(lines) > 0:
        for i in range(len(lines)):
            d.append(float(lines[i]))
        x_file_button.config(state=NORMAL)
        print("The 'd' file has been loaded.")
        d = np.array(d)
    else:
        print("The 'd' file is empty.")

def select_training_data(x_compleat, d_compleat):
    i=0
    x_train = []
    d_train = []
    while i < len(x_compleat):
        x_train.append([x_compleat[i,0],x_compleat[i,1]])
        d_train.append(d_compleat[i])
        i+=5
    x_train = np.matrix(x_train)
    d_train = np.array(d_train)
    return x_train, d_train

#---------------------------------------------------------------------------------------------------------
#Inicializar los pesos de las neuronas en funcion a cuantas neuronas haya
def initializeWeights():
    global W_hide, W_out, num_neurons
    W_hide = np.matrix(np.random.rand(num_neurons.get(),2))
    W_out = np.random.rand(num_neurons.get()+1)

#---------------------------------------------------------------------------------------------------------
#Reimprimir la grafica principal con su plano y tamaño correspondiente
def print_axis():
    global ax, ax_e
    ejeX = [-5,5]
    ejeY = [-5,5]
    zeros = [0,0]
    ax.plot(ejeX, zeros, 'k')
    ax.plot(zeros, ejeY, 'k')
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)

#---------------------------------------------------------------------------------------------------------
#Limpiar pantalla y valores
def clean_screen():
    global X, W_hide, W_out, d
    X = []
    d = []
    W_hide = []
    W_out = []

    ax.cla()
    ax_e.cla()
    ax_e.set_xticklabels([])

    print_axis()
    canvas.draw()
    canvas_e.draw()

    x_file_button.config(state=DISABLED)
    start_button.config(state=DISABLED)

#---------------------------------------------------------------------------------------------------------
#Funcion de activacion
def ActivationFunc(x, w, func):
    global a_gui
    a = float(a_gui.get())

    v = np.dot(x,w)

    if func == 0:
        #Logistica
        F_u = 1/(1+np.exp(-a*v))

    elif func == 1:
        #Tangente hiperbolica
        F_u = np.tanh(v)
    
    elif func == 2:
        #Lineal
        F_u = a*v

    return F_u

#---------------------------------------------------------------------------------------------------------
#Funcion de activacion derevada
def ActivationFuncDerivated(Y, func):
    global a_gui
    a = float(a_gui.get())

    if func == 0:
        #Logistica
        F_u = a*Y*(1-Y)

    elif func == 1:
        #Tangente hiperbolica
        F_u = 1-(Y**2)

    elif func == 2:
        #Lineal
        F_u = a

    return F_u

#---------------------------------------------------------------------------------------------------------
#Clasificar datos
def dataClasification():
    global eta_gui, epoch_gui, X, W_hide, W_out, d, func_value
    eta = float(eta_gui.get())
    epoch = int(epoch_gui.get())
    epoch_cont = 0
    error = True
    error_graph = []
    initializeWeights()
    func = func_value.get()
    linear = 2

    X_compleat = X
    d_compleat = d
    X, d = select_training_data(X, d)
    print(X,d)

    while epoch and error:
        error = False
        
        salida_oculta = ActivationFunc(X, np.transpose(W_hide), func)
        X_hide = np.c_[np.ones(len(salida_oculta)),salida_oculta]
        salida = ActivationFunc(X_hide, np.array(W_out).flatten(), linear)
        errors = d - salida

        #capa salida
        delta_out = []
        for i in range(len(X_hide)):
            salida_der = ActivationFuncDerivated(np.array(salida).flatten()[i], linear)
            delta_out.append(salida_der*np.array(errors).flatten()[i])
            W_out = W_out + np.dot(X_hide[i],eta*delta_out[-1])

        #capa oculta
        for i in range(len(X)):
            for j in range(len(W_hide)):
                salida_der = ActivationFuncDerivated(salida_oculta[i,j], func)
                delta_hide = np.array(W_out).flatten()[j+1]*np.array(delta_out).flatten()[i]*salida_der
                W_hide[j] = W_hide[j] + np.dot(X[i],eta*delta_hide)

        ax.cla()

        square_error =  np.average(np.power(errors,2))
        if square_error > float(min_error.get()):
            error = True
        error_graph.append(square_error)

        #Imprimimos los puntos en la grafica
        salida_oculta = ActivationFunc(X_compleat, np.transpose(W_hide), func)
        X_hide = np.c_[np.ones(len(salida_oculta)),salida_oculta]
        salida = ActivationFunc(X_hide, np.array(W_out).flatten(), linear)

        ax.plot(np.array(X_compleat[:,1]),d_compleat, 'o', color="black")
        ax.plot(np.array(X[:,1]),np.array(d).flatten(), 'o', color="green")
        ax.plot(np.array(X_compleat[:,1]),np.array(salida).flatten(), '--', color="red")
 
        ax_e.cla()
        ax_e.plot(error_graph)
        ax_e.set_xticklabels([])
        
        epoch-=1
        epoch_cont+=1
        print_axis()
        canvas.draw()
        canvas_e.draw()

        error_actual.config(text="Error: "+str(square_error))
        epoch_count.config(text="Epoca: "+str(epoch_cont))
    
#---------------------------------------------------------------------------------------------------------
#Inicio de programa + interfaz
X = []
W_hide = []
W_out = []
d = []

colors = ['red', 'green']

#Inizializamos la grafica de matplotlib
fig_e, ax_e= plt.subplots(facecolor='#8D96DA')
ax_e.set_xticklabels([])
fig, ax= plt.subplots(facecolor='#8D96DA')
print_axis()

mainwindow = Tk()
mainwindow.geometry('910x600')
mainwindow.wm_title('Red neuronal multicapa')
eta_gui = StringVar(mainwindow, 0)
epoch_gui = StringVar(mainwindow, 0)
a_gui = StringVar(mainwindow, 0)
min_error = StringVar(mainwindow, 0)
func_value = IntVar(mainwindow, 0)
num_neurons = IntVar(mainwindow, 0)

#Colocamos la grafica en la interfaz
canvas = FigureCanvasTkAgg(fig, master = mainwindow)
canvas.get_tk_widget().place(x=10, y=10, width=580, height=580)

#Colocamos las etiquetas, cuadros de entrada y boton
a_label = Label(mainwindow, text = "A: ")
a_label.place(x=600, y=60)
a_entry = Entry(mainwindow, textvariable=a_gui)
a_entry.place(x=600, y=80)

Eta_label = Label(mainwindow, text = "Eta: ")
Eta_label.place(x=750, y=60)
Eta_entry = Entry(mainwindow, textvariable=eta_gui)
Eta_entry.place(x=750, y=80) 

Epoch_label = Label(mainwindow, text = "Num. Epoch: ")
Epoch_label.place(x=600, y=110)
Epoch_entry = Entry(mainwindow, textvariable=epoch_gui)
Epoch_entry.place(x=600, y=130)

error_label = Label(mainwindow, text = "Min. Error: ")
error_label.place(x=750, y=110)
error_entry = Entry(mainwindow, textvariable=min_error)
error_entry.place(x=750, y=130)

neurons_label = Label(mainwindow, text = "Num. Neuronas: ")
neurons_label.place(x=680, y=160)
neurons_entry = Entry(mainwindow, textvariable=num_neurons)
neurons_entry.place(x=680, y=180)

start_button = Button(mainwindow, text="Go!", command=lambda:threading.Thread(target=dataClasification).start(), state=DISABLED)
start_button.place(x=600, y=210)

x_file_button = Button(mainwindow, text="Select X file", command=select_x_file, state=DISABLED)
x_file_button.place(x=600, y=240)

d_file_button = Button(mainwindow, text="Select D file", command=select_d_file)
d_file_button.place(x=600, y=270)

clean_button = Button(mainwindow, text="Clean", command=clean_screen)
clean_button.place(x=600, y=300)

logistica_rb = Radiobutton(mainwindow, text="Logistica", variable=func_value, value=0)
logistica_rb.place(x=600, y=330)
tangente_rb = Radiobutton(mainwindow, text="Tangente", variable=func_value, value=1)
tangente_rb.place(x=600, y=350)
Lineal_rb = Radiobutton(mainwindow, text="Lineal", variable=func_value, value=2)
Lineal_rb.place(x=600, y=370)

epoch_count = Label(mainwindow, text="Epoca: 0")
epoch_count.place(x=600, y=420)

error_actual = Label(mainwindow, text="Error: 0")
error_actual.place(x=700, y=420)

canvas_e = FigureCanvasTkAgg(fig_e, master = mainwindow)
canvas_e.get_tk_widget().place(x=600, y=460, width=300, height=130)

#Mostramos la interfaz
mainwindow.mainloop()
