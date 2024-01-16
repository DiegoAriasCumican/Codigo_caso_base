import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from pandapower.pf.runpp_3ph import runpp_3ph
import pandapower.plotting as plot
from pandapower.plotting.plotly import simple_plotly
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

##########################################################################################################################################
#Definicion de los parametros del panel solar jinko
Potencia_nominal = 625  
Eficiencia_conversion = 0.22  
G_stc = 1000 
T_stc = 25  
alpha = -0.29/100 

##########################################################################################################################################

#Se crea el objeto con el dataframe de la red
net = pn.create_cigre_network_lv()

##########################################################################################################################################

#De la pregunta anterior, se toma el dataframe, con los datos de las cargas residenciales, comerciales, e industriales
Modelo_Cargas = pd.DataFrame({
    
    'Carga_Residencial': [30,27, 25,23, 20,20, 22,30,40,43,45,51,50,55,60,60,55,50,65,85,100,90,75,55,30],
    
    'Carga_Industrial': [35,33,30,27,25,30,40,55,75,90,100,100,90,100,100,95,90,75,63,55,50,45,40,38,35],
    
    'Carga_Comercial': [22,21,20,21,22,23,25,35,50,65,80,85,90,93,93,88,85,90,100,80,70,60,50,30,20],

    'Hora': ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", 
             "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00",
             "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00"]
})

##########################################################################################################################################

#Se toman los datos de las condiciones climaticas
Data_clima = pd.DataFrame({
    "Hora":             ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00",
                        "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00",
                        "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00"],
    "irradiancia_enero":[0, 0, 0, 0, 0, 0, 0, 0.98415, 132.9594, 289.1711, 449.3114,
                         547.806, 655.0924, 708.4984, 699.1695, 645.9715, 538.3517, 355.7143,
                         173.7246, 6.4881, 0, 0, 0, 0, 0],
    "temperatura_enero":[14.23279, 13.55445, 12.96318, 12.46165, 11.97344, 11.4217, 11.59185, 12.49347, 
                         13.91809, 15.68968, 17.50937, 19.33868, 20.94959, 22.2611, 23.01217, 23.22459, 
                         22.90132, 22.23493, 21.23292, 19.89685, 18.13699, 16.8129, 15.75275, 14.93132, 14.5716]
})
Hora= ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", 
             "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00",
             "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00"]
#Se genera en el dataframe, los calculos de las potencias
Data_clima["potencia_real"]=(Potencia_nominal/G_stc)*(Data_clima["irradiancia_enero"])*(1+alpha*(Data_clima["temperatura_enero"]-T_stc))

##########################################################################################################################################

#Se estipula la hora como el indice del dataframe "Modelo_Cargas"
Modelo_Cargas.set_index('Hora', inplace=True)
Data_clima.set_index('Hora', inplace=True)

##########################################################################################################################################

#Se generan indices para los interruptores
# indice_interruptor_S_2 = 0
# indice_interruptor_S_3 = 1

# # Cambiar el estado de los interruptores S2 y S3 a "abierto" (OFF)
# net.switch.loc[indice_interruptor_S_2, 'closed'] = False
# net.switch.loc[indice_interruptor_S_3, 'closed'] = False

##########################################################################################################################################

#Se generan dataframes para almacenar los datos con los consumos en las cargas y barras del alimentador residencial
df_res_bus = pd.DataFrame()
Dataframe_Voltaje = pd.DataFrame()
Dataframe_Corriente=pd.DataFrame()

#se generan dos diccionarios vacios, con el objetivo de poder ordenar todos los datos generados, e ingresarlos al dataframe
dicc1= {}
dicc2= {}

#Se genera una copia del estado de carga, con el objetivo de que en cada ciclo for, 
#se actualice esta copia, y no el valor actual modificado del estado de carga
Estado_Carga = net.load.copy()
##########################################################################################################################################

#Se generan los objetos con las generaciones, y se instalan en todas las barrras del alimentador residencial 
#(primeras 19 barras)
# for barra in range(0,20):
#     nombre_generacion = f"Generacion_Barrra_{barra}"
#     pp.create_sgen(net, bus=barra, p_mw=0,q_mvar=0, name=nombre_generacion)

#Se crean los objetos para las generaciones
pp.create_sgen(net, bus=2, p_mw=0.0, name="Generador0")
pp.create_sgen(net, bus=12, p_mw=0.0, name="Generador1")
pp.create_sgen(net, bus=16, p_mw=0.0, name="Generador2")
pp.create_sgen(net, bus=17, p_mw=0.0, name="Generador3")
pp.create_sgen(net, bus=18, p_mw=0.0, name="Generador4")
pp.create_sgen(net, bus=19, p_mw=0.0, name="Generador5")

##########################################################################################################################################
#Se itera para poder conseguir los valores de las cargas segun el consumo en funcion de la hora
for Hora,  row in Modelo_Cargas.iterrows():
    #Se hace uso de la copia para que en cada iteracion se actualice el valor original, y no el valor modificado posterior al ciclo for
    net.load=Estado_Carga
    net.load['Hora'] = Hora
    Estado_Carga = net.load.copy()
    
    #Se generan los valores nuevos de las cargas, en funcion del consumo a lo largo del dia
    for indice_x, load in net.load.iterrows():
        #Se filtran las barras con cargas, en funcion del nombre, y se toman en consideracion solo las barras del sector residencial
        if 'R' in str(load['name']):
            Potencia_Activa_Residencial_Nueva=net.load.at[indice_x, 'p_mw']
            Potencia_Reactiva_Residencial_Nueva=net.load.at[indice_x, 'q_mvar']
            #Se actualiza la lista de potencias activas y reactivas en los datos de la red, utilizando el dataframe de carga residencial
            net.load.at[indice_x, 'p_mw']= Potencia_Activa_Residencial_Nueva*row["Carga_Residencial"]/100
            net.load.at[indice_x, 'q_mvar']=Potencia_Reactiva_Residencial_Nueva*row["Carga_Residencial"]/80
        elif 'C' in str(load['name']):
            Potencia_Activa_Comercial_Nueva=net.load.at[indice_x, 'p_mw']
            Potencia_Reactiva_Comercial_Nueva=net.load.at[indice_x, 'q_mvar']
            #Se actualiza la lista de potencias activas y reactivas en los datos de la red, utilizando el dataframe de carga residencial
            net.load.at[indice_x, 'p_mw']= Potencia_Activa_Comercial_Nueva*row["Carga_Comercial"]/100
            net.load.at[indice_x, 'q_mvar']=Potencia_Reactiva_Comercial_Nueva*row["Carga_Comercial"]/100
        elif 'I' in str(load['name']):
            Potencia_Activa_Industrial_Nueva=net.load.at[indice_x, 'p_mw']
            Potencia_Reactiva_Industrial_Nueva=net.load.at[indice_x, 'q_mvar']
            #Se actualiza la lista de potencias activas y reactivas en los datos de la red, utilizando el dataframe de carga residencial
            net.load.at[indice_x, 'p_mw']= Potencia_Activa_Industrial_Nueva*row["Carga_Industrial"]/100
            net.load.at[indice_x, 'q_mvar']=Potencia_Reactiva_Industrial_Nueva*row["Carga_Industrial"]/100
   
    
    #Se pasa las unidades del generador a una potencia en MW
    Potencia_generador_hora=Data_clima["potencia_real"].loc[f'{Hora}']/1e6
    net.sgen.at[2, 'p_mw'] = 444*Potencia_generador_hora
    pp.runpp(net,max_iteration=100)
   
    print("#########################")
    print(net.res_bus.loc[16, 'vm_pu'])
    barra=18
    tension_barra_16 = net.res_bus.loc[16, 'vm_pu']
    # net.sgen.at[2, 'p_mw'] = 444*Potencia_generador_hora
    #Se genera la modificacion de las potencias
    # net.sgen.at[0, 'p_mw'] = 44*Potencia_generador_hora
    # net.sgen.at[1, 'p_mw'] = 44*Potencia_generador_hora
        
    a_1=0.999
    a_2=0
    if net.res_bus.loc[16, 'vm_pu']>1.05:
        for i in range(1, 301):
            if net.res_bus.loc[16, 'vm_pu']>1.05:
                a_1=a_1-0.0015
                a_2=444*Potencia_generador_hora*a_1
                net.sgen.at[2, 'p_mw'] =a_2
                pp.runpp(net,max_iteration=1000)
                print("-------------------------")
                print(net.sgen.at[2, 'p_mw'])
                print(net.res_bus.loc[16, 'vm_pu'])
            else:
                a_1=a_1+0.0015
                a_2=444*Potencia_generador_hora*a_1
                net.sgen.at[2, 'p_mw'] =a_2
                pp.runpp(net,max_iteration=1000)
                # net.sgen.at[2, 'p_mw']=a_2
                # pp.runpp(net,max_iteration=100)
    print("elvalorsera::")            
    print(a_2)
    print(net.res_bus.loc[16, 'vm_pu'])    
    # print(net.bus.at[16, "vn_kv"])
    # net.sgen.at[3, 'p_mw'] = 44*Potencia_generador_hora
    # net.sgen.at[4, 'p_mw'] = 44*Potencia_generador_hora
    # net.sgen.at[5, 'p_mw'] = 352*Potencia_generador_hora  
    #se corre el flujo de potencia en cada iteracion para las horas 
    # pp.runpp(net,max_iteration=100)
    
    # print(Hora)
    # print(net.res_bus)
    # print(net.sgen)
    #se toman los datos de las barras y las lineas
    Flujo_Potencia_Barras = net.res_bus
    Flujo_Potencia_Lineas = net.res_line
    
    
    Voltaje_barra=net.res_bus['vm_pu']
    Corriente_Linea=net.res_line["i_ka"]
    
    #Se agregan los valores al dataframe vacio para tener los datos
    Dataframe_Voltaje['Voltaje'] =  Voltaje_barra
    Dataframe_Corriente['Corriente'] =  Corriente_Linea
    
    #en cada iteracion se genera un nombre con la hora respectiva
    nombre_columna = f'Columna_{Hora}'
    
    #se agrega al diccionario hecho anteriormente y se incluye en un nuevo dataframe
    dicc1[nombre_columna] = Voltaje_barra
    dicc2[nombre_columna]= Corriente_Linea
    
    #Se genera un nuevo dataframe con los valores respectivos de los voltajes en los buses y las corrientes de linea
    df1 = pd.DataFrame(dicc1)
    df2 = pd.DataFrame(dicc2)

#Se generan dataframes para poder graficar los comportamientos de los diferentes alimentadores
#(los numeros impares son las graficas de voltaje en las diferentes barras, y los pares para las corrientes de linea)
#df_limitado1,df_limitado3,df_limitado5 = voltaje en barrasalimentadores 1,2,3
#df_limitado2,df_limitado4,df_limitado6 = Corriente de linea alimentadores 1,2,3

df_limitado1 = pd.DataFrame()
df_limitado2 = pd.DataFrame()

df_limitado3 = pd.DataFrame()
df_limitado4 = pd.DataFrame()

df_limitado5 = pd.DataFrame()
df_limitado6 = pd.DataFrame()

#Se llenan los dataframes del primer alimentador
filas1 = 20
filas2 = 17

for columna in df1.columns:
    df_limitado1[columna] = df1[columna].head(filas1)

for columna in df1.columns:
    df_limitado2[columna] = df2[columna].head(filas2)
    
#Se filtran los valores de los alimentadores 2 y 3
alimentador16=df1.iloc[16]
filas_3_a_agregar = df1.iloc[20:23] 
filas_4_a_agregar = df2.iloc[17] 
filas_5_a_agregar = df1.iloc[23:44] 
filas_6_a_agregar = df2.iloc[18:37] 

# Agrega las filas al DataFrame nuevo junto con todas las columnas
df_limitado3 = df_limitado3.append(filas_3_a_agregar)
df_limitado4 = df_limitado4.append(filas_4_a_agregar)  
df_limitado5 = df_limitado5.append(filas_5_a_agregar) 
df_limitado6 = df_limitado6.append(filas_6_a_agregar) 


#Se grafica la linea para poder ver los indices de las barras
plot.simple_plotly(net)


#Finalmente, se grafican las magnitudes de voltaje y corriente
plt.figure(figsize=(30,12))
sns.heatmap(df_limitado1, annot=True, cmap="coolwarm", fmt=".5f")
plt.title('Heatmap de magnitudes de voltaje por barra y hora del alimentador 1')
plt.xlabel('Hora')
plt.ylabel('Barra')
plt.show()


# #Finalmente, se grafican las magnitudes de voltaje y corriente
# plt.figure(figsize=(30,12))
# sns.heatmap(df_limitado3, annot=True, cmap="coolwarm", fmt=".5f")
# plt.title('Heatmap de magnitudes de voltaje por barra y hora del alimentador 2')
# plt.xlabel('Hora')
# plt.ylabel('Barra')
# plt.show()


# #Finalmente, se grafican las magnitudes de voltaje y corriente
# plt.figure(figsize=(30,12))
# sns.heatmap(df_limitado5, annot=True, cmap="coolwarm", fmt=".5f")
# plt.title('Heatmap de magnitudes de voltaje por barra y hora del alimentador 3')
# plt.xlabel('Hora')
# plt.ylabel('Barra')
# plt.show()


plt.figure(figsize=(30,12))
sns.heatmap(df_limitado2, annot=True, cmap="coolwarm", fmt=".5f")
plt.title('Heatmap de magnitudes de Corriente por linea y hora del alimentador 1')
plt.xlabel('Hora')
plt.ylabel('Indice de Linea')
plt.show()

# plt.figure(figsize=(30,12))
# sns.heatmap(df_limitado4, annot=True, cmap="coolwarm", fmt=".5f")
# plt.title('Heatmap de magnitudes de Corriente por linea y hora del alimentador 2')
# plt.xlabel('Hora')
# plt.ylabel('Indice de Linea')
# plt.show()

# plt.figure(figsize=(30,12))
# sns.heatmap(df_limitado6, annot=True, cmap="coolwarm", fmt=".5f")
# plt.title('Heatmap de magnitudes de Corriente por linea y hora del alimentador 3')
# plt.xlabel('Hora')
# plt.ylabel('Indice de Linea')
# plt.show()



#Adicionalmente, se generan graficas en 3d, para ver la evolucion de estas barras
num_barras, num_horas = df_limitado1.shape

# Crear una figura 3D
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

# Crear una cuadrícula de puntos para las barras y las horas
barras, horas = np.meshgrid(range(num_barras), range(num_horas))

# Obtener los valores de voltaje del DataFrame en forma de un arreglo unidimensional
voltajes = df_limitado1.values.flatten()

#Se generan arreglos con las dimensiones de las horas y las barras
horas = np.arange(num_horas)
barras= np.arange(num_barras)
valores_voltajes_barras=[]

#Se grafica cada curva en un grafico en conjunto
for indice, fila in df_limitado1.iterrows():
    lista_fila = fila.tolist() 
    ax.plot(np.full(num_horas, indice),horas , lista_fila, label=columna)

# Configurar etiquetas y leyendas
ax.set_xlabel('Indice de Barra')
ax.set_ylabel('Horas')
ax.set_zlabel('Voltaje')
ax.set_title('Voltajes en diferentes barras a lo largo del tiempo en el alimentador 1')

#Se genera una linea de codigo para poder modificar la vista de la grafica
ax.view_init(elev=50, azim=-30)

# Mostrar el gráfico 3D
plt.show()


################################################################################################################

#Se define el tamaño de la figura
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

#Se establece las dimensiones de los vectores
num_barras2, num_horas2 = df_limitado2.shape

# Obtener los valores de voltaje del DataFrame
voltajes2 = df_limitado2.values.flatten()

#Se crean vectores con el largo del numero de horas y lineas (se mantuvieron los nombres de las variables por simplicidad aunque sean lineas)
horas2 = np.arange(num_horas)
barras2= np.arange(num_barras)
valores_voltajes_barras2=[]

#Se grafica cada curva dentro de un for
for indice, fila in df_limitado2.iterrows():
    lista_fila = fila.tolist() 
    ax.plot(np.full(num_horas2, indice),horas2 , lista_fila, label=columna)


# Configurar etiquetas y leyendas
ax.set_xlabel('Indice de linea')
ax.set_ylabel('Horas')
ax.set_zlabel('Corriente')
ax.set_title('Corrientes en diferentes lineas a lo largo del tiempo en el alimentador 1')

#Se configura la perspectiva de la vista
ax.view_init(elev=50, azim=-30)

# Mostrar el gráfico 3D
plt.show()

################################################################################################################

#Se define el tamaño de la figura
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

#Se establece las dimensiones de los vectores
num_barras3, num_horas3 = df_limitado3.shape

# Obtener los valores de voltaje del DataFrame
voltajes3 = df_limitado3.values.flatten()

#Se crean vectores con el largo del numero de horas y lineas (se mantuvieron los nombres de las variables por simplicidad aunque sean lineas)
horas3 = np.arange(num_horas)
barras3= np.arange(num_barras)
valores_voltajes_barras3=[]

#Se grafica cada curva dentro de un for
for indice, fila in df_limitado3.iterrows():
    lista_fila = fila.tolist() 
    ax.plot(np.full(num_horas3, indice),horas3 , lista_fila, label=columna)


# Configurar etiquetas y leyendas
ax.set_xlabel('Indice de Barra')
ax.set_ylabel('Horas')
ax.set_zlabel('Voltaje')
ax.set_title('Voltajes en diferentes barras a lo largo del tiempo en el alimentador 2')

#Se configura la perspectiva de la vista
ax.view_init(elev=50, azim=-30)

# Mostrar el gráfico 3D
plt.show()

################################################################################################################

#Se define el tamaño de la figura
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

#Se establece las dimensiones de los vectores
num_barras4, num_horas4 = df_limitado4.shape

# Obtener los valores de voltaje del DataFrame
voltajes4 = df_limitado4.values.flatten()

#Se crean vectores con el largo del numero de horas y lineas (se mantuvieron los nombres de las variables por simplicidad aunque sean lineas)
horas4 = np.arange(num_horas)
barras4= np.arange(num_barras)
valores_voltajes_barras4=[]

#Se grafica cada curva dentro de un for
for indice, fila in df_limitado4.iterrows():
    lista_fila = fila.tolist() 
    ax.plot(np.full(num_horas4, indice),horas4 , lista_fila, label=columna)


# Configurar etiquetas y leyendas
ax.set_xlabel('Indice de linea')
ax.set_ylabel('Horas')
ax.set_zlabel('Corriente')
ax.set_title('Corrientes en diferentes lineas a lo largo del tiempo en el alimentador 2')

#Se configura la perspectiva de la vista
ax.view_init(elev=50, azim=-30)

# Mostrar el gráfico 3D
plt.show()

################################################################################################################

#Se define el tamaño de la figura
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

#Se establece las dimensiones de los vectores
num_barras5, num_horas5 = df_limitado5.shape

# Obtener los valores de voltaje del DataFrame
voltajes5 = df_limitado5.values.flatten()

#Se crean vectores con el largo del numero de horas y lineas (se mantuvieron los nombres de las variables por simplicidad aunque sean lineas)
horas5 = np.arange(num_horas)
barras5= np.arange(num_barras)
valores_voltajes_barras5=[]

#Se grafica cada curva dentro de un for
for indice, fila in df_limitado5.iterrows():
    lista_fila = fila.tolist() 
    ax.plot(np.full(num_horas5, indice),horas5 , lista_fila, label=columna)


# Configurar etiquetas y leyendas
ax.set_xlabel('Indice de Barra')
ax.set_ylabel('Horas')
ax.set_zlabel('Voltaje')
ax.set_title('Voltajes en diferentes barras a lo largo del tiempo en el alimentador 3')

#Se configura la perspectiva de la vista
ax.view_init(elev=50, azim=-30)

# Mostrar el gráfico 3D
plt.show()

################################################################################################################

#Se define el tamaño de la figura
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111, projection='3d')

#Se establece las dimensiones de los vectores
num_barras6, num_horas6 = df_limitado6.shape

# Obtener los valores de voltaje del DataFrame
voltajes6 = df_limitado6.values.flatten()

#Se crean vectores con el largo del numero de horas y lineas (se mantuvieron los nombres de las variables por simplicidad aunque sean lineas)
horas6 = np.arange(num_horas)
barras6= np.arange(num_barras)
valores_voltajes_barras6=[]

#Se grafica cada curva dentro de un for
for indice, fila in df_limitado6.iterrows():
    lista_fila = fila.tolist() 
    ax.plot(np.full(num_horas6, indice),horas6 , lista_fila, label=columna)


# Configurar etiquetas y leyendas
ax.set_xlabel('Indice de linea')
ax.set_ylabel('Horas')
ax.set_zlabel('Corriente')
ax.set_title('Corrientes en diferentes lineas a lo largo del tiempo en el alimentador 3')

#Se configura la perspectiva de la vista
ax.view_init(elev=50, azim=-30)

# Mostrar el gráfico 3D
plt.show()

# Trazar la segunda gráfica en la misma figura
plt.plot(horas6, alimentador16, label='Estado de carga')
# Etiquetas de los ejes y leyenda
plt.xlabel('Horas')
plt.ylabel('Amplitud')
plt.title('Estado de carga')
plt.legend()

# Mostrar la figura
plt.show()

