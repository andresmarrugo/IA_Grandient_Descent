import numpy as np
import csv,random as rnd
import matplotlib.pyplot as plt




class datos:

    def llenarX(self, x1,x2,x3,x4,x5,x6,x7,x8):
        self.X.append([x1,x2,x3,x4,x5,x6,x7,x8])

    def llenarXwithOnes(self, x1, x2, x3, x4, x5, x6, x7, x8):
        self.X1.append([1.0,x1, x2, x3, x4, x5, x6, x7, x8])
    def llenarY(self, y1,y2):
        self.Y1.append(y1)
        self.Y2.append(y2)
    def __init__(self):
        self.X = []
        self.X1 =[]
        self.Y1 = []
        self.Y2 = []

    def getX(self):
        return np.asarray(self.X)
    def getXWithOnes(self):
        return np.asarray(self.X1)
    def getY1(self):
        return np.asarray(self.Y1)
    def getY2(self):
        return np.asarray(self.Y2)

class genData:

    def __init__(self):
        self.train= datos()
        self.test = datos()
        self.train1 = datos()
        self.nums = []
        self.X1 = []
        self.X2 = []
        self.X3 = []
        self.X4 = []
        self.X5 = []
        self.X6 = []
        self.X7 = []
        self.X8 = []
        self.Y1 = []
        self.Y2 = []
    def crearConjuntos(self):
        with open('data.csv') as csvarchivo:
            entrada = csv.DictReader(csvarchivo)
            for reg in entrada:
                self.X1.append(float(reg["X1"]))
                self.X2.append(float(reg["X2"]))
                self.X3.append(float(reg["X3"]))
                self.X4.append(float(reg["X4"]))
                self.X5.append(float(reg["X5"]))
                self.X6.append(float(reg["X6"]))
                self.X7.append(float(reg["X7"]))
                self.X8.append(float(reg["X8"]))
                self.Y1.append(float(reg["Y1"]))
                self.Y2.append(float(reg["Y2"]))

        for i in range(768):
            self.nums.append(i)
        rnd.shuffle(self.nums)

        for i in range(len(self.X1)):
            if(i<462):
                #print(i,"Llenando train",nums[i])
                self.train.llenarX(self.nomalizar(i, self.X1), self.nomalizar(i, self.X2), self.nomalizar(i, self.X3),
                                  self.nomalizar(i, self.X4), self.nomalizar(i, self.X5), self.nomalizar(i, self.X6),
                                  self.nomalizar(i, self.X7), self.nomalizar(i, self.X8))

                self.train.llenarXwithOnes(self.nomalizar(i, self.X1), self.nomalizar(i, self.X2), self.nomalizar(i, self.X3),
                                   self.nomalizar(i, self.X4), self.nomalizar(i, self.X5), self.nomalizar(i, self.X6),
                                   self.nomalizar(i, self.X7), self.nomalizar(i, self.X8))

                self.train.llenarY(self.nomalizar(i, self.Y1), self.nomalizar(i, self.Y2))
            else:
                #print(i,"llenando test",nums[i])
                self.test.llenarX(self.nomalizar(i, self.X1), self.nomalizar(i, self.X2), self.nomalizar(i, self.X3),
                                  self.nomalizar(i, self.X4), self.nomalizar(i, self.X5), self.nomalizar(i, self.X6),
                                  self.nomalizar(i, self.X7), self.nomalizar(i, self.X8))

                self.test.llenarY(self.nomalizar(i, self.Y1), self.nomalizar(i, self.Y2))

    def nomalizar(self,i, valArray):
        norm=(valArray[self.nums[i]]-np.mean(valArray))/(np.amax(valArray)-np.amin(valArray))
        return norm

    def getTrainX(self):
        return self.train.getX()
    def getTrainXones(self):
        return self.train.getXWithOnes()
    def getTrainY1(self):
        return self.train.getY1()

    def getTrainY2(self):
        return self.train.getY2()

    def getTestX(self):
        return self.test.getX()

    def getTestY1(self):
        return self.test.getY1()

    def getTestY2(self):
        return self.test.getY2()

    def gradientDescent( self,x, y, theta, alpha, m, numIterations,ActivarLog=False):
        J_history = np.zeros(shape=(numIterations, 1))
        xTrans = x.transpose()
        for i in range(0, numIterations):
            hypothesis = np.dot(x, theta)
            loss = hypothesis - y

            cost = np.sum(loss ** 2) / (2 * m)  # Funcion de costo

            (ActivarLog)and print("Iteration %d | Cost: %f" % (i, cost))  # gradient descent per iteration is displayed
            gradient = np.dot(xTrans, loss) / m
            #actualizar al tiempo
            theta = theta - alpha * gradient
            J_history[i][0] = cost
        return theta, J_history

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = (y_true, y_pred)

            ## Note: does not handle mix 1d representation
            # if _is_1d(y_true):
            #    y_true, y_pred = _check_1d_array(y_true, y_pred)

            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

misDatos=genData()

misDatos.crearConjuntos()

TrainX=misDatos.getTrainX()
TrainX1=misDatos.getTrainXones()
TrainY1=misDatos.getTrainY1()
TrainY2=misDatos.getTrainY2()

TestX=misDatos.getTestX()
TestY1=misDatos.getTestY1()
TestY2=misDatos.getTestY2()

print("conjunto de entrenamiento: \n cantidad de registros: ",len(TrainX),"\n train X:",len(TrainX)," train Y1",len(TrainY1)," train Y2",len(TrainY2))
print("conjunto de prueba: \n cantidad de registros: ",len(TestX),"\n test X:",len(TestX)," test Y1",len(TestY1)," test Y2",len(TestY2))

print("forma TrainX: ",np.shape(TrainX))
print("forma TrainY1: ",np.shape(TrainY1))
print("forma TrainY2: ",np.shape(TrainY1))

print("forma TestX: ",np.shape(TestX))
print("forma TestY1: ",np.shape(TestY1))
print("forma TestY2: ",np.shape(TestY2))

#Parametros del Algoritmo

alpha=0.01
numIterations=200000
m, n = np.shape(TrainX)

theta=np.ones(8) #Vector de parametros

theta, history = misDatos.gradientDescent(TrainX, TrainY1, theta, alpha,m,numIterations,False)
print ("Resultado\n Vator de paratros theta: ",theta)

hx = TestX.dot(theta)
i=0
for reg in hx:
    print("Y1:",TestY1[i],"    Ycal: ",reg,"   diferencia: ",TestY1[i]-reg)
    i=i+1

plt.scatter(range(len(TestY1)),TestY1,color="blue",alpha=0.8)
plt.scatter(range(len(TestY1)),hx,color="magenta",alpha=0.8)
plt.show()