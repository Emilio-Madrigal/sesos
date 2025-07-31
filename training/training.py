import numpy as np
import data.dataset_and as AND
x=AND.entradas
y=AND.salidas
def train(self,X,y,epochs,taza_aprendizaje):
    for epoch in range(epochs):
        loss = 0
        for i in range(len(x)):
            salida=self.forward(X[i])
            loss+=np.mean((y[i]-salida)**2)
            loss_gradient=2*(salida-y[i])
            self.backward(loss_gradient, taza_aprendizaje)
    loss/= len(X)
    self.loss_list.append(loss)
    print