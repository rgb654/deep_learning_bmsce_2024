import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers,models,optimizers

def create_data():
    X=np.random.randn(1000,10)
    y=np.random.randn(1000,1)
    return X,y

def create_model():
  model = models.Sequential([
      layers.Dense(200,activation='relu',input_shape=(10,)),
      layers.Dense(100,activation='tanh'),
      layers.Dense(50,activation='sigmoid'),
      layers.Dense(1)
  ])
  return model

def train_model(model,optimizer,X,y,batch_size,epochs,optimizer_name):
  model.compile(optimizer=optimizer,loss='mean_squared_error')
  history=[]

  for epoch in range(epochs):
    hist=model.fit(X,y,batch_size=batch_size,epochs=1,verbose=0)
    loss=hist.history['loss'][0]
    history.append(loss)
    print(f"Epoch {epoch+1}/{epochs}-{optimizer_name} Loss:{loss:4f}")
  return history

X,y = create_data()
model_sgd = create_model()
model_adam = create_model()
model_adagrad = create_model()

optimizer_sgd = optimizers.SGD(learning_rate=0.05)
optimizer_adam = optimizers.Adam(learning_rate=0.001)
optimizer_adagrad = optimizers.Adagrad(learning_rate=0.1)

epochs = 50
batch_size = 16

print("\nTraining with SGD Optimizer:")
sgd_loss = train_model(model_sgd, optimizer_sgd, X, y, batch_size, epochs, 'SGD')

print("\nTraining with Adam Optimizer:")
adam_loss = train_model(model_adam, optimizer_adam, X, y, batch_size, epochs, 'Adam')

print("\nTraining with Adagrad Optimizer:")
adagrad_loss = train_model(model_adagrad, optimizer_adagrad, X, y, batch_size, epochs, 'Adagrad')

plt.plot(range(1,epochs+1),sgd_loss,label='SGD',color='blue')
plt.plot(range(1,epochs+1),adam_loss,label='Adam',color='orange')
plt.plot(range(1,epochs+1),adagrad_loss,label='Adagrad',color='red')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss comparision')
plt.legend()
plt.grid(True)
plt.show()
