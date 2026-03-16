import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras import models,layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_data=ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)
traindata_load=train_data.flow_from_directory(
    "C:/Users/D.RAHUL/Downloads/datasets",
    target_size=(180,180),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    subset="training",
    seed=42
    )
valdata_load=train_data.flow_from_directory(
    "C:/Users/D.RAHUL/Downloads/datasets",
    target_size=(180,180),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    subset="validation",
    seed=42
    )
num_classes = traindata_load.num_classes
model=models.Sequential([
    layers.Input(shape=(180,180,3)),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2),2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2),2),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2),2),
    layers.Flatten(),
    layers.Dense(units=128,activation='relu'),
    layers.Dense(num_classes,activation='softmax')
    ])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(traindata_load,
          epochs=10,
          validation_data=valdata_load)
model.save("cnn_model.keras")
print("Training complete & model saved!")
loss,acc=model.evaluate(valdata_load)
print("accuracy_score :",acc)
model = load_model("cnn_model.keras")
path=input("enter image path:")
img=image.load_img(path,target_size=(180,180))
img_array=image.img_to_array(img)
img_array=img_array/255.0
img_array=np.expand_dims(img_array,axis=0)
pred=model.predict(img_array)
class_labels = list(traindata_load.class_indices.keys())
predicted_class = class_labels[np.argmax(pred)]
print("Predicted Class:", predicted_class)

    
    
