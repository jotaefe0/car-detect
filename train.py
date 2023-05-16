import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the data directories
data_dir = '/opt/ml/input/data/train'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')


data = tf.keras.utils.image_dataset_from_directory(data_dir)
data = data.map(lambda x,y: (x/255, y))

train_size = int(len(data)*.8)
val_size = int(len(data)*.1) + 1


train = data.take(train_size) 
val = data.skip(train_size).take(val_size)



# Set up the image generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='binary')

# val_generator = val_datagen.flow_from_directory(
#         val_dir,
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='binary')

# Define the model architecture
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train,
          epochs=10,
          validation_data=val)

# Save the trained model to the /opt/ml/model directory
model.save('/opt/ml/model/model.h5')
