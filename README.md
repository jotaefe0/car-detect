# Car Interior/Exterior Classification using CNN

This notebook demonstrates the use of a Convolutional Neural Network (CNN) to classify images as either interior or exterior of a car. The CNN is trained on a labeled dataset of car images and learns to differentiate between the two categories.

## Dataset

The dataset used for training and evaluation consists of a collection of car images labeled as "interior" or "exterior". The dataset should be split into training and testing sets, with a sufficient number of images in each category. This images were scrapped from different car dealership's websites.

## Model Architecture

The CNN model used for this classification task typically consists of multiple convolutional layers, followed by max-pooling layers to reduce the spatial dimensions. This is then followed by fully connected layers leading to the final classification layer. The specific architecture and hyperparameters can vary depending on the dataset and problem at hand.

## Model Training

1. Load and preprocess the dataset.
2. Split the dataset into training and testing sets.
3. Design the CNN architecture.
4. Compile the model with appropriate loss function and optimizer.
5. Train the model on the training data.
6. Evaluate the model on the testing data.
7. Fine-tune the model and adjust hyperparameters as needed.

It was trainged usaing a ml.p2.xlarge

## Model Evaluation

loss: 0.0168 - accuracy: 0.9951 - val_loss: 0.0344 - val_accuracy: 0.9919

## Try the Model

You can try the trained model by visiting the following URL:

[Car Interior/Exterior Classification Model](https://huggingface.co/spaces/jotaefe/car-detect)

## Conclusion

Using a CNN, we can effectively classify images as interior or exterior of a car. By training the model on a labeled dataset and optimizing its architecture and hyperparameters, we can achieve accurate predictions. This can be useful in various applications, such as data pipelines for dealership's website where they need to label these images.
