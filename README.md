# Tomato Leaf Disease Classification

A project of Machine Learning and Artificial Intelligence, our purpose for this project is to classifying the types of tomato leaf disease infections based on images using the Convolutional Neural Network (CNN) method. By using the Convolutional Neural Network, the results of these three model measures about 92%-99% of the F1-Score.

Below is the description of project files and folders.
1. Utility Files
- `dataset_builder.py`: Contain class definition to prepare the raw dataset from certain directory and convert it to TFRecord for the use with `tensorflow_datasets` (TFDS) library. 
- `utils.py`: Contain helper functions for preprocessing the dataset and building the CNN model. 
- `plots.py`: Contain specific functions for visualizing the results.
2. Notebook Files
- `binary_classification.ipynb`: Doing 2 class classification based on infection status (healhy, diseased)
- `quinary_classification.ipynb`: Doing 5 class classification based on pathogen type (health, bacteria, fungi, mite, virus)
- `main_classification.ipynb`: Doing 10 class classification based on pathogen species or equivalent to disease type (healthy, bacterial spot, early blight, late blight, leaf mold, target spot, septoria leaf spot, spider mite, tomato yellow leaf curl virus, tomato mosaic virus)
- `visualization.ipynb`: Visualize the training history and results from all models.

Dataset used (You can download here):

[tomato_leaf_disease_binary](https://www.kaggle.com/datasets/habiburrohman/tomato-leaf-disease-binary) <br>
[tomato_leaf_disease_quinary](https://www.kaggle.com/datasets/habiburrohman/tomato-leaf-disease-quinary) <br>
[tomato_leaf_disease_main](https://www.kaggle.com/datasets/habiburrohman/tomato-leaf-disease) <br>
- Extract each of the zipped dataset in the `dataset` folder, then rename them acccording to the hyperlink texts above.

Other members:

Habiburrohman
