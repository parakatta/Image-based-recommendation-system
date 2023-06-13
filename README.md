# Image based Recommendation System using Fashion Product Dataset
The objective of the project was to create a system that could suggest visually similar fashion items based on an input image using KNN.  
*Method Used*: K Nearest Neighbors (KNN)  

![Screenshot 2023-06-13 164841](https://github.com/parakatta/Image-based-recommendation-system/assets/83866928/798360d1-4758-4f72-9894-77b2e53e635a)

For the implementation of the recommendation system, we employed the K Nearest Neighbors (KNN) algorithm. KNN is a popular machine learning algorithm used for classification and regression tasks. In this project, we adapted it for the task of image similarity matching.

The KNN algorithm operates by storing a set of labeled training samples, where each sample consists of an image and its corresponding fashion item label. During the recommendation phase, given an input image, the KNN algorithm identifies the k nearest neighbors from the training samples based on a distance metric, such as Euclidean distance or cosine similarity. The output is a set of k visually similar fashion items that are considered potential recommendations.

## Feature Extraction using Normalization
To extract meaningful features from the fashion dataset, we utilized a pre-trained ResNet framework. ResNet (Residual Neural Network) is a deep learning architecture commonly used for image classification tasks. By leveraging the power of ResNet, we could benefit from its hierarchical feature extraction capabilities.

Normalization is an essential step in feature extraction to enhance the model's performance. It involves scaling the features to a standard range to ensure consistency and facilitate accurate comparisons. By normalizing the extracted features, we could eliminate any bias introduced by varying scales or magnitudes in the dataset.

## Novelty and Potential Enhancements
The novelty of this project lies in the application of KNN and feature extraction using ResNet for fashion recommendation. By combining these techniques, we created an image-based recommendation system that considered visual similarities between fashion items.

While this project used KNN and ResNet, there are other methods and approaches that can be explored to achieve image-based recommendation systems. Some potential alternatives include:

- Deep Learning Architectures: Besides ResNet, other deep learning architectures like Convolutional Neural Networks (CNNs) or Siamese Networks can be used for feature extraction and image similarity matching.

- Content-Based Filtering: In addition to visual features, other metadata or attributes associated with fashion items, such as brand, color, style, or price, can be incorporated to enhance recommendation accuracy.

- Hybrid Approaches: Combining multiple recommendation techniques, such as collaborative filtering (utilizing user preferences) and content-based filtering (using image features), can result in more robust and accurate recommendation systems.

- Transfer Learning: Instead of training a model from scratch, pre-trained models on large-scale datasets (e.g., ImageNet) can be fine-tuned or utilized as feature extractors to boost performance.

By further exploring these approaches and incorporating user feedback and evaluation metrics, the recommendation system can be continually improved in terms of accuracy, efficiency, and user satisfaction.

Overall, the project aimed to leverage KNN, ResNet, and normalization techniques to develop an image-based fashion recommendation system. The project's novelty stemmed from its unique combination of these methods, while also leaving room for further enhancements and exploration of alternative approaches in the field of recommendation systems.

## Implementation  

Install the dependencies  
 ```
 pip install -r requirements.txt
 ```  
 Run the streamlit app  
 ```
 streamlit run app.py
 ```  
 
 Kaggle [notebook](https://www.kaggle.com/code/aleemaparakatta/image-based-recommendation-system-fashion/)
 
