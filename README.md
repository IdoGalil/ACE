# Attack on Confidence Estimation (ACE)
<p align="center">
  <img src="https://github.com/IdoGalil/ACE/blob/main/Intuition.png" width="640" height="360">
</p>
This repository is the official implementation of "Disrupting Deep Uncertainty Estimation Without Harming Accuracy"(*provide arxiv link*).

## Overview
ACE is an algorithm for crafting adversarial examples that disrupt a model's uncertainy estimation performance without harming its accuracy.
The figure above conceptually illustrates how ACE works. Consider a classifier for cats vs. dogs that uses its prediction's softmax score as its uncertainty estimation measurement. An end user asks the model to classify several images, and output only the ones in which it has the most confidence. Since softmax quantifies the margin from an instance to the decision boundary, we visualize it on a 2D plane where the instances' distance to the decision boundary reflect their softmax score. In the example shown in the figure above, the classifier was mistaken about one image of a dog, classifying it as a cat, but fortunately its confidence in this prediction is the lowest among its predictions. A malicious attacker targeting the images in which the model has the most confidence would want to increase the confidence in the mislabeled instance by pushing it away from the decision boundary, and decrease the confidence in the correctly labeled instances by pushing them closer to the decision boundary. 

## Example
example.py shows a simple demonstration of how ACE decreases an EfficientNetB0's confidence (measured by max softmax score) in a corrent prediction (tank image), and how it increases its confidence in an incorrect prediction (binoculars incorrectly labeled as a tank). 

<img src="https://github.com/IdoGalil/ACE/blob/main/demonstration.PNG" width="627" height="457">
To use it, simply run:

```example
python example.py
```

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
