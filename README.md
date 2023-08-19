# Recommendation
Recommendation is a collection of state-of-the-art recommendation system models, ranging from traditional content-based methods to advanced deep learning techniques.

## Table of Contents
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)

# Models
* **Content-Based Recommendation**: Utilizing item features and user preferences to suggest relevant items.
  * Content Based Recommendation (Base): A fundamental approach to content-based recommendations. Defined in ```content_based_recommendation_base.py```.
  * Content Based Recommendation (BERT): Enhancing content-based recommendations using BERT. Defined in ```content_based_recommendation_bert.py```.
* **Matrix Factorization**: Classical collaborative filtering approach for recommendation. Defined in ```matrix_factorization_based_recommendation.py```.
* **Deep Learning Based Recommendation**: Using dense neural networks to model complex patterns in user-item interactions. Defined in ```deep_learning_based_recommendation.py```.
* **Hybrid Model**: Designed to incorporate the strengths of both deep learning-based recommendation models and matrix factorization techniques to provide better and more accurate item recommendations to users. It balances between these two techniques using a learnable parameter, alpha. Defined in ```hybrid_recommendation.py```
* **Examples & Tests**: Comprehensive usage examples and unit tests for each recommendation model.

# Installation
1. Clone the repository:
  `git clone https://github.com/shoaibur/Recommendation.git`
2. Navigate to the repository: `cd Recommendation`
3. Install the required packages: `pip install -r requirements.txt`

# Usage
To use any of the recommendation models, first import the required module from the **`models`** directory. For a step-by-step guide on how to use these models, refer to the **`example_usages.ipynb`** Jupyter notebook.

# Testing
Unit tests are provided for each recommendation model. To run the tests:
1. Navigate to the tests directory: `cd tests`
2. Run the tests using pytest (make sure you have it installed): `pytest`

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
