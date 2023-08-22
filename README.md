# Recommendation
**Recommendation** is a collection of state-of-the-art recommendation system models, ranging from traditional content-based methods to advanced deep learning techniques.

## Table of Contents
- [Models](#models)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

# Models
1. **Matrix Factorization Based Recommendation**

    **File**: ```matrix_factorization_based_recommendation.py```
  
    **Description**: Matrix factorization is a collaborative filtering method which decomposes the user-item interaction matrix into the product of two lower-dimensional matrices - one representing users and the other representing items. It's one of the most popular recommendation techniques, especially for systems with sparse datasets, such as Netflix or MovieLens.

2. **Deep Learning Based Recommendation**

    **File**: ```deep_learning_based_recommendation.py```

    **Description**: This recommendation method uses dense neural networks to uncover and model the underlying patterns in user-item interactions. It's powerful for capturing non-linear relationships and can combine multiple types of information (like user demographic data and item features) to improve recommendation accuracy.

3. **Hybrid Model**

    **File**: ```hybrid_recommendation.py```

    **Description**: The Hybrid Model is designed to incorporate the strengths of both deep learning-based recommendation models and matrix factorization techniques. It provides a balanced and potentially more accurate recommendation system by weighing contributions from both models using a learnable parameter, alpha.

4. **Content-Based Recommendation**

    **Definition**: Content-based recommendation methods suggest items by comparing the content of the items and the user preferences, with content being described in terms of multiple descriptors or terms that are inherent to the item.

  * Content Based Recommendation (Base):

      * **File**: ```content_based_recommendation_base.py```

      * **Description**: This is the fundamental approach to content-based recommendations. The model leverages item features and known user preferences to predict which items a user may find relevant. It typically employs techniques like term frequency-inverse document frequency (TF-IDF) and cosine similarity to measure the relevance between user preferences and item features.
  
  * Content Based Recommendation (BERT)
      
      * **File**: ```content_based_recommendation_bert.py```
      
      * **Description**: This model enhances traditional content-based recommendations using BERT (Bidirectional Encoder Representations from Transformers). BERT allows for more sophisticated feature extraction from item content, capturing context and semantics in a way that traditional methods might miss. This model can be particularly useful for recommending text-based items like articles or books.

# Data
For effective recommendation, the system requires rich datasets to understand and model the underlying patterns in user-item interactions. This repository leverages three example datasets. The generation of these synthetic datasets are outlined in **generate_data.ipynb** notebook.

* **products.json**: This is a json file that includes the details about different products or items. The key of the json object represents product_id and each item has three fields: name, description, and category of the items.
* **users.csv**: This file provides details about each user. The attributes include: id, gender, location, age, occupation, and preferred_category.
* **interactions.csv**: This dataset captures the interactions between users and products. It provides a snapshot of which user interacted with which product and can also include details like ratings, timestamps, etc., depending on the specifics of the dataset. Key attributes are: user_id, product_id, and rating.

**Examples & Tests**

For each recommendation model, comprehensive usage examples and unit tests are provided. These are designed to showcase how each model can be set up, trained, and used for making predictions. The unit tests also ensure the correctness and robustness of each model's implementation. Users are encouraged to refer to these examples and tests when incorporating these models into their own projects or when seeking to understand the finer details of each model's operation.

# Installation
1. Clone the repository:

    `git clone https://github.com/shoaibur/Recommendation.git`
  
2. Navigate to the repository:

    `cd Recommendation`

4. Install the required packages:

    `pip install -r requirements.txt`

# Usage
To use any of the recommendation models, first import the required module from the **`models`** directory. For a step-by-step guide on how to use these models, refer to the **`example_usages.ipynb`** Jupyter notebook.

# Testing
Unit tests are provided for each recommendation model. To run the tests:
1. Navigate to the tests directory: `cd tests`
2. Run the tests using pytest (make sure you have it installed): `pytest`

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# License
This project is licensed under the terms specified in the license available at the following link: [LICENSE](https://github.com/shoaibur/Recommendation/blob/main/LICENSE). Please ensure to review the license terms before using, modifying, or distributing the code or its derivatives.
