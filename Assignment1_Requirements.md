# AIAA 3102 — Assignment 1: CIFAR-10 Image Classification

Use the provided data to train models for performing 10-class image classification. You may design your own feature processing, data augmentation, hyperparameter tuning methods, and experimental workflow. Submit the notebook and the trained models.

## Data and Models

The data package `student_data.zip` contains 50,000 original training images and 5,000 validation images, all with labels. The images are 32 × 32 RGB images, with class labels numbered from 0 to 9. After extraction, the files are organized as follows:

| Directory                     | Image and Label Files        |
| ----------------------------- | ---------------------------- |
| `CIFAR10_Data/`               | `x_train.npy`, `y_train.npy` |
| `validation_set_to_students/` | `images.npy`, `labels.npy`   |

Use **scikit-learn** and select specific algorithms from the following model categories. You may also combine models or use ensemble methods:

- **Linear classifiers**: e.g., Logistic Regression, `RidgeClassifier`, `SGDClassifier`, etc.
- **SVMs**: e.g., linear SVM, RBF-kernel SVM, polynomial-kernel SVM, etc.
- **Tree-based models**: e.g., Decision Tree, Random Forest, Extra Trees, Gradient Boosting, etc.
- **MLP**: Multi-Layer Perceptron.

Only the data provided by the course may be used. External data and pretrained models are not allowed.

## Evaluation and Grading

After fixing your final data-processing pipeline, model, and hyperparameters, train three models from scratch using the random seeds **42, 2026, and 3102**, respectively.

The course instructors will evaluate the submitted models using 5,000 hidden test images. The test set contains both original and corrupted images. Some corruption types or corruption strengths may differ from those in the validation set.

The total score is **10 points**, consisting of model performance (60%) and notebook execution (40%).

**Model Performance (60%, maximum 6 points):** Students will be ranked from highest to lowest according to the average test accuracy of the three submitted models.

| Ranking Range                                                                           | Score |
| --------------------------------------------------------------------------------------- | -----:|
| Top 20%                                                                                 | 6     |
| (20%, 45%]                                                                              | 5     |
| (45%, 70%]                                                                              | 4     |
| Bottom 30%                                                                              | 3     |
| `Model_Evaluation.ipynb` cannot successfully load the models or cannot run successfully | 1     |

**Notebook Execution (40%, maximum 4 points):** The notebook should be able to complete data processing, model training, and model saving successfully using **Run All**. If the notebook cannot run successfully using **Run All**, this component will receive **0 points**.

### Software Environment Requirements

To ensure compatibility with the course evaluation environment, the final submitted `.joblib` model files must be generated using the following package versions:

- **Python:** 3.14.x

- **NumPy:** 2.4.x

- **SciPy:** 1.18.x

- **scikit-learn:** 1.9.0

- **joblib:** 1.5.x

The exact environment used for evaluation is Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0, scikit-learn 1.9.0, and joblib 1.5.3.

You may use **Windows, macOS, or Linux**. Other software environments may be used during development and experimentation, but the final submitted models must be generated using the environment specified above.

You are responsible for verifying that your submitted models can be successfully loaded and evaluated using the provided `Model_Evaluation.ipynb`.

## Submission Requirements

Submit to **Canvas** one ZIP file named:

`NAME-IN-PINYIN-StudentID-AIAA3102-Assignment1.zip`

For example:

`QIN, Chengwei-StudentID-AIAA3102-Assignment1.zip`

The ZIP file should contain:

1. **`train_models.ipynb`**: Python code for training and saving the three models. Clearly specify the running environment and data paths, retain the execution outputs, and briefly describe the final method and the reasons for choosing it at the beginning of the notebook.

2. - `model_seed42.joblib` 
   - `model_seed2026.joblib`
   - `model_seed3102.joblib`
   
   These files should correspond to random seeds 42, 2026, and 3102, respectively. The **size** of each model file should not exceed **100 MB**, or else, 1 point will be deducted. You do not need to submit the dataset.

## Due: 23:59, Oct 19, 2026.

## Note 1

If your model requires data preprocessing before prediction, the saved model should include the data-preprocessing module. The following is an example:

```python
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC


# Number of features after flattening one CIFAR-10 image.
N_FEATURES = 32 * 32 * 3


model = Pipeline([
    # Convert input from
    # (batch_size, 32, 32, 3)
    # to
    # (batch_size, 3072)
    (
        "flatten",
        FunctionTransformer(
            np.reshape,
            kw_args={"shape": (-1, N_FEATURES)},
            validate=False
        )
    ),

    # Standardize the flattened features.
    (
        "scaler",
        StandardScaler()
    ),

    # SVM classifier.
    (
        "classifier",
        SVC()
    )
])


# X_train can have shape:
# (n_samples, 32, 32, 3)
model.fit(X_train, y_train)


# Save the whole pipeline.
joblib.dump(model, "model_seed42.joblib")
```

## Note 2

The code in `Model_Evaluation.ipynb` will be used to calculate model performance. Before submitting your models, make sure that this code runs successfully using **the model files that you are going to submit** and the data files in the `validation_set_to_students` folder.

## Late Submission Policy

A **2-point penalty** will be applied to submissions received within **48 hours after the deadline**. Submissions received **more than 48 hours after the deadline will not be accepted**.

## AI Policy

The use of AI tools is permitted for this assignment; however, students are responsible for understanding and verifying all submitted work.
