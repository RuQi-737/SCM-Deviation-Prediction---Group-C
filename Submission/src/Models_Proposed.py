import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We need to split the dataset into "Test" and "Training" :
    1. Library importation
    2. Creation of "Test" and "Training" subdataset
    3. Preparation of some varaibles : normalization, centralization

    But we have to modify the format of some variables into categorical one : "Product Article Number", "Supplier"

    2 possibilities :

    - One-hot encoding = add a new column for every product and every supplier and the vraible can be "1" ( is this product/supplier) or "0" ( it is not this product/supplier)
    - frequency encoding = replace by the frequency of the product/suppliers

    Finally I have chosen to use the One-hot encoding because our number of differents number are not so great and the most suited method for our case is one-hot encoding

    !!! Maybe need to improve the split between "Test" and "Training" subdataset, because now we are not sure that the distribution and both subdataset are the same but just with a smaller quantity of data  !!!
    """)
    return


@app.cell
def _():
    import pathlib
    import sys

    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier



    return (
        DecisionTreeClassifier,
        GridSearchCV,
        MLPClassifier,
        MultiOutputClassifier,
        RandomForestClassifier,
        SVC,
        StandardScaler,
        accuracy_score,
        classification_report,
        pd,
    )


@app.cell
def _():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # =========================
    # 1. Load dataset
    # =========================
    data = pd.read_csv("Company_A_Cleaned.csv") #TO CHANGE

    # =========================
    # 2. Define target variables
    # =========================
    target_cols = ["Delivery_status", "Quantity_status"] #TO CHANGE IF NOT THE RIGHT NAME 

    # Separate features (X) and targets (y)
    X = data.drop(columns=target_cols)
    y = data[target_cols]

    # =========================
    # 3. Basic preprocessing 
    # =========================

    # Handle missing values
    # (simple strategy: fill numeric with median, categorical with mode)
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())

    # Encode categorical variables (one-hot encoding)
    X = pd.get_dummies(X, drop_first=True)

    # =========================
    # 4. Train / Test Split
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # =========================
    # 5. One-Hot Encoding
    # =========================

    categorical_cols = ["Supplier", "Product Article Number"]

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

    # Align train and test columns (VERY IMPORTANT)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # =========================
    # 6. Output shapes (sanity check)
    # =========================
    print("Training set shape:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nTest set shape:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    return X_test, X_train, pd, target_cols, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # First Model : Decision Tree
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A Decision Tree is a simple supervised learning model that makes predictions by splitting data into branches based on feature values. Each split represents a decision rule, and the final leaves give the predicted class.

    Here it is used to predict Delevery status + Quantity status

    Main parameters to tune :
    - max_depth: limits how deep the tree grows (controls overfitting)
    - min_samples_split: minimum samples required to split a node
    - min_samples_leaf: minimum samples in a leaf node
    - criterion: function to measure split quality (Gini or entropy)

    Possible issues :

    - Overfitting: the tree may become too complex and perform well on training data but poorly on test data
    - Instability: small changes in data can lead to very different trees
    - Bias toward dominant classes: if data is imbalanced, predictions may be skewed
    -
    Overall, Decision Trees are easy to interpret but often need tuning to achieve good performance.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    MultiOutputClassifier,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    pd,
    target_cols,
    y_test,
    y_train,
):
    # =========================
    # A1 Build Decision Tree model
    # =========================
    base_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10  # you can tune this
    )

    model = MultiOutputClassifier(base_model)

    # =========================
    # A2 Train model
    # =========================
    model.fit(X_train, y_train)

    # =========================
    # A3 Predictions
    # =========================
    y_pred_DT = model.predict(X_test)

    # Convert predictions back to DataFrame for readability
    y_pred_DT = pd.DataFrame(y_pred_DT, columns=target_cols, index=y_test.index)

    # =========================
    # A4 Evaluation
    # =========================
    print("\n=== Accuracy per target ===")

    for col in target_cols:
        acc = accuracy_score(y_test[col], y_pred_DT[col])
        print(f"{col}: {acc:.4f}")

    print("\n=== Detailed classification report ===")

    for col in target_cols:
        print(f"\n--- {col} ---")
        print(classification_report(y_test[col], y_pred_DT[col]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Random Forest
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A Random Forest is an ensemble model that combines multiple Decision Trees. Each tree is trained on random samples of the data, and the final prediction is made by majority vote.

    In this project, it is used to predict **Delivery_status** and **Quantity_status**.

    Key parameters:

    * **n_estimators**: number of trees in the forest
    * **max_depth**: controls how deep each tree can grow
    * **min_samples_split / min_samples_leaf**: control tree complexity and help reduce overfitting
    * **max_features**: number of features used at each split

    Data preprocessing:

    Categorical variables (e.g., Supplier, Product ID) must be converted using **one-hot encoding**, which transforms each category into binary columns (0/1).

    Possible issues:

    * Overfitting if trees are too deep
    * High computational cost with many trees
    * Large number of features after one-hot encoding
    * Sensitivity to imbalanced classes

    Random Forest is generally more accurate and stable than a single Decision Tree for tabular data.
    """)
    return


@app.cell
def _(
    MultiOutputClassifier,
    RandomForestClassifier,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    pd,
    y_test,
    y_train,
):
    #The one-hot encoding has already been applied 

    # =========================
    # B1. Build Random Forest model
    # =========================

    base_model = RandomForestClassifier(
        n_estimators=200,      # number of trees
        max_depth=15,          # depth of each tree
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )

    model = MultiOutputClassifier(base_model)

    # =========================
    # B2. Train model
    # =========================

    model.fit(X_train, y_train)

    # =========================
    # B3. Predictions
    # =========================

    y_pred_RF = model.predict(X_test)

    # Convert predictions into DataFrame for readability
    y_pred_RF = pd.DataFrame(y_pred_RF, columns=y_test.columns, index=y_test.index)

    # =========================
    # B4. Evaluation
    # =========================

    print("\n=== Accuracy per target ===")

    for col in y_test.columns:
        acc = accuracy_score(y_test[col], y_pred_RF[col])
        print(f"{col}: {acc:.4f}")

    print("\n=== Classification report ===")

    for col in y_test.columns:
        print(f"\n--- {col} ---")
        print(classification_report(y_test[col], y_pred_RF[col]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The main parameters of a Random Forest control the model’s complexity and performance.
    n_estimators: number of trees in the model. More trees usually improve accuracy but increase computation time.

    - max_depth: limits how deep each tree can grow. Lower values reduce overfitting, higher values increase model complexity.
    - min_samples_split: minimum number of samples needed to split a node. Higher values make the model more conservative.
    - min_samples_leaf: minimum number of samples in a leaf. Helps reduce overfitting and makes predictions more stable.
    - max_features: number of features considered at each split. Smaller values increase diversity between trees and can improve generalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SVM : Support Vector Machine
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **Support Vector Machine (SVM)** is a supervised learning model used for classification. It finds the best boundary (hyperplane) that separates classes with the largest margin. It is like regression but in an higher dimensional space ----> hyperplane.

    ### Main parameters:

    * **C**: regularization parameters, controls the balance between correct classification and a simple model (high C can cause overfitting)
    * **kernel**: defines the shape of the decision boundary (RBF is most common)
    * **gamma**: controls how much influence each data point has (high gamma can overfit)

    ### Possible problems:

    * Slow on large datasets
    * Sensitive to feature scaling
    * Can overfit if parameters are not well tuned

    ### Categorical variables:

    They must be converted into numbers using **one-hot encoding**, where each category becomes a binary column (0/1).

    SVM is powerful but needs good preprocessing and parameter tuning.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    MultiOutputClassifier,
    SVC,
    StandardScaler,
    accuracy_score,
    classification_report,
    pd,
    y_test,
    y_train,
):
    # =========================
    # C1. Feature scaling 
    # =========================

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # C2. Define SVM model
    # =========================

    svm = SVC()

    model = MultiOutputClassifier(svm)

    # =========================
    # C3. Hyperparameter tuning
    # =========================

    param_grid = {
        "estimator__C": [0.1, 1, 10, 100],
        "estimator__gamma": [0.001, 0.01, 0.1, 1],
        "estimator__kernel": ["rbf"]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    # =========================
    # C4. Train model with tuning
    # =========================

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)

    # =========================
    # C5. Predictions
    # =========================

    y_pred_SVM = best_model.predict(X_test)

    y_pred_SVM = pd.DataFrame(y_pred_SVM, columns=y_test.columns, index=y_test.index)

    # =========================
    # C6. Evaluation
    # =========================

    print("\n=== Accuracy per target ===")

    for col in y_test.columns:
        acc = accuracy_score(y_test[col], y_pred_SVM[col])
        print(f"{col}: {acc:.4f}")

    print("\n=== Classification report ===")

    for col in y_test.columns:
        print(f"\n--- {col} ---")
        print(classification_report(y_test[col], y_pred_SVM[col]))
    return X_test, X_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Influence of hyperparameters :

    * **C**: controls the balance between correct classification and simplicity. High C can overfit, low C makes the model simpler.
    * **gamma**: controls how far each point influences the model. High gamma → complex model, low gamma → smoother model.
    * **kernel**: defines the type of decision boundary (RBF for non-linear problems, linear for simple ones).

    👉 In short: C controls flexibility, gamma controls complexity, and kernel defines the shape of the model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multilayer Percepton : Simple Neurone Network
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A **Multilayer Perceptron (MLP)** is a type of neural network used for classification and regression. It is made of layers of connected nodes (neurons): an input layer, one or more hidden layers, and an output layer.

    The model learns patterns by adjusting weights between neurons using a method called **backpropagation**.

    Key parameters:

    * **hidden_layer_sizes**: number and size of hidden layers
    * **activation**: function used in neurons (e.g., ReLU, sigmoid)
    * **alpha**: regularization to reduce overfitting
    * **learning_rate**: controls how fast the model learns

    Possible issues:

    * Needs feature scaling
    * Can overfit if too complex
    * Requires more data and computation than simpler models
    """)
    return


@app.cell
def _(
    GridSearchCV,
    MLPClassifier,
    MultiOutputClassifier,
    StandardScaler,
    accuracy_score,
    classification_report,
    pd,
    y_test,
    y_train,
):
    # =========================
    # C1. Feature scaling (IMPORTANT for MLP)
    # =========================

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # C2. Define base MLP model
    # =========================

    mlp = MLPClassifier(max_iter=300, random_state=42)
    model = MultiOutputClassifier(mlp)

    # =========================
    # C3. Hyperparameter tuning
    # =========================

    param_grid = {
        "estimator__hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "estimator__activation": ["relu", "tanh"],
        "estimator__alpha": [0.0001, 0.001, 0.01],
        "estimator__learning_rate_init": [0.001, 0.01]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    # =========================
    # C4. Train model with tuning
    # =========================

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)

    # =========================
    # C5. Predictions
    # =========================

    y_pred_MLP = best_model.predict(X_test)

    y_pred_MLP = pd.DataFrame(y_pred_MLP, columns=y_test.columns, index=y_test.index)

    # =========================
    # C6. Evaluation
    # =========================

    print("\n=== Accuracy per target ===")

    for col in y_test.columns:
        acc = accuracy_score(y_test[col], y_pred_MLP[col])
        print(f"{col}: {acc:.4f}")

    print("\n=== Classification report ===")

    for col in y_test.columns:
        print(f"\n--- {col} ---")
        print(classification_report(y_test[col], y_pred_MLP[col]))
    return X_test, X_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Influence of every hyperparameters :
    - hidden_layer_sizes: defines the number and size of hidden layers. More neurons/layers → more complex model, but higher risk of overfitting.
    - activation function used in neurons (e.g., ReLU, tanh). It affects how the model learns non-linear patterns.
    - alpha: regularization parameter. Higher values reduce overfitting by penalizing large weights.
    - learning_rate_init: initial learning speed. High value → faster but less stable learning; low value → slower but more stable.
    - max_iter: maximum number of training iterations. Too low → underfitting; too high → longer training time.
    """)
    return


if __name__ == "__main__":
    app.run()
