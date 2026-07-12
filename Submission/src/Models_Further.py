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
    from xgboost import XGBClassifier
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder
    from pytorch_tabnet.tab_model import TabNetClassifier
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    return (
        GridSearchCV,
        LabelEncoder,
        MultiOutputClassifier,
        StandardScaler,
        TabNetClassifier,
        XGBClassifier,
        accuracy_score,
        classification_report,
        pd,
        tf,
        train_test_split,
    )


@app.cell
def _(pd, train_test_split):
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
    # 6. Output shapes (sanity check)
    # =========================
    print("Training set shape:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nTest set shape:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # XGBoost
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    **XGBoost** is a machine learning algorithm based on decision trees. It builds trees one after another, and each new tree tries to correct the mistakes made by the previous ones. This usually leads to high prediction accuracy.

    Main hyperparameters:

    * **n_estimators**: number of trees in the model.
    * **max_depth**: maximum depth of each tree. Higher values increase model complexity.
    * **learning_rate**: controls how much each tree contributes to the final model. Lower values require more trees but often improve performance.
    * **subsample**: percentage of training data used to build each tree. Lower values can reduce overfitting.

    Possible issues:

    * Can overfit if the trees are too deep.
    * Training can be slower than a single Decision Tree.
    * Requires hyperparameter tuning to achieve the best performance.

    Categorical variables:

    Categorical variables should be converted using **one-hot encoding** before training the model.
    """)
    return


@app.cell
def _(
    GridSearchCV,
    MultiOutputClassifier,
    XGBClassifier,
    X_test,
    X_train,
    accuracy_score,
    classification_report,
    pd,
    y_test,
    y_train,
):

    # =========================
    # D1. Define XGBoost model
    # =========================

    xgb = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        random_state=42
    )

    model = MultiOutputClassifier(xgb)

    # =========================
    # D2. Hyperparameter tuning
    # =========================

    param_grid = {
        "estimator__n_estimators": [100, 200],
        "estimator__max_depth": [3, 6, 9],
        "estimator__learning_rate": [0.01, 0.1],
        "estimator__subsample": [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    # =========================
    # D3. Train model with tuning
    # =========================

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)

    # =========================
    # D4. Predictions
    # =========================

    y_pred_XGB = best_model.predict(X_test)

    y_pred_XGB = pd.DataFrame(y_pred_XGB, columns=y_test.columns, index=y_test.index)

    # =========================
    # D7. Evaluation
    # =========================

    print("\n=== Accuracy per target ===")

    for col in y_test.columns:
        acc = accuracy_score(y_test[col], y_pred_XGB[col])
        print(f"{col}: {acc:.4f}")

    print("\n=== Classification report ===")

    for col in y_test.columns:
        print(f"\n--- {col} ---")
        print(classification_report(y_test[col], y_pred_XGB[col]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## TabNet
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **TabNet**

    **TabNet** is a deep learning model designed for **tabular data**. It learns by selecting the most important features at each decision step using an **attention mechanism**. This allows the model to focus on the most useful information for making predictions.

    In this project, it can be used to predict **Delivery_status** and **Quantity_status**.

    Main hyperparameters:

    * **n_d**: size of the decision layer. Higher values increase the model's capacity.
    * **n_a**: size of the attention layer. Controls how the model selects important features.
    * **n_steps**: number of decision steps. More steps allow the model to learn more complex patterns.
    * **learning_rate**: controls how fast the model learns.
    * **batch_size**: number of samples processed before updating the model weights.

    Possible issues:

    * Requires more training time than tree-based models.
    * Needs careful hyperparameter tuning to avoid overfitting.
    * Performs better with larger datasets.
    * Requires feature scaling for the best performance.

    Categorical variables:

    Categorical variables should be converted into numerical values, usually using **label encoding** or **embeddings**. Unlike many traditional models, **one-hot encoding is generally not recommended** because TabNet can learn efficient representations of categorical features.

    !!! We train two separate models because TabNet does not natively support multi-output classification !!!
    """)
    return


@app.cell
def _(
    LabelEncoder,
    TabNetClassifier,
    X_test,
    X_train,
    accuracy_score,
    y_test,
    y_train,
):
    # =========================
    # E1. Encode categorical variables
    # =========================

    categorical_cols = ["Supplier", "Product Article Number"]

    for col in categorical_cols:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    # =========================
    # E2. Convert to numpy
    # =========================

    X_train_np = X_train.values
    X_test_np = X_test.values

    # =========================
    # E3. Define TabNet model
    # =========================

    model_TN = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        optimizer_fn="adam",
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax"
    )

    # =========================
    # E4. Train model (Delivery_status)
    # =========================

    model_TN.fit(
        X_train_np, y_train["Delivery_status"].values,
        eval_set=[(X_test_np, y_test["Delivery_status"].values)],
        max_epochs=50,
        patience=10,
        batch_size=256
    )

    # =========================
    # E5. Predictions (Delivery_status)
    # =========================

    y_pred_delivery_TN = model_TN.predict(X_test_np)

    # =========================
    # E6. Train second model (Quantity_status)
    # =========================

    model2_TN = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        optimizer_fn="adam",
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax"
    )

    model2_TN.fit(
        X_train_np, y_train["Quantity_status"].values,
        eval_set=[(X_test_np, y_test["Quantity_status"].values)],
        max_epochs=50,
        patience=10,
        batch_size=256
    )

    # =========================
    # E7. Predictions (Quantity_status)
    # =========================

    y_pred_quantity_TN = model2_TN.predict(X_test_np)

    # =========================
    # 8. Evaluation
    # =========================

    print("\nDelivery Status Accuracy:")
    print(accuracy_score(y_test["Delivery_status"], y_pred_delivery_TN))

    print("\nQuantity Status Accuracy:")
    print(accuracy_score(y_test["Quantity_status"], y_pred_quantity_TN))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## DNN
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Deep Neural Networks (DNN)**

    A **Deep Neural Network (DNN)** is an advanced type of neural network with **multiple hidden layers** between the input and output. Each layer learns more complex patterns from the data, allowing the model to capture non-linear relationships in large datasets.

    In your project, a DNN can be used to predict **Delivery_status** and **Quantity_status** based on procurement features.

    Main hyperparameters:

    * **number of layers**: defines how deep the network is. More layers → more complex learning but higher risk of overfitting.
    * **number of neurons per layer**: controls the learning capacity of each layer.
    * **activation function**: e.g., ReLU, sigmoid. It defines how neurons process information.
    * **learning rate**: controls how fast the model updates its weights.
    * **batch size**: number of samples used before updating the model.
    * **epochs**: number of times the model sees the full dataset.
    * **dropout rate**: randomly disables neurons to reduce overfitting.

    Possible problems:

    * Needs **large amounts of data** to perform well
    * Can easily **overfit** on small datasets (like 9,000 samples)
    * Requires **feature scaling**
    * Training can be **slow and computationally expensive**
    * Hard to interpret compared to tree-based models

    We train two separate DNN models because it works better if your target is binary or already encoded
    """)
    return


@app.cell
def _(StandardScaler, X_test, X_train, accuracy_score, tf, y_test, y_train):
    # =========================
    # F1. Encode categorical variables already made for the previous model so 
    #     we can use the same subdatastes 
    # =========================

    # =========================
    # F2. Scale features (IMPORTANT for DNN)
    # =========================

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # =========================
    # F3. Build DNN model function
    # =========================

    def build_model(input_dim):
        model_DNN = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu", input_shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid")
        ])

        model_DNN.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model_DNN

    # =========================
    # F4. Model for Delivery_status
    # =========================

    model_delivery_DNN = build_model(X_train_scaled.shape[1])

    model_delivery_DNN.fit(
        X_train_scaled,
        y_train["Delivery_status"],
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Predictions
    y_pred_delivery_DNN = (model_delivery_DNN.predict(X_test_scaled) > 0.5).astype(int)

    # =========================
    # F5. Model for Quantity_status
    # =========================

    model_quantity_DNN = build_model(X_train_scaled.shape[1])

    model_quantity_DNN.fit(
        X_train_scaled,
        y_train["Quantity_status"],
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Predictions
    y_pred_quantity_DNN = (model_quantity_DNN.predict(X_test_scaled) > 0.5).astype(int)

    # =========================
    # F6. Evaluation
    # =========================

    print("\nDelivery Status Accuracy:")
    print(accuracy_score(y_test["Delivery_status"], y_pred_delivery_DNN))

    print("\nQuantity Status Accuracy:")
    print(accuracy_score(y_test["Quantity_status"], y_pred_quantity_DNN))

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
