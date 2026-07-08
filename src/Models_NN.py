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

    We transform the data with a One-Hot Encoding for the MLP
    And for the TabNet we use the categrocial data directly because the model is suited for this kind of data
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
    import numpy as np
    import pandas as pd
    import torch

    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
    )
    from sklearn.impute import SimpleImputer
    from sklearn.utils.class_weight import (
        compute_class_weight, compute_sample_weight
    )
    from sklearn.model_selection import (
        train_test_split, GridSearchCV, RandomizedSearchCV, ParameterSampler
    )
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        make_scorer, f1_score
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.multioutput import MultiOutputClassifier

    from imblearn.over_sampling import RandomOverSampler, SMOTE

    from pytorch_tabnet.tab_model import TabNetClassifier

    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Dropout, Input, Embedding, Flatten, Concatenate
    )
    from tensorflow.keras.callbacks import EarlyStopping
    from imblearn.pipeline import Pipeline as ImbPipeline
    from scipy.stats import randint, uniform
    from sklearn.neighbors import NearestNeighbors

    from imblearn.over_sampling import SMOTENC

    return (
        ColumnTransformer,
        ImbPipeline,
        LabelEncoder,
        MLPClassifier,
        MultiOutputClassifier,
        OneHotEncoder,
        ParameterSampler,
        Pipeline,
        RandomOverSampler,
        RandomizedSearchCV,
        SMOTE,
        SMOTENC,
        SimpleImputer,
        StandardScaler,
        TabNetClassifier,
        accuracy_score,
        classification_report,
        compute_class_weight,
        confusion_matrix,
        f1_score,
        make_scorer,
        np,
        pd,
        randint,
        torch,
        train_test_split,
        uniform,
    )


@app.cell
def _(pd):
    # ============================================================
    # Load datasets
    # ============================================================

    df_company_a = pd.read_csv('/Users/yanisbrocard/Desktop/TUM/Courses/Summer semester 2026/KI für Produktion/aiinpe_exercise/data/company_a_ML.csv')
    df_company_b = pd.read_csv('/Users/yanisbrocard/Desktop/TUM/Courses/Summer semester 2026/KI für Produktion/aiinpe_exercise/data/company_b_ML.csv')

    print("Company A shape:", df_company_a.shape)
    print("Company B shape:", df_company_b.shape)

    # ============================================================
    # Merge datasets
    #============================================================

    #df = pd.concat(
    #    [df_company_a, df_company_b],
    #    ignore_index=True
    #)

    df=df_company_a
    #df=df_company_b

    print("Merged dataset shape:", df.shape)

    for col in ["Supplier", "Unique Product Identifier"]:
        print(f"\n{col}")
        print(df[col].map(type).value_counts())

    df["Supplier"] = (
        df["Supplier"]
        .fillna("Unknown")
        .astype(str)
    )

    df["Unique Product Identifier"] = (
        df["Unique Product Identifier"]
        .fillna("Unknown")
        .astype(str)
    )
    return df, df_company_b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Functions
    """)
    return


@app.cell
def _(accuracy_score, classification_report, confusion_matrix, np):
    # ============================================================
    # DELIVERY EVALUATION
    # ============================================================

    def evaluate_delivery(y_train, train_predictions, y_test, predictions):
        print("\n===================================================")
        print("DELIVERY PERFORMANCE")
        print("===================================================")

        train_accuracy = accuracy_score(
            y_train["Delivery Status"],
            train_predictions["Delivery Status"]
        )

        test_accuracy = accuracy_score(
            y_test["Delivery Status"],
            predictions["Delivery Status"]
        )

        overfitting = train_accuracy - test_accuracy

        print(f"Train Accuracy : {train_accuracy:.4f}")
        print(f"Test Accuracy  : {test_accuracy:.4f}")
        print(f"Overfitting Gap: {overfitting:.4f}")

        if overfitting > 0.05:
            print("⚠️ Overfitting detected.")
        else:
            print("✓ No significant overfitting.")

        print("\nClassification Report")
        print(
            classification_report(
                y_test["Delivery Status"],
                predictions["Delivery Status"],
                zero_division=0
            )
        )

        print("\nConfusion Matrix")
        print(
            confusion_matrix(
                y_test["Delivery Status"],
                predictions["Delivery Status"]
            )
        )


    # ============================================================
    # QUANTITY EVALUATION
    # ============================================================

    def evaluate_quantity(y_train, train_predictions, y_test, predictions):
        print("\n===================================================")
        print("QUANTITY PERFORMANCE")
        print("===================================================")

        train_accuracy = accuracy_score(
            y_train["Quantity Status"],
            train_predictions["Quantity Status"]
        )

        test_accuracy = accuracy_score(
            y_test["Quantity Status"],
            predictions["Quantity Status"]
        )

        overfitting = train_accuracy - test_accuracy

        print(f"Train Accuracy : {train_accuracy:.4f}")
        print(f"Test Accuracy  : {test_accuracy:.4f}")
        print(f"Overfitting Gap: {overfitting:.4f}")

        if overfitting > 0.05:
            print("⚠️ Overfitting detected.")
        else:
            print("✓ No significant overfitting.")

        print("\nClassification Report")
        print(
            classification_report(
                y_test["Quantity Status"],
                predictions["Quantity Status"],
                zero_division=0
            )
        )

        print("\nConfusion Matrix")
        print(
            confusion_matrix(
                y_test["Quantity Status"],
                predictions["Quantity Status"]
            )
        )

    # ============================================================
    # CYCLIC FEATURES
    # ============================================================

    def add_cyclic_features(df):

        df = df.copy()


        df["month_sin"] = np.sin(
            2*np.pi*df["Planned Month"]/12
        )

        df["month_cos"] = np.cos(
            2*np.pi*df["Planned Month"]/12
        )


        df["quarter_sin"] = np.sin(
            2*np.pi*df["Planned Quarter"]/4
        )

        df["quarter_cos"] = np.cos(
            2*np.pi*df["Planned Quarter"]/4
        )


        df["weekday_sin"] = np.sin(
            2*np.pi*df["Planned Weekday"]/7
        )

        df["weekday_cos"] = np.cos(
            2*np.pi*df["Planned Weekday"]/7
        )


        df["day_sin"] = np.sin(
            2*np.pi*df["Planned Day"]/31
        )

        df["day_cos"] = np.cos(
            2*np.pi*df["Planned Day"]/31
        )


        return df


    return add_cyclic_features, evaluate_delivery, evaluate_quantity


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multilayer Percepton : Simple Neurone Network
    """)
    return


@app.cell
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

    Influence of every hyperparameters :
    - hidden_layer_sizes: defines the number and size of hidden layers. More neurons/layers → more complex model, but higher risk of overfitting.
    - activation function used in neurons (e.g., ReLU, tanh). It affects how the model learns non-linear patterns.
    - alpha: regularization parameter. Higher values reduce overfitting by penalizing large weights.
    - learning_rate_init: initial learning speed. High value → faster but less stable learning; low value → slower but more stable.
    - max_iter: maximum number of training iterations. Too low → underfitting; too high → longer training time.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    LabelEncoder,
    MLPClassifier,
    MultiOutputClassifier,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    df,
    evaluate_delivery,
    evaluate_quantity,
    pd,
    train_test_split,
):
    # ============================================================
    # Features and Targets
    # ============================================================

    mlp_feature_cols = [
        'Order Complexity',
        'Supplier',
        'Unique Product Identifier',
        'Planned Year',
        'Planned Month',
        'Planned Quarter',
        'Planned Day',
        'Planned Weekday',
        'Ordered Quantity'
    ]

    mlp_target_cols = [
        'Delivery Status',
        'Quantity Status'
    ]

    mlp_X = df[mlp_feature_cols].copy()
    mlp_y = df[mlp_target_cols].copy()

    delivery_encoder = LabelEncoder()
    quantity_encoder = LabelEncoder()

    mlp_y["Delivery Status"] = delivery_encoder.fit_transform(
        mlp_y["Delivery Status"]
    )

    mlp_y["Quantity Status"] = quantity_encoder.fit_transform(
        mlp_y["Quantity Status"]
    )

    # ============================================================
    # Variable types
    # ============================================================

    mlp_categorical_features = [
        'Supplier',
        'Unique Product Identifier'
    ]

    mlp_numeric_features = [
        col for col in mlp_feature_cols
        if col not in mlp_categorical_features
    ]

    # ============================================================
    # Preprocessing
    # ============================================================

    mlp_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                mlp_numeric_features
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                mlp_categorical_features
            )
        ]
    )

    # ============================================================
    # MLP Train / Test Split + Deal with imbalanced Quantity Status
    # ============================================================

    mlp_X_train, mlp_X_test, mlp_y_train, mlp_y_test = train_test_split(
        mlp_X,
        mlp_y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )


    print(f"Training samples : {len(mlp_X_train)}")
    print(f"Test samples     : {len(mlp_X_test)}")

    # ============================================================
    # MLP Model (single architecture)
    # ============================================================

    mlp_classifier = MLPClassifier(
        hidden_layer_sizes=(128,64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )

    mlp_model = Pipeline([
        ("preprocessing", mlp_preprocessor),
        ("classifier", MultiOutputClassifier(mlp_classifier))
    ])

    # ============================================================
    # Training
    # ============================================================

    print("Training MLP...")

    mlp_model.fit(mlp_X_train, mlp_y_train)

    print("Training finished.")

    # ============================================================
    # Predictions on the training set
    # ============================================================

    mlp_predictions = mlp_model.predict(mlp_X_test)

    print("\nPredictions shape :", mlp_predictions.shape)

    # ============================================================
    # Model Evaluation
    # ============================================================

    # Predictions for test 
    mlp_predictions = mlp_model.predict(mlp_X_test)

    mlp_predictions = pd.DataFrame(
        mlp_predictions,
        columns=mlp_target_cols,
        index=mlp_y_test.index
    )

    #Predictions for train 
    train_predictions = pd.DataFrame(
        mlp_model.predict(mlp_X_train),
        columns=mlp_y_train.columns,
        index=mlp_y_train.index
    )


    evaluate_quantity(mlp_y_train,train_predictions, mlp_y_test,mlp_predictions)
    evaluate_delivery(mlp_y_train,train_predictions, mlp_y_test,mlp_predictions)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Find the best parameters
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    LabelEncoder,
    MLPClassifier,
    OneHotEncoder,
    Pipeline,
    RandomizedSearchCV,
    StandardScaler,
    accuracy_score,
    add_cyclic_features,
    classification_report,
    compute_class_weight,
    confusion_matrix,
    df,
    f1_score,
    make_scorer,
    np,
    train_test_split,
):
    # ============================================================
    # FEATURES & TARGETS
    # ============================================================

    mlp_feature_cols_best = [
        'Order Complexity',
        'Supplier',
        'Unique Product Identifier',
        'Planned Year',
        'Planned Month',
        'Planned Quarter',
        'Planned Day',
        'Planned Weekday',
        'Ordered Quantity'
    ]

    mlp_X_best = df[mlp_feature_cols_best].copy()
    mlp_y_best = df[['Delivery Status', 'Quantity Status']].copy()

    # ============================================================
    # LABEL ENCODING TARGETS
    # ============================================================

    delivery_encoder_best = LabelEncoder()
    quantity_encoder_best = LabelEncoder()

    mlp_y_best["Delivery Status"] = delivery_encoder_best.fit_transform(
        mlp_y_best["Delivery Status"]
    )

    mlp_y_best["Quantity Status"] = quantity_encoder_best.fit_transform(
        mlp_y_best["Quantity Status"]
    )

    # ============================================================
    # CYCLIC FEATURES
    # ============================================================

    mlp_X_best = add_cyclic_features(mlp_X_best)

    mlp_X_best = mlp_X_best.drop(columns=[
        "Planned Month",
        "Planned Quarter",
        "Planned Weekday",
        "Planned Day"
    ])

    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================

    X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(
        mlp_X_best,
        mlp_y_best,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # ============================================================
    # PREPROCESSING
    # ============================================================

    categorical_best = [
        "Supplier",
        "Unique Product Identifier"
    ]

    numeric_best = [
        col
        for col in X_train_best.columns
        if col not in categorical_best
    ]

    preprocessor_best = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("scaler", StandardScaler())
                ]),
                numeric_best
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_best
            )
        ]
    )

    # ============================================================
    # CLASS WEIGHTS (FOR INFORMATION)
    # ============================================================

    classes_best = np.unique(
        y_train_best["Quantity Status"]
    )

    weights_best = compute_class_weight(
        class_weight="balanced",
        classes=classes_best,
        y=y_train_best["Quantity Status"]
    )

    class_weight_dict_best = dict(
        zip(classes_best, weights_best)
    )

    print("Quantity Status class weights:")
    print(class_weight_dict_best)

    # ============================================================
    # BASE MODEL
    # ============================================================

    mlp_base_best = MLPClassifier(
        max_iter=500,
        early_stopping=True,
        random_state=42
    )

    # ============================================================
    # HYPERPARAMETER SEARCH SPACE
    # ============================================================

    param_grid_best = {
        "hidden_layer_sizes": [
            (64,),
            (128,),
            (128, 64),
            (256, 128),
            (256, 128, 64)
        ],
        "alpha": [
            1e-5,
            1e-4,
            1e-3
        ],
        "learning_rate_init": [
            1e-4,
            5e-4,
            1e-3
        ],
        "activation": [
            "relu",
            "tanh"
        ]
    }

    # ============================================================
    # SCORING
    # ============================================================

    scorer_best = make_scorer(
        f1_score,
        average="macro"
    )

    # ============================================================
    # GRID SEARCH FUNCTION
    # ============================================================

    def train_mlp_best(target_best):

        print("\n" + "=" * 60)
        print(f"TRAINING MLP FOR: {target_best}")
        print("=" * 60)

        model_best = Pipeline([
            ("preprocess", preprocessor_best),
            ("mlp", mlp_base_best)
        ])

        search_best = RandomizedSearchCV(
            estimator=model_best,
            param_distributions={
                "mlp__hidden_layer_sizes":
                    param_grid_best["hidden_layer_sizes"],
                "mlp__alpha":
                    param_grid_best["alpha"],
                "mlp__learning_rate_init":
                    param_grid_best["learning_rate_init"],
                "mlp__activation":
                    param_grid_best["activation"]
            },
            n_iter=20,
            cv=3,
            scoring=scorer_best,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )

        search_best.fit(
            X_train_best,
            y_train_best[target_best]
        )

        print("\nBest params:")
        print(search_best.best_params_)

        print("\nBest CV score:")
        print(search_best.best_score_)

        return search_best.best_estimator_

    # ============================================================
    # TRAIN BOTH MODELS
    # ============================================================

    best_delivery_mlp = train_mlp_best(
        "Delivery Status"
    )

    best_quantity_mlp = train_mlp_best(
        "Quantity Status"
    )

    # ============================================================
    # EVALUATION WITH OVERFITTING ANALYSIS
    # ============================================================

    def evaluate_best(model_best, target_best):

        # --------------------------------------------------------
        # TRAIN PERFORMANCE
        # --------------------------------------------------------

        train_pred_best = model_best.predict(
            X_train_best
        )

        train_accuracy_best = accuracy_score(
            y_train_best[target_best],
            train_pred_best
        )

        # --------------------------------------------------------
        # TEST PERFORMANCE
        # --------------------------------------------------------

        test_pred_best = model_best.predict(
            X_test_best
        )

        test_accuracy_best = accuracy_score(
            y_test_best[target_best],
            test_pred_best
        )

        # --------------------------------------------------------
        # OVERFITTING GAP
        # --------------------------------------------------------

        overfitting_gap_best = (
            train_accuracy_best -
            test_accuracy_best
        )

        # --------------------------------------------------------
        # RESULTS
        # --------------------------------------------------------

        print("\n" + "=" * 60)
        print(target_best)
        print("=" * 60)

        print(
            f"Train Accuracy : "
            f"{train_accuracy_best:.4f}"
        )

        print(
            f"Test Accuracy  : "
            f"{test_accuracy_best:.4f}"
        )

        print(
            f"Overfitting Gap: "
            f"{overfitting_gap_best:.4f}"
        )

        if overfitting_gap_best > 0.05:
            print("⚠️ Overfitting detected.")
        else:
            print("✓ No significant overfitting.")

        print("\nClassification Report")

        print(
            classification_report(
                y_test_best[target_best],
                test_pred_best,
                zero_division=0
            )
        )

        print("\nConfusion Matrix")

        print(
            confusion_matrix(
                y_test_best[target_best],
                test_pred_best
            )
        )

    # ============================================================
    # FINAL EVALUATION
    # ============================================================

    evaluate_best(
        best_delivery_mlp,
        "Delivery Status"
    )

    evaluate_best(
        best_quantity_mlp,
        "Quantity Status"
    )
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
        "\"\")
    """)
    return


@app.cell(hide_code=True)
def _(
    LabelEncoder,
    TabNetClassifier,
    accuracy_score,
    classification_report,
    confusion_matrix,
    df,
    train_test_split,
):
    #============================================================
    # Features and Targets
    # ============================================================

    tabnet_feature_cols = [
        'Order Complexity',
        'Supplier',
        'Unique Product Identifier',
        'Planned Year',
        'Planned Month',
        'Planned Quarter',
        'Planned Day',
        'Planned Weekday',
        'Ordered Quantity'
    ]

    tabnet_target_cols = [
        'Delivery Status',
        'Quantity Status'
    ]

    tabnet_X = df[tabnet_feature_cols].copy()
    tabnet_y = df[tabnet_target_cols].copy()

    # ============================================================
    # Encode targets
    # ============================================================

    tabnet_delivery_encoder = LabelEncoder()
    tabnet_quantity_encoder = LabelEncoder()

    tabnet_y["Delivery Status"] = (
        tabnet_delivery_encoder.fit_transform(
            tabnet_y["Delivery Status"]
        )
    )

    tabnet_y["Quantity Status"] = (
        tabnet_quantity_encoder.fit_transform(
            tabnet_y["Quantity Status"]
        )
    )

    # ============================================================
    # Encode categorical features
    # ============================================================

    tabnet_categorical_features = [
        'Supplier',
        'Unique Product Identifier'
    ]

    tabnet_feature_encoders = {}

    for tabnet_feature in tabnet_categorical_features:

        tabnet_encoder = LabelEncoder()

        tabnet_X[tabnet_feature] = (
            tabnet_encoder.fit_transform(
                tabnet_X[tabnet_feature].astype(str)
            )
        )

        tabnet_feature_encoders[tabnet_feature] = tabnet_encoder

    # ============================================================
    # Train / Test Split
    # ============================================================

    (
        tabnet_X_train,
        tabnet_X_test,
        tabnet_y_train,
        tabnet_y_test
    ) = train_test_split(
        tabnet_X,
        tabnet_y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Training samples : {len(tabnet_X_train)}")
    print(f"Test samples     : {len(tabnet_X_test)}")

    # ============================================================
    # Convert to numpy
    # ============================================================

    tabnet_X_train_np = tabnet_X_train.values
    tabnet_X_test_np = tabnet_X_test.values

    tabnet_y_delivery_train = (
        tabnet_y_train["Delivery Status"].values
    )

    tabnet_y_delivery_test = (
        tabnet_y_test["Delivery Status"].values
    )

    tabnet_y_quantity_train = (
        tabnet_y_train["Quantity Status"].values
    )

    tabnet_y_quantity_test = (
        tabnet_y_test["Quantity Status"].values
    )

    # ============================================================
    # Delivery Status TabNet
    # ============================================================

    tabnet_delivery_model = TabNetClassifier(
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        seed=42
    )

    print("\nTraining Delivery Status TabNet...")

    tabnet_delivery_model.fit(
        tabnet_X_train_np,
        tabnet_y_delivery_train,
        eval_set=[
            (
                tabnet_X_test_np,
                tabnet_y_delivery_test
            )
        ],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128
    )

    # ============================================================
    # Quantity Status TabNet
    # ============================================================

    tabnet_quantity_model = TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=7,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        seed=42
    )

    print("\nTraining Quantity Status TabNet...")

    tabnet_quantity_model.fit(
        tabnet_X_train_np,
        tabnet_y_quantity_train,
        eval_set=[
            (
                tabnet_X_test_np,
                tabnet_y_quantity_test
            )
        ],
        max_epochs=300,
        patience=20,
        batch_size=1024,
        virtual_batch_size=128
    )

    # ============================================================
    # Predictions
    # ============================================================

    tabnet_delivery_predictions = (
        tabnet_delivery_model.predict(
            tabnet_X_test_np
        )
    )

    tabnet_quantity_predictions = (
        tabnet_quantity_model.predict(
            tabnet_X_test_np
        )
    )

    # ============================================================
    # Delivery Status Evaluation
    # ============================================================

    print("\n===================================================")
    print("TABNET DELIVERY STATUS PERFORMANCE")
    print("===================================================")

    tabnet_delivery_accuracy = accuracy_score(
        tabnet_y_delivery_test,
        tabnet_delivery_predictions
    )

    print(
        f"Accuracy : {tabnet_delivery_accuracy:.4f}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            tabnet_y_delivery_test,
            tabnet_delivery_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix")

    tabnet_delivery_confusion_matrix = (
        confusion_matrix(
            tabnet_y_delivery_test,
            tabnet_delivery_predictions
        )
    )

    print(tabnet_delivery_confusion_matrix)

    # ============================================================
    # Quantity Status Evaluation
    # ============================================================

    print("\n===================================================")
    print("TABNET QUANTITY STATUS PERFORMANCE")
    print("===================================================")

    tabnet_quantity_accuracy = accuracy_score(
        tabnet_y_quantity_test,
        tabnet_quantity_predictions
    )

    print(
        f"Accuracy : {tabnet_quantity_accuracy:.4f}"
    )

    print("\nClassification Report")

    print(
        classification_report(
            tabnet_y_quantity_test,
            tabnet_quantity_predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix")

    tabnet_quantity_confusion_matrix = (
        confusion_matrix(
            tabnet_y_quantity_test,
            tabnet_quantity_predictions
        )
    )

    print(tabnet_quantity_confusion_matrix)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TabNet (improved : best hyperparameters + imbalanecd data )
    """)
    return


@app.cell
def _():
    return


@app.cell
def _(
    LabelEncoder,
    ParameterSampler,
    TabNetClassifier,
    accuracy_score,
    classification_report,
    confusion_matrix,
    df,
    f1_score,
    randint,
    torch,
    train_test_split,
    uniform,
):
    # ============================================================
    # FEATURES & TARGETS
    # ============================================================

    feature_cols = [
        'Order Complexity',
        'Supplier',
        'Unique Product Identifier',
        'Planned Year',
        'Planned Month',
        'Planned Quarter',
        'Planned Day',
        'Planned Weekday',
        'Ordered Quantity'
    ]

    X = df[feature_cols].copy()
    y = df[['Delivery Status', 'Quantity Status']].copy()

    # ============================================================
    # LABEL ENCODING TARGETS
    # ============================================================

    delivery_encoder_TN_improved = LabelEncoder()
    quantity_encoder_TN_improved = LabelEncoder()

    y["Delivery Status"] = delivery_encoder_TN_improved.fit_transform(
        y["Delivery Status"]
    )

    y["Quantity Status"] = quantity_encoder_TN_improved.fit_transform(
        y["Quantity Status"]
    )

    # ============================================================
    # CATEGORICAL ENCODING
    # ============================================================

    cat_cols = [
        "Supplier",
        "Unique Product Identifier"
    ]

    cat_idxs = [
        X.columns.get_loc(col)
        for col in cat_cols
    ]

    cat_dims = []

    for column in cat_cols:
        le = LabelEncoder()
        X[column] = le.fit_transform(
            X[column].astype(str)
        )
        cat_dims.append(
            X[column].nunique()
        )

    # ============================================================
    # TRAIN / TEST SPLIT
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y["Delivery Status"]
    )

    # ============================================================
    # SEARCH SPACE
    # ============================================================

    param_grid = {
        "n_d": randint(16, 64),
        "n_a": randint(16, 64),
        "n_steps": randint(3, 10),
        "gamma": uniform(1.0, 1.8),
        "lambda_sparse": uniform(1e-6, 1e-3),
        "mask_type": ["entmax", "sparsemax"],
        "learning_rate": [0.005, 0.01, 0.02],
        "batch_size": [512, 1024, 2048]
    }

    # ============================================================
    # TABNET SEARCH FUNCTION
    # ============================================================

    def train_tabnet(target_name):

        print("\n" + "=" * 70)
        print(f"OPTIMIZING TABNET FOR {target_name}")
        print("=" * 70)

        best_score = -999
        best_model = None
        best_params = None

        y_train_target = y_train[target_name].values
        y_test_target = y_test[target_name].values

        candidates = list(
            ParameterSampler(
                param_grid,
                n_iter=30,
                random_state=42
            )
        )

        for i, params in enumerate(candidates):

            print(f"\nTrial {i+1}/{len(candidates)}")

            model = TabNetClassifier(
                n_d=params["n_d"],
                n_a=params["n_a"],
                n_steps=params["n_steps"],
                gamma=params["gamma"],
                lambda_sparse=params["lambda_sparse"],
                mask_type=params["mask_type"],
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(
                    lr=params["learning_rate"]
                ),
                scheduler_params={
                    "step_size": 20,
                    "gamma": 0.9
                },
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                verbose=0
            )

            model.fit(
                X_train.values,
                y_train_target,
                eval_set=[
                    (X_test.values, y_test_target)
                ],
                eval_metric=["balanced_accuracy"],
                max_epochs=200,
                patience=30,
                batch_size=params["batch_size"],
                virtual_batch_size=128
            )

            preds = model.predict(
                X_test.values
            )

            score = f1_score(
                y_test_target,
                preds,
                average="macro"
            )

            print(
                f"Macro F1 = {score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

        print("\nBest Parameters")
        print(best_params)

        print(
            f"\nBest Macro F1: {best_score:.4f}"
        )

        return best_model

    # ============================================================
    # TRAIN DELIVERY MODEL
    # ============================================================

    delivery_model = train_tabnet(
        "Delivery Status"
    )

    # ============================================================
    # TRAIN QUANTITY MODEL
    # ============================================================

    quantity_model = train_tabnet(
        "Quantity Status"
    )

    # ============================================================
    # EVALUATION FUNCTION
    # ============================================================

    def evaluate_model(
        model,
        target_name
    ):

        y_train_target = y_train[target_name].values
        y_test_target = y_test[target_name].values

        train_pred = model.predict(
            X_train.values
        )

        test_pred = model.predict(
            X_test.values
        )

        train_acc = accuracy_score(
            y_train_target,
            train_pred
        )

        test_acc = accuracy_score(
            y_test_target,
            test_pred
        )

        gap = train_acc - test_acc

        print("\n" + "=" * 70)
        print(target_name)
        print("=" * 70)

        print(
            f"Train Accuracy : {train_acc:.4f}"
        )

        print(
            f"Test Accuracy  : {test_acc:.4f}"
        )

        print(
            f"Overfitting Gap: {gap:.4f}"
        )

        if gap > 0.05:
            print("⚠️ Overfitting detected.")
        else:
            print("✓ No significant overfitting.")

        print("\nClassification Report")

        print(
            classification_report(
                y_test_target,
                test_pred,
                zero_division=0
            )
        )

        print("\nConfusion Matrix")

        print(
            confusion_matrix(
                y_test_target,
                test_pred
            )
        )

    # ============================================================
    # RESULTS
    # ============================================================

    evaluate_model(
        delivery_model,
        "Delivery Status"
    )

    evaluate_model(
        quantity_model,
        "Quantity Status"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Add some features to improve the model
    """)
    return


@app.cell
def _(df_company_b, np, pd):

    # ===============================
    # Chargement des données
    # ===============================
    #df_new_features = df_company_a
    df_new_features = df_company_b

    # =====================================================
    # 1. Variables temporelles
    # =====================================================

    # Date prévue
    df_new_features["Planned_Date"] = pd.to_datetime({
        "year": df_new_features["Planned Year"],
        "month": df_new_features["Planned Month"],
        "day": df_new_features["Planned Day"]
    })

    # Date d'arrivée
    df_new_features["Arrival_Date"] = pd.to_datetime({
        "year": df_new_features["Arrival Year"],
        "month": df_new_features["Arrival Month"],
        "day": df_new_features["Arrival Day"]
    })

    # Jour de l'année
    df_new_features["Planned_DayOfYear"] = df_new_features["Planned_Date"].dt.dayofyear
    df_new_features["Arrival_DayOfYear"] = df_new_features["Arrival_Date"].dt.dayofyear

    # Semaine de l'année
    df_new_features["Planned_Week"] = df_new_features["Planned_Date"].dt.isocalendar().week.astype(int)
    df_new_features["Arrival_Week"] = df_new_features["Arrival_Date"].dt.isocalendar().week.astype(int)

    # Fin de semaine
    df_new_features["Planned_IsWeekend"] = (df_new_features["Planned Weekday"] >= 5).astype(int)
    df_new_features["Arrival_IsWeekend"] = (df_new_features["Arrival Weekday"] >= 5).astype(int)

    # Début / fin de mois
    df_new_features["Planned_IsMonthEnd"] = df_new_features["Planned_Date"].dt.is_month_end.astype(int)
    df_new_features["Planned_IsMonthStart"] = df_new_features["Planned_Date"].dt.is_month_start.astype(int)

    # =====================================================
    # 2. Variables de quantité
    # =====================================================

    # Ratio livré / commandé
    df_new_features["Delivery_Ratio"] = (
        df_new_features["Delivered Quantity"] /
        df_new_features["Ordered Quantity"].replace(0, np.nan)
    )

    # Pourcentage non livré
    df_new_features["Missing_Ratio"] = (
        (df_new_features["Ordered Quantity"] - df_new_features["Delivered Quantity"]) /
        df_new_features["Ordered Quantity"].replace(0, np.nan)
    )

    # Quantité absolue manquante
    df_new_features["Missing_Quantity"] = (
        df_new_features["Ordered Quantity"] - df_new_features["Delivered Quantity"]
    )

    # =====================================================
    # 3. Variables liées au délai
    # =====================================================

    df_new_features["Absolute_Delay"] = df_new_features["Delivery_Delay_Days"].abs()

    df_new_features["Late_Delivery"] = (
        df_new_features["Delivery_Delay_Days"] > 0
    ).astype(int)

    df_new_features["Very_Late"] = (
        df_new_features["Delivery_Delay_Days"] > 7
    ).astype(int)

    df_new_features["Early_Delivery"] = (
        df_new_features["Delivery_Delay_Days"] < 0
    ).astype(int)

    # =====================================================
    # 4. Encodage de la saison
    # =====================================================

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"

    df_new_features["Season"] = df_new_features["Planned Month"].apply(get_season)

    # =====================================================
    # 5. Fréquence des fournisseurs
    # =====================================================

    supplier_freq = df_new_features["Supplier"].value_counts()

    df_new_features["Supplier_Frequency"] = df_new_features["Supplier"].map(supplier_freq)

    # =====================================================
    # 6. Fréquence des produits
    # =====================================================

    product_freq = df_new_features["Unique Product Identifier"].value_counts()

    df_new_features["Product_Frequency"] = (
        df_new_features["Unique Product Identifier"].map(product_freq)
    )

    # =====================================================
    # 7. Taille de la commande
    # =====================================================

    df_new_features["Large_Order"] = (
        df_new_features["Ordered Quantity"] >
        df_new_features["Ordered Quantity"].median()
    ).astype(int)

    # =====================================================
    # 8. Interactions
    # =====================================================

    df_new_features["Delay_x_Quantity"] = (
        df_new_features["Delivery_Delay_Days"] *
        df_new_features["Ordered Quantity"]
    )

    df_new_features["Complexity_x_Quantity"] = (
        df_new_features["Order Complexity"] *
        df_new_features["Ordered Quantity"]
    )

    df_new_features["Complexity_x_Delay"] = (
        df_new_features["Order Complexity"] *
        df_new_features["Delivery_Delay_Days"]
    )

    # =====================================================
    # 9. Encodage cyclique des mois et jours
    # =====================================================

    df_new_features["Month_sin"] = np.sin(
        2 * np.pi * df_new_features["Planned Month"] / 12
    )
    df_new_features["Month_cos"] = np.cos(
        2 * np.pi * df_new_features["Planned Month"] / 12
    )

    df_new_features["Weekday_sin"] = np.sin(
        2 * np.pi * df_new_features["Planned Weekday"] / 7
    )
    df_new_features["Weekday_cos"] = np.cos(
        2 * np.pi * df_new_features["Planned Weekday"] / 7
    )

    feature_cols_new_features = [
        # Variables d'origine
        'Order Complexity',
        'Supplier',
        'Unique Product Identifier',
        'Planned Year',
        'Planned Month',
        'Planned Quarter',
        'Planned Day',
        'Planned Weekday',
        'Ordered Quantity',

        # Nouvelles features
        'Planned_DayOfYear',
        'Planned_Week',
        'Planned_IsWeekend',
        'Planned_IsMonthEnd',
        'Planned_IsMonthStart',
        'Season',
        'Supplier_Frequency',
        'Product_Frequency',
        'Large_Order',
        'Month_sin',
        'Month_cos',
        'Weekday_sin',
        'Weekday_cos'
    ]

    target_cols_new_features = [
        'Delivery Status',
        'Quantity Status'
    ]

    categorical_features_new_features = [
        'Supplier',
        'Unique Product Identifier',
        'Season'
    ]

    return df_new_features, feature_cols_new_features, target_cols_new_features


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##New MLP
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    LabelEncoder,
    MLPClassifier,
    MultiOutputClassifier,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    df_new_features,
    evaluate_delivery,
    evaluate_quantity,
    feature_cols_new_features,
    pd,
    target_cols_new_features,
    train_test_split,
):
    # ============================================================
    # Data
    # ============================================================

    mlp_X_new_features = df_new_features[feature_cols_new_features].copy()
    mlp_y_new_features = df_new_features[target_cols_new_features].copy()

    delivery_encoder_new_features = LabelEncoder()
    quantity_encoder_new_features = LabelEncoder()

    mlp_y_new_features["Delivery Status"] = (
        delivery_encoder_new_features.fit_transform(
            mlp_y_new_features["Delivery Status"]
        )
    )

    mlp_y_new_features["Quantity Status"] = (
        quantity_encoder_new_features.fit_transform(
            mlp_y_new_features["Quantity Status"]
        )
    )

    # ============================================================
    # Variable types
    # ============================================================

    mlp_categorical_features_new_features = [
        "Supplier",
        "Unique Product Identifier",
        "Season"
    ]

    mlp_numeric_features_new_features = [
        col for col in feature_cols_new_features
        if col not in mlp_categorical_features_new_features
    ]

    # ============================================================
    # Preprocessing
    # ============================================================

    mlp_preprocessor_new_features = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                mlp_numeric_features_new_features
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                mlp_categorical_features_new_features
            )
        ]
    )

    # ============================================================
    # Train / Test Split
    # ============================================================

    (
        mlp_X_train_new_features,
        mlp_X_test_new_features,
        mlp_y_train_new_features,
        mlp_y_test_new_features
    ) = train_test_split(
        mlp_X_new_features,
        mlp_y_new_features,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    print(f"Training samples : {len(mlp_X_train_new_features)}")
    print(f"Test samples     : {len(mlp_X_test_new_features)}")

    # ============================================================
    # MLP Model
    # ============================================================

    mlp_classifier_new_features = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )

    mlp_model_new_features = Pipeline([
        ("preprocessing", mlp_preprocessor_new_features),
        ("classifier", MultiOutputClassifier(mlp_classifier_new_features))
    ])

    # ============================================================
    # Training
    # ============================================================

    print("Training MLP with new features...")

    mlp_model_new_features.fit(
        mlp_X_train_new_features,
        mlp_y_train_new_features
    )

    print("Training finished.")

    # ============================================================
    # Predictions
    # ============================================================

    mlp_predictions_new_features = mlp_model_new_features.predict(
        mlp_X_test_new_features
    )

    print("\nPredictions shape :", mlp_predictions_new_features.shape)

    # ============================================================
    # Predictions DataFrames
    # ============================================================

    mlp_predictions_new_features = pd.DataFrame(
        mlp_predictions_new_features,
        columns=target_cols_new_features,
        index=mlp_y_test_new_features.index
    )

    train_predictions_new_features = pd.DataFrame(
        mlp_model_new_features.predict(mlp_X_train_new_features),
        columns=mlp_y_train_new_features.columns,
        index=mlp_y_train_new_features.index
    )

    # ============================================================
    # Evaluation
    # ============================================================

    evaluate_quantity(
        mlp_y_train_new_features,
        train_predictions_new_features,
        mlp_y_test_new_features,
        mlp_predictions_new_features
    )

    evaluate_delivery(
        mlp_y_train_new_features,
        train_predictions_new_features,
        mlp_y_test_new_features,
        mlp_predictions_new_features
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Best parameters
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    LabelEncoder,
    MLPClassifier,
    MultiOutputClassifier,
    OneHotEncoder,
    Pipeline,
    RandomizedSearchCV,
    SimpleImputer,
    StandardScaler,
    classification_report,
    confusion_matrix,
    df_new_features,
    f1_score,
    feature_cols_new_features,
    make_scorer,
    np,
    target_cols_new_features,
    train_test_split,
):
    # ============================================================
    # Data
    # ============================================================

    mlp_X_new_features_best = (
        df_new_features[
            feature_cols_new_features
        ].copy()
    )


    mlp_y_new_features_best = (
        df_new_features[
            target_cols_new_features
        ].copy()
    )



    # ============================================================
    # Label Encoding
    # ============================================================


    delivery_encoder_new_features_best = LabelEncoder()

    quantity_encoder_new_features_best = LabelEncoder()



    mlp_y_new_features_best[
        "Delivery Status"
    ] = (

        delivery_encoder_new_features_best.fit_transform(

            mlp_y_new_features_best[
                "Delivery Status"
            ]
        )
    )

    mlp_y_new_features_best[
        "Quantity Status"
    ] = (

        quantity_encoder_new_features_best.fit_transform(

            mlp_y_new_features_best[
                "Quantity Status"
            ]
        )
    )

    # ============================================================
    # Variable types
    # ============================================================

    mlp_categorical_features_new_features_best = [
        "Supplier",

        "Unique Product Identifier",

        "Season"
    ]

    mlp_numeric_features_new_features_best = [

        col

        for col in feature_cols_new_features

        if col not in mlp_categorical_features_new_features_best
    ]

    # ============================================================
    # Preprocessing
    # ============================================================

    mlp_preprocessor_new_features_best = ColumnTransformer(

        transformers=[
            (
                "num",

                Pipeline([

                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                mlp_numeric_features_new_features_best
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                mlp_categorical_features_new_features_best
            )
        ]
    )

    # ============================================================
    # Train / Test Split
    # ============================================================

    (
        mlp_X_train_new_features_best,
        mlp_X_test_new_features_best,
        mlp_y_train_new_features_best,
        mlp_y_test_new_features_best

    ) = train_test_split(

        mlp_X_new_features_best,

        mlp_y_new_features_best,

        test_size=0.2,

        random_state=42,

        shuffle=True

    )

    print(
        f"Training samples : "
        f"{len(mlp_X_train_new_features_best)}"
    )

    print(
        f"Test samples : "
        f"{len(mlp_X_test_new_features_best)}"
    )

    # ============================================================
    # Base MLP
    # ============================================================

    mlp_classifier_new_features_best = MLPClassifier(

        max_iter=1000,

        early_stopping=True,

        validation_fraction=0.1,

        n_iter_no_change=20,

        random_state=42
    )

    mlp_model_new_features_best = Pipeline([
        (
            "preprocessing",
            mlp_preprocessor_new_features_best
        ),
        (
            "classifier",
            MultiOutputClassifier(
                mlp_classifier_new_features_best
            )
        )
    ])

    # ============================================================
    # Hyperparameter search space
    # ============================================================

    param_distributions_new_features_best = {
        "classifier__estimator__hidden_layer_sizes": [

            (32,),

            (64,),

            (128,),

            (64,32),

            (128,64),

            (256,128),

            (128,64,32)

        ],
        "classifier__estimator__activation": [

            "relu",

            "tanh"
        ],
        "classifier__estimator__alpha": [

            1e-5,

            1e-4,

            1e-3,

            1e-2

        ],
        "classifier__estimator__learning_rate_init": [

            1e-4,

            5e-4,

            1e-3
        ],
        "classifier__estimator__batch_size": [

            32,

            64,

            128
        ]
    }

    # ============================================================
    # Custom scoring for multi-output
    # ============================================================

    def multioutput_macro_f1_new_features_best(
            y_true_new_features_best,
            y_pred_new_features_best):

        delivery_f1_new_features_best = f1_score(

            y_true_new_features_best[:,0],

            y_pred_new_features_best[:,0],

            average="macro"
        )

        quantity_f1_new_features_best = f1_score(

            y_true_new_features_best[:,1],

            y_pred_new_features_best[:,1],

            average="macro"
        )
        return (
            delivery_f1_new_features_best
            +
            quantity_f1_new_features_best
        ) / 2
    scorer_new_features_best = make_scorer(

        multioutput_macro_f1_new_features_best
    )

    # ============================================================
    # Random Search
    # ============================================================

    mlp_random_search_new_features_best = RandomizedSearchCV(

        estimator=mlp_model_new_features_best,

        param_distributions=
        param_distributions_new_features_best,

        n_iter=40,

        cv=5,

        scoring=scorer_new_features_best,

        n_jobs=-1,

        verbose=2,

        random_state=42
    )

    # ============================================================
    # Training Search
    # ============================================================


    print(
        "Searching best MLP parameters..."
    )


    mlp_random_search_new_features_best.fit(

        mlp_X_train_new_features_best,

        mlp_y_train_new_features_best

    )

    print(
        "Search finished."
    )

    # ============================================================
    # Best parameters
    # ============================================================

    print("\nBEST PARAMETERS")

    print(
        mlp_random_search_new_features_best.best_params_
    )

    print("\nBEST CV SCORE")

    print(
        mlp_random_search_new_features_best.best_score_
    )

    # ============================================================
    # Best final model
    # ============================================================

    best_mlp_model_new_features_best = (

        mlp_random_search_new_features_best.best_estimator_
    )

    # ============================================================
    # FINAL EVALUATION
    # ============================================================

    def evaluate_mlp_new_features_best(
            model_new_features_best,
            target_names_new_features_best):
        # --------------------------------------------------------
        # Predictions
        # --------------------------------------------------------
        train_predictions_new_features_best = (
            model_new_features_best.predict(
                mlp_X_train_new_features_best
            )
        )
        test_predictions_new_features_best = (
            model_new_features_best.predict(
                mlp_X_test_new_features_best
            )
        )
        # --------------------------------------------------------
        # Accuracy calculation
        # --------------------------------------------------------
        train_accuracy_new_features_best = np.mean(

            np.all(
                train_predictions_new_features_best
                ==
                mlp_y_train_new_features_best.values,

                axis=1
            )
        )
        test_accuracy_new_features_best = np.mean(
            np.all(
                test_predictions_new_features_best
                ==
                mlp_y_test_new_features_best.values,

                axis=1
            )
        )
        # --------------------------------------------------------
        # Overfitting gap
        # --------------------------------------------------------
        overfitting_gap_new_features_best = (

            train_accuracy_new_features_best
            -
            test_accuracy_new_features_best
        )
    
        # --------------------------------------------------------
        # Results
        # --------------------------------------------------------
    
        print("\n" + "="*70)

        print("FINAL MODEL EVALUATION")

        print("="*70)

        print(

            f"Train Accuracy : "
            f"{train_accuracy_new_features_best:.4f}"

        )
    
        print(

            f"Test Accuracy  : "
            f"{test_accuracy_new_features_best:.4f}"

        )
    
        print(

            f"Overfitting Gap: "
            f"{overfitting_gap_new_features_best:.4f}"
        )
        # --------------------------------------------------------
        # Overfitting classification
        # --------------------------------------------------------

        if overfitting_gap_new_features_best < 0.02:

            print(
                "✓ Excellent generalization "
                "(No significant overfitting)"
            )
        elif overfitting_gap_new_features_best < 0.05:

            print(
                "✓ Low overfitting level "
                "(Acceptable)"
            )

        elif overfitting_gap_new_features_best < 0.10:

            print(
                "⚠ Moderate overfitting detected"
            )
        else:

            print(
                "❌ Strong overfitting detected"
            )

        # --------------------------------------------------------
        # Individual target evaluation
        # --------------------------------------------------------
    
        for i, target_new_features_best in enumerate(
            target_names_new_features_best
        ):


            print("\n" + "-"*70)

            print(
                target_new_features_best
            )

            print("-"*70)



            print("\nClassification Report")


            print(

                classification_report(

                    mlp_y_test_new_features_best.iloc[:,i],

                    test_predictions_new_features_best[:,i],

                    zero_division=0

                )

            )

            print("\nConfusion Matrix")

            print(
                confusion_matrix(

                    mlp_y_test_new_features_best.iloc[:,i],

                    test_predictions_new_features_best[:,i]

                )

            )
    # ============================================================
    # RUN FINAL EVALUATION
    # ============================================================

    evaluate_mlp_new_features_best(

        best_mlp_model_new_features_best,

        target_cols_new_features
    )
    return


@app.cell
def _():
    ## New TabNet to test with the new features 
    return


@app.cell
def _(
    LabelEncoder,
    TabNetClassifier,
    accuracy_score,
    classification_report,
    confusion_matrix,
    df_new_features,
    feature_cols_new_features,
    target_cols_new_features,
    train_test_split,
):
    # ============================================================
    # Data
    # ============================================================

    tabnet_X_new_features_tabnet = (
        df_new_features[feature_cols_new_features].copy()
    )

    tabnet_y_new_features_tabnet = (
        df_new_features[target_cols_new_features].copy()
    )

    # ============================================================
    # Encode targets
    # ============================================================

    tabnet_delivery_encoder_new_features_tabnet = LabelEncoder()
    tabnet_quantity_encoder_new_features_tabnet = LabelEncoder()

    tabnet_y_new_features_tabnet["Delivery Status"] = (
        tabnet_delivery_encoder_new_features_tabnet.fit_transform(
            tabnet_y_new_features_tabnet["Delivery Status"]
        )
    )

    tabnet_y_new_features_tabnet["Quantity Status"] = (
        tabnet_quantity_encoder_new_features_tabnet.fit_transform(
            tabnet_y_new_features_tabnet["Quantity Status"]
        )
    )

    # ============================================================
    # Encode categorical variables
    # ============================================================

    tabnet_categorical_features_new_features_tabnet = [
        "Supplier",
        "Unique Product Identifier",
        "Season"
    ]

    tabnet_feature_encoders_new_features_tabnet = {}

    for feature_new_features_tabnet in tabnet_categorical_features_new_features_tabnet:

        encoder_new_features_tabnet = LabelEncoder()

        tabnet_X_new_features_tabnet[feature_new_features_tabnet] = (
            encoder_new_features_tabnet.fit_transform(
                tabnet_X_new_features_tabnet[
                    feature_new_features_tabnet
                ].astype(str)
            )
        )

        tabnet_feature_encoders_new_features_tabnet[
            feature_new_features_tabnet
        ] = encoder_new_features_tabnet

    # ============================================================
    # Train / Test Split
    # ============================================================

    (
        tabnet_X_train_new_features_tabnet,
        tabnet_X_test_new_features_tabnet,
        tabnet_y_train_new_features_tabnet,
        tabnet_y_test_new_features_tabnet
    ) = train_test_split(

        tabnet_X_new_features_tabnet,

        tabnet_y_new_features_tabnet,

        test_size=0.20,

        random_state=42,

        shuffle=True

    )

    # ============================================================
    # Convert to numpy
    # ============================================================

    tabnet_X_train_np_new_features_tabnet = (
        tabnet_X_train_new_features_tabnet.values
    )

    tabnet_X_test_np_new_features_tabnet = (
        tabnet_X_test_new_features_tabnet.values
    )

    tabnet_y_delivery_train_new_features_tabnet = (
        tabnet_y_train_new_features_tabnet[
            "Delivery Status"
        ].values
    )

    tabnet_y_delivery_test_new_features_tabnet = (
        tabnet_y_test_new_features_tabnet[
            "Delivery Status"
        ].values
    )

    tabnet_y_quantity_train_new_features_tabnet = (
        tabnet_y_train_new_features_tabnet[
            "Quantity Status"
        ].values
    )

    tabnet_y_quantity_test_new_features_tabnet = (
        tabnet_y_test_new_features_tabnet[
            "Quantity Status"
        ].values
    )

    # ============================================================
    # Delivery TabNet
    # ============================================================

    tabnet_delivery_model_new_features_tabnet = TabNetClassifier(

        n_d=32,

        n_a=32,

        n_steps=5,

        gamma=1.3,

        lambda_sparse=1e-3,

        optimizer_params=dict(lr=1e-2),

        mask_type="entmax",

        seed=42

    )

    print("Training Delivery TabNet...")

    tabnet_delivery_model_new_features_tabnet.fit(

        tabnet_X_train_np_new_features_tabnet,

        tabnet_y_delivery_train_new_features_tabnet,

        eval_set=[

            (

                tabnet_X_test_np_new_features_tabnet,

                tabnet_y_delivery_test_new_features_tabnet

            )

        ],

        max_epochs=250,

        patience=30,

        batch_size=512,

        virtual_batch_size=64

    )

    # ============================================================
    # Quantity TabNet
    # ============================================================

    tabnet_quantity_model_new_features_tabnet = TabNetClassifier(

        n_d=64,

        n_a=64,

        n_steps=6,

        gamma=1.3,

        lambda_sparse=1e-3,

        optimizer_params=dict(lr=1e-2),

        mask_type="entmax",

        seed=42

    )

    print("Training Quantity TabNet...")

    tabnet_quantity_model_new_features_tabnet.fit(

        tabnet_X_train_np_new_features_tabnet,

        tabnet_y_quantity_train_new_features_tabnet,

        eval_set=[

            (

                tabnet_X_test_np_new_features_tabnet,

                tabnet_y_quantity_test_new_features_tabnet

            )

        ],

        max_epochs=300,

        patience=30,

        batch_size=512,

        virtual_batch_size=64

    )

    # ============================================================
    # Predictions
    # ============================================================

    delivery_train_pred_new_features_tabnet = (
        tabnet_delivery_model_new_features_tabnet.predict(
            tabnet_X_train_np_new_features_tabnet
        )
    )

    delivery_test_pred_new_features_tabnet = (
        tabnet_delivery_model_new_features_tabnet.predict(
            tabnet_X_test_np_new_features_tabnet
        )
    )

    quantity_train_pred_new_features_tabnet = (
        tabnet_quantity_model_new_features_tabnet.predict(
            tabnet_X_train_np_new_features_tabnet
        )
    )

    quantity_test_pred_new_features_tabnet = (
        tabnet_quantity_model_new_features_tabnet.predict(
            tabnet_X_test_np_new_features_tabnet
        )
    )

    # ============================================================
    # Evaluation function
    # ============================================================

    def evaluate_tabnet_new_features_tabnet(
        y_train,
        train_pred,
        y_test,
        test_pred,
        title
    ):

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        gap = train_acc - test_acc

        print("\n" + "="*60)
        print(title)
        print("="*60)

        print(f"Train Accuracy : {train_acc:.4f}")
        print(f"Test Accuracy  : {test_acc:.4f}")
        print(f"Overfitting Gap: {gap:.4f}")

        if gap < 0.02:
            print("✓ Excellent generalization")
        elif gap < 0.05:
            print("✓ Low overfitting")
        elif gap < 0.10:
            print("⚠ Moderate overfitting")
        else:
            print("❌ Strong overfitting")

        print("\nClassification Report")

        print(
            classification_report(
                y_test,
                test_pred,
                zero_division=0
            )
        )

        print("\nConfusion Matrix")

        print(
            confusion_matrix(
                y_test,
                test_pred
            )
        )

    # ============================================================
    # Final Evaluation
    # ============================================================

    evaluate_tabnet_new_features_tabnet(

        tabnet_y_delivery_train_new_features_tabnet,

        delivery_train_pred_new_features_tabnet,

        tabnet_y_delivery_test_new_features_tabnet,

        delivery_test_pred_new_features_tabnet,

        "DELIVERY STATUS"

    )

    evaluate_tabnet_new_features_tabnet(

        tabnet_y_quantity_train_new_features_tabnet,

        quantity_train_pred_new_features_tabnet,

        tabnet_y_quantity_test_new_features_tabnet,

        quantity_test_pred_new_features_tabnet,

        "QUANTITY STATUS"

    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Final MLP model with dealing with imbalanced data

    **MLP Performance Improvement**

    To improve the MLP model, an imbalance handling technique called **SMOTENC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)** was applied.

    The original dataset contains highly imbalanced classes, especially for **Quantity Status**, where some classes have very few samples. This causes the MLP model to focus mainly on the majority class and perform poorly on minority classes.

    The improvement process includes:

    * Splitting the dataset into training and testing sets while preserving the class distribution using **stratified sampling**.
    * Applying **SMOTENC** only on the training data to generate synthetic samples for minority classes while correctly handling both numerical and categorical variables.
    * Keeping the test set unchanged to ensure a fair evaluation of the model.
    * Using the optimized hyperparameters obtained during the tuning process:

      * Hidden layer size: (64,)
      * Activation function: ReLU
      * Learning rate: 0.001
      * Batch size: 128
      * Regularization (alpha): 0.001
    * Using **early stopping** to reduce overfitting and improve generalization.

    This approach aims to increase the model's ability to recognize minority classes, improve recall and F1-score for rare categories, and achieve a better balance between overall accuracy and class-level performance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Model for company A
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Explore the dataset
    """)
    return


@app.cell
def _(df_new_features, pd):
    # ============================================================
    # Dataset Exploration - Company A
    # ============================================================

    import matplotlib.pyplot as plt
    import seaborn as sns


    # ============================================================
    # Load data
    # ============================================================

    exploration_df_new_features_improved_A = (
        df_new_features.copy()
    )


    # ============================================================
    # Dataset information
    # ============================================================

    print("===== DATASET INFORMATION =====")

    print(
        "Number of rows:",
        len(exploration_df_new_features_improved_A)
    )

    print(
        "Number of columns:",
        exploration_df_new_features_improved_A.shape[1]
    )


    print("\nColumn names:")
    print(
        exploration_df_new_features_improved_A.columns.tolist()
    )


    # ============================================================
    # Target distribution
    # ============================================================

    target_cols_new_features_improved_A = [
        "Delivery Status",
        "Quantity Status"
    ]


    for target_new_features_improved_A in target_cols_new_features_improved_A:

        print("\n===================================")
        print(target_new_features_improved_A)
        print("===================================")


        distribution_new_features_improved_A = (
            exploration_df_new_features_improved_A[
                target_new_features_improved_A
            ]
            .value_counts()
        )


        percentage_new_features_improved_A = (
            exploration_df_new_features_improved_A[
                target_new_features_improved_A
            ]
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )


        result_distribution_new_features_improved_A = pd.DataFrame({

            "Count":
            distribution_new_features_improved_A,

            "Percentage (%)":
            percentage_new_features_improved_A

        })


        print(
            result_distribution_new_features_improved_A
        )


    # ============================================================
    # Visualization
    # ============================================================

    for target_new_features_improved_A in target_cols_new_features_improved_A:


        plt.figure(figsize=(7,5))


        sns.countplot(
            data=exploration_df_new_features_improved_A,
            x=target_new_features_improved_A
        )


        plt.title(
            f"Distribution of {target_new_features_improved_A}"
        )


        plt.xlabel(
            target_new_features_improved_A
        )


        plt.ylabel(
            "Number of samples"
        )


        plt.xticks(rotation=45)

        plt.grid(axis="y")

        plt.show()



    # ============================================================
    # Missing values
    # ============================================================

    print("\n===== MISSING VALUES =====")


    missing_values_new_features_improved_A = (
        exploration_df_new_features_improved_A
        .isnull()
        .sum()
        .sort_values(
            ascending=False
        )
    )


    print(
        missing_values_new_features_improved_A[
            missing_values_new_features_improved_A > 0
        ]
    )


    # ============================================================
    # Feature types
    # ============================================================

    print("\n===== FEATURE TYPES =====")


    print(
        exploration_df_new_features_improved_A.dtypes
    )
    return


@app.cell
def _():
    ##Model for company A 
    return


@app.cell
def _(
    ColumnTransformer,
    ImbPipeline,
    LabelEncoder,
    MLPClassifier,
    OneHotEncoder,
    Pipeline,
    RandomOverSampler,
    SMOTENC,
    SimpleImputer,
    StandardScaler,
    df_new_features,
    evaluate_delivery,
    evaluate_quantity,
    feature_cols_new_features,
    pd,
    target_cols_new_features,
    train_test_split,
):
    # ============================================================
    # Data
    # ============================================================

    mlp_X_new_features_MLP_improved = (
        df_new_features[feature_cols_new_features].copy()
    )

    mlp_y_new_features_MLP_improved = (
        df_new_features[target_cols_new_features].copy()
    )


    # ============================================================
    # Encode targets
    # ============================================================

    delivery_encoder_new_features_MLP_improved = LabelEncoder()

    quantity_encoder_new_features_MLP_improved = LabelEncoder()


    mlp_y_new_features_MLP_improved["Delivery Status"] = (
        delivery_encoder_new_features_MLP_improved.fit_transform(
            mlp_y_new_features_MLP_improved["Delivery Status"]
        )
    )


    mlp_y_new_features_MLP_improved["Quantity Status"] = (
        quantity_encoder_new_features_MLP_improved.fit_transform(
            mlp_y_new_features_MLP_improved["Quantity Status"]
        )
    )



    # ============================================================
    # Feature types
    # ============================================================

    mlp_categorical_features_MLP_improved = [
        "Supplier",
        "Unique Product Identifier",
        "Season"
    ]


    mlp_numeric_features_MLP_improved = [

        col for col in feature_cols_new_features

        if col not in mlp_categorical_features_MLP_improved

    ]


    # ============================================================
    # Preprocessing
    # ============================================================

    mlp_preprocessor_MLP_improved = ColumnTransformer(

        transformers=[

            (
                "num",

                Pipeline([

                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )

                ]),

                mlp_numeric_features_MLP_improved

            ),


            (

                "cat",

                Pipeline([

                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),

                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )

                ]),

                mlp_categorical_features_MLP_improved

            )

        ]

    )



    # ============================================================
    # ============================================================
    # DELIVERY STATUS MODEL
    # ============================================================
    # ============================================================


    (
        mlp_X_train_delivery_new_features_MLP_improved,
        mlp_X_test_delivery_new_features_MLP_improved,
        mlp_y_train_delivery_new_features_MLP_improved,
        mlp_y_test_delivery_new_features_MLP_improved

    ) = train_test_split(

        mlp_X_new_features_MLP_improved,

        mlp_y_new_features_MLP_improved["Delivery Status"],

        test_size=0.2,

        random_state=42,

        stratify=
        mlp_y_new_features_MLP_improved["Delivery Status"]

    )


    print(
        "Delivery training samples:",
        len(mlp_X_train_delivery_new_features_MLP_improved)
    )



    delivery_model_new_features_MLP_improved = ImbPipeline(

        steps=[


            (
                "oversampling",

                RandomOverSampler(

                    random_state=42

                )

            ),


            (
                "preprocessing",

                mlp_preprocessor_MLP_improved

            ),


            (
                "classifier",

                MLPClassifier(

                    hidden_layer_sizes=(64,),

                    activation="relu",

                    learning_rate_init=0.001,

                    batch_size=128,

                    alpha=0.001,

                    max_iter=1000,

                    early_stopping=True,

                    validation_fraction=0.1,

                    n_iter_no_change=20,

                    random_state=42

                )

            )

        ]

    )



    print(
        "Training Delivery Status MLP..."
    )


    delivery_model_new_features_MLP_improved.fit(

        mlp_X_train_delivery_new_features_MLP_improved,

        mlp_y_train_delivery_new_features_MLP_improved

    )


    print(
        "Delivery training finished."
    )



    # Predictions

    delivery_predictions_new_features_MLP_improved = (

        delivery_model_new_features_MLP_improved.predict(

            mlp_X_test_delivery_new_features_MLP_improved

        )

    )



    delivery_train_predictions_new_features_MLP_improved = (

        delivery_model_new_features_MLP_improved.predict(

            mlp_X_train_delivery_new_features_MLP_improved

        )

    )



    # ============================================================
    # ============================================================
    # QUANTITY STATUS MODEL
    # ============================================================
    # ============================================================



    (
        mlp_X_train_quantity_new_features_MLP_improved,
        mlp_X_test_quantity_new_features_MLP_improved,
        mlp_y_train_quantity_new_features_MLP_improved,
        mlp_y_test_quantity_new_features_MLP_improved

    ) = train_test_split(

        mlp_X_new_features_MLP_improved,

        mlp_y_new_features_MLP_improved["Quantity Status"],

        test_size=0.2,

        random_state=42,

        stratify=
        mlp_y_new_features_MLP_improved["Quantity Status"]

    )



    print(
        "Quantity training samples:",
        len(mlp_X_train_quantity_new_features_MLP_improved)
    )



    categorical_indexes_MLP_improved = [

        mlp_X_new_features_MLP_improved.columns.get_loc(col)

        for col in mlp_categorical_features_MLP_improved

    ]



    quantity_model_new_features_MLP_improved = ImbPipeline(

        steps=[


            (
                "smote",

                SMOTENC(

                    categorical_features=
                    categorical_indexes_MLP_improved,

                    random_state=42,

                    k_neighbors=3

                )

            ),


            (
                "preprocessing",

                mlp_preprocessor_MLP_improved

            ),


            (
                "classifier",

                MLPClassifier(

                    hidden_layer_sizes=(64,),

                    activation="relu",

                    learning_rate_init=0.001,

                    batch_size=128,

                    alpha=0.001,

                    max_iter=1000,

                    early_stopping=True,

                    validation_fraction=0.1,

                    n_iter_no_change=20,

                    random_state=42

                )

            )

        ]

    )



    print(
        "Training Quantity Status MLP..."
    )



    quantity_model_new_features_MLP_improved.fit(

        mlp_X_train_quantity_new_features_MLP_improved,

        mlp_y_train_quantity_new_features_MLP_improved

    )



    print(
        "Quantity training finished."
    )



    # Predictions

    quantity_predictions_new_features_MLP_improved = (

        quantity_model_new_features_MLP_improved.predict(

            mlp_X_test_quantity_new_features_MLP_improved

        )

    )



    quantity_train_predictions_new_features_MLP_improved = (

        quantity_model_new_features_MLP_improved.predict(

            mlp_X_train_quantity_new_features_MLP_improved

        )

    )



    # ============================================================
    # Evaluation DataFrames
    # ============================================================


    delivery_test_df_new_features_MLP_improved = pd.DataFrame({

        "Delivery Status":
        mlp_y_test_delivery_new_features_MLP_improved

    })


    delivery_pred_df_new_features_MLP_improved = pd.DataFrame({

        "Delivery Status":
        delivery_predictions_new_features_MLP_improved

    })


    delivery_train_df_new_features_MLP_improved = pd.DataFrame({

        "Delivery Status":
        mlp_y_train_delivery_new_features_MLP_improved

    })


    delivery_train_pred_df_new_features_MLP_improved = pd.DataFrame({

        "Delivery Status":
        delivery_train_predictions_new_features_MLP_improved

    })



    quantity_test_df_new_features_MLP_improved = pd.DataFrame({

        "Quantity Status":
        mlp_y_test_quantity_new_features_MLP_improved

    })


    quantity_pred_df_new_features_MLP_improved = pd.DataFrame({

        "Quantity Status":
        quantity_predictions_new_features_MLP_improved

    })


    quantity_train_df_new_features_MLP_improved = pd.DataFrame({

        "Quantity Status":
        mlp_y_train_quantity_new_features_MLP_improved

    })


    quantity_train_pred_df_new_features_MLP_improved = pd.DataFrame({

        "Quantity Status":
        quantity_train_predictions_new_features_MLP_improved

    })



    # ============================================================
    # Evaluation
    # ============================================================


    print("\n========== DELIVERY STATUS ==========")


    evaluate_delivery(

        delivery_train_df_new_features_MLP_improved,

        delivery_train_pred_df_new_features_MLP_improved,

        delivery_test_df_new_features_MLP_improved,

        delivery_pred_df_new_features_MLP_improved

    )



    print("\n========== QUANTITY STATUS ==========")


    evaluate_quantity(

        quantity_train_df_new_features_MLP_improved,

        quantity_train_pred_df_new_features_MLP_improved,

        quantity_test_df_new_features_MLP_improved,

        quantity_pred_df_new_features_MLP_improved

    )
    return


@app.cell
def _():
    ##Model for company B 
    return


@app.cell
def _(
    ColumnTransformer,
    LabelEncoder,
    MLPClassifier,
    OneHotEncoder,
    Pipeline,
    RandomOverSampler,
    SMOTE,
    SimpleImputer,
    StandardScaler,
    df_new_features,
    evaluate_delivery,
    evaluate_quantity,
    feature_cols_new_features,
    pd,
    target_cols_new_features,
    train_test_split,
):
    # ============================================================
    # Data
    # ============================================================

    mlp_X_new_features_MLP_improved_B = (
        df_new_features[feature_cols_new_features].copy()
    )


    mlp_y_new_features_MLP_improved_B = (
        df_new_features[target_cols_new_features].copy()
    )



    # ============================================================
    # Encode targets
    # ============================================================

    delivery_encoder_new_features_MLP_improved_B = LabelEncoder()

    quantity_encoder_new_features_MLP_improved_B = LabelEncoder()



    mlp_y_new_features_MLP_improved_B["Delivery Status"] = (

        delivery_encoder_new_features_MLP_improved_B.fit_transform(

            mlp_y_new_features_MLP_improved_B["Delivery Status"]

        )

    )



    mlp_y_new_features_MLP_improved_B["Quantity Status"] = (

        quantity_encoder_new_features_MLP_improved_B.fit_transform(

            mlp_y_new_features_MLP_improved_B["Quantity Status"]

        )

    )



    # ============================================================
    # Features
    # ============================================================

    categorical_features_new_features_MLP_improved_B = [

        "Supplier",

        "Unique Product Identifier",

        "Season"

    ]


    numeric_features_new_features_MLP_improved_B = [

        col for col in feature_cols_new_features

        if col not in categorical_features_new_features_MLP_improved_B

    ]



    # ============================================================
    # Preprocessor
    # ============================================================


    preprocessor_new_features_MLP_improved_B = ColumnTransformer(

        transformers=[

            (
                "num",

                Pipeline([

                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )

                ]),

                numeric_features_new_features_MLP_improved_B

            ),


            (

                "cat",

                Pipeline([

                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),

                    (

                        "encoder",

                        OneHotEncoder(

                            handle_unknown="ignore",

                            sparse_output=False

                        )

                    )

                ]),

                categorical_features_new_features_MLP_improved_B

            )

        ]

    )



    # ============================================================
    # DELIVERY STATUS
    # ============================================================


    (
    X_train_delivery_new_features_MLP_improved_B,
    X_test_delivery_new_features_MLP_improved_B,
    y_train_delivery_new_features_MLP_improved_B,
    y_test_delivery_new_features_MLP_improved_B

    )=train_test_split(

        mlp_X_new_features_MLP_improved_B,

        mlp_y_new_features_MLP_improved_B["Delivery Status"],

        test_size=0.2,

        random_state=42,

        stratify=
        mlp_y_new_features_MLP_improved_B["Delivery Status"]

    )



    print("Preprocessing Delivery...")


    X_train_delivery_new_features_MLP_improved_B = (

        preprocessor_new_features_MLP_improved_B.fit_transform(

            X_train_delivery_new_features_MLP_improved_B

        )

    )


    X_test_delivery_new_features_MLP_improved_B = (

        preprocessor_new_features_MLP_improved_B.transform(

            X_test_delivery_new_features_MLP_improved_B

        )

    )



    print(
        "Train shape:",
        X_train_delivery_new_features_MLP_improved_B.shape
    )


    print(
        "Test shape:",
        X_test_delivery_new_features_MLP_improved_B.shape
    )



    # Oversampling

    ros_new_features_MLP_improved_B = RandomOverSampler(

        random_state=42

    )



    X_train_delivery_new_features_MLP_improved_B, y_train_delivery_new_features_MLP_improved_B = (

        ros_new_features_MLP_improved_B.fit_resample(

            X_train_delivery_new_features_MLP_improved_B,

            y_train_delivery_new_features_MLP_improved_B

        )

    )



    # MLP Delivery


    delivery_model_new_features_MLP_improved_B = MLPClassifier(

        hidden_layer_sizes=(64,),

        activation="relu",

        learning_rate_init=0.001,

        batch_size=128,

        alpha=0.01,

        max_iter=1000,

        early_stopping=True,

        validation_fraction=0.1,

        n_iter_no_change=10,

        random_state=42

    )



    print("Training Delivery...")


    delivery_model_new_features_MLP_improved_B.fit(

        X_train_delivery_new_features_MLP_improved_B,

        y_train_delivery_new_features_MLP_improved_B

    )



    print("Delivery finished")



    # ============================================================
    # QUANTITY STATUS
    # ============================================================


    (
    X_train_quantity_new_features_MLP_improved_B,
    X_test_quantity_new_features_MLP_improved_B,
    y_train_quantity_new_features_MLP_improved_B,
    y_test_quantity_new_features_MLP_improved_B

    )=train_test_split(

        mlp_X_new_features_MLP_improved_B,

        mlp_y_new_features_MLP_improved_B["Quantity Status"],

        test_size=0.2,

        random_state=42,

        stratify=
        mlp_y_new_features_MLP_improved_B["Quantity Status"]

    )



    print("Preprocessing Quantity...")


    X_train_quantity_new_features_MLP_improved_B = (

        preprocessor_new_features_MLP_improved_B.fit_transform(

            X_train_quantity_new_features_MLP_improved_B

        )

    )


    X_test_quantity_new_features_MLP_improved_B = (

        preprocessor_new_features_MLP_improved_B.transform(

            X_test_quantity_new_features_MLP_improved_B

        )

    )



    print(
        "Quantity train shape:",
        X_train_quantity_new_features_MLP_improved_B.shape
    )



    # SMOTE after preprocessing

    smote_new_features_MLP_improved_B = SMOTE(

        random_state=42,

        k_neighbors=5

    )



    X_train_quantity_new_features_MLP_improved_B, y_train_quantity_new_features_MLP_improved_B = (

        smote_new_features_MLP_improved_B.fit_resample(

            X_train_quantity_new_features_MLP_improved_B,

            y_train_quantity_new_features_MLP_improved_B

        )

    )



    # MLP Quantity


    quantity_model_new_features_MLP_improved_B = MLPClassifier(

        hidden_layer_sizes=(64,),

        activation="relu",

        learning_rate_init=0.001,

        batch_size=128,

        alpha=0.001,

        max_iter=1000,

        early_stopping=True,

        validation_fraction=0.1,

        n_iter_no_change=20,

        random_state=42

    )



    print("Training Quantity...")


    quantity_model_new_features_MLP_improved_B.fit(

        X_train_quantity_new_features_MLP_improved_B,

        y_train_quantity_new_features_MLP_improved_B

    )



    print("Quantity finished")



    # ============================================================
    # EVALUATION
    # ============================================================


    delivery_predictions_new_features_MLP_improved_B = (

        delivery_model_new_features_MLP_improved_B.predict(

            X_test_delivery_new_features_MLP_improved_B

        )

    )


    delivery_train_predictions_new_features_MLP_improved_B = (

        delivery_model_new_features_MLP_improved_B.predict(

            X_train_delivery_new_features_MLP_improved_B

        )

    )



    quantity_predictions_new_features_MLP_improved_B = (

        quantity_model_new_features_MLP_improved_B.predict(

            X_test_quantity_new_features_MLP_improved_B

        )

    )


    quantity_train_predictions_new_features_MLP_improved_B = (

        quantity_model_new_features_MLP_improved_B.predict(

            X_train_quantity_new_features_MLP_improved_B

        )

    )



    # DataFrames evaluation

    delivery_train_df_new_features_MLP_improved_B = pd.DataFrame({

        "Delivery Status":
        y_train_delivery_new_features_MLP_improved_B

    })


    delivery_train_pred_df_new_features_MLP_improved_B = pd.DataFrame({

        "Delivery Status":
        delivery_train_predictions_new_features_MLP_improved_B

    })


    delivery_test_df_new_features_MLP_improved_B = pd.DataFrame({

        "Delivery Status":
        y_test_delivery_new_features_MLP_improved_B

    })


    delivery_pred_df_new_features_MLP_improved_B = pd.DataFrame({

        "Delivery Status":
        delivery_predictions_new_features_MLP_improved_B

    })



    quantity_train_df_new_features_MLP_improved_B = pd.DataFrame({

        "Quantity Status":
        y_train_quantity_new_features_MLP_improved_B

    })


    quantity_train_pred_df_new_features_MLP_improved_B = pd.DataFrame({

        "Quantity Status":
        quantity_train_predictions_new_features_MLP_improved_B

    })


    quantity_test_df_new_features_MLP_improved_B = pd.DataFrame({

        "Quantity Status":
        y_test_quantity_new_features_MLP_improved_B

    })


    quantity_pred_df_new_features_MLP_improved_B = pd.DataFrame({

        "Quantity Status":
        quantity_predictions_new_features_MLP_improved_B

    })



    print("\n========== DELIVERY STATUS ==========")


    evaluate_delivery(

        delivery_train_df_new_features_MLP_improved_B,

        delivery_train_pred_df_new_features_MLP_improved_B,

        delivery_test_df_new_features_MLP_improved_B,

        delivery_pred_df_new_features_MLP_improved_B

    )



    print("\n========== QUANTITY STATUS ==========")


    evaluate_quantity(

        quantity_train_df_new_features_MLP_improved_B,

        quantity_train_pred_df_new_features_MLP_improved_B,

        quantity_test_df_new_features_MLP_improved_B,

        quantity_pred_df_new_features_MLP_improved_B

    )
    return


if __name__ == "__main__":
    app.run()
