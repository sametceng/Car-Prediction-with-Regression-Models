# ======================================================
# IMPORTS
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor



# ======================================================
# LOAD
# ======================================================

df = pd.read_csv("data/ford.csv")

target = "price"



# ======================================================
# COLUMN TYPES
# ======================================================

cat_cols = []
num_cols = []

for col in df.columns:
    if col == target:
        continue
    if df[col].dtype == "object":
        cat_cols.append(col)
    else:
        num_cols.append(col)



# ======================================================
# SPLIT
# ======================================================

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# ======================================================
# PREPROCESSOR
# ======================================================

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])



# ======================================================
# MODELS 
# ======================================================

models = {

    "Ridge": Ridge(alpha=1.0),

    "Lasso": Lasso(alpha=0.001),

    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5),

    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(),


    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}



# ======================================================
# CROSS VALIDATION
# ======================================================

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

best_score = np.inf
best_name = None

print("===== CROSS VALIDATION =====")

for name, reg in models.items():

    pipe = Pipeline([
        ("prep", preprocessor),
        ("reg", reg)
    ])

    scores = cross_val_score(
        pipe,
        X_train,
        y_train,
        cv=kfold,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    rmse = -scores.mean()

    print(f"{name:20s} RMSE = {rmse:.2f}")

    if rmse < best_score:
        best_score = rmse
        best_name = name


print("\nBest model:", best_name)



# ======================================================
# FINAL TRAINING 
# ======================================================

if best_name == "XGBoost":
    
 
    preprocessor.fit(X_train)
    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)
    
  
    final_reg = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        random_state=42
    )
    
    final_reg.fit(
        X_train_prep,
        y_train,
        eval_set=[(X_test_prep, y_test)],
        verbose=False
    )
    

    final_model = Pipeline([
        ("prep", preprocessor),
        ("reg", final_reg)
    ])

else:
    final_model = Pipeline([
        ("prep", preprocessor),
        ("reg", models[best_name])
    ])
    final_model.fit(X_train, y_train)



# ======================================================
# TEST EVALUATION
# ======================================================

preds = final_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n===== TEST RESULTS =====")
print("RMSE:", rmse)
print("R2  :", r2)
##


# ======================================================
# FEATURE IMPORTANCE
# ======================================================

reg = final_model.named_steps["reg"]

if hasattr(reg, "feature_importances_"):

    feature_names = final_model.named_steps["prep"].get_feature_names_out()

    importances = pd.Series(reg.feature_importances_, index=feature_names)

    importances.sort_values(ascending=False).head(15).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.show()