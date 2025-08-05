import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from typing import Tuple

def interactions(X: pd.DataFrame, y: pd.Series, n_samples: int = 1000) -> dict:
    """
    Calculate interactions between features and target using LGBM and SHAP.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.record_evaluation({})]
    )
    
    if len(X) > n_samples:
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X.iloc[sample_indices]
    else:
        X_sample = X
    
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_sample)
    
    interactions = {}
    
    for i in range(len(X.columns)):
        for j in range(i + 1, len(X.columns)):
            feature_i = X.columns[i]
            feature_j = X.columns[j]
            
            interaction_values = shap_interaction_values[:, i, j]
            interaction_strength = np.abs(interaction_values).mean()
            
            if interaction_strength > 0.001:
                interactions[f"{feature_i}__{feature_j}"] = {
                    'strength': interaction_strength,
                    'mean_interaction': interaction_values.mean(),
                    'std_interaction': interaction_values.std()
                }
    
    return dict(sorted(interactions.items(), 
                      key=lambda x: x[1]['strength'], reverse=True))


n = 1000
X = np.random.randn(n, 2)
y = X[:,0] + X[:,1]
X = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
y = pd.Series(y, name='target')

result = interactions(X, y)
print(result)
