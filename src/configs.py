"""Model registry. 'input' decides which arrays a model receives:
   ordinal : X_ord (all features scaled, categoricals as scaled ordinal codes)  -- thesis input
   mixed   : X_num + X_cat (categoricals as integer indices for embeddings)
   tree    : X_tree (raw values + ordinal codes)
Training defaults are shared by all neural models; 'tune' lists the small validation grid
searched on seed 42 (same budget order for every model)."""

TRAIN_DEFAULTS = dict(optimizer="adam", lr=5e-4, weight_decay=0.0, batch_size=256,
                      max_epochs=60, patience=10)

MODELS = {
    # ---------------- proposed ----------------
    "proposed": dict(kind="torch", cls="FeatureTokenTransformer", input="mixed",
                     params=dict(num_embedding="per_feature", cat_mode="embed", identity=False),
                     tune=dict(lr=[1e-3, 5e-4, 2e-4])),
    # ---------------- ablations (all share the thesis encoder: d=64, 4 heads, 2 layers) ----------------
    "thesis_ftt": dict(kind="torch", cls="FeatureTokenTransformer", input="ordinal",   # thesis Exp05 design
                       params=dict(num_embedding="shared", cat_mode="ordinal", identity=True),
                       tune=dict(lr=[1e-3, 5e-4, 2e-4])),
    "single_token": dict(kind="torch", cls="FeatureTokenTransformer", input="ordinal",  # thesis Exp01 design
                         params=dict(single_token=True), tune=dict(lr=[1e-3, 5e-4, 2e-4])),
    "abl_shared_noid": dict(kind="torch", cls="FeatureTokenTransformer", input="ordinal",
                            params=dict(num_embedding="shared", cat_mode="ordinal", identity=False), tune=None),
    "abl_perfeat_ordinal": dict(kind="torch", cls="FeatureTokenTransformer", input="ordinal",
                                params=dict(num_embedding="per_feature", cat_mode="ordinal", identity=False), tune=None),
    "abl_shared_catemb": dict(kind="torch", cls="FeatureTokenTransformer", input="mixed",
                              params=dict(num_embedding="shared", cat_mode="embed", identity=True), tune=None),
    # ---------------- baselines ----------------
    "mlp": dict(kind="torch", cls="MLP", input="mixed", params=dict(cat_mode="embed"),
                tune=dict(lr=[1e-3, 5e-4], dropout=[0.1, 0.3])),
    "cnn1d": dict(kind="torch", cls="CNN1D", input="ordinal", params=dict(),
                  tune=dict(lr=[1e-3, 5e-4, 2e-4])),
    "ft_transformer": dict(kind="torch", cls="FTTransformer", input="mixed", params=dict(),
                           train=dict(optimizer="adamw", lr=1e-4, weight_decay=1e-5),
                           tune=dict(lr=[3e-4, 1e-4])),
    "rf": dict(kind="rf", input="tree", params=dict(n_estimators=300, max_features="sqrt", min_samples_leaf=1),
               tune=dict(min_samples_leaf=[1, 5], max_features=["sqrt", 0.5])),
    "xgboost": dict(kind="xgb", input="tree",
                    params=dict(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.8, early_stopping_rounds=100),
                    tune=dict(max_depth=[6, 8, 10])),
    "lightgbm": dict(kind="lgbm", input="tree",
                     params=dict(n_estimators=3000, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8),
                     tune=dict(num_leaves=[31, 63, 127])),
}

# ablation variants inherit the learning rate tuned for the thesis model so only the design changes
ABLATION_LR_FROM = {"abl_shared_noid": "thesis_ftt", "abl_perfeat_ordinal": "thesis_ftt",
                    "abl_shared_catemb": "thesis_ftt"}

GROUPS = {
    "main": ["rf", "xgboost", "mlp", "cnn1d", "ft_transformer", "proposed"],
    "ablation": ["thesis_ftt", "single_token", "abl_shared_noid", "abl_perfeat_ordinal",
                 "abl_shared_catemb", "proposed"],
    "shortcut": ["proposed", "xgboost"],         # run with --dataset unsw_noshortcut
    "extra": ["lightgbm"],
}
TUNE_ORDER = ["proposed", "thesis_ftt", "single_token", "mlp", "cnn1d", "ft_transformer", "rf", "xgboost"]
