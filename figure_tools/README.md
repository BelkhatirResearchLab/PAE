# Optional revised SHAP figures
These scripts change presentation only, not the fitted models or SHAP values.
Input arrays and patient-level derived data are intentionally NOT included.

`render_shap_figures.py --arrays /approved/path/verified_shap_arrays.npz --output /approved/path/figures`

The NPZ contains `shap_values`, `transformed_features`, and `feature_names`. Its arrays were independently refitted and checked against the stage-11 ranking (maximum absolute difference 8.33e-17). The renderer writes PNG, PDF and SVG.

`render_smoothed_shap.py --input-dir /approved/path/shap_inputs --output /approved/path/smoothed`

The second script requires that NPZ plus `feature_label_mapping.csv` in the input directory. It overlays Gaussian KDE (Scott bandwidth) on real SHAP points. It is a smoothed density visualization, not an unmodified beeswarm. The original model has discrete SHAP values; smoothing must be disclosed.

The original stage-11 computation and original plots are fully included in `Code-Final/src/11_feature_importance.py`. These optional renderers require precomputed local arrays; they do not fit the model or export those arrays themselves.
