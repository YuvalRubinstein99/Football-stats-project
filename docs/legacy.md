# Historical notebooks and models

The eight original notebooks are in `notebooks/legacy/`; original fitted estimators and encoders are in `models/legacy/`. Their contents and old outputs are preserved, except for a new setup cell and model-file paths needed by the layout change.

The setup cell locates the repository and changes the working directory to `data/`, so original CSV and league-directory references continue to resolve. Saved-model references point to `../models/legacy/`. Running an old notebook can still overwrite its historical data/model outputs, just as before.

These notebooks are reference material, not the supported reproduction entry point. They may need additional packages such as XGBoost, matplotlib, and seaborn, and retain pre-existing schema and estimator-version assumptions. In particular, the original prediction notebook expects a league column absent from the current prepared data. They were not rerun during cleanup.

Use `python run.py train` and `notebooks/prediction_bivariate.ipynb` for the maintained experiment. Only load serialized models you trust; the current trained model is reproducible and excluded from new commits. The old models remain versioned to preserve historical work.
