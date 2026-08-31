import os
import coremltools as ct

model_path = "models/macd_bidirectional_gru_forecaster.mlpackage"
if os.path.exists(model_path):
    model = ct.models.MLModel(model_path)
    print("Metadata:")
    for k, v in model.user_defined_metadata.items():
        print(f"  {k}: {v}")
else:
    print(f"Model not found at {model_path}")
