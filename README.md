# Model Stealing Assignment 2 Submission

This repository contains the code and notebook for the Trustworthy Machine Learning Assignment 2: Model Stealing.

## Files
- `a2.ipynb`: Main Jupyter notebook for the assignment, including all steps from data loading to model submission.
- `notebook_submission_assignment2.py`: Python script version of the notebook, following the format of `example_submission_assignment2.py`.


## Workflow
1. **Dataset Loading**: Load the provided dataset and apply normalization transforms.
2. **API Request**: Request a new API session to obtain your `SEED` and `PORT`.
3. **Model Stealing**: Query the API with images to obtain representations. Save the output to `outv1.pickle`.
4. **Data Augmentation**: Use `augmented_outv1.pickle` for training with augmented data.
5. **Model Creation**: Train or create a model that outputs 1024-dimensional features. Save as `stolen_model.pth`.
6. **Export to ONNX**: Exported the model to ONNX format (`stolen_model.onnx` or `dummy_submission.onnx`).
7. **Validation**: Tested the ONNX model using `onnxruntime` to ensure correct input/output shapes.
8. **Submission**: Submited ONNX model to the server using the provided API endpoint.

## Example Usage
Run the notebook `a2.ipynb` step by step, or execute the script:
