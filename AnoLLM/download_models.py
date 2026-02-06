"""
download_models.py

A simple, safe wrapper script for modelscope.snapshot_download.
- Checks import errors
- Catches download exceptions and prints friendly messages

Usage:
    python3 download_models.py

Note: Before running, make sure `modelscope` is installed, network access is available, and the model ID exists.
"""

def download_model():
    try:
        from modelscope import snapshot_download
    except Exception as e:
        print('Failed to import modelscope:', e)
        print('Please install modelscope (e.g., pip install modelscope) or check PYTHONPATH.')
        raise

    model_id = 'HuggingFaceTB/SmolLM-360M'
    try:
        # Keep original call; use variable name models to stay compatible with user-provided code
        models = snapshot_download(model_id)
        print('Model downloaded to:', models)
        return models
    except Exception as e:
        print('Error calling snapshot_download:', e)
        print('Possible reasons: network issues, model ID not found, or auth/permission required.')
        raise


if __name__ == '__main__':
    download_model()


