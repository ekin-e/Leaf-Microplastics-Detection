# Leaf-Microplastics-Detection

# Setup
Install using the pyproject.toml. `pip`, `uv`, or `Poetry` works. We recommend [uv](https://docs.astral.sh/uv/).
```bash
uv python install
uv venv
uv pip install -r pyproject.toml
source .venv/bin/activate
```

# To make inference with the trained model
1. Put the images you want to predict on inside inference/old_img
2. Run the inference_yolo_sahi.py script
```bash
cd inference
uv run inference_yolo_sahi.py
```
3. Visualization of the predictions will be inside inference/new_img
4. Area information of the microplastics will be inside final_output.json
