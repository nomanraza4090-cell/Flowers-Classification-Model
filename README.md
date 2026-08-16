# CNN Flower Classifier

End-to-end flower image classification project designed for Google Colab GPU training and local/online deployment with Streamlit and Flask.

## Dataset
The training notebook inspects the uploaded ZIP and automatically detects classes and existing train/validation/test folders.

## Train in Google Colab
1. Open `notebooks/CNN_Training.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run cells from top to bottom.
4. Upload the original dataset ZIP when requested.
5. The best/final model and `class_names.json` are saved to Google Drive.

## Run Streamlit
After copying the generated `models/final_model.keras` and `models/class_names.json` into this project:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Run Flask
```bash
python app/flask_api.py
```

Open `http://127.0.0.1:5000`.

API:
- `GET /health`
- `POST /predict` with multipart field `image`

## Important
The clean project ZIP does not contain a trained `.keras` model. Train the model in Colab first; the notebook creates the model files automatically.
