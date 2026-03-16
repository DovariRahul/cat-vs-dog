import zipfile

with zipfile.ZipFile("model.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write("cnn_model.keras")
