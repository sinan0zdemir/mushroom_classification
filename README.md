# Mushroom Classification Project

This project aims to classify the 20 most observed different species of mushrooms in Italy region using deep learning techniques. The workflow includes data collection, preprocessing, model training, and deployment of a web API for mushroom identification.

## Project Structure
```
mushroom_classification/
├── code/
│   ├── data_scraper.py
│   ├── mushroom_classifier.ipynb
│   └── website/
│       ├── index.html
│       └── mushroom_api.py
├── models/
│   ├── mushroom_mobilenet_finetuned.h5
│   └── mushroom_mobilenet_frozen.h5
```

## How It Works
1. **Data Collection**: Images are fetched from iNaturalist API and organized by species.
2. **Data Preprocessing**: Images are split into training, validation, and test sets.
3. **Model Training**: A MobileNet-based neural network is fine-tuned on the dataset to classify mushroom species.
4. **Model Deployment**: The trained model is served via a simple web API, allowing users to upload images and receive predictions.

## Getting Started

### Option 1: Train Your Own Model
1. Clone the repository.
```bash
git clone https://github.com/sinan0zdemir/mushroom_classification
```
2. Install required Python packages. Required packages include:
```bash
    tensorflow
    numpy
    seaborn
    scikit-learn
    matplotlib
    pillow
    flask
    flask_cors
    requests
    tqdm
    pyinaturalist
    visualkeras 
```

3. Download the dataset:
   - Option A: Run the data scraper (`code/data_scraper.py`) to download images from iNaturalist API.
   - Option B: Download the prepared dataset archive from the provided link and extract it to the project directory. 
    (https://drive.google.com/file/d/1ZuKf5-fXRvdbYMGIvJOeHB3Yr2gl3kx5/view?usp=drive_link)
4. Use the Jupyter notebook (`code/mushroom_classifier.ipynb`) to train or evaluate the model.
5. Save the trained model to the `models/` directory.
6. Start the API server in `code/website/mushroom_api.py` and open `index.html` to use the web interface.

### Option 2: Use the Pretrained Model
1. Clone the repository.
2. Install required Python packages (see above).
3. Download or use the provided pretrained model in the `models/` directory (e.g., `mushroom_mobilenet_finetuned.h5`).
4. Start the API server in `code/website/mushroom_api.py` and open `index.html` to use the web interface.

## Notes
- The project is for educational and research purposes; model accuracy may vary depending on the dataset and training.
