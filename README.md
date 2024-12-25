
# Car Price Prediction App

This project demonstrates a machine learning pipeline to predict the selling price of used cars based on their features. The pipeline includes data preprocessing, model training, fine-tuning, and deployment as a web application using Streamlit.

## Features

- **Interactive Web App**: Users can input car details and get a predicted selling price instantly.
- **Machine Learning Model**: The app uses a Random Forest Regressor trained on historical car data.
- **Preprocessing**: Includes data cleaning, feature engineering, and scaling for consistent predictions.

## Dataset

The dataset used for training the model is publicly available on Kaggle: [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho).

- **Description**: Contains information about used cars, such as year, price, kilometers driven, fuel type, and more.
- **Source**: Kaggle.

*Note*: The raw dataset is not included in this repository. Please download it directly from the Kaggle link above.

## Repository Structure

```
car-price-prediction/
├── app.py                 # Streamlit app for deployment
├── main.py                # Model building and training script
├── random_forest_model.pkl # Trained Random Forest model
├── scaler.pkl             # Fitted StandardScaler for input scaling
├── features.pkl           # List of features used in the model
├── requirements.txt       # List of dependencies
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/car-price-prediction.git
   cd car-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) and save it locally if you want to run `main.py`.

## Usage

### Train the Model
1. Place the dataset in the project directory.
2. Update the file path in `main.py` to point to the dataset.
3. Run `main.py` to preprocess the data, train the model, and save the artifacts:
   ```bash
   python main.py
   ```

### Run the Web App
1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser and input car details to get predictions.

## Deployment

The app is deployed on Streamlit Cloud Community for public access. You can also deploy it yourself by following these steps:
1. Push your repository to GitHub.
2. Link your repository to Streamlit Cloud Community and deploy the app.

For more information, see [Streamlit Cloud](https://streamlit.io/cloud).

## Future Enhancements

- Add more features to improve prediction accuracy.
- Implement additional machine learning algorithms for comparison.
- Include visualization tools for exploratory data analysis in the app.

## Acknowledgments

- Dataset by [Nehal Birla](https://www.kaggle.com/nehalbirla) on Kaggle.
- Machine learning framework: [Scikit-learn](https://scikit-learn.org/).
- Web app framework: [Streamlit](https://streamlit.io/).

---

### Author
Henry Ha
[GitHub Profile](https://github.com/Minhhoang2606)
[LinkedIn Profile](https://www.linkedin.com/in/ha-minh-hoang/)
