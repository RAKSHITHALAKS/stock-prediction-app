features
SVM: Predicts stock direction (Up/Down) based on historical trends.
MLP: Provides continuous price predictions using historical closing data.
LSTM: Captures sequential patterns for accurate time series forecasting.
Visualize predicted vs actual stock prices.
Compare multiple model predictions easily.


Folder Structure
stock-prediction-app/
│
├── app.py                  
├── README.md               
├── requirements.txt        
├── sample_stock_data.csv   
├── models/scalers                 
             

installation

Clone the repo:
git clone https://github.com/RAKSHITHALAKS/stock-prediction-app.git
cd stock-prediction-app


Install dependencies:
pip install -r requirements.txt

Download Pre-trained Models
Due to GitHub file size limits, pre-trained models are hosted externally:

SVM Model
MLP Model
LSTM Model
Scalers
Download these files and place them in the models/ and scalers/ folders.
Usage
Run the Streamlit app:
streamlit run app.py


Upload a CSV file with a close column.
Select a model: SVM, MLP, or LSTM.
View predicted stock prices and compare with actual values.
