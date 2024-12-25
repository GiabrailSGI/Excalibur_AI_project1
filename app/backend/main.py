from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
from joblib import load
# import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

# from tensorflow.keras.models import load_model

app = FastAPI()

# Load your pre-trained LSTM model
model = load("models/lstm_model.joblib")


def preprocess_data(data):
    data.drop(['index'], axis= 1, inplace= True)
    #transform data to numeric 
    data['Global_active_power'] = pd.to_numeric(data['Global_active_power'], errors='coerce')
    data['Global_reactive_power'] = pd.to_numeric(data['Global_reactive_power'], errors='coerce')
    data['Voltage'] = pd.to_numeric(data['Voltage'], errors='coerce')
    data['Global_intensity'] = pd.to_numeric(data['Global_intensity'], errors='coerce')
    data['Sub_metering_1'] = pd.to_numeric(data['Sub_metering_1'], errors='coerce')
    data['Sub_metering_2'] = pd.to_numeric(data['Sub_metering_2'], errors='coerce')
    data['Sub_metering_3'] = pd.to_numeric(data['Sub_metering_3'], errors='coerce')
# transform time
    data['Date'] = pd.to_datetime(data['Date'], format='mixed')
    data['Time'] = pd.to_datetime(data['Time'],format='mixed')
    #adugarea indicatorilor
    data['RSI'] = ta.rsi(close = data['Global_reactive_power'], length=5) 
    data['EMAF'] = ta.ema(close = data['Global_reactive_power'], length=20) 
    data['EMAM'] = ta.ema(close = data['Global_reactive_power'], length=50) 
    data['EMAS'] = ta.ema(close = data['Global_reactive_power'], length=100)

    data['TargetNextDay'] = data['Global_reactive_power'].shift(-1)
    data_set =data.iloc[:,0:15]
    pd.set_option('display.max_columns', None)
    data.isna().sum()

    data_set.dropna(inplace = True)
    data_set.reset_index(inplace = True)

    # df.drop(['Time'], axis= 1, inplace= True)
    # data_set.drop(['index'], axis= 1, inplace= True)

    data_set['Date'] = data_set['Date'].astype('int64') // 10**9
    data_set['Time'] = data_set['Time'].astype('int64') // 10**9

    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data_set)
    print(data_set_scaled)
    
    X = []
    backcandles = 30
    print (data_set_scaled.shape[0])
    for j in range(14):
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):
            X[j].append(data_set_scaled[i-backcandles:i, j])
    X= np.moveaxis(X, [0], [1])
    X,yi = np.array(X), np.array(data_set_scaled[backcandles:,-1])
    y= np.reshape(yi,(len(yi),1))

    return X, y

@app.post("/forecast/")
async def forecast_endpoint(
    user_input: str = Form(None), 
    file: UploadFile = None
):
    data = None
    if not user_input and not file:
        raise HTTPException(status_code=400, detail="Either user input or file is required.")

    if file:
        print(f"File: {file.filename}")
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    else:
        data = io.StringIO(user_input)
        print(f"Data: {user_input}")
        df = pd.read_csv(data)

    if "Global_reactive_power" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'usGlobal_reactive_powerage' column.")

    print(f"Data before: {data}")
    data, labels = preprocess_data(df)
    # print(f"Data after: {data}")

    # Make predictions using the loaded model
    predictions = model.predict(data)  # Adjust the shape as needed
    for i in range(20):
        print(predictions[i], labels[i])
    return JSONResponse(content={"forecast": predictions.tolist()})


@app.post("/predict/day")
async def forecast_endpoint(
    user_input: str = Form(None), 
    file: UploadFile = None
):
    if not user_input and not file:
        raise HTTPException(status_code=400, detail="Either user input or file is required.")

    if file:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    else:
        data = io.StringIO(user_input)
        df = pd.read_csv(data)

    train_and_save_model(df)
    return JSONResponse(content={"msg": "Model trained and saved successfuly"}) 


@app.get("/predict/day")
async def forecast_endpoint(month: int, day: int):

    # Model takes [[month,day]]
    pred_data = read_model_and_predict([[month,day]])
    
    # Return the prediction result as JSON
    return JSONResponse(content={"data": pred_data})



if __name__ == "__main__":
    # Start the FastAPI app using uvicorn
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8000)