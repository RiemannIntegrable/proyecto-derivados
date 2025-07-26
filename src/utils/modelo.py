import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

def garch():
    
    sp500_path = "../../data/input/sp500.csv"
    vix_path = "../../data/input/vix.csv"
    output_path = "../../data/output/predict.csv"
    
    start_date = "2025-07-18"
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        sp500_data = yf.download("^GSPC", start=start_date, end=today)
        vix_data = yf.download("^VIX", start=start_date, end=today)
        
        if not sp500_data.empty:
            sp500_new = sp500_data.reset_index()
            sp500_new = sp500_new[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            
            sp500_existing = pd.read_csv(sp500_path)
            sp500_existing['Date'] = pd.to_datetime(sp500_existing['Date'])
            sp500_new['Date'] = pd.to_datetime(sp500_new['Date'])
            
            max_existing_date = sp500_existing['Date'].max()
            sp500_truly_new = sp500_new[sp500_new['Date'] > max_existing_date]
            
            if not sp500_truly_new.empty:
                sp500_combined = pd.concat([sp500_existing, sp500_truly_new]).sort_values('Date')
                sp500_combined.to_csv(sp500_path, index=False)
        
        if not vix_data.empty:
            vix_new = vix_data.reset_index()
            vix_new = vix_new[['Date', 'Close', 'High', 'Low', 'Open', 'Volume']]
            
            vix_existing = pd.read_csv(vix_path)
            vix_existing['Date'] = pd.to_datetime(vix_existing['Date'])
            vix_new['Date'] = pd.to_datetime(vix_new['Date'])
            
            max_existing_date = vix_existing['Date'].max()
            vix_truly_new = vix_new[vix_new['Date'] > max_existing_date]
            
            if not vix_truly_new.empty:
                vix_combined = pd.concat([vix_existing, vix_truly_new]).sort_values('Date')
                vix_combined.to_csv(vix_path, index=False)
        
        sp500 = pd.read_csv(sp500_path)["Close"]
        vix = pd.read_csv(vix_path)["Close"]
        
        retornos = np.log(sp500 / sp500.shift(1)) * 100
        retornos = retornos.dropna()
        
        modelo_garch = arch_model(
            retornos,
            vol='GARCH',
            p=1,
            q=1,
            mean='Constant',
            dist='Normal',
            rescale=True
        )
        
        resultado_garch = modelo_garch.fit(disp='off')
        
        forecast = resultado_garch.forecast(horizon=1)
        volatilidad_garch = forecast.variance.iloc[-1, 0] ** 0.5
        
        vix_actual = vix.iloc[-1] if len(vix) > 0 else 0
        
        result_df = pd.DataFrame({
            'garch': [volatilidad_garch],
            'vix': [vix_actual]
        })
        
        result_df.to_csv(output_path, index=False)
        
        return {
            'garch_volatility': volatilidad_garch,
            'vix_volatility': vix_actual,
            'forecast_date': today
        }
        
    except Exception as e:
        print(f"Error en el modelo: {e}")
        return None 