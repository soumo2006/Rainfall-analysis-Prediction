import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")
client=Groq(api_key=my_api_key)

def get_ai_suggestion(region,month,annual_rainfall,monthly_rainfall):
    if annual_rainfall>1500:
        flood_risk="High"
    elif annual_rainfall>1000:
        flood_risk="Moderate"
    else:
        flood_risk="Low"
        
        
        
        
        
    prompt = f"""
You are an experienced agricultural and rainfall advisor for India.

The following rainfall values were predicted by a Machine Learning model
trained on historical IMD rainfall data (1901–2015).

Prediction Details:

Region: {region}

Month: {month}

Monthly Rainfall:
{monthly_rainfall:.2f} mm

Predicted Annual Rainfall:
{annual_rainfall:.2f} mm

Flood Risk:
{flood_risk}

Provide your answer in Markdown using these headings:

## 🌧️ Rainfall Analysis
Explain whether rainfall is low, moderate or high.

## 🌾 Suitable Crops
Recommend crops suitable for this rainfall.

## 💧 Irrigation Advice
Suggest irrigation practices.

## 🌱 Water Conservation
Provide practical water-saving tips.

## ⚠️ Flood / Drought Precautions
Mention precautions farmers should take.

## 👨‍🌾 Farmer Recommendation
Give simple advice in easy English.

Important:
- Keep the response practical.
- Do not use tables.
- Do not make unrealistic claims.
- Mention that the advice is based on an ML prediction using historical IMD data and should be used together with official weather forecasts.
"""    




    response=client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are an experienced agricultural and rainfall advisor for India."
        },
        {
            "role": "user",
            "content": prompt   
        }
    ],
    temperature=0.4,
    max_tokens=700
    
)  
    return response.choices[0].message.content 
if __name__ == "__main__":
    print(
        get_ai_suggestion(
            region="Gangetic West Bengal",
            month="July",
            annual_rainfall=1250,
            monthly_rainfall=250
        )
    ) 



