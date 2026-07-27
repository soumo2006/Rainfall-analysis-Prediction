import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("api error")
client=Groq(api_key=my_api_key)
def get_annual_ai_suggestion(region,current_rainfall,predicted_annual_rainfall):
    change=predicted_annual_rainfall-current_rainfall
    if predicted_annual_rainfall>2000:
        flood_risk="High"
    elif predicted_annual_rainfall>1500:
        flood_risk="Moderate"
    else:
        flood_risk="Low"
    prompt = f"""
You are an experienced agricultural and rainfall advisor for India.

The following annual rainfall values were predicted by a Machine Learning model
trained on historical IMD rainfall data (1901–2015).

Prediction Details:
- Region: {region}
- This Year's Rainfall: {current_rainfall:.2f} mm
- Next Year's Predicted Rainfall: {predicted_annual_rainfall:.2f} mm
- Expected Change:{change:+.2f} mm

- Flood Risk Level: {flood_risk}

Provide your answer in Markdown using these exact headings:

## 🌧️ Annual Rainfall Analysis
Explain whether the forecasted annual rainfall is low, moderate, or high compared to normal annual trends in {region}.

## 🌾 Suitable Crops & Cropping Strategy
Recommend crops suitable specifically for {region}'s climate and this rainfall level.

## 💧 Long-Term Irrigation Advice
Suggest practical, long-term irrigation and rainwater storage methods.

## 🌱 Water Conservation
Provide practical annual water-saving tips for farmers.

## ⚠️ Flood / Drought Precautions
Mention long-term precautions farmers should take for the predicted {flood_risk} risk level.

## 👨‍🌾 Farmer Recommendation
Give simple, actionable advice in plain, easy English (6th-grade reading level).

Important Constraints:
- Keep the advice realistic, practical, and grounded in Indian farming practices.
- Do not use tables.
- Mention explicitly that this advice is based on ML predictions using historical IMD data and must be cross-verified with official IMD weather forecasts.
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
    temperature=0.7,
    max_tokens=700
)
    return response.choices[0].message.content
if __name__ == "__main__":
    print(
        get_annual_ai_suggestion(
            region="TELANGANA",
            current_rainfall=3000.0,
            predicted_annual_rainfall=2759.3,
        )
    )
