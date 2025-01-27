from flask import Flask, request, jsonify
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from collections import Counter

app = Flask(__name__)

# Initialize Gemini API
GOOGLE_API_KEY = 'AIzaSyDcVab_1FxaMmYZrvn_MwwUS5B0uxpXhW4'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro-vision')

def extract_portfolio_data(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract company cards/sections
        companies = []
        # Common portfolio item class patterns
        portfolio_items = soup.find_all(['div', 'article'], 
            class_=lambda x: x and ('portfolio' in x or 'company' in x))
        
        for item in portfolio_items:
            # Extract company name
            name = item.find(['h2', 'h3', 'h4'])
            name = name.text.strip() if name else ''
            
            # Extract company image
            img = item.find('img')
            img_url = img.get('src', '') if img else ''
            
            # Extract description
            desc = item.find(['p', 'div'], class_=lambda x: x and 'description' in x)
            desc = desc.text.strip() if desc else ''
            
            companies.append({
                'name': name,
                'image_url': img_url,
                'description': desc
            })
            
        return companies
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return []

def analyze_with_gemini(company):
    try:
        # Download image
        if company['image_url']:
            img_response = requests.get(company['image_url'])
            img = Image.open(BytesIO(img_response.content))
            
            # Analyze with Gemini
            prompt = f"""
            Analyze this company:
            Name: {company['name']}
            Description: {company['description']}
            Image: [attached]
            
            Is this an AI company? Return only 'yes' or 'no'.
            """
            
            response = model.generate_content([prompt, img])
            return response.text.strip().lower() == 'yes'
    except Exception as e:
        print(f"Error analyzing with Gemini: {str(e)}")
        return False

@app.route('/analyze', methods=['POST'])
def analyze_portfolios():
    urls = request.json['urls']
    all_companies = []
    
    for url in urls:
        companies = extract_portfolio_data(url)
        for company in companies:
            if analyze_with_gemini(company):
                all_companies.append(company['name'])
    
    # Find companies that appear in multiple portfolios
    company_counts = Counter(all_companies)
    recurring_companies = {
        company: count for company, count 
        in company_counts.items() 
        if count > 1
    }
    
    return jsonify({
        'recurring_ai_companies': recurring_companies
    })

if __name__ == '__main__':
    app.run(debug=True)
