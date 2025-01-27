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
GOOGLE_API_KEY = ''
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
        # Initialize NLP pipeline for text analysis
        nlp_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Analyze company description with NLP
        text_analysis = nlp_classifier(company['description'])
        text_score = text_analysis[0]['score'] if text_analysis[0]['label'] == 'POSITIVE' else 0
        
        # Process image if available
        image_score = 0
        if company['image_url']:
            img_response = requests.get(company['image_url'])
            img = Image.open(BytesIO(img_response.content))
            
            # Use NeuralVision for image analysis
            vision_model = pipeline("image-classification", model="microsoft/resnet-50")
            image_results = vision_model(img)
            
            # Check for AI/tech related classifications
            ai_keywords = ['computer', 'technology', 'digital', 'software', 'electronic']
            image_score = sum(result['score'] for result in image_results 
                            if any(keyword in result['label'].lower() for keyword in ai_keywords))

        # Combine text and image analysis with Gemini
        prompt = f"""
        Analyze this company based on the following scores:
        Name: {company['name']}
        Text Analysis Score: {text_score}
        Image Analysis Score: {image_score}
        Description: {company['description']}
        
        Given these analysis scores and the description, is this an AI company? Return only 'yes' or 'no'.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip().lower() == 'yes'
        
    except Exception as e:
        print(f"Error in advanced analysis: {str(e)}")
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
