from bs4 import BeautifulSoup
import pandas as pd
from collections import Counter
import re

def analyze_portfolio_sites(websites_data):
    """
    Analyze VC portfolio websites to find commonly occurring companies that have IPO'd
    
    Parameters:
    websites_data: List of dictionaries containing {'url': url, 'html_content': content}
    
    Returns:
    DataFrame with company frequencies and IPO status
    """
    # Store all company mentions
    company_mentions = []
    
    for site in websites_data:
        soup = BeautifulSoup(site['html_content'], 'html.parser')
        
        # Extract company names from the HTML content
        # This assumes companies are typically in headings or specific div classes
        company_elements = soup.find_all(['h1', 'h2', 'h3', 'div'], 
                                       class_=re.compile(r'company|portfolio|startup', re.I))
        
        for element in company_elements:
            company_name = element.get_text().strip()
            if company_name:
                company_mentions.append(company_name)
    
    # Count frequencies
    company_counts = Counter(company_mentions)
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(company_counts, orient='index', columns=['frequency'])
    df.index.name = 'company'
    df = df.reset_index()
    
    # Sort by frequency
    df = df.sort_values('frequency', ascending=False)
    
    # Add placeholder for IPO status (would need to be verified against actual market data)
    df['is_public'] = False  # Default to False
    
    # Function to check IPO status (you would need to implement this with real market data)
    def check_ipo_status(company_name):
        try:
            # Initialize Gemini API
            # You'll need to set GOOGLE_API_KEY in your environment variables
            import google.generativeai as genai
            import os
            import re
            
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Please set GOOGLE_API_KEY environment variable")
                
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            prompt = f"""
            Is {company_name} a publicly traded company as of today? 
            Please respond with ONLY 'Yes' or 'No'.
            Base this on whether they have had an IPO and are currently trading on a major stock exchange.
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip().lower()
            
            # Use regex to look for yes/no answer
            if re.search(r'\byes\b', response_text):
                return True
            elif re.search(r'\bno\b', response_text):
                return False
            else:
                # If response is unclear, assume not public
                return False
                
        except Exception as e:
            print(f"Error checking IPO status for {company_name}: {str(e)}")
            return True
    
    # Check IPO status for each company
    df['is_public'] = df['company'].apply(check_ipo_status)
    
    # Filter for public companies and return top 20
    public_companies = df[df['is_public']].head(20)
    
    return public_companies

def process_results(companies_df):
    """
    Process and format the results for presentation
    """
    summary = []
    for _, row in companies_df.iterrows():
        summary.append({
            'company': row['company'],
            'portfolio_appearances': row['frequency'],
            'public_status': 'Public'
        })
    
    return summary
