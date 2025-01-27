import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';

const PortfolioAnalyzer = () => {
  const [urls, setUrls] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const urlList = urls.split('\n').filter(url => url.trim());
      
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ urls: urlList }),
      });
      
      if (!response.ok) {
        throw new Error('Analysis failed');
      }
      
      const data = await response.json();
      setResults(data.recurring_ai_companies);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 max-w-4xl mx-auto">
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>VC Portfolio AI Company Analyzer</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">
              Enter VC Portfolio URLs (one per line):
            </label>
            <textarea
              className="w-full h-32 p-2 border rounded"
              value={urls}
              onChange={(e) => setUrls(e.target.value)}
              placeholder="https://vc1.com/portfolio&#10;https://vc2.com/portfolio"
            />
          </div>
          <Button 
            onClick={handleAnalyze}
            disabled={loading || !urls.trim()}
            className="w-full"
          >
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              'Analyze Portfolios'
            )}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <div className="text-red-600 mb-4 p-4 bg-red-50 rounded">
          Error: {error}
        </div>
      )}

      {results && (
        <Card>
          <CardHeader>
            <CardTitle>Recurring AI Companies</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {Object.entries(results).map(([company, count]) => (
                <div key={company} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="font-medium">{company}</span>
                  <span className="text-sm text-gray-600">
                    Found in {count} portfolios
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default PortfolioAnalyzer;
