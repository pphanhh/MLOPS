import requests
import json
# Similar functionality to the send_request.py
def test_sentiment_api(text):
    """Test the sentiment analysis API with the given text"""
    url = "http://localhost:5001/invocations"
    headers = {"Content-Type": "application/json"}
    
    # Format data theo yêu cầu của MLflow serving API
    data = json.dumps({"inputs": [text]})
    
    print(f"Sending request to {url} with text: '{text}'")
    
    try:
        response = requests.post(url, headers=headers, data=data)
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction result: {result}")
            
            # Phân tích và hiển thị kết quả một cách dễ hiểu hơn
            sentiment = "positive" if result[0] > 0.5 else "negative"
            confidence = result[0] if result[0] > 0.5 else 1 - result[0]
            
            print(f"Text: '{text}'")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 50)
            
            return result
        else:
            print(f"Error: {response.status_code}")
            print(f"Response text: {response.text}")
    except Exception as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    print("=== Sentiment Analysis API Test ===\n")
    
    # Test với một vài ví dụ
    test_sentiment_api("I love Trump, he's amazing!")
    test_sentiment_api("This is the worst election ever.")
    
    # Nhập từ người dùng
    while True:
        user_input = input("\nEnter text to analyze (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        test_sentiment_api(user_input)