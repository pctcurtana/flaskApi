from flask import Flask, request, jsonify
from flask_cors import CORS
from loadModel import get_recommendations

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "RecBole Two-Stage Recommendation API is running!"

@app.route('/recommends', methods=['POST'])
def recommends():
    """
    Endpoint gợi ý hoàn chỉnh, thực hiện cả 2 bước Retrieval và Ranking.
    """
    list_items = ["242", "302", "377"]
    print(get_recommendations(list_items, 5))

    candidate_items = get_recommendations(list_items, 5)    

    return jsonify({
        "status": "success",
        "data": str(candidate_items)
    }), 200


if __name__ == '__main__':
    # Khi deploy thực tế, hãy dùng Gunicorn thay vì app.run()
    # Ví dụ: gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
    app.run(host='0.0.0.0', port=5000, debug=False)