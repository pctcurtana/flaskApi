from flask import Flask, request, jsonify
from flask_cors import CORS
from loadModel import get_recommendations
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    js = {
    "id": "685e24c762e692bdfa5c7f13",
    "userId": "683537506dfa74193bb28c31",
    "productId": "685ba96f09ed89b856650872",
    "action": "VIEW",
    "created_at": "undefined",
    "updated_at": "undefined"
    } 
    return jsonify(js)

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)