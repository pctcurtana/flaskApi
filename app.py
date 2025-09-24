from flask import Flask, request, jsonify
from flask_cors import CORS
from loadModel import get_recommendations
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    js = {
    "id": "9999",
    "name": "thatx2",
    "class": "KTPM2211",
    "action": "VIEW",
    "language" : "Javascript"
    } 
    return jsonify(js)

@app.route('/recommends', methods=['POST'])
def recommends():

    try:
        # Kiểm tra xem request có chứa JSON không
        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Request must be JSON"
            }), 400

        # Lấy dữ liệu từ request body
        data = request.get_json()
        
        # Validate dữ liệu đầu vào
        if not data:
            return jsonify({
                "status": "error",
                "message": "Have no data in request"
            }), 400

        # Lấy các thông tin cần thiết từ frontend
        user_id = data.get('userId')
        product_ids = data.get('productIds', [])  # Danh sách product IDs
        
        # Validate các tham số bắt buộc
        if not user_id:
            return jsonify({
                "status": "error",
                "message": "User ID is required"
            }), 400

        # Nếu không có product_ids, trả lỗi
        if not product_ids:
            return jsonify({
                "status": "error",
                "message": "Product IDs are required"
            }), 400

        # Log để debug
        print(f"User ID: {user_id}")
        print(f"Product IDs: {product_ids}")
        
        # Gọi model để lấy gợi ý 
        candidate_items = get_recommendations(product_ids, 5)

        print(f"Recommendations: {candidate_items}")

        # Trả về kết quả
        return jsonify({
            "status": "success",
            "data": {
                "userId": user_id,
                "recommendations": candidate_items
            }
        }), 200

    except Exception as e:
        # Xử lý lỗi
        print(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)