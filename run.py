# run.py
from app.main import app  # Flask 애플리케이션 임포트

if __name__ == '__main__':
    app.run(debug=True)  # 개발 모드에서 실행 (debug=True)