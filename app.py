from flask import Flask, request, render_template
import fitz
from transformers import pipeline
import os
import torch

app = Flask(__name__)
# テンプレートを表示する
@app.route('/')
def index():
    return render_template('index.html')

# ファイルをアップロードする
@app.route('/upload', methods=['POST'])
def upload():
    # ファイルを取得する
    file = request.files['file']
    filename = file.filename

    # PDFファイルからテキストを抽出する
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    text = extract_text_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], filename))


    # テキストを前処理する
    text = preprocess_text(text)
    
    # Hugging faceのt5-baseモデルを用いてtextの要約を行う
    summarizer = pipeline(
        "summarization",
        "pszemraj/long-t5-tglobal-base-16384-book-summary",
        device=0 if torch.cuda.is_available() else -1,
    )

    result = summarizer(text)
    print(result[0]["summary_text"])  




    # 結果をHTMLページに表示する
    return render_template('result.html', input_text=text,summary_text=result[0]["summary_text"])

# テキストを前処理する
def preprocess_text(text):
    # 特定の文字列を削除するなどの前処理を行う
    # 改行を削除する
    text = text.replace('\n',' ')
    # 連続した空白を削除する
    import re
    text = re.sub(r'\s+', ' ', text) 
    return text

# PDFファイルからテキストを抽出する
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if __name__ == '__main__':
    UPLOAD_FOLDER = 'static/uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(debug=True)
