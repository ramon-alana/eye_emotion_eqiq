#!/usr/bin/env python3
"""
简单的图片上传服务器
在服务器上运行此脚本，然后在浏览器中访问上传页面
"""

import http.server
import socketserver
import cgi
import os
from pathlib import Path
import urllib.parse

DEMO_DIR = Path(__file__).parent.parent / "data" / "demo_images"
DEMO_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8000


class UploadHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>图片上传</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
        }}
        input[type="file"] {{
            margin: 20px 0;
            padding: 10px;
            width: 100%;
        }}
        button {{
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        .message {{
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
        }}
        .success {{
            background-color: #d4edda;
            color: #155724;
        }}
        .error {{
            background-color: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 上传眼部图片到 Demo</h1>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" multiple required>
            <button type="submit">上传图片</button>
        </form>
        <div id="message"></div>
    </div>
</body>
</html>
            """
            self.wfile.write(html.encode('utf-8'))
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/upload':
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST'}
            )
            
            uploaded_files = []
            errors = []
            
            if 'file' in form:
                file_field = form['file']
                if isinstance(file_field, list):
                    files = file_field
                else:
                    files = [file_field]
                
                for file_item in files:
                    if file_item.filename:
                        filename = os.path.basename(file_item.filename)
                        filepath = DEMO_DIR / filename
                        
                        try:
                            with open(filepath, 'wb') as f:
                                f.write(file_item.file.read())
                            uploaded_files.append(filename)
                        except Exception as e:
                            errors.append(f"{filename}: {str(e)}")
            
            # 返回结果页面
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            result_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>上传结果</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .success {{
            color: #155724;
            background-color: #d4edda;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .error {{
            color: #721c24;
            background-color: #f8d7da;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        a {{
            display: inline-block;
            margin-top: 20px;
            color: #4CAF50;
            text-decoration: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>上传结果</h1>
"""
            if uploaded_files:
                result_html += '<div class="success"><strong>✓ 成功上传:</strong><ul>'
                for f in uploaded_files:
                    result_html += f'<li>{f}</li>'
                result_html += '</ul></div>'
            
            if errors:
                result_html += '<div class="error"><strong>✗ 上传失败:</strong><ul>'
                for e in errors:
                    result_html += f'<li>{e}</li>'
                result_html += '</ul></div>'
            
            result_html += '<a href="/">← 返回继续上传</a></div></body></html>'
            self.wfile.write(result_html.encode('utf-8'))
        else:
            self.send_error(404)


if __name__ == '__main__':
    import socket
    
    # 获取本机 IP 地址
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    with socketserver.TCPServer(("", PORT), UploadHandler) as httpd:
        print("=" * 60)
        print("图片上传服务器已启动")
        print("=" * 60)
        print(f"本地访问: http://localhost:{PORT}")
        print(f"局域网访问: http://{local_ip}:{PORT}")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器")
        print("=" * 60)
        httpd.serve_forever()

