
from http.server import HTTPServer, BaseHTTPRequestHandler
import ssl
from io import BytesIO
import os



class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def processPath(self, path):
        if(path.startswith("/")):
            path = path[1:]
        while "//" in path:
            path = path.replace("//", "/")
        return path


    def do_GET(self):

        if("?" in self.path):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'Query parameter not supported')
            return
        if(self.path.startswith("/results/")):
            pass

        self.send_response(200)
        self.end_headers()
        self.wfile.write(bytes("Request: %s\n" % self.path, "utf-8"))
        self.wfile.write(b'Hello, world!')

    def do_POST(self):
        cleanPath = self.processPath(self.path)
        if(self.headers['Content-Type'].startswith("image")):
            print(os.path.abspath(os.getcwd()))
            print(cleanPath)
            file_path = os.path.join(os.path.abspath(os.getcwd()), cleanPath)
            print(file_path)
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            os.makedirs(os.path.dirname(file_path))
            with open(file_path, 'wb') as f:
                f.write(body)
            pass

        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        response = BytesIO()
        response.write(b'This is POST request. ')
        response.write(bytes("Request: %s" % self.path, "utf-8"))
        response.write(b'Received: ')
        response.write(body)
        self.wfile.write(response.getvalue())

    def do_DELETE(self):
        response = BytesIO()
        response.write(b'This is DELETE request. ')
        response.write(bytes("Request: %s" % self.path, "utf-8"))
        self.wfile.write(response.getvalue())



hostName = "localhost"
serverPort = 8080

if __name__ == "__main__":        
    httpd = HTTPServer((hostName, serverPort), SimpleHTTPRequestHandler)

    #httpd.socket = ssl.wrap_socket (httpd.socket, keyfile="./serverkey.pem", certfile='./servercert.pem', server_side=True)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print("Server stopped.")