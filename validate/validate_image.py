from flask import jsonify
import io
from PIL import Image
import http



EXTENSIONS_IMAGE_ACEPT = ['jpeg', 'png', 'jpg']
MAX_SIZE_IMAGE = 50 * 1024 * 1024

def validate_request_image(request_file):
    if not request_file.files["image"]:
        data = {"erro:": "é necessário enviar uma imagem para a análise", 
                "code": http.HTTPStatus.UNPROCESSABLE_ENTITY}  
        return jsonify(data), http.HTTPStatus.UNPROCESSABLE_ENTITY
    return None
    
    
def validate_image_extension(file):
    if not file.filename.endswith((".png", ".jpeg", ".jpg")):
        print(file.filename)
        data = {"erro":f"extensão da imagem é inválida. São aceitas apenas imagens com extensões {EXTENSIONS_IMAGE_ACEPT}",
                "code": http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE}
        return jsonify(data), http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    return None
    
    
def convert_image_for_bytes_in_memory_and_open(file):
    #BytesIO trasforma a imagem em um fluxo de bytes na memoria. assim, não é preciso salvar a imagem no host
    img = Image.open(io.BytesIO(file.read()))
    return img    