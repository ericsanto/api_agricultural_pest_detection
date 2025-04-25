from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import http
import os

app = Flask(__name__)
model = YOLO("best.pt")

# Lista de nomes das classes que fazem parte do modelo
class_names = [
    'rice leaf roller', 'rice leaf caterpillar', 'paddy stem maggot', 'asiatic rice borer', 'yellow rice borer',
    'rice gall midge', 'Rice Stemfly', 'brown plant hopper', 'white backed plant hopper', 'small brown plant hopper',
    'rice water weevil', 'rice leafhopper', 'grain spreader thrips', 'rice shell pest', 'grub', 'mole cricket', 'wireworm',
    'white margined moth', 'black cutworm', 'large cutworm', 'yellow cutworm', 'red spider', 'corn borer', 'army worm', 'aphids',
    'Potosiabre vitarsis', 'peach borer', 'english grain aphid', 'green bug', 'bird cherry-oataphid', 'wheat blossom midge',
    'penthaleus major', 'longlegged spider mite', 'wheat phloeothrips', 'wheat sawfly', 'cerodonta denticornis', 'beet fly',
    'flea beetle', 'cabbage army worm', 'beet army worm', 'Beet spot flies', 'meadow moth', 'beet weevil', 'sericaorient alismots chulsky',
    'alfalfa weevil', 'flax budworm', 'alfalfa plant bug', 'tarnished plant bug', 'Locustoidea', 'lytta polita', 'legume blister beetle',
    'blister beetle', 'therioaphis maculata Buckton', 'odontothrips loti', 'Thrips', 'alfalfa seed chalcid', 'Pieris canidia',
    'Apolygus lucorum', 'Limacodidae', 'Viteus vitifoliae', 'Colomerus vitis', 'Brevipoalpus lewisi McGregor', 'oides decempunctata',
    'Polyphagotars onemus latus', 'Pseudococcus comstocki Kuwana', 'parathrene regalis', 'Ampelophaga', 'Lycorma delicatula', 'Xylotrechus',
    'Cicadella viridis', 'Miridae', 'Trialeurodes vaporariorum', 'Erythroneura apicalis', 'Papilio xuthus', 'Panonchus citri McGregor',
    'Phyllocoptes oleiverus ashmead', 'Icerya purchasi Maskell', 'Unaspis yanonensis', 'Ceroplastes rubens', 'Chrysomphalus aonidum',
    'Parlatoria zizyphus Lucus', 'Nipaecoccus vastalor', 'Aleurocanthus spiniferus', 'Tetradacus c Bactrocera minax', 'Dacus dorsalis(Hendel)',
    'Bactrocera tsuneonis', 'Prodenia litura', 'Adristyrannus', 'Phyllocnistis citrella Stainton', 'Toxoptera citricidus', 'Toxoptera aurantii',
    'Aphis citricola Vander Goot', 'Scirtothrips dorsalis Hood', 'Dasineura sp', 'Lawana imitata Melichar', 'Salurnis marginella Guerr',
    'Deporaus marginatus Pascoe', 'Chlumetia transversa', 'Mango flat beak leafhopper', 'Rhytidodera bowrinii white', 'Sternochetus frigidus',
    'Cicadellidae'
]

EXTENSIONS_IMAGE_ACEPT = ['jpeg', 'png', 'jpg']
MAX_SIZE_IMAGE = 50 * 1024 * 1024


@app.route('/detect', methods=['POST'])
def detect():
    
    if not request.files["image"]:
        data = {"erro:": "é necessário enviar uma imagem para a análise", 
                "code": http.HTTPStatus.UNPROCESSABLE_ENTITY}
        
        return jsonify(data), 422
        
    file = request.files['image']
    
    if not file.filename.endswith((".png", ".jpeg", ".jpg")):
            print(file.filename)
            data = {"erro":f"extensão da imagem é inválida. São aceitas apenas imagens com extensões {EXTENSIONS_IMAGE_ACEPT}",
                    "code": http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE}
            return jsonify(data), http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    
    try:
    #BytesIO trasforma a imagem em um fluxo de bytes na memoria. assim, não é preciso salvar a imagem no host
        img = Image.open(io.BytesIO(file.read()))
    except:
        data = {"erro": "imagem corrompida ou no formato incorreto", "code":http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE}
        return jsonify(data, http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
    
    #Aqui será a imagem sera processada por um modelo YOLO que foi definido acima
    results = model(img)

    #detection é uma lista onde contém informações sobre o objeto detectado
    #decttions retornará uma lista dessa forma: [x1, x2, y1, y2, confidence, int(class_name)]
    #essas classes já estão definidas no modelo treinado. a lista class_names que foi declarada acima é importante para retornar o nome dela
    #x1, x2, y1, y2 são as corrdenadas do objeto que foi detectado na imagem. Tudo que estiver dentro dessas coordenadas é considerado parte do objeto
    detections = results[0].boxes.data.cpu().numpy().tolist()
    
    try:
        detections_with_names = []
        for detection in detections:
            if detection[4] >= 0.6:
                class_index = int(detection[5])
                class_name = class_names[class_index]
                hit_percentage = round(detection[4] * 100)
                hit_percentage_formated = str(f'{hit_percentage}%')
                detections_with_names.append({
                    "pest": class_name,
                    "confidence": detection[4],
                    "hit_percentage": hit_percentage,
                    "hit_percentage_formated": hit_percentage_formated
                })
                    
        return jsonify({"detections": detections_with_names})
    except: 
        data = {"erro":"erro interno no servidor", "code":http.HTTPStatus.INTERNAL_SERVER_ERROR}
        return jsonify, http.HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=True, port=6000)
