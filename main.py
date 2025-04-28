from flask import Flask, request, jsonify
from ultralytics import YOLO
import http
from transformers import AutoImageProcessor, AutoModelForImageClassification
from validate.validate_image import validate_image_extension, validate_request_image, convert_image_for_bytes_in_memory_and_open
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
model = YOLO("best.pt")

# Lista de nomes das classes que fazem parte do modelo YOLO para detacção de pragas
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


plant_disease_detection_model_url = os.getenv("PLANT_DISEASE_DETECTION")

@app.route('/detect-plant-pest', methods=['POST'])
def detect():

    validate_image = validate_request_image(request)
    
    if validate_image:
        return validate_image
    
    file = request.files['image']
    
    validate_image = validate_image_extension(file)
    
    if validate_image:
        return validate_image
    
    try:
        img = convert_image_for_bytes_in_memory_and_open(file)
    except:
        data = {"erro": "imagem corrompida ou no formato incorreto", "code":http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE}
        return jsonify(data), http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    
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
    
    
    
@app.route("/detect-plant-disease", methods=["POST"])
def detect_plant_disease():


    validate_request = validate_request_image(request)
    
    if validate_request:
      return validate_request
    
    file = request.files["image"]
    
    
    try:
        model = AutoModelForImageClassification.from_pretrained(plant_disease_detection_model_url)   
        preprocessor = AutoImageProcessor.from_pretrained(plant_disease_detection_model_url)
    except:
        data = {"erro:": "Não foi possível encontrar o modelo", "code:":http.HTTPStatus.BAD_REQUEST}
        return jsonify(data), http.HTTPStatus.BAD_REQUEST
    
    try:
        image = convert_image_for_bytes_in_memory_and_open(file)
    
    except:
        data = {"erro": "imagem corrompida ou no formato incorreto", "code":http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE}
        return jsonify(data), http.HTTPStatus.BAD_REQUEST
        
    inputs = preprocessor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits

    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
    return jsonify({"Predicted class": model.config.id2label[predicted_class_idx]})


if __name__ == '__main__':
    app.run(debug=True, port=6000)
