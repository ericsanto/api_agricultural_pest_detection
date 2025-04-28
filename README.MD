
# API de Detecção de Pragas

Esta é uma API REST baseada em Flask que utiliza um modelo YOLO pré-treinado do HugginFace para detectar pragas em imagens enviadas pelos usuários. O modelo foi treinado com imagens de várias pragas que afetam as lavouras de arroz e retorna informações sobre as pragas detectadas com uma pontuação de confiança.

## Créditos

O modelo YOLO de detecção de pragas utilizado nesta aplicação foi desenvolvido por [Yudsky](https://huggingface.co/Yudsky/pest-detection-yolo11/tree/main) e está disponível no [Hugging Face](https://huggingface.co/Yudsky/pest-detection-yolo11/tree/main). Agradecemos ao autor por disponibilizar o modelo pré-treinado.

o modelo para detecção de doeças de plantas utilizado nessa aplicação foi desenvolvido por Diginsa(https://huggingface.co/Diginsa) e está disponível no [Huggin Face](https://huggingface.co/Diginsa/Plant-Disease-Detection-Project). Agradecemos ao autor por disponibilizar o modelo pré-treinado

## Funcionalidades

- Aceita o envio de imagens (formatos JPEG, PNG, JPG).
- Usa o modelo YOLO (You Only Look Once) para detecção de objetos.
- Retorna os nomes das pragas detectadas junto com a porcentagem de confiança.
- Aplicação dockerizada para fácil implantação.

## Requisitos

- Python 3.10
- Docker
- Docker Compose

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seunome/pest-detection-api.git
cd pest-detection-api
```

### 2. Construa e execute com Docker Compose

Para construir a aplicação usando Docker Compose e iniciar o serviço:

```bash
docker-compose up --build
```

Isso criará um container Docker para a aplicação Flask e iniciará o servidor.

### 3. Acesse a Aplicação

A aplicação estará acessível em `http://localhost:6000`.

## Documentação da API

### Endpoints

#### `POST /detect-plant-pest`

Este endpoint aceita um arquivo de imagem, processa-o usando o modelo YOLO e retorna uma lista de pragas detectadas junto com seus níveis de confiança.

##### Requisição

- **Content-Type**: `multipart/form-data`
- **Body**: A requisição deve incluir um arquivo de imagem nos dados do formulário. A imagem deve estar em um dos seguintes formatos: `.jpg`, `.jpeg` ou `.png`.

Exemplo de Requisição:
```bash
curl -X POST -F "image=@caminho/para/sua/imagem.jpg" http://localhost:6000/detect-plant-pest
```

##### Resposta

A resposta será um objeto JSON contendo uma lista de pragas detectadas, sua pontuação de confiança e a porcentagem formatada de acerto.

- **Status 200 OK**: Os resultados da detecção foram processados com sucesso.

Exemplo de Resposta:
```json
{
  "detections": [
    {
      "pest": "rice leaf roller",
      "confidence": 0.91,
      "hit_percentage": 91,
      "hit_percentage_formated": "91%"
    },
    {
      "pest": "yellow rice borer",
      "confidence": 0.88,
      "hit_percentage": 88,
      "hit_percentage_formated": "88%"
    }
  ]
}
```


#### `POST /detect-plant-disease`

Este endpoint aceita um arquivo de imagem, processa-o usando o modelo pre treinado e retorna uma a doença da planta detectada

##### Requisição

- **Content-Type**: `multipart/form-data`
- **Body**: A requisição deve incluir um arquivo de imagem nos dados do formulário. A imagem deve estar em um dos seguintes formatos: `.jpg`, `.jpeg` ou `.png`.

Exemplo de Requisição:
```bash
curl -X POST -F "image=@caminho/para/sua/imagem.jpg" http://localhost:6000/detect-plant-disease
```

##### Resposta

A resposta será um objeto JSON contendo a doença detectada

- **Status 200 OK**: Os resultados da detecção foram processados com sucesso.

Exemplo de Resposta:
```json
{
  "disease": "Corn (maize) Cercospora leaf spot Gray leaf spot"
}
```

- **Status 415 UNSUPPORTED_MEDIA_TYPE**: Quando o arquivo enviado não é uma imagem válida ou o formato não é aceito.

Exemplo de Resposta de Erro:
```json
{
  "erro": "extensão da imagem é inválida. São aceitas apenas imagens com extensões ['jpeg', 'png', 'jpg']",
  "code": 415
}
```

- **Status 422 UNPROCESSABLE ENTITY**: Quando não é enviado um arquivo de imagem na requisição.

Exemplo de Resposta de Erro:
```json
{
  "erro": "é necessário enviar uma imagem para a análise",
  "code": 422
}
```

- **Status 415 UNSUPPORTED MEDIA TYPE**: Quando a imagem enviada está corrompida ou em um formato inválido.

Exemplo de Resposta de Erro:
```json
{
  "erro": "imagem corrompida ou no formato incorreto",
  "code": 415
}
```

- **Status 500 INTERNAL SERVER ERROR**: Quando ocorre um erro interno no servidor.

Exemplo de Resposta de Erro:
```json
{
  "erro": "erro interno no servidor",
  "code": 500
}
```

## Testando Localmente em ambiente Linux

Se você quiser testar localmente, basta rodar a aplicação Flask diretamente, siga os seguintes comandos:

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```

A aplicação estará acessível em `http://localhost:6000`.

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

```