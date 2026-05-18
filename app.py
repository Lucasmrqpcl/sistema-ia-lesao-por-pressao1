# app.py

from flask import Flask, render_template, request, jsonify

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import os
import base64

from PIL import Image
from io import BytesIO

app = Flask(__name__)

# PASTA UPLOADS
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MODELO IA
modelo = load_model('modelo_lesao_lpp.h5')

# CLASSES
classes = [
    'Estágio 1',
    'Estágio 2',
    'Estágio 3',
    'Estágio 4'
]

# HOME
@app.route('/')
def index():

    return render_template('index.html')

# ANÁLISE
@app.route('/analisar', methods=['POST'])
def analisar():

    try:

        dados = request.get_json()

        imagem_base64 = dados['imagem']

        respostas = dados['respostas']

        # REMOVER HEADER BASE64
        imagem_base64 = imagem_base64.split(',')[1]

        # DECODIFICAR IMAGEM
        imagem_bytes = base64.b64decode(imagem_base64)

        imagem = Image.open(BytesIO(imagem_bytes))

        # CAMINHO
        caminho = os.path.join(
            app.config['UPLOAD_FOLDER'],
            'captura.png'
        )

        # SALVAR
        imagem.save(caminho)

        # PREPARAR IMAGEM
        img = image.load_img(
            caminho,
            target_size=(224,224)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = img_array / 255.0

        # PREVISÃO IA
        previsao = modelo.predict(img_array)

        probabilidades = previsao[0]

        # PONTUAÇÃO IA
        pontuacao_final = {

            'Estágio 1': float(probabilidades[0]),

            'Estágio 2': float(probabilidades[1]),

            'Estágio 3': float(probabilidades[2]),

            'Estágio 4': float(probabilidades[3])

        }

        # =========================
        # TRIAGEM CLÍNICA
        # =========================

        # Vermelhidão
        if respostas[0] == 'sim':

            pontuacao_final['Estágio 1'] += 0.30

        # Superficial
        if respostas[1] == 'sim':

            pontuacao_final['Estágio 2'] += 0.35

        # Gordura/cavidade
        if respostas[2] == 'sim':

            pontuacao_final['Estágio 3'] += 0.45

        # Músculo/osso/tendão
        if respostas[3] == 'sim':

            pontuacao_final['Estágio 4'] += 0.55

        # Necrose
        if respostas[4] == 'sim':

            pontuacao_final['Estágio 3'] += 0.20
            pontuacao_final['Estágio 4'] += 0.20

        # RESULTADO FINAL
        resultado = max(
            pontuacao_final,
            key=pontuacao_final.get
        )

        # RETORNO
        return jsonify({

            'resultado': f'Lesão por Pressão - {resultado}'

        })

    except Exception as erro:

        return jsonify({

            'resultado': f'Erro: {str(erro)}'

        })

# EXECUTAR
if __name__ == '__main__':

    app.run(debug=True)