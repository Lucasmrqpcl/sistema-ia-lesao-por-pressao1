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

        resultado_triagem = dados['triagem']

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

        # IA
        previsao = modelo.predict(img_array)[0]

        indice = np.argmax(previsao)

        resultado_ia = classes[indice]

        maior_probabilidade = float(np.max(previsao))

        # =========================
        # RESULTADO FINAL
        # =========================

        # TRIAGEM TEM PRIORIDADE
        if resultado_triagem is not None:

            resultado_final = resultado_triagem

            # verificar compatibilidade
            if resultado_triagem == resultado_ia:

                compatibilidade = 'Alta'

            else:

                compatibilidade = 'Moderada'

        else:

            resultado_final = resultado_ia

            # confiança IA
            if maior_probabilidade >= 0.80:

                compatibilidade = 'Alta'

            elif maior_probabilidade >= 0.60:

                compatibilidade = 'Moderada'

            else:

                compatibilidade = 'Baixa'

        # RETORNO
        return jsonify({

            'resultado': f'Lesão por Pressão - {resultado_final}',

            'compatibilidade': compatibilidade

        })

    except Exception as erro:

        return jsonify({

            'resultado': f'Erro: {str(erro)}',

            'compatibilidade': 'Erro'

        })

# EXECUTAR
if __name__ == '__main__':

    app.run(debug=True)