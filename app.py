from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Pasta uploads
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carregar modelo treinado
modelo = load_model('modelo_lesao_lpp.h5')

# Classes da IA
classes = [
    'Estágio 1',
    'Estágio 2',
    'Estágio 3',
    'Estágio 4'
]

# Página principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota análise
@app.route('/analisar', methods=['POST'])
def analisar():

    try:

        dados = request.get_json()

        imagem_base64 = dados['imagem']
        respostas = dados['respostas']

        # Remove cabeçalho base64
        imagem_base64 = imagem_base64.split(',')[1]

        # Converter imagem
        imagem_bytes = base64.b64decode(imagem_base64)

        imagem = Image.open(BytesIO(imagem_bytes))

        # Caminho
        caminho = os.path.join(
            app.config['UPLOAD_FOLDER'],
            'captura.png'
        )

        # Salvar
        imagem.save(caminho)

        # Preparar imagem
        img = image.load_img(
            caminho,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        img_array = img_array / 255.0

        # IA
        previsao = modelo.predict(img_array)

        indice = np.argmax(previsao)

        resultado_ia = classes[indice]

        # =========================
        # TRIAGEM CLÍNICA
        # =========================

        # respostas:
        # 0 = vermelhidão
        # 1 = superficial
        # 2 = gordura
        # 3 = músculo/osso/tendão
        # 4 = necrose

        if respostas[3] == 'sim':

            resultado = 'Lesão por Pressão - Estágio 4'

        elif respostas[2] == 'sim':

            resultado = 'Lesão por Pressão - Estágio 3'

        elif respostas[1] == 'sim':

            resultado = 'Lesão por Pressão - Estágio 2'

        elif respostas[0] == 'sim':

            resultado = 'Lesão por Pressão - Estágio 1'

        else:

            resultado = resultado_ia

        return jsonify({
            'resultado': resultado
        })

    except Exception as erro:

        return jsonify({
            'resultado': f'Erro: {str(erro)}'
        })

if __name__ == '__main__':
    app.run(debug=True)