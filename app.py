# app.py
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


MODEL_PATH = r"./models/pipeline_model.pkl"

def carregar_modelo():
    try:
        model = joblib.load('./models/pipeline_model.pkl')
        print("Modelo carregado com sucesso.")
        return model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None

model = carregar_modelo()

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

@app.route('/')
def index():
    print("Renderizando index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        print("Modelo não carregado corretamente.")
        return jsonify(error="Modelo não carregado corretamente."), 500

    data = request.form

    try:
        # Extrair dados do formulário
        
        input_data = [
            [
                data.get("dificuldade_materiais_pessoas"),
                data.get("realizou_obra"),
                data.get("desafios"),
                data.get("satisfacao_atual"),
                data.get("realizaria_reforma"),
                data.get("condicoes_financeiras"),
                data.get("valorizacao_sustentabilidade")
            ]
        ]
        

# print(input_data)


        input_df = pd.DataFrame(input_data, columns=[
            "Dificuldade_Materiais_Pessoas",
            "Realizou_Obra",
            "Desafios",
            "Satisfacao_Atual",
            "Realizaria_Reforma",
            "Condicoes_Financeiras",
            "Valorizacao_Sustentabilidade"
        ])

        prediction = model.predict(input_df)
        
        prediction_result = int(prediction[0])
        
        if prediction_result == 1:
            message = "Parabéns! Seu perfil se encaixa perfeitamente com o que oferecemos. Estamos ansiosos para te ajudar a alcançar seus objetivos!"
        else:
            message = "Obrigado por compartilhar! Fique atento para novidades!"
        
        return jsonify({"prediction": prediction_result, "message": message})
    
    except Exception as e:
        print(f"Erro ao processar a previsão: {str(e)}")
        return jsonify(error=f"Erro ao processar a previsão: {str(e)}"), 400

if __name__ == '__main__':
    app.run(port=int(os.getenv('PORT', 5500)), debug=False)