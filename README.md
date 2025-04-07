# Obesity Level Prediction

Este projeto utiliza algoritmos de Machine Learning para prever o nível de obesidade de um indivíduo com base em características pessoais e hábitos de vida. O dataset é baseado em dados reais coletados de pessoas em diferentes regiões e com diferentes estilos de vida.

## 📊 Dataset

O dataset utilizado é o [Obesity Level dataset](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams), que inclui variáveis como:

- Idade, altura e peso
- Frequência de consumo de alimentos (FAVC, FCVC, NCP, CAEC)
- Atividade física e transporte (FAF, TUE, MTRANS)
- Fatores comportamentais (SMOKE, family_history_with_overweight)
- Variável alvo: **NObeyesdad**, representando o nível de obesidade

## ⚙️ Tecnologias e Bibliotecas

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Seaborn, Matplotlib

## 🔍 Pré-processamento

- **Label Encoding** para colunas categóricas binárias
- **One-Hot Encoding** para colunas com múltiplas categorias
- **Padronização com StandardScaler**

## 🧠 Modelos Utilizados

- KNN
- SVC
- Logistic Regression
- Random Forest (com GridSearch para tuning)
- XGBoost
- Redes Neurais (ANN com Keras)

## 🧪 Exemplo de Previsão

```python
print(best_forest.predict(scaler.transform([
    [
        0,      # Gender (encoded)
        21,     # Age
        1.90,   # Height
        64,     # Weight
        1, 0,   # family_history_with_overweight, FAVC
        2.0,    # FCVC
        3.0,    # NCP
        0, 1.0, # CAEC, SMOKE
        1, 0, 0, 1, 0, 0, 0, 0,  # CALC + MTRANS (one-hot)
        1, 0, 0, 0, 1, 0         # MTRANS cont. + outras features
    ]
])))
# Output: [0] -> Insufficient_Weight
