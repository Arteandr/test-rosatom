from catboost import CatBoostClassifier, CatBoostRegressor
import torch
import pickle as pkl
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_selection import mutual_info_regression, SelectKBest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

select_class = SelectKBest(mutual_info_regression,k=1000)
with open("../files/select_class.pkl", "rb") as f:
   select_class = pkl.load(f)

MODEL_NAME = 'intfloat/multilingual-e5-large'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
emb_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = emb_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

class ClassifierModel:
    def __init__(self, model_path):
        self._model = CatBoostClassifier()
        self._model.load_model(model_path)
        
    def predict(self, text):
       emb = get_embeddings(text)
       predict = self._model.predict([emb]) 
       print(predict)
       return predict[0]
        
clf = ClassifierModel("../files/classification_model.cbm")
clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

class RegressionModel:
    def __init__(self, model_path):
        self._model = CatBoostRegressor()
        self._model.load_model(model_path)
        
    def predict(self, text):
       emb = get_embeddings(text)
       matrix = select_class.transform([emb])
       print(matrix)
       predict = self._model.predict(matrix) 
       print(predict)
       return clamp(predict[0].round(), 0, 10)
        
reg = RegressionModel("../files/regression_model.cbm")