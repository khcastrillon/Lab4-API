from joblib import load

class Model:
    def __init__(self,columns):
        self.model = load("assets/pipeline.joblib")

    def make_predictions(self, data):
        print(self.model)
        result = self.model.predict(data)
        return result

    def R2(self, data, y):
        r2 = self.model.score(data,y)
        return r2
