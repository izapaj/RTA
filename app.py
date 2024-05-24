from flask import Flask

# Create a flask
app = Flask(__name__)

import numpy as np
import random

class Perceptron():
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        
        #self.w_ = np.zeros(1+X.shape[1])
        self.w_ = [random.uniform(-1.0, 1.0) for _ in range(1+X.shape[1])] 
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                #print(xi, target)
                update = self.eta*(target-self.predict(xi))
                #print(update)
                self.w_[1:] += update*xi
                self.w_[0] += update
                #print(self.w_)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X)>=0.0, 1, -1)

# Create an API end point
@app.route('/')
def result():
    model = Perceptron()
    X = np.array([1, 2, 3, 4]).reshape((-1, 1))
    y = np.array([-1, -1, 1, 1])
    model.fit(X,y)
    result = model.predict(np.array([-1, 0, 5, 6]).reshape(-1, 1))
    return 'Predykcje z modelu:' + ' '.join(map(str,result.tolist()))


if __name__ == '__main__':
    app.run() # domyślnie ustawia localhost i port 5000
    # app.run(host='0.0.0.0', port=8000)
