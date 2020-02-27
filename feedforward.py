import numpy as np
import copy

'''
Discoveries:
    An adaptive learning parameter is required, since as the neural net converges on the minimum the error surface becomes increasingly jagged. The learning rate must diminish or training will perpetually overshoot. Decreasing the learning rate is equivalent to "zooming in" on the error surface.

    There are long stretches where the error seems to not be dimishing. Relatedly, the adaptive delta rapidly diminishes to below the representational capacity of python.

    In the case of XOR, The net achieves linear seperability on the classes without diminishing the error to zero (or near zero)
'''

class Layer():
    def __init__(self, inputDim, outputDim):
        self.dim = [inputDim, outputDim]
        self.input = np.zeros(self.dim[0]+1)
        self.input[-1] = 1
        self.weights = np.multiply(np.random.normal(size=(self.dim[0]+1, self.dim[1])), 3)
        self.prevW = copy.deepcopy(self.weights)
        self.derivative = np.zeros((self.dim[0]+1, self.dim[1]))
        self.output = np.zeros(self.dim[1])

    def enter(self, vec):
        for i in range(self.dim[0]):
            self.input[i] = vec[i]

    def forward(self):
        self.output = np.matmul(self.input, self.weights)
        return self.output

    def backward(self, vec):
        for i in range(self.dim[0]+1):
            for j in range(self.dim[1]):
                self.derivative[i,j] += self.input[i]*vec[j]
        return np.multiply(vec, self.weights)[:-1]

    def gradient(self, delta):
        self.prevW = copy.deepcopy(self.weights)
        self.weights = np.add(self.weights, np.multiply(-1*delta, self.derivative))
        self.derivative = np.zeros((self.dim[0]+1, self.dim[1]))

    def revert(self):
        self.weights = self.prevW


class Sigmoidal():
    def __init__(self, dim):
        self.dim = dim
        self.input = np.zeros(self.dim)
        self.output = np.zeros(self.dim)

    def enter(self, vec):
        for i in range(self.dim):
            self.input[i] = vec[i]

    def forward(self):
        self.output = np.divide(np.ones(self.dim), np.add(np.exp(self.input),1))
        return self.output

    def backward(self, vec):
        return np.multiply(np.multiply(self.output, np.subtract(1, self.output)), vec)

    def gradient(self, delta):
        pass

    def revert(self):
        pass

'''
model = [Layer(1,1), Sigmoidal(1)]
X = [np.array([x]) for x in range(2)]
Y = [np.array([x]) for x in range(2)]

def train(model, X, Y):
    for i,x in enumerate(X):
        model[0].enter(x)
        hid = model[0].forward()

        model[1].enter(hid)
        out = model[1].forward()

        derror = -2*(out[0]-Y[i][0])
        #print(derror)
        signal = model[1].backward(np.array([derror]))
        #print(signal)
        model[0].backward(signal)
        #print(model[0].derivative)
        #sys.exit()

    for x in model:
        x.gradient(0.1)

def validate(model, X, Y):
    sqError = 0
    for i,x in enumerate(X):
        model[0].enter(x)
        hid = model[0].forward()

        model[1].enter(hid)
        out = model[1].forward()

        sqError += (out[0]-Y[i][0])**2
    return sqError
'''

def revert(model):
    for x in model:
        x.revert()

def train(model, X, Y, delta=0.01):
    for i,x in enumerate(X):
        model[0].enter(x)
        hid = model[0].forward()

        model[1].enter(hid)
        hid = model[1].forward()

        model[2].enter(np.concatenate((x,hid)))
        out = model[2].forward()

        model[3].enter(out)
        out = model[3].forward()

        signal = model[3].backward(np.array([-2*(out[0]-Y[i][0])]))
        signal = model[2].backward(signal)

        signal = model[1].backward(signal[-1])
        model[0].backward(signal)
    for x in model:
        x.gradient(delta)

def validate(model, X, Y, verbose=False):
    sqError = 0
    store = []
    for i,x in enumerate(X):
        model[0].enter(x)
        hid = model[0].forward()
        model[1].enter(hid)
        hid = model[1].forward()
        
        model[2].enter(np.concatenate((x,hid)))
        out = model[2].forward()
        

        model[3].enter(out)
        out = model[3].forward()
        
        store.append(out[0])

        sqError += (out[0]-Y[i][0])**2
    
        if verbose:
            print(np.concatenate((x,hid)))
            print(out, Y[i])
            print(model[0].input)
            print(model[0].weights)
    if verbose:
        print(sqError)
        print()

    test1 = store[1] > store[0] and store[1] > store[3]
    test2 = store[2] > store[0] and store[2] > store[3]
    if verbose:
        print(test1, test2)

        #print() 
    #print()
    return sqError, test1 and test2

lucky = False
prevErr = 10000
delta = .1
for j in range(100000000):
    model = [Layer(2, 1), Sigmoidal(1), Layer(3,1), Sigmoidal(1)]
    #model[0].weights[0,0] = 2
    #model[0].weights[1,0] = 2
    #model[0].weights[2,0] = -3
    #model[0].weights[0,0] = j%10*.1-0.5
    #model[0].weights[1,0] = j//10%10*.1-0.5
    #model[0].weights[2,0] = j//10**2%10*.1-0.5
    #model[2].weights[0,0] = j//10**3%10*.1-0.5
    #model[2].weights[1,0] = j//10**4%10*.1-0.5
    #model[2].weights[2,0] = j//10**5%10*.1-0.5
    #model[2].weights[3,0] = j//10**6%10*.1-0.5
    #print(model[0].weights)
    #print(model[2].weights)
    X = [np.array([i, j])  for i in range(2) for j in range(2)]
    Y = [np.array([int(i!=j)]) for i in range(2) for j in range(2)]
    count = 0
    print()
    print("before training:")
    validate(model, X, Y)
    while True:
        train(model, X, Y, delta)
        err, sep = validate(model, X, Y, lucky)
        print(prevErr, delta)
        if prevErr < err:
            delta /= 2
            revert(model)
        #elif prevErr == err:
        #    validate(model, X, Y, True)
        #    sys.exit(0)
        else:
            prevErr = err
        if count % 100 == 99:
            print("after training:")
            if not sep and not lucky:
                break
            else:
                print("LUCKY!")
                lucky = True
        count += 1
