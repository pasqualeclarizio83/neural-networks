# Reti Neurali

Breve guida sulle Reti Neurali

## Descrizione

Reti Neurali e Introduzione


```bash

Le reti neurali sono un tipo di modello computazionale ispirato al funzionamento del cervello umano. Si tratta di un sistema di algoritmi e matematica che imita il modo in cui il cervello elabora le informazioni.

Immagina una rete neurale come un insieme di "neuroni artificiali" collegati tra loro, organizzati in strati. Ogni neurone riceve delle informazioni di input, le elabora e passa il risultato agli altri neuroni connessi ad esso. Questo processo è iterato attraverso più strati fino a produrre un'output.

La formazione di una rete neurale coinvolge due fasi principali: l'addestramento e l'utilizzo. Durante l'addestramento, la rete neurale impara dai dati, regolando i pesi delle connessioni tra i neuroni in modo da minimizzare gli errori tra l'output previsto e quello reale. Una volta addestrata, la rete può essere utilizzata per fare previsioni o classificazioni su nuovi dati.

Le reti neurali sono ampiamente utilizzate in molte applicazioni, come il riconoscimento vocale, il riconoscimento facciale, la traduzione automatica, il riconoscimento di pattern in immagini e molto altro ancora. La loro capacità di imparare dai dati li rende molto potenti e versatili per risolvere una vasta gamma di problemi complessi.


```


## Esempi di Reti Neurali - Classificatore

```bash

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Caricamento del dataset di esempio (Iris)
iris = load_iris()
X = iris.data
y = iris.target

# Divisione dei dati in set di training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione del modello XGBoost
model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, random_state=42)

# Addestramento del modello
model.fit(X_train, y_train)

# Predizione sui dati di test
y_pred = model.predict(X_test)

# Calcolo dell'accuratezza
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

```







## un algoritmo di boosting binario simile a XGBoost utilizzando solo Python puro. Abbiamo definito un ## dataset di esempio X e y. Poi, abbiamo inizializzato i pesi degli alberi e addestrato una serie di ## alberi decisionali attraverso l'iterazione su un numero fissato di stimatori

```bash

import numpy as np

# Definizione del dataset di esempio
X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])
y = np.array([0, 0, 1, 1])

# Definizione dei parametri dell'algoritmo
learning_rate = 0.1
n_estimators = 100
max_depth = 3

# Inizializzazione dei pesi degli alberi
trees_weights = np.ones(n_estimators)

# Addestramento degli alberi
for i in range(n_estimators):
    # Calcolo del gradiente
    gradients = 2 * (y - 1 / (1 + np.exp(-X.dot(trees_weights))))
    
    # Calcolo degli Hessiani
    hessians = 2 * np.exp(X.dot(trees_weights)) / ((1 + np.exp(X.dot(trees_weights)))**2)
    
    # Calcolo del valore da ottimizzare
    obj_value = np.sum(gradients) / np.sum(hessians)
    
    # Aggiornamento dei pesi degli alberi
    trees_weights[i] += learning_rate * obj_value
    
# Predizione
def predict(X, trees_weights):
    return np.round(1 / (1 + np.exp(-X.dot(trees_weights))))

# Esempio di predizione
X_test = np.array([[1, 2], [2, 3]])
predictions = predict(X_test, trees_weights)
print("Predictions:", predictions)

```




## In questo esempio, stiamo creando e addestrando una rete neurale con un layer nascosto. 
## Utilizziamo la funzione di attivazione sigmoide sia per il layer nascosto che per l'output. 
## Abbiamo implementato il forward propagation, il calcolo dell'errore, e il backpropagation

```bash

import numpy as np

# Definizione della funzione di attivazione (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definizione della derivata della funzione di attivazione
def sigmoid_derivative(x):
    return x * (1 - x)

# Definizione dei dati di input e output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Inizializzazione dei pesi
np.random.seed(1)
weights_input_hidden = np.random.uniform(size=(2, 3))
weights_hidden_output = np.random.uniform(size=(3, 1))

# Addestramento della rete neurale
learning_rate = 0.5
epochs = 10000

for epoch in range(epochs):
    # Forward propagation
    input_layer = X
    hidden_layer_input = np.dot(input_layer, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_input)
    
    # Calcolo dell'errore
    error = y - output
    
    # Backpropagation
    d_output = error * sigmoid_derivative(output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Aggiornamento dei pesi
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += input_layer.T.dot(d_hidden_layer) * learning_rate

# Predizione
input_layer = X
hidden_layer_input = np.dot(input_layer, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_input)

print("Output dopo l'addestramento:")
print(output)

```





## Un esempio molto semplice di Rete Neurale

```bash

import numpy as np

# Definizione della funzione di attivazione ReLU
def relu(x):
    return np.maximum(0, x)

# Definizione dei dati di input e output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inizializzazione dei pesi e dei bias
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Addestramento della rete neurale
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    output = relu(z2)

    # Calcolo della loss
    loss = np.square(output - y).mean()

    # Backpropagation
    grad_output = 2 * (output - y)
    grad_z2 = grad_output * (z2 > 0)
    grad_W2 = np.dot(a1.T, grad_z2)
    grad_b2 = np.sum(grad_z2, axis=0, keepdims=True)
    grad_a1 = np.dot(grad_z2, W2.T)
    grad_z1 = grad_a1 * (z1 > 0)
    grad_W1 = np.dot(X.T, grad_z1)
    grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

    # Aggiornamento dei pesi e dei bias
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1
    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Output finale:")
print(output)

```











## Rete Neurale e semplice Neurone

```bash

import numpy as np

# Definiamo i dati di input e output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inizializziamo i pesi e il bias casualmente
np.random.seed(0)
weights = np.random.randn(2, 1)
bias = np.random.randn(1)

# Definiamo la funzione di attivazione (funzione di identità)
def activation(x):
    return x

# Addestramento della rete neurale
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # Forward pass
    output = activation(np.dot(X, weights) + bias)

    # Calcolo della loss (errore quadratico medio)
    loss = np.mean((output - y) ** 2)

    # Backpropagation
    d_weights = np.dot(X.T, (output - y))
    d_bias = np.sum(output - y)

    # Aggiornamento dei pesi e del bias
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Output finale:")
print(output)

```





## Ecco un esempio ancora più semplice di una rete neurale con un 
## singolo neurone utilizzando solamente Python e algebra lineare

```bash

import numpy as np

# Definiamo il dato di input e di output
X = np.array([1])
y = np.array([0])

# Inizializziamo casualmente il peso e il bias
weight = np.random.randn()
bias = np.random.randn()

# Definiamo la funzione di attivazione (funzione di identità)
def activation(x):
    return x

# Addestramento della rete neurale
learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    output = activation(X * weight + bias)

    # Calcolo della loss (errore quadratico medio)
    loss = np.mean((output - y) ** 2)

    # Backpropagation
    d_weight = 2 * X * (output - y)
    d_bias = 2 * (output - y)

    # Aggiornamento del peso e del bias
    weight -= learning_rate * d_weight
    bias -= learning_rate * d_bias

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Output finale:")
print(output)

```


## Una rete neurale è un modello computazionale ispirato al funzionamento del cervello umano. 
## Consiste in una serie di "neuroni" artificiali organizzati in strati, ognuno dei quali trasforma l'input ## in un output attraverso operazioni matematiche. Questi strati sono chiamati "layer", e la rete può avere ## diversi tipi di layer, come layer di input, layer nascosti e layer di output.

## In una rete neurale, l'input viene passato attraverso i vari layer, con ciascun layer che esegue una ## trasformazione lineare seguita da una funzione non lineare chiamata "funzione di attivazione". Questo ## processo di trasformazione viene chiamato "forward pass". Durante l'addestramento, la rete neurale cerca ## di migliorare la sua capacità di produrre output accurati attraverso un processo chiamato ## "backpropagation", che calcola l'errore tra l'output previsto e l'output desiderato e aggiorna i pesi ## dei neuroni per ridurre questo errore.

```bash

import numpy as np

# Dati di input e output
X = np.array([[0], [1], [2], [3], [4], [5]])
y = np.array([[0], [2], [4], [6], [8], [10]])

# Inizializzazione dei pesi e del bias
np.random.seed(0)
weight = np.random.randn()
bias = np.random.randn()

# Definizione della funzione di attivazione (identità)
def activation(x):
    return x

# Addestramento della rete neurale
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    output = activation(X * weight + bias)

    # Calcolo della loss (errore quadratico medio)
    loss = np.mean((output - y) ** 2)

    # Backpropagation
    d_weight = 2 * np.mean(X * (output - y))
    d_bias = 2 * np.mean(output - y)

    # Aggiornamento dei pesi e del bias
    weight -= learning_rate * d_weight
    bias -= learning_rate * d_bias

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Output finale:")
print(output)

```



## Una rete Neurale
##  un modello che simula in modo molto semplice il pensiero, possiamo creare una rete neurale con un ## singolo neurone

```bash

import numpy as np

# Dati di input e output
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Input: matrice di numeri
y = np.array([[0], [1], [1]])  # Output desiderato: 0 se la somma è <= 6, altrimenti 1

# Inizializzazione dei pesi e del bias
np.random.seed(0)
weights = np.random.randn(3)  # Iniziamo con pesi casuali per i 3 input
bias = np.random.randn()

# Definizione della funzione di attivazione (funzione di soglia)
def activation(x):
    return 1 if x > 0 else 0

# Addestramento della rete neurale
learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    output = activation(np.dot(X, weights) + bias)

    # Calcolo della loss (errore quadratico medio)
    loss = np.mean((output - y) ** 2)

    # Backpropagation
    d_weights = np.dot(X.T, (output - y))
    d_bias = np.sum(output - y)

    # Aggiornamento dei pesi e del bias
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

print("Output finale:")
print(output)

```





## Una rete Neurale - del pensiero
##  creeremo una piccola rete neurale con un solo strato nascosto per classificare frasi 
## come "positive" o "negative" utilizzando un approccio di bag of words.

```bash

import numpy as np

# Frasi di esempio
sentences = [
    'Mi sento felice oggi',
    'Che bella giornata!',
    'Non mi piace affatto questo cibo',
    'Mi sento molto triste',
    'Sto molto bene adesso'
]

# Etichette corrispondenti: 1 per positive, 0 per negative
labels = np.array([1, 1, 0, 0, 1])

# Costruzione del vocabolario
vocab = set()
for sentence in sentences:
    words = sentence.lower().split()
    vocab.update(words)

# Mappatura delle parole ai loro indici nel vocabolario
word_to_index = {word: i for i, word in enumerate(vocab)}

# Creazione di una funzione per rappresentare le frasi come vettori one-hot
def sentence_to_vector(sentence, word_to_index):
    vector = np.zeros(len(word_to_index))
    words = sentence.lower().split()
    for word in words:
        if word in word_to_index:
            vector[word_to_index[word]] = 1
    return vector

# Conversione delle frasi in vettori
X_train = np.array([sentence_to_vector(sentence, word_to_index) for sentence in sentences])

# Definizione della rete neurale
class SimpleNN:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weights_1 = np.random.randn(self.input_size, self.hidden_size)
        self.bias_1 = np.zeros(self.hidden_size)
        self.weights_2 = np.random.randn(self.hidden_size, 1)
        self.bias_2 = np.zeros(1)

    def forward(self, X):
        self.hidden_layer = np.maximum(0, np.dot(X, self.weights_1) + self.bias_1)
        return np.dot(self.hidden_layer, self.weights_2) + self.bias_2

    def train(self, X, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calcolo della loss
            loss = np.mean((output - y) ** 2)
            
            # Calcolo del gradiente
            grad_output = 2 * (output - y)
            grad_weights_2 = np.dot(self.hidden_layer.T, grad_output)
            grad_bias_2 = np.sum(grad_output, axis=0)
            grad_hidden = np.dot(grad_output, self.weights_2.T)
            grad_hidden[self.hidden_layer <= 0] = 0
            grad_weights_1 = np.dot(X.T, grad_hidden)
            grad_bias_1 = np.sum(grad_hidden, axis=0)
            
            # Aggiornamento dei pesi
            self.weights_1 -= learning_rate * grad_weights_1
            self.bias_1 -= learning_rate * grad_bias_1
            self.weights_2 -= learning_rate * grad_weights_2
            self.bias_2 -= learning_rate * grad_bias_2
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Creazione e addestramento della rete neurale
input_size = len(word_to_index)
hidden_size = 10
model = SimpleNN(input_size, hidden_size)
model.train(X_train, labels.reshape(-1, 1), epochs=1000, learning_rate=0.01)

# Test su nuove frasi
test_sentences = [
    'Mi sento fantastico oggi',
    'Questa giornata è pessima'
]

# Conversione delle frasi di test in vettori
X_test = np.array([sentence_to_vector(sentence, word_to_index) for sentence in test_sentences])

# Predizione
predictions = model.forward(X_test)

# Output delle predizioni
for sentence, pred in zip(test_sentences, predictions):
    print(sentence)
    if pred >= 0.5:
        print("Sentimento positivo")
    else:
        print("Sentimento negativo")

```


## Una rete Neurale - estrazione numeri

```bash

import numpy as np

# Definizione della funzione di attivazione ReLU
def relu(x):
    return np.maximum(0, x)

# Definizione della funzione di attivazione Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

# Caricamento del dataset MNIST
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']

# Normalizzazione dei dati
X = X / 255.0

# Conversione delle etichette in numeri interi
y = y.astype(int)

# Divisone dei dati in set di addestramento e test
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Inizializzazione dei pesi e dei bias
np.random.seed(0)
weights = np.random.randn(784, 10) * 0.01
biases = np.zeros((1, 10))

# Definizione del numero di epoche e del tasso di apprendimento
epochs = 10
learning_rate = 0.01

# Addestramento del modello
for epoch in range(epochs):
    # Forward pass
    z = np.dot(X_train, weights) + biases
    a = softmax(z)
    
    # Calcolo della loss
    loss = -np.mean(np.log(a[np.arange(len(a)), y_train]))
    
    # Calcolo del gradiente della loss rispetto a z
    dz = a
    dz[np.arange(len(a)), y_train] -= 1
    dz /= len(X_train)
    
    # Backpropagation
    dw = np.dot(X_train.T, dz)
    db = np.sum(dz, axis=0, keepdims=True)
    
    # Aggiornamento dei pesi e dei bias
    weights -= learning_rate * dw
    biases -= learning_rate * db
    
    # Stampa della loss
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

# Test del modello
z_test = np.dot(X_test, weights) + biases
a_test = softmax(z_test)
predictions = np.argmax(a_test, axis=1)

# Calcolo dell'accuratezza
accuracy = np.mean(predictions == y_test)
print(f'\nTest accuracy: {accuracy}')

```



## Una rete Neurale - predizione della memoria

```bash

import numpy as np

# Funzione di attivazione ReLU
def relu(x):
    return np.maximum(0, x)

# Funzione di attivazione Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

# Funzione di perdita Cross-entropy
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# Funzione per la trasformazione one-hot delle etichette
def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

# Definizione della rete neurale
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inizializzazione dei pesi e dei bias
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))
    
    def forward(self, X):
        # Forward pass
        self.hidden_output = relu(np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        self.output = softmax(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        return self.output
    
    def train(self, X, y, learning_rate=0.01, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calcolo della loss
            loss = cross_entropy_loss(y, output)
            
            # Calcolo del gradiente
            m = y.shape[0]
            grad_output = (output - y) / m
            grad_hidden_output = np.dot(grad_output, self.weights_hidden_output.T)
            grad_hidden = grad_hidden_output * (self.hidden_output > 0)
            
            grad_weights_hidden_output = np.dot(self.hidden_output.T, grad_output)
            grad_bias_output = np.sum(grad_output, axis=0, keepdims=True)
            grad_weights_input_hidden = np.dot(X.T, grad_hidden)
            grad_bias_hidden = np.sum(grad_hidden, axis=0, keepdims=True)
            
            # Aggiornamento dei pesi e dei bias
            self.weights_hidden_output -= learning_rate * grad_weights_hidden_output
            self.bias_output -= learning_rate * grad_bias_output
            self.weights_input_hidden -= learning_rate * grad_weights_input_hidden
            self.bias_hidden -= learning_rate * grad_bias_hidden
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Dati di input e output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
y_one_hot = one_hot_encode(y, 2)  # Codifica one-hot delle etichette

# Creazione e addestramento della rete neurale
input_size = 2
hidden_size = 3
output_size = 2
model = NeuralNetwork(input_size, hidden_size, output_size)
model.train(X, y_one_hot)

# Test della rete neurale
predictions = model.forward(X)
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted Labels:", predicted_labels)

```



## Una rete Neurale
## per capire se una persona passerà o non passerà un esame

```bash

    Definizione del problema: Il nostro obiettivo è predire se una persona passerà o meno l'esame. Questo è un problema di classificazione binaria, in cui l'output desiderato è 0 se la persona non supera l'esame e 1 se la persona supera l'esame.

    Rappresentazione dei dati: I nostri dati consistono nel numero di ore di studio e di sonno della persona prima dell'esame, insieme all'etichetta che indica se la persona ha superato o meno l'esame.

    Struttura della rete neurale: Per questo problema, useremo una rete neurale molto semplice con un solo neurone di input (uno per le ore di studio e uno per le ore di sonno) e un singolo neurone di output. Non useremo alcuno strato nascosto per mantenere il modello il più semplice possibile.

    Addestramento del modello: Utilizzeremo un algoritmo di apprendimento per regolare i pesi e i bias della rete neurale in modo che possa fare predizioni accurate. L'algoritmo che useremo è la discesa del gradiente.

    Predizione: Una volta addestrata, la nostra rete neurale sarà in grado di prendere in input il numero di ore di studio e di sonno di una persona e produrre un'output che rappresenta la probabilità che quella persona superi l'esame.

-----------------------------------------------------------------------------------------------


import numpy as np

# Definizione della funzione di attivazione Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definizione della derivata della funzione di attivazione Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Dati di addestramento (ore di studio, ore di sonno, esito dell'esame)
training_data = np.array([
    [2, 9, 0],  # Non passa l'esame
    [4, 8, 0],  # Non passa l'esame
    [5, 6, 0],  # Non passa l'esame
    [6, 5, 1],  # Passa l'esame
    [7, 4, 1],  # Passa l'esame
    [8, 3, 1]   # Passa l'esame
])

# Normalizzazione dei dati
X = training_data[:, :-1] / 10
y = training_data[:, -1]

# Inizializzazione dei pesi e dei bias
np.random.seed(1)
weights = np.random.randn(2)
bias = np.random.randn()

# Addestramento della rete neurale
learning_rate = 0.1
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    output = sigmoid(np.dot(X, weights) + bias)
    
    # Calcolo della loss
    loss = np.mean((output - y) ** 2)
    
    # Calcolo del gradiente
    dloss_doutput = 2 * (output - y)
    doutput_dz = sigmoid_derivative(output)
    dz_dw = X.T
    dloss_dz = dloss_doutput * doutput_dz
    grad_weights = np.dot(dz_dw, dloss_dz)
    grad_bias = np.sum(dloss_dz)
    
    # Aggiornamento dei pesi e dei bias
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Predizione
hours_studied = 6
hours_slept = 5
input_data = np.array([hours_studied / 10, hours_slept / 10])
prediction = sigmoid(np.dot(input_data, weights) + bias)

print(f'Probabilità di superare l\'esame: {prediction}')


----------------------------------------------------------------------


    Definiamo la funzione di attivazione sigmoidale, che sarà utilizzata per la nostra neurone di output. Questa funzione mappa qualsiasi valore reale nell'intervallo tra 0 e 1, che possiamo interpretare come la probabilità che la persona superi l'esame.
    Normalizziamo i dati di addestramento (le ore di studio e di sonno) per assicurarci che si trovino nell'intervallo tra 0 e 1.
    Inizializziamo i pesi e il bias in modo casuale. Questi sono i parametri che verranno adattati durante l'addestramento per fare previsioni accurate.
    Iteriamo attraverso il numero di epoche specificato. Ad ogni epoca, calcoliamo l'output previsto dalla rete neurale, calcoliamo la loss (il nostro obiettivo è minimizzare questa loss durante l'addestramento) e aggiorniamo i pesi e il bias in base al gradiente della loss rispetto a questi parametri.
    Alla fine, facciamo una previsione utilizzando le ore di studio e di sonno specificate e stampiamo la probabilità che la persona passi l'esame.


```



## Una rete Neurale
## creeremo una rete neurale con un solo neurone di input e un solo neurone di output. La nostra rete ## neurale dovrà imparare a mappare un input x a un output y, dove y è il doppio di x.


```bash

import numpy as np

# Funzione di attivazione Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dati di addestramento (input e output)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]])

# Inizializzazione dei pesi e dei bias
np.random.seed(0)
weights = np.random.randn(1)
bias = np.random.randn()

# Addestramento della rete neurale
learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    output = sigmoid(np.dot(X, weights) + bias)
    
    # Calcolo della loss
    loss = np.mean((output - y) ** 2)
    
    # Calcolo del gradiente
    dloss_doutput = 2 * (output - y)
    doutput_dz = sigmoid(output) * (1 - sigmoid(output))
    dz_dw = X.T
    dloss_dz = dloss_doutput * doutput_dz
    grad_weights = np.dot(dz_dw, dloss_dz)
    grad_bias = np.sum(dloss_dz)
    
    # Aggiornamento dei pesi e dei bias
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Test della rete neurale
x_test = 11
predicted_output = sigmoid(np.dot(x_test, weights) + bias)
print(f'Input: {x_test}, Output predetto: {predicted_output}')

```












