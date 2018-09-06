#from keras.applications import InceptionV3,ResNet50
# from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils
from keras.models import load_model
import numpy as np
import pandas as pd
#from gensim.models import KeyedVectors
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
import tensorflow as tf
#from PIL import Image
#from yoloimage import yoloImageCrop
import flask
import io



import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords




# Initialize Flask application
app = flask.Flask(__name__)
global model
global category_model
global graph
graph = tf.get_default_graph()


MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300

EMBEDDING_FILE = "GoogleNews-vectors-negative300.bin"
category_index = {"clothing":0, "camera":1, "home-appliances":2}
category_reverse_index = dict((y,x) for (x,y) in category_index.items())
STOPWORDS = set(stopwords.words("english"))


clothing = pd.read_csv('product-titles-cnn-data/clothing.tsv', sep='\t')
cameras = pd.read_csv('product-titles-cnn-data/cameras.tsv', sep='\t')
home_appliances = pd.read_csv('product-titles-cnn-data/home.tsv', sep='\t')
#
datasets = [clothing, cameras, home_appliances]

print("Make sure there are no null values in the datasets")
for data in datasets:
    print("Has null values: ", data.isnull().values.any())

def preprocess(text):
    text= text.strip().lower().split()
    text = filter(lambda word: word not in STOPWORDS, text)
    return " ".join(text)

for dataset in datasets:
    dataset['title'] = dataset['title'].apply(preprocess)

all_texts = clothing['title'] + cameras['title'] + home_appliances['title']
all_texts = all_texts.drop_duplicates(keep=False)
#
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_texts)

clothing_sequences = tokenizer.texts_to_sequences(clothing['title'])
electronics_sequences = tokenizer.texts_to_sequences(cameras['title'])
home_appliances_sequences = tokenizer.texts_to_sequences(home_appliances['title'])

clothing_data = pad_sequences(clothing_sequences, maxlen=MAX_SEQUENCE_LENGTH)
electronics_data = pad_sequences(electronics_sequences, maxlen=MAX_SEQUENCE_LENGTH)
home_appliances_data = pad_sequences(home_appliances_sequences, maxlen=MAX_SEQUENCE_LENGTH)
#
print(clothing_data)

word_index = tokenizer.word_index
test_string = "sports action spy pen camera"
print("word\t\tid")
print("-" * 20)
for word in test_string.split():
    print("%s\t\t%s" % (word, word_index[word]))

test_sequence = tokenizer.texts_to_sequences(["sports action camera", "spy pen camera"])
padded_sequence = pad_sequences(test_sequence, maxlen=MAX_SEQUENCE_LENGTH)
print("Text to Vector", test_sequence)
print("Padded Vector", padded_sequence)

print("clothing: \t\t", to_categorical(category_index["clothing"], 3))
print("camera: \t\t", to_categorical(category_index["camera"], 3))
print("home appliances: \t", to_categorical(category_index["home-appliances"], 3))

print("clothing shape: ", clothing_data.shape)
print("electronics shape: ", electronics_data.shape)
print("home appliances shape: ", home_appliances_data.shape)

data = np.vstack((clothing_data, electronics_data, home_appliances_data))
category = pd.concat([clothing['category'], cameras['category'], home_appliances['category']]).values
category = to_categorical(category)
print("-"*10)
print("combined data shape: ", data.shape)
print("combined category/label shape: ", category.shape)



def init_model(to_load=None):
    """
    Initialize the model to use for predicting
    :param to_load: file name of the model to load
    :param is_17: whether it is the 17 class model or the 102 class model
    :return: a tuple containing the (model, classdict)
    """

    print('Loading model', to_load, '\n\n')
    model = load_model('model.h5')
    print(model.summary())

    dict = category_reverse_index

    print('Model loaded successfully.\n')
    return model, dict

# def load_model():
#     # Load Pretrained Model
#     global model
#     # model = InceptionV3(weights = "imagenet")
#     model = ResNet50(weights = "imagenet")
#     print(model.summary())

# preprocess Step

def preprocess(text):
    text= text.strip().lower().split()
    text = filter(lambda word: word not in STOPWORDS, text)
    return " ".join(text)


category_model, catoegory_dict = init_model()

@app.route('/')
def homepage():
    #load_model()
    return """Welcome To Text Classifier"""


@app.route("/predict", methods=["POST"])
def predict():
    print("Inside Predict Method")
    if flask.request.method == "POST":
        # category_model, catoegory_dict = init_model()
        # print(category_model.summary())
        example_product = "Nikon Coolpix A10 Point and Shoot Camera (Black)"
        example_product = preprocess(example_product)
        example_sequence = tokenizer.texts_to_sequences([example_product])
        example_padded_sequence = pad_sequences(example_sequence, maxlen=MAX_SEQUENCE_LENGTH)

        with graph.as_default():

            print("-"*10)
            print("Predicted category: ", category_reverse_index[category_model.predict_classes(example_padded_sequence, verbose=0)[0]])
            print("-"*10)
            probabilities = category_model.predict(example_padded_sequence, verbose=0)

        probabilities = probabilities[0]
        return_data = {}
        return_data['Clothing Probability'] = str(probabilities[category_index["clothing"]])
        return_data['Camera Probability'] = str(probabilities[category_index["camera"]])
        return_data['home appliance Probability'] = str(probabilities[category_index["home-appliances"]])
        # print("Clothing Probability: ",probabilities[category_index["clothing"]] )
        # print("Camera Probability: ",probabilities[category_index["camera"]] )
        # print("home appliances probability: ",probabilities[category_index["home-appliances"]] )


    return flask.jsonify(return_data)



# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    #init_model()
    app.run(debug=True,host='0.0.0.0',port=5000)
