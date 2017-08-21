# EmoAtt: Inner attention sentence embedding for Emotion Intensity

This is a TensorFlow implementation of the EmoAtt system used for the WASSA 2017 Shared task on Emotion Intensity. Please check https://arxiv.org/abs/1708.05521 for more details.


1. Clone this repository
    * `cd ~; git clone https://github.com/epochx/emoatt`

2. Download the needed data and software
   * Downoad the EmoInt Dataset from http://saifmohammad.com/WebPages/EmotionIntensity-SharedTask.html
   * download and install TweeboParser from https://github.com/ikekonglp/TweeboParser
   * download GloVe Twitter pre-trained word embeddings 
        * `cd ~/emoatt; wget http://nlp.stanford.edu/data/glove.twitter.27B.zip; unzip glove.twitter.27B.zip`
   
   
3. Create environment
    * `sh create_data_folder.sh /path/for/data/`
    * modify ~/emoatt/enlp/settings.py accordingly


4. Pre-process datasets
    * ``sh preprocess_data /path/to/json/``

5. Run models
* ``python run.py --json_path /path/to/json/TwiboParser.FearTrainFearValidFearTest.GloveTwitter50.json --results_path /path/to/results  --use_binary_features True --loss pc --optimizer adam --dropout_keep_prob 0.9 --size 100 --regularization_lambda 0.05``
* ``python run.py --json_path /path/to/json/TwiboParser.AngerTrainAngerValidAngerTest.GloveTwitter50.json --results_path /path/to/results --use_binary_features True --loss pc --optimizer adam --dropout_keep_prob 0.5 --size 100 --regularization_lambda 0.01``
* ``python run.py --json_path /path/to/json/TwiboParser.SadnessTrainSadnessValidSadnessTest.GloveTwitter50.json --results_path /path/to/results --use_binary_features True --loss pc --optimizer adam --dropout_keep_prob 0.8 --size 50 --regularization_lambda 0.2``
* ``python run.py --json_path /path/to/json/TwiboParser.JoyTrainJoyValidJoyTest.GloveTwitter50.json --results_path /path/to/results --use_binary_features True --loss pc --optimizer adam --dropout_keep_prob 0.8 --size 100 --regularization_lambda 0.2``

