# Detecting Sentiment Based on OpenAi Solution

https://openai.com/blog/unsupervised-sentiment-neuron/

This solution consists on using a multiplicative LSTM (mLSTM) to simply predict the next charactere on product reviews. After training the recurrent network, use it's hidden units to train a classifier.

In my case, I lack the resources and time available in OpenAi (as long as the knowledge), so I am going to do my best with a few days training on a GeForce GTX 1050 instead of "one month across four NVIDIA Pascal GPUs [...] processing 12,500 characters per second."

Instead of using amazon reviews, I use a portuguese language review dataset obtained from https://www.kaggle.com/olistbr/brazilian-ecommerce.

At the notebooks folder you can check some of the code used to prototype, train and test the models. At "src" I have placed all definitive code.

The .pdf file at the root folder consists on a shor paper written to explain what was done. That is an initial version and might be replaced later.


## How to use


1. Install Conda (python3.7 version)
2. Install Pytorch with Cuda (Cpu version still not woking)
3. Install Unidecode (Hope you have pip installed)
    ```console
    pip install unidecode
    ```
4. At the root folder, download the models executing
    ```console
    make download
    ```
5. To start detecting sentiment type
   ```console
    make detect_sentiment
    ```

That is still a simple user input program, but could be easily adapted to an API, for exemple. Just follow the instructions on your prompt and everything should be fine. Only remember that the model was trained on portuguese language text, and will probably not work for other languages.