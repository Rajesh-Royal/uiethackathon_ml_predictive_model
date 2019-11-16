import flask
import pandas as pd
from joblib import dump, load


with open(f'../model/shares_prediction', 'rb') as f:
    model = load(f)


app = flask.Flask(__name__, template_folder='')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('index.html'))

    if flask.request.method == 'POST':
        n_non_stop_words = flask.request.form['n_non_stop_words']
        self_reference_avg_sharess = flask.request.form['self_reference_avg_sharess']
        abs_title_sentiment_polarity = flask.request.form['abs_title_sentiment_polarity']
        n_tokens_content = flask.request.form['n_tokens_content']
        rate_negative_words = flask.request.form['rate_negative_words']

        input_variables = pd.DataFrame([[n_non_stop_words, self_reference_avg_sharess, abs_title_sentiment_polarity, n_tokens_content, rate_negative_words]],
                                       columns=['n_non_stop_words', 'self_reference_avg_sharess', 'abs_title_sentiment_polarity', 'n_tokens_content', 'rate_negative_words'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('index.html', original_input={'n_non_stop_words': n_non_stop_words, 'self_reference_avg_sharess': self_reference_avg_sharess, 'abs_title_sentiment_polarity': abs_title_sentiment_polarity, 'n_tokens_content' :n_tokens_content,
        'rate_negative_words': rate_negative_words},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)
