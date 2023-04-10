from flask import Flask, render_template, request
import kmeans_model
import matrix_factorization
import dnn

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve form data
        selected_option = request.form.get('option')
        entered_number = request.form.get('number')
        tdf = generate_table_data(selected_option, entered_number)
        table_data = [(row['movie_id'], row['movie_title'], row['movie_rating']) for index, row in tdf.iterrows()]
        option_text = {'A': 'K-Means', 'B': 'Matrix Factorization', 'C': 'Deep Neural Network'}
        selected_option_text = option_text.get(selected_option, 'Unknown Option')

        return render_template('index.html',selected_option_text=selected_option_text, number=entered_number, table_data=table_data)
    else:
        return render_template('index.html')

def generate_table_data(selected_option, entered_number):
    # Function to generate table data based on form inputs
    if selected_option == 'A':
        data = get_kmeans_prediction(entered_number)
    elif selected_option == 'B':
        data = get_svd_prediction(entered_number)
    else:
        data = [(f'Option {selected_option} - Row {i}', entered_number+i) for i in range(1, 6)]
    return data


def get_svd_prediction(pid):
    data_k = matrix_factorization.get_movie_recommendations(int(pid))
    return data_k

def get_kmeans_prediction(pid):
    data_k = kmeans_model.get_movie_recommendations(int(pid))
    return data_k

def get_dnn_prediction(pid):
    data_k = dnn.get_movie_recommendations(int(pid))
    return data_k

if __name__ == "__main__":
    app.run(debug=True)
