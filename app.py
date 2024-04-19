from flask import Flask, render_template, request, send_file, jsonify
import pickle
import io
import base64
import math
from joblib import load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np


app = Flask(__name__)

# Load the database
# dataset_filename = "dataset_MELT_ONLY_2024-04-01 10_47_42.502500.pkl"
dataset_filename = "dataset_MELT_ONLY_2024-04-19 19:05:39.408182.pkl"
with open(dataset_filename, 'rb') as f:
    database = pickle.load(f)
    database = database.to_dict(orient='index') #TODO: change subsequent code to just expect dataframe format, not dict
# Load trained model
model = load('model_saved_1.joblib')


@app.route('/predict_melting_point/<compound_name>')
def predict_melting_point(compound_name):
    compound = database.get(compound_name)
    if not compound:
        return jsonify({'error': 'Compound not found'}), 404

    spectral_data = compound['spectral']
    spectral_data = spectral_data[:500,1] # TODO: This is just hack to get expected shape. Should actually interpolate
    prediction = model.predict([spectral_data])
    return jsonify({'melting_point': prediction[0]})


def plot_spectral_data(spectral_data):
    # Extracting Raman shift and intensity values
    raman_shifts = []
    intensities = []
    for point in spectral_data:
        shift, intensity = point
        raman_shifts.append(shift)
        intensities.append(intensity)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(raman_shifts, intensities, '-o', markersize=4, label='Abelsonite')
    plt.title('Raman Spectrum of Abelsonite')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    img = io.BytesIO() # Create an in-memory bytes buffer
    plt.savefig(img, format='png')  # Save plot to the buffer
    plt.close()
    img.seek(0) # Go to the beginning of the IO stream
    return img


@app.route('/get_figure/<compound_name>')
def get_figure(compound_name):
    # Fetch neccesary data from the database
    if compound_name in database:
        props = database[compound_name]
        spectral_data = props.get('spectral')
        if spectral_data is None:
            return "Spectral data not found", 404
        img_data = plot_spectral_data(spectral_data)
        return send_file(img_data, mimetype='image/png')
    return "Compound not found", 404


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query'].strip().lower()
    min_mp = request.form.get('min_mp', -math.inf)  # Returns None if not specified
    max_mp = request.form.get('max_mp', math.inf)  # Returns None if not specified
    has_mp = 'has_mp' in request.form  # Check if checkbox was ticked

    # Convert min and max melting points to integers, handling empty inputs
    try:
        min_mp = int(min_mp) if min_mp is not None and min_mp != '' else None
        max_mp = int(max_mp) if max_mp is not None and max_mp != '' else None
    except ValueError:
        # Handle case where the input cannot be converted to an integer
        min_mp = None#-math.inf
        max_mp = None#math.inf

    # Filter results based on the query and melting point range
    results = {}
    for name, props in database.items():
        melting_point = props.get("melting_point")
        # If no melting point recorded but user filters by melting point, skip this result
        if (melting_point is None) and (min_mp or max_mp):
            continue
        if (query in name.lower() or query in str(props['pubchem_cid'])) and \
           (not has_mp or melting_point) and \
           (min_mp is None or props['melting_point'] >= min_mp) and \
           (max_mp is None or props['melting_point'] <= max_mp):
            results[name] = props
            # results[name]['plot_img'] = plot_spectral_data(props['spectral']) # This is dealt with elsewhere now

    return render_template('results.html', query=query, results=results)



if __name__ == '__main__':
    app.run(debug=True)
