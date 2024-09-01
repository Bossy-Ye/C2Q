import tempfile

from flask import Flask, render_template, request, jsonify
from qiskit.visualization import circuit_drawer
from qiskit import qasm2
from applications.graph.Ising import Ising
from Framework.generator import QASMGenerator
import os
from werkzeug.utils import secure_filename
from classical_to_quantum.utils import *

app = Flask(__name__)

# Dummy parser function for example
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'gset', 'txt', 'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/manual_input')
def manual_input():
    return render_template('manual_input.html')


@app.route('/graph_problem')
def graph_problem():
    return render_template('graph_problem.html')


@app.route('/generate_circuit', methods=['POST'])
def generate_circuit():
    data = request.json
    classical_code = data['classical_code']

    generator = QASMGenerator()
    qasm_codes, img_ios = generator.qasm_generate(classical_code, verbose=False)
    qasm_code_results = {}
    circuit_images = {}
    for key, qasm_code in qasm_codes.items():
        # Convert QASM 2.0 code to QuantumCircuit
        circuit = qasm2.loads(qasm_code,
                              custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

        # Transpile the circuit for visualization
        transpiled_circuit = circuit

        # Generate circuit diagram as an image
        img = circuit_drawer(transpiled_circuit, output='mpl')

        img_str = img_gen_img_io(img)
        # Store the results in the dictionaries
        qasm_code_results[key] = qasm_code.strip()
        circuit_images[key] = f"data:image/png;base64,{img_str}"
    for key, img_io in img_ios.items():
        img_ios[key] = f"data:image/png;base64,{img_io}"
    return jsonify({
        'qasm_code': qasm_code_results,
        'circuit_image': circuit_images,
        'img_ios': img_ios
    })


@app.route('/graph_problem', methods=['POST'])
def process_graph():
    graph_problem_type = request.form['problem_type']
    if 'gset_file' in request.files:
        file = request.files['gset_file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

    elif 'gset_data' in request.form:
        manual_input = request.form['gset_data']
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gset') as temp_file:
            temp_file.write(manual_input.encode('utf-8'))
            file_path = temp_file.name
    if not file_path:
        return jsonify({'error': 'No valid input provided'}), 400
    problem_instance = Ising(input_data=file_path,
                             class_name=graph_problem_type)
    problem_instance.run(verbose=True)
    circuit, qasm_code = problem_instance.generate_qasm()
    # Generate a QuantumCircuit from QASM code
    # circuit = qiskit.qasm2.loads(qasm_code)
    # Generate circuit diagram as an image
    img = circuit_drawer(circuit.decompose(), output='mpl')

    # Convert image to base64 string
    buf = BytesIO()
    img.figure.savefig(buf, format='png')
    buf.seek(0)
    circuit_img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    problem_instance.plot_graph_solution()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return jsonify({
        'qasm_code': qasm_code.strip(),
        'circuit_image': f"data:image/png;base64,{circuit_img_str}",
        'graph_image': f"data:image/png;base64,{graph_img_str}"
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
