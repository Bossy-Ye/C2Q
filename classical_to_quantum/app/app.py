import tempfile

from flask import Flask, render_template, request, jsonify
from qiskit.visualization import circuit_drawer
import qiskit.qasm3 as qasm3
from io import BytesIO
import base64

from applications.graph.ising_applications.max_cut import MaxCut
from applications.graph.ising_applications.cliquep import CliqueP
from applications.graph.ising_applications.partition import Partition
from applications.graph.ising_applications.tspp import TspP
from applications.graph.ising_applications.vertex_coverp import VertexCoverp
from classical_to_quantum.qasm_generate import QASMGenerator
import os
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt


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
    problem_type = data['problem_type']

    generator = QASMGenerator()
    qasm_code = generator.qasm_generate(classical_code, verbose=True)

    # Convert QASM 3.0 code to QuantumCircuit
    circuit = qasm3.loads(qasm_code)

    # Transpile the circuit for visualization
    transpiled_circuit = circuit.decompose()

    # Generate circuit diagram as an image
    img = circuit_drawer(transpiled_circuit, output='mpl')

    # Convert image to base64 string
    buf = BytesIO()
    img.figure.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return jsonify({
        'qasm_code': qasm_code.strip(),
        'circuit_image': f"data:image/png;base64,{img_str}"
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
    problem_class_map = {
        'maxcut': MaxCut,
        'clique': CliqueP,
        'partition': Partition,
        'tsp': TspP,
        'vertex_cover': VertexCoverp
    }
    problem_instance = problem_class_map[graph_problem_type](file_path)

    #max_cut = MaxCut(file_path)
    problem_instance.run(verbose=True)
    qasm_code = problem_instance.generate_qasm()

    # Generate a QuantumCircuit from QASM code
    circuit = qasm3.loads(qasm_code)

    # Generate circuit diagram as an image
    img = circuit_drawer(circuit.decompose(), output='mpl')

    # Convert image to base64 string
    buf = BytesIO()
    img.figure.savefig(buf, format='png')
    buf.seek(0)
    circuit_img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    problem_instance.plot_res(transmission=True)
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
