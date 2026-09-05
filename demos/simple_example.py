from pathlib import Path

from qiskit import QuantumCircuit

from disqco import (
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco import PartitionedCircuitExtractor
import networkx as nx
from disqco.scheduling.evaluator import evaluate_quantum_runtime
from disqco.scheduling.utils import plot_schedule
from disqco.scheduling.greedy_scheduler import greedy_scheduler

demo_dir = Path(__file__).parent

# circuit = QuantumCircuit(5)
# for i in range(5):
#     circuit.h(i)
# circuit.cx(0, 4)
# circuit.cx(1, 4)
# circuit.x(1)
# circuit.x(3)
# circuit.cx(0, 3)
# circuit.x(2)
# circuit.cx(3, 4)
# circuit.x(2)

# Dense cp circuit: cp gates are diagonal, so gates sharing a qubit commute
# freely -- the commutation graph gains large unordered groups, giving the
# greedy scheduler real choices (which EPR-bound chain to start first).
from qiskit import transpile
from disqco.circuits.cp_fraction import cp_fraction

circuit = cp_fraction(num_qubits=10, depth=20, fraction=0.6, seed=7)
circuit = transpile(circuit, basis_gates=["u", "cp"])

circuit.draw(output="mpl", style="bw", fold=40,
             filename=str(demo_dir / "circuit.png"))
print(f"Saved original circuit to {demo_dir / 'circuit.png'}")

hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)

NUM_COMM = 6

def ring_with_ports(n_data, n_comm):
    topo = nx.path_graph(n_data)
    comm_nodes = range(n_data, n_data + n_comm)
    for c in comm_nodes:
        topo.add_edge(n_data - 1, c)
    for a in comm_nodes:
        for b in comm_nodes:
            if a < b:
                topo.add_edge(a, b)
    return topo

network = QuantumNetwork(
    {0: 6, 1: 6},
    qpu_topologies={
        0: ring_with_ports(6, NUM_COMM),
        1: ring_with_ports(6, NUM_COMM),
    },
    # every comm qubit of QPU 0 serves the link to QPU 1 and vice versa
    comm_links={0: {k: 1 for k in range(NUM_COMM)},
                1: {k: 0 for k in range(NUM_COMM)}},
    comm_sizes=[NUM_COMM, NUM_COMM],
)

assignment = set_initial_partition_assignment(hypergraph, network, round_robin=True)

hypergraph_path = demo_dir / "partition_result.png"
hypergraph.draw(
    network=network,
    assignment=assignment,
    show_labels=False,
    output="mpl",
    dpi=150,
    save_path=str(hypergraph_path),
)

extractor = PartitionedCircuitExtractor(
    hypergraph, network, partition_assignment=assignment
)
partitioned_circuit = extractor.extract_partitioned_circuit()

print("Number of e-bits requested:", partitioned_circuit.count_ops().get("EPR", 0))
print("Local routing SWAPs inserted:", extractor.local_swap_count)

partitioned_circuit_path = demo_dir / "partitioned_circuit.png"
partitioned_circuit.draw(
    output="mpl", style="bw", fold=50, filename=str(partitioned_circuit_path)
)

# Durations in units of one single-qubit gate. Two literature-grounded
# regimes (sweep EPR between them - the conclusion depends on it):
#  - Trapped-ion + optical network (demonstrated DQC, Main et al.,
#    arXiv:2407.00835): 1q ~10us, 2q ~5-10x 1q, measurement ~10x,
#    heralded EPR ~5-10ms => EPR ~ 500-1000x a 1q gate.
#  - Superconducting + microwave link (deterministic, e.g. Kurpiers et al.,
#    npj QI 2019): EPR only ~2-20x a 2q gate => EPR ~ 20.
EPR_DURATION = 500          # ion-optical headline; set ~20 for SC-microwave
DURATIONS = {
    'u': 1,                 # single-qubit gate (reference unit)
    'cp': 5, 'cx': 5, 'cz': 5,   # two-qubit gate ~5x 1q (ion-trap ratio)
    'swap': 15,             # 3 back-to-back cx
    'measure': 10,          # readout ~10x 1q
    'reset': 5,
    'x': 1, 'z': 1,         # classical-feedforward corrections
    'EPR': EPR_DURATION,
}

runtime, schedule = evaluate_quantum_runtime(partitioned_circuit, DURATIONS)
print("Total quantum runtime of circuit: ", runtime)

plot_schedule(partitioned_circuit, schedule, runtime, save_path=demo_dir / "circuit_schedule.png")

print("---------------------------------------------")

runtime, schedule = greedy_scheduler(partitioned_circuit, DURATIONS)
print("Total greedy quantum runtime of circuit: ", runtime)

plot_schedule(partitioned_circuit, schedule, runtime, save_path=demo_dir / "circuit_schedule_greedy.png")

