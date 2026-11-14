from pathlib import Path

import networkx as nx
from qiskit import transpile

from disqco import (
    PartitionedCircuitExtractor,
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco.circuits.cp_fraction import cp_fraction
from disqco.scheduling.draw_schedule import plot_schedule
from disqco.scheduling.errors import ErrorModel
from disqco.scheduling.evaluator import evaluate_quantum_runtime
from disqco.scheduling.scheduler import rename_classical_bits, schedule
from disqco.scheduling.stochastic_scheduler import simulate_many

demo_dir = Path(__file__).parent

DURATIONS = {"u": 1, "cp": 2, "cx": 2, "swap": 6, "measure": 5, "EPR": 100, "distill": 7}
ERROR_MODEL = ErrorModel(T2_data=20000, T2_comm=2000, F_epr=0.98, F_meas=0.99, p2=0.003)
MAX_TIMEOUTS_PAIR_KEPT_BEYOND_CUTOFF = 3

NUM_QPUS = 3
QPU_SIZE = 4
NUM_COMM = 4
SPARE = 4

TRIALS = 20
RANDOM_PAIRS = dict(trials=TRIALS, mean=DURATIONS["EPR"], p=0.1, cutoff=100, error_model=ERROR_MODEL)
BASELINE = dict(commuting=False, buffer=False, lazy_first_instructions=False, max_timeouts_pair_kept_beyond_cutoff=0)
ALL_FEATURES = dict(commuting=True, buffer=True, lazy_first_instructions=True,
                    max_timeouts_pair_kept_beyond_cutoff=MAX_TIMEOUTS_PAIR_KEPT_BEYOND_CUTOFF)


def build_network(data_qubits_per_qpu):
    """Three QPUs, every QPU able to entangle with every other.

    Each QPU's topology names all its qubits (data + comm), which makes the
    network comm_constrained: the extractor refuses rather than invent
    communication qubits the machine does not have.
    """
    return QuantumNetwork(
        {qpu: data_qubits_per_qpu for qpu in range(NUM_QPUS)},
        qpu_topologies={qpu: nx.complete_graph(data_qubits_per_qpu + NUM_COMM) for qpu in range(NUM_QPUS)},
        comm_sizes=[NUM_COMM] * NUM_QPUS,
    )


def extract(logical_circuit, network, assignment):
    """The partitioned circuit: local gates, EPR pairs and the corrections of each remote gate."""
    hypergraph = QuantumCircuitHyperGraph(logical_circuit, group_gates=True)
    extractor = PartitionedCircuitExtractor(hypergraph, network, partition_assignment=assignment)
    return extractor.extract_partitioned_circuit()


def percent_off(runtime, against):
    return 100 * (against - runtime) / against


def print_stage(label, runtime, against):
    print("{:32}{:5}   {:.1f}%".format(label, runtime, percent_off(runtime, against)))


def print_summary(label, summary, against=None):
    shorter = "" if against is None else "{:.1f}%".format(percent_off(summary["mean"], against["mean"]))
    print("    {:28}{:5.0f}   {:6}   discards {:4.1f}   F {:.3f}".format(
        label, summary["mean"], shorter, summary["discards"], summary["fidelity"]))


circuit = cp_fraction(num_qubits=10, depth=14, fraction=0.6, seed=7)
circuit = transpile(circuit, basis_gates=["u", "cp"])

hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
assignment = set_initial_partition_assignment(hypergraph, build_network(QPU_SIZE))

network = build_network(QPU_SIZE + SPARE)
before_circuit = extract(circuit, network, assignment)

before_runtime, before_schedule = evaluate_quantum_runtime(before_circuit, DURATIONS)
renamed_runtime, _ = evaluate_quantum_runtime(rename_classical_bits(before_circuit), DURATIONS)
scheduled_runtime, scheduled_schedule, scheduled_circuit = schedule(
    before_circuit, DURATIONS, network=network, commuting=True, buffer=True)

plot_schedule(before_circuit, before_schedule, before_runtime, save_path=demo_dir / "before_schedule.png")
plot_schedule(scheduled_circuit, scheduled_schedule, scheduled_runtime, save_path=demo_dir / "scheduled_schedule.png")

print("EPR pairs requested:", before_circuit.count_ops().get("EPR", 0))
print("---------------------------------------------")
print("{:32}{:5}".format("as extracted:", before_runtime))
print_stage("classical bits renamed:", renamed_runtime, before_runtime)
print_stage("scheduled:", scheduled_runtime, renamed_runtime)
print("total improvement: {:.1f}%".format(percent_off(scheduled_runtime, before_runtime)))

print("---------------------------------------------")
print("random pair generation, mean of {} runs:".format(TRIALS))
for pairs_per_operation in (1, 2):
    baseline = simulate_many(before_circuit, DURATIONS, network=network, pairs_per_operation=pairs_per_operation,
                             **RANDOM_PAIRS, **BASELINE)
    all_features = simulate_many(before_circuit, DURATIONS, network=network, pairs_per_operation=pairs_per_operation,
                                 **RANDOM_PAIRS, **ALL_FEATURES)
    print("EPR pairs per operation: {}".format(pairs_per_operation))
    print_summary("baseline:", baseline)
    print_summary("all features:", all_features, baseline)
