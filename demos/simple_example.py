"""Partition a circuit across three QPUs, then schedule it with the deterministic scheduler.

Every EPR pair takes exactly DURATIONS["EPR"] time units, so one run gives one schedule.
The demo prints the makespan (the time the last instruction finishes) after each stage,
with the percentage it took off the stage before:

    as extracted               the circuit the extractor produced
    classical bits renamed     each classical bit written only once
    scheduled                  a session may use any communication qubit of its QPU;
                               the session with the longest path to the end claims first
    scheduled + commuting      the same, with commuting gates free to reorder
    buffer                     on QPUs with spare data qubits, a waiting half-pair is
                               parked on a spare qubit so its communication qubit is free

Saved next to this file:

    circuit.png                the logical circuit we started from
    partition_result.png       the hypergraph, coloured by which QPU each qubit went to
    <stage>_circuit.png        the circuit after each stage
    <stage>_schedule.png       when each instruction runs, one lane per qubit
    circuits.qpy               every stage's circuit, to load back with qpy.load
"""

from pathlib import Path

import networkx as nx
from qiskit import qpy, transpile

from disqco import (
    PartitionedCircuitExtractor,
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco.circuits.cp_fraction import cp_fraction
from disqco.scheduling.draw_schedule import plot_schedule
from disqco.scheduling.evaluator import evaluate_quantum_runtime
from disqco.scheduling.schedule_to_circuit import schedule_to_circuit
from disqco.scheduling.scheduler import (
    claim_order_by_priority,
    host_chooser,
    prepare,
    rename_classical_bits,
    run_without_stalling,
    schedule,
)

demo_dir = Path(__file__).parent

DURATIONS = {"u": 1, "cp": 2, "cx": 2, "swap": 6, "measure": 5, "EPR": 100}

NUM_QPUS = 3
QPU_SIZE = 4
NUM_COMM = 4
SPARE = 4


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


def save(circuit, name, runtime, schedule_entries):
    """Store one stage: a drawing of the gates and a drawing of the timing."""
    circuit.draw(output="mpl", style="bw", fold=50, filename=str(demo_dir / f"{name}_circuit.png"))
    plot_schedule(circuit, schedule_entries, runtime, save_path=demo_dir / f"{name}_schedule.png")


def scheduled_pass(partitioned_circuit, network, commuting):
    """One scheduling pass, longest path first, on the strict or the commuting graph.

    Returns (circuit, runtime, schedule entries), or (None, None, None) if the pass stalled.
    """
    renamed, sessions, session_of, meta, preds, succs, pool, priority = prepare(
        partitioned_circuit, DURATIONS, network, commuting=commuting)

    order = claim_order_by_priority(sessions, meta, priority)
    state = run_without_stalling(renamed, sessions, session_of, meta, preds, succs,
                                 priority, DURATIONS, pool, host_chooser(order, meta))
    if state is None:
        return None, None, None

    scheduled_circuit = schedule_to_circuit(renamed, state, session_of, sessions)
    runtime, entries = evaluate_quantum_runtime(scheduled_circuit, DURATIONS)
    return scheduled_circuit, runtime, entries


def percent_off(runtime, against):
    return 100 * (against - runtime) / against


def print_stage(label, runtime, against):
    print("{:26}{:5}   {:.1f}%".format(label, runtime, percent_off(runtime, against)))


circuit = cp_fraction(num_qubits=10, depth=14, fraction=0.6, seed=7)
circuit = transpile(circuit, basis_gates=["u", "cp"])
circuit.draw(output="mpl", style="bw", fold=40, filename=str(demo_dir / "circuit.png"))

network = build_network(QPU_SIZE)
hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)

assignment = set_initial_partition_assignment(hypergraph, network)
hypergraph.draw(network=network, assignment=assignment, show_labels=False, output="mpl", dpi=150,
                save_path=str(demo_dir / "partition_result.png"))

before_circuit = extract(circuit, network, assignment)
before_runtime, before_schedule = evaluate_quantum_runtime(before_circuit, DURATIONS)
save(before_circuit, "before", before_runtime, before_schedule)

print("EPR pairs requested:", before_circuit.count_ops().get("EPR", 0))

renamed_circuit = rename_classical_bits(before_circuit)
renamed_runtime, renamed_schedule = evaluate_quantum_runtime(renamed_circuit, DURATIONS)
save(renamed_circuit, "renamed", renamed_runtime, renamed_schedule)

scheduled_circuit, scheduled_runtime, scheduled_schedule = scheduled_pass(
    before_circuit, network, commuting=False)
if scheduled_circuit is not None:
    save(scheduled_circuit, "scheduled", scheduled_runtime, scheduled_schedule)

commuting_circuit, commuting_runtime, commuting_schedule = scheduled_pass(
    before_circuit, network, commuting=True)
if commuting_circuit is not None:
    save(commuting_circuit, "scheduled_commuting", commuting_runtime, commuting_schedule)

roomy_network = build_network(QPU_SIZE + SPARE)
roomy_circuit = extract(circuit, roomy_network, assignment)

roomy_runtime, _, _ = schedule(roomy_circuit, DURATIONS, network=roomy_network, commuting=True)
buffered_runtime, buffered_schedule, buffered_circuit = schedule(
    roomy_circuit, DURATIONS, network=roomy_network, commuting=True, buffer=True)
save(buffered_circuit, "buffered", buffered_runtime, buffered_schedule)

stages = [before_circuit, renamed_circuit, scheduled_circuit, commuting_circuit, buffered_circuit]
with open(demo_dir / "circuits.qpy", "wb") as handle:
    qpy.dump([stage for stage in stages if stage is not None], handle)

print("---------------------------------------------")
print("{:26}{:5}".format("as extracted:", before_runtime))
print_stage("classical bits renamed:", renamed_runtime, before_runtime)
if scheduled_runtime is None:
    print("scheduled:                stalled on this circuit")
else:
    print_stage("scheduled:", scheduled_runtime, renamed_runtime)
if commuting_runtime is None:
    print("scheduled + commuting:    stalled on this circuit")
else:
    print_stage("scheduled + commuting:", commuting_runtime, scheduled_runtime or renamed_runtime)
    print("total improvement: {:.1f}%".format(percent_off(commuting_runtime, before_runtime)))

print("--- with {} spare data qubits per QPU".format(SPARE))
print("{:26}{:5}".format("no buffer:", roomy_runtime))
print_stage("buffer:", buffered_runtime, roomy_runtime)
moves = buffered_circuit.count_ops().get("swap", 0)
print("half-pairs moved onto slots:", moves, "of", buffered_circuit.count_ops().get("EPR", 0) * 2, "sessions")

assert buffered_circuit.count_ops().get("reset", 0) == roomy_circuit.count_ops().get("reset", 0) + moves
assert len(buffered_circuit.data) == len(roomy_circuit.data) + 2 * moves
