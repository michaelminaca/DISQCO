"""Partition a circuit across three QPUs, then schedule it with the deterministic scheduler.

Every EPR pair takes exactly DURATIONS["EPR"] time units, so one run gives one schedule.
The demo prints the makespan (the time the last instruction finishes) after each stage,
with the percentage it took off the stage before, and the estimated fidelity F of each
scheduled stage under ERROR_MODEL:

    as extracted               the circuit the extractor produced
    classical bits renamed     each classical bit written only once
    scheduled                  a session may use any communication qubit of its QPU;
                               the session with the longest path to the end claims first
    scheduled + commuting      the same, with commuting gates free to reorder
    buffer                     on QPUs with spare data qubits, a waiting half-pair is
                               parked on a spare qubit whenever another session wants
                               its communication qubit
    buffer, swap if it pays    the same, but a half-pair is parked only when the wait it
                               faces would cost more fidelity than the swap; the online
                               loop run with p = 1, so its timing is deterministic too

Then the error budget of three of those schedules, and what schedule() keeps when asked
for the shortest schedule and when asked for the highest-fidelity one.

Saved next to this file:

    circuit.png                the logical circuit we started from
    partition_result.png       the hypergraph, coloured by which QPU each qubit went to
    <stage>_circuit.png        the circuit after each stage
    <stage>_schedule.png       when each instruction runs, one lane per qubit
    circuits.qpy               every stage's circuit, to load back with qpy.load
"""

import math
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
from disqco.scheduling.buffer import spare_slots_by_qpu
from disqco.scheduling.draw_schedule import plot_schedule
from disqco.scheduling.errors import ErrorModel, log_fidelity
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
from disqco.scheduling.stochastic_scheduler import run as run_online

demo_dir = Path(__file__).parent

DURATIONS = {"u": 1, "cp": 2, "cx": 2, "swap": 6, "measure": 5, "EPR": 100}
ERROR_MODEL = ErrorModel(T2_data=20000, T2_comm=2000, F_epr=0.98, F_meas=0.99, p2=0.003)

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


def finish_pass(state, renamed, sessions, session_of):
    """What every pass returns from its state: circuit, runtime, schedule entries, fidelity, log terms."""
    scheduled_circuit = schedule_to_circuit(renamed, state, session_of, sessions)
    runtime, entries = evaluate_quantum_runtime(scheduled_circuit, DURATIONS)
    total, logs = log_fidelity(state, renamed, sessions, session_of, ERROR_MODEL)
    return scheduled_circuit, runtime, entries, math.exp(total), logs


def scheduled_pass(partitioned_circuit, network, commuting, buffer=False):
    """One deterministic scheduling pass, longest path first, on the strict or the commuting graph.

    Returns (circuit, runtime, schedule entries, fidelity, log terms by error source),
    or five Nones if the pass stalled.
    """
    renamed, sessions, session_of, meta, preds, succs, pool, priority = prepare(
        partitioned_circuit, DURATIONS, network, commuting=commuting)

    order = claim_order_by_priority(sessions, meta, priority)
    spare = spare_slots_by_qpu(renamed, network) if buffer else None
    state = run_without_stalling(renamed, sessions, session_of, meta, preds, succs,
                                 priority, DURATIONS, pool, host_chooser(order, meta), spare)
    if state is None:
        return None, None, None, None, None
    return finish_pass(state, renamed, sessions, session_of)


def gated_buffer_pass(partitioned_circuit, network):
    """The online loop with p = 1 and no cutoff: deterministic timing, but the buffer swaps a
    half-pair onto a slot only when the wait it faces would cost more fidelity than the swap."""
    renamed, sessions, session_of, meta, preds, succs, pool, priority = prepare(
        partitioned_circuit, DURATIONS, network, commuting=True)

    order = claim_order_by_priority(sessions, meta, priority)
    spare = spare_slots_by_qpu(renamed, network)
    state = run_online(renamed, sessions, session_of, meta, preds, succs, priority, DURATIONS, pool,
                       host_chooser(order, meta), spare, mean=DURATIONS["EPR"], p=1.0, cutoff=None,
                       error_model=ERROR_MODEL)
    return finish_pass(state, renamed, sessions, session_of)


def percent_off(runtime, against):
    return 100 * (against - runtime) / against


def print_stage(label, runtime, against, fidelity=None):
    line = "{:32}{:5}   {:.1f}%".format(label, runtime, percent_off(runtime, against))
    if fidelity is not None:
        line += "   F {:.3f}".format(fidelity)
    print(line)


def print_error_budget(label, logs):
    """-log F by source: the fidelity lost to each kind of error."""
    print("error budget of {}:".format(label))
    for source, value in logs.items():
        print("    {:18}{:7.3f}".format(source, -value))
    print("    {:18}{:7.3f}   F {:.3f}".format("total", -sum(logs.values()), math.exp(sum(logs.values()))))


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

scheduled_circuit, scheduled_runtime, scheduled_schedule, scheduled_fidelity, _ = scheduled_pass(
    before_circuit, network, commuting=False)
if scheduled_circuit is not None:
    save(scheduled_circuit, "scheduled", scheduled_runtime, scheduled_schedule)

commuting_circuit, commuting_runtime, commuting_schedule, commuting_fidelity, commuting_logs = scheduled_pass(
    before_circuit, network, commuting=True)
if commuting_circuit is not None:
    save(commuting_circuit, "scheduled_commuting", commuting_runtime, commuting_schedule)

roomy_network = build_network(QPU_SIZE + SPARE)
roomy_circuit = extract(circuit, roomy_network, assignment)

_, roomy_runtime, _, roomy_fidelity, _ = scheduled_pass(roomy_circuit, roomy_network, commuting=True)
buffered_circuit, buffered_runtime, buffered_schedule, buffered_fidelity, buffered_logs = scheduled_pass(
    roomy_circuit, roomy_network, commuting=True, buffer=True)
save(buffered_circuit, "buffered", buffered_runtime, buffered_schedule)

gated_circuit, gated_runtime, gated_schedule, gated_fidelity, gated_logs = gated_buffer_pass(
    roomy_circuit, roomy_network)
save(gated_circuit, "buffered_if_it_pays", gated_runtime, gated_schedule)

by_makespan_runtime, _, _ = schedule(roomy_circuit, DURATIONS, network=roomy_network, commuting=True, buffer=True)
by_fidelity_runtime, _, _ = schedule(roomy_circuit, DURATIONS, network=roomy_network, commuting=True, buffer=True,
                                     model=ERROR_MODEL)

stages = [before_circuit, renamed_circuit, scheduled_circuit, commuting_circuit, buffered_circuit, gated_circuit]
with open(demo_dir / "circuits.qpy", "wb") as handle:
    qpy.dump([stage for stage in stages if stage is not None], handle)

print("---------------------------------------------")
print("{:32}{:5}".format("as extracted:", before_runtime))
print_stage("classical bits renamed:", renamed_runtime, before_runtime)
if scheduled_runtime is None:
    print("scheduled:                      stalled on this circuit")
else:
    print_stage("scheduled:", scheduled_runtime, renamed_runtime, scheduled_fidelity)
if commuting_runtime is None:
    print("scheduled + commuting:          stalled on this circuit")
else:
    print_stage("scheduled + commuting:", commuting_runtime, scheduled_runtime or renamed_runtime, commuting_fidelity)
    print("total improvement: {:.1f}%".format(percent_off(commuting_runtime, before_runtime)))

print("--- with {} spare data qubits per QPU".format(SPARE))
print("{:32}{:5}           F {:.3f}".format("no buffer:", roomy_runtime, roomy_fidelity))
print_stage("buffer:", buffered_runtime, roomy_runtime, buffered_fidelity)
print("    half-pairs moved onto slots:", buffered_circuit.count_ops().get("swap", 0),
      "of", buffered_circuit.count_ops().get("EPR", 0) * 2, "sessions")
print_stage("buffer, swap only if it pays:", gated_runtime, roomy_runtime, gated_fidelity)
print("    half-pairs moved onto slots:", gated_circuit.count_ops().get("swap", 0),
      "of", gated_circuit.count_ops().get("EPR", 0) * 2, "sessions")

print("---------------------------------------------")
print_error_budget("scheduled + commuting", commuting_logs)
print_error_budget("buffer", buffered_logs)
print_error_budget("buffer, swap only if it pays", gated_logs)

print("---------------------------------------------")
print("schedule() keeps the shortest:         makespan", by_makespan_runtime)
print("schedule() keeps the highest fidelity: makespan", by_fidelity_runtime)

moves = buffered_circuit.count_ops().get("swap", 0)
assert buffered_circuit.count_ops().get("reset", 0) == roomy_circuit.count_ops().get("reset", 0) + moves
assert len(buffered_circuit.data) == len(roomy_circuit.data) + 2 * moves
