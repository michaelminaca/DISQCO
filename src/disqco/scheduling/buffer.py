from qiskit import QuantumCircuit
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.scheduling.errors import swap_error_cost
from disqco.scheduling.helper import DATA_REG
from enum import IntEnum

class BufferEvent(IntEnum):
    SWAP_DONE = -1
    RESET_DONE = -2

def spare_slots_by_qpu(circuit: QuantumCircuit, network: QuantumNetwork | None=None):
    used = {qubit for inst in circuit.data for qubit in inst.qubits}
    spare = {}
    for register in circuit.qregs:
        match = DATA_REG.match(register.name)
        if match is None:
            continue
        qpu = int(match.group(1))
        slots = [qubit for qubit in register if qubit not in used]
        if network is not None and network.comm_constrained(qpu):
            topology = network.qpu_topologies[qpu]
            comm_nodes = range(network.qpu_sizes[qpu], network.qpu_sizes[qpu] + network.comm_sizes[qpu])
            slots = [qubit for qubit in slots
                     if all(topology.has_edge(circuit.find_bit(qubit).registers[0][1], node) for node in comm_nodes)]
        spare[qpu] = slots
    return spare

def epr_waiting_to_generate(state, qpu, meta, epr_of):
    for session_id, session_meta in enumerate(meta):
        if session_meta["qpu"] == qpu and session_id not in state.host and epr_of[session_id] in state.ready:
            return True
    return False

def maybe_move_tail(state, session_id, meta, spare, epr_of, swap_time, reset_time):
    qpu = meta[session_id]["qpu"]
    if not epr_waiting_to_generate(state, qpu, meta, epr_of):
        return
    free_slots = [slot for slot in spare.get(qpu, []) if slot not in state.held and state.is_free(slot)]
    if not free_slots:
        return
    slot = max(free_slots, key=lambda s: state.free_at.get(s, 0))
    comm = state.host[session_id]

    state.moved[session_id] = comm
    state.moved_at[session_id] = state.clock
    state.host[session_id] = slot
    state.held[slot] = session_id
    del state.held[comm]
    state.occupy(comm, state.clock + swap_time + reset_time)
    state.occupy(slot, state.clock + swap_time)
    state.push_running(BufferEvent.SWAP_DONE, state.clock + swap_time)
    state.push_running(BufferEvent.RESET_DONE, state.clock + swap_time + reset_time)

def move_tail_if_it_pays(state, session_id, meta, spare, epr_of, swap_time, reset_time,
                         consumer_of_session, eprs_before_consumer, preds):
    """move the tail of a session to a buffer slot if its fidelity would be better than leaving it on the comm qubit """
    qpu = meta[session_id]["qpu"]
    if not epr_waiting_to_generate(state, qpu, meta, epr_of):
        return
    if not holding_costs_more_than_a_swap(state, session_id, consumer_of_session, eprs_before_consumer, preds):
        return
    free_slots = [slot for slot in spare.get(qpu, []) if slot not in state.held and state.is_free(slot)]
    if not free_slots:
        return
    slot = max(free_slots, key=lambda s: state.free_at.get(s, 0))
    comm = state.host[session_id]

    state.moved[session_id] = comm
    state.moved_at[session_id] = state.clock
    state.host[session_id] = slot
    state.held[slot] = session_id
    del state.held[comm]
    state.occupy(comm, state.clock + swap_time + reset_time)
    state.occupy(slot, state.clock + swap_time)
    state.push_running(BufferEvent.SWAP_DONE, state.clock + swap_time)
    state.push_running(BufferEvent.RESET_DONE, state.clock + swap_time + reset_time)

def expected_wait_for_consumer(state, consumer, eprs_before_consumer, preds):
    """The expected wait for the consumer of epr pair to be executed (from the predecessors of the consumer)"""
    pairs_still_missing = [epr for epr in eprs_before_consumer[consumer] if epr not in state.finished]
    if pairs_still_missing:
        return len(pairs_still_missing) * state.mean
    still_running = [state.end[pred] - state.clock for pred in preds[consumer]
                     if state.start[pred] is not None and state.end[pred] > state.clock]
    return max(still_running, default=0)

def expected_wait_until_pair_is_used(state, epr, sessions_opening_from_instruction, consumer_of_session, eprs_before_consumer, preds):
    """estimated time for how long until the pair is used, the nearer of its two halves' consumers."""
    return min(expected_wait_for_consumer(state, consumer_of_session[session_id], eprs_before_consumer, preds)
               for session_id in sessions_opening_from_instruction[epr])

def holding_costs_more_than_a_swap(state, session_id, consumer_of_session, eprs_before_consumer, preds):
    """true if the expected wait for the consumer costs more than the fidelity of a swap to a buffer slot, false if a swap to buffer slot would be better"""
    extra_decay_per_unit_on_comm = (1 / state.error_model.T2_comm - 1 / state.error_model.T2_data) / 2
    expected_wait = expected_wait_for_consumer(state, consumer_of_session[session_id], eprs_before_consumer, preds)
    return expected_wait * extra_decay_per_unit_on_comm > swap_error_cost(state.error_model)
