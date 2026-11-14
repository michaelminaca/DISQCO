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
                         consumer_of_session, preds, duration_of_instruction, eprs, expected_finish_of=None):
    """move the tail of a session to a buffer slot if its fidelity would be better than leaving it on the comm qubit """
    qpu = meta[session_id]["qpu"]
    if not spare.get(qpu) or not state.is_free(state.host[session_id]):
        return
    a_pair_waits_for_the_comm_qubit = epr_waiting_to_generate(state, qpu, meta, epr_of)
    if not (holding_costs_more_than_a_swap(state, session_id, consumer_of_session, preds, duration_of_instruction, eprs, expected_finish_of,
                                           a_pair_waits_for_the_comm_qubit)
            or (a_pair_waits_for_the_comm_qubit
                and blocks_a_pair_it_waits_for(state, session_id, meta, epr_of, consumer_of_session, preds, eprs)
                and standstill_costs_more_than_a_swap(state, qpu, meta))):
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

def expected_finish(state, idx, preds, duration_of_instruction, eprs, expected_finish_of):
    if idx in state.finished:
        return state.clock
    if state.start[idx] is not None:
        return state.clock + state.mean if idx in eprs else max(state.end[idx], state.clock)
    duration = state.mean if idx in eprs else duration_of_instruction[idx]
    return max([state.clock] + [expected_finish_of[pred] for pred in preds[idx]]) + duration

def instructions_waited_on(state, consumer, preds, expected_finish_of):
    needed = set()
    to_visit = list(preds[consumer])
    while to_visit:
        idx = to_visit.pop()
        if idx in needed or idx in expected_finish_of:
            continue
        needed.add(idx)
        if state.start[idx] is None:
            to_visit.extend(preds[idx])
    return sorted(needed)

def expected_wait_via_predecessors(state, consumer, preds, duration_of_instruction, eprs, expected_finish_of):
    for idx in instructions_waited_on(state, consumer, preds, expected_finish_of):
        expected_finish_of[idx] = expected_finish(state, idx, preds, duration_of_instruction, eprs, expected_finish_of)
    return max([0] + [expected_finish_of[pred] - state.clock for pred in preds[consumer]])

def holding_costs_more_than_a_swap(state, session_id, consumer_of_session, preds, duration_of_instruction, eprs, expected_finish_of=None,
                                   a_pair_waits_for_the_comm_qubit=True):
    """true if the expected wait for the consumer costs more than the fidelity of a swap to a buffer slot, false if a swap to buffer slot would be better"""
    expected_wait = expected_wait_via_predecessors(state, consumer_of_session[session_id], preds, duration_of_instruction, eprs,
                                                                 {} if expected_finish_of is None else expected_finish_of)
    decay_per_unit_on_comm = 1 / (2 * state.error_model.T2_comm)
    decay_per_unit_on_data = 1 / (2 * state.error_model.T2_data)
    half_decays_on_comm = expected_wait * decay_per_unit_on_comm
    a_data_qubit_idles_waiting_for_the_comm_qubit = expected_wait * decay_per_unit_on_data if a_pair_waits_for_the_comm_qubit else 0
    half_decays_on_slot = expected_wait * decay_per_unit_on_data
    return half_decays_on_comm + a_data_qubit_idles_waiting_for_the_comm_qubit > swap_error_cost(state.error_model) + half_decays_on_slot

def pairs_the_consumer_waits_for(state, consumer, preds, eprs):
    pairs, seen, to_visit = set(), set(), list(preds[consumer])
    while to_visit:
        idx = to_visit.pop()
        if idx in seen:
            continue
        seen.add(idx)
        if state.start[idx] is None:
            if idx in eprs:
                pairs.add(idx)
            to_visit.extend(preds[idx])
    return pairs

def pair_waits_for_comm_qubit(state, epr, qpu, meta, epr_of):
    if epr not in state.ready:
        return False
    return any(epr_of[session_id] == epr and session_meta["qpu"] == qpu and session_id not in state.host
               for session_id, session_meta in enumerate(meta))

def blocks_a_pair_it_waits_for(state, session_id, meta, epr_of, consumer_of_session, preds, eprs):
    qpu = meta[session_id]["qpu"]
    pairs = pairs_the_consumer_waits_for(state, consumer_of_session[session_id], preds, eprs)
    return any(pair_waits_for_comm_qubit(state, epr, qpu, meta, epr_of) for epr in pairs)

def data_qubits_with_remote_gates(qpu, meta):
    return len(set().union(*(session_meta["data"] for session_meta in meta if session_meta["qpu"] == qpu)))

def standstill_costs_more_than_a_swap(state, qpu, meta):
    decay_per_unit_on_comm = 1 / (2 * state.error_model.T2_comm)
    decay_per_unit_on_data = 1 / (2 * state.error_model.T2_data)
    half_decays_on_comm = state.mean * decay_per_unit_on_comm
    data_qubits_of_the_qpu_idle = state.mean * decay_per_unit_on_data * data_qubits_with_remote_gates(qpu, meta)
    return half_decays_on_comm + data_qubits_of_the_qpu_idle > swap_error_cost(state.error_model)

def tails_on_comm(state, epr_of, consumer_of_session):
    return [session_id for session_id, host in state.host.items()
            if session_id not in state.moved and state.held.get(host) == session_id
            and epr_of[session_id] in state.finished and state.start[consumer_of_session[session_id]] is None]

def move_tails_on_comm_if_it_pays(state, meta, spare, epr_of, swap_time, reset_time, consumer_of_session, preds, duration_of_instruction, eprs):
    if not spare:
        return
    expected_finish_of = {}
    tails = [session_id for session_id in tails_on_comm(state, epr_of, consumer_of_session) if meta[session_id]["qpu"] in spare]
    tails.sort(key=lambda session_id: expected_wait_via_predecessors(state, consumer_of_session[session_id], preds, duration_of_instruction, eprs, expected_finish_of),
               reverse=True)
    for session_id in tails:
        move_tail_if_it_pays(state, session_id, meta, spare, epr_of, swap_time, reset_time,
                             consumer_of_session, preds, duration_of_instruction, eprs, expected_finish_of)
