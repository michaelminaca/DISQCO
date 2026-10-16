class ScheduledCircuitDiffers(RuntimeError):
    """the schedule is invalid compared to original circuit"""

def validate_schedule(state, preds, circuit, sessions, session_of, swap_time, reset_time):
    check_dependencies(state, preds)
    check_no_qubit_is_used_twice_at_once(time_each_qubit_is_used(state, circuit, sessions, session_of, swap_time, reset_time))

def check_dependencies(state, preds):
    for idx, earlier_instructions in enumerate(preds):
        if state.start[idx] is None or state.end[idx] is None:
            raise ScheduledCircuitDiffers("instruction %d was never scheduled" % idx)
        for earlier in earlier_instructions:
            if state.end[earlier] > state.start[idx]:
                raise ScheduledCircuitDiffers("instruction %d starts at %s, before instruction %d it depends on finishes at %s" % (idx, state.start[idx], earlier, state.end[earlier]))

def time_each_qubit_is_used(state, circuit, sessions, session_of, swap_time, reset_time):
    used = {}
    for idx, instruction in enumerate(circuit.data):
        for qubit in instruction.qubits:
            if (idx, qubit) not in session_of:
                used.setdefault(qubit, []).append((state.start[idx], state.end[idx], "instruction %d" % idx))
    for session_id, instructions in enumerate(sessions):
        start, end, what = state.start[instructions[0]], max(state.end[idx] for idx in instructions), "session %d" % session_id
        if session_id in state.moved:
            moved_at = state.moved_at[session_id]
            used.setdefault(state.moved[session_id], []).append((start, moved_at + swap_time + reset_time, what))
            used.setdefault(state.host[session_id], []).append((moved_at, end, what))
        else:
            used.setdefault(state.host[session_id], []).append((start, end, what))
    return used

def check_no_qubit_is_used_twice_at_once(used):
    for spans in used.values():
        spans.sort()
        for (_, earlier_end, earlier), (later_start, _, later) in zip(spans, spans[1:]):
            if earlier_end > later_start:
                raise ScheduledCircuitDiffers("%s and %s use the same qubit at the same time" % (earlier, later))
