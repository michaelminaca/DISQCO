import heapq
import random
from enum import IntEnum
from disqco.scheduling.buffer import BufferEvent, maybe_move_tail, spare_slots_by_qpu
from disqco.scheduling.evaluator import duration_of, wires_of
from disqco.scheduling.scheduler import (ScheduleState, claim_order_by_priority, claims_to_start, host_chooser,
                                         instruction_finished, prepare, start_instruction)
from disqco.scheduling.sessions import sessions_ending_at, sessions_opening_at


class EPREvent(IntEnum):
    TIMED_OUT = -3

class StochasticScheduleState(ScheduleState):
    def __init__(self, num_instructions, preds, mean, p, cutoff, trial):
        super().__init__(num_instructions, preds)
        self.attempt = mean * p
        self.p = p
        self.cutoff = cutoff
        self.trial = trial
        self.random_of_epr = {}
        self.epr_generated_at = {}
        self.discards = 0
        self.finished = set()

def random_duration_of_epr(state, epr):
    """Calculate total time to generate an epr """
    if epr not in state.random_of_epr:
        state.random_of_epr[epr] = random.Random("trial %d, epr %d" % (state.trial, epr))
    stream = state.random_of_epr[epr]
    attempts = 1
    while stream.random() > state.p:
        attempts += 1
    return attempts * state.attempt

def durations_for_instruction(state, idx, durations, sessions_opening_from_instruction):
    """durations of each type of instruction but overrides the EPR with a stochastic generation"""
    if idx not in sessions_opening_from_instruction:
        return durations
    durations_with_overridden_epr = dict(durations)
    durations_with_overridden_epr["EPR"] = random_duration_of_epr(state, idx)
    return durations_with_overridden_epr

def first_instruction_using_the_pair(circuit, sessions, session_of, session_id):
    "returns the first inst for a given session that touches a data qubit"
    tail = sessions[session_id][1:]
    for idx in tail:
        if any(session_of.get((idx, qubit)) != session_id for qubit in circuit.data[idx].qubits):
            return idx
    return tail[-1]

def epr_used(state, epr, circuit, sessions, session_of, sessions_opening_from_instruction):
    """The epr is used if any of the 2 sessions have a multi-qubit gate from them to a data qubit"""
    for session_id in sessions_opening_from_instruction[epr]:
        idx = first_instruction_using_the_pair(circuit, sessions, session_of, session_id)
        if state.start[idx] is not None:
            return True
    return False

def epr_generated(state, epr):
    """called when an epr is successfully generated. Bookkeeping as well as adds the timeout event to the heap."""
    state.epr_generated_at[epr] = state.clock
    if state.cutoff is not None:
        heapq.heappush(state.running, (state.clock + state.cutoff, EPREvent.TIMED_OUT, epr))

def undo_instruction(state, idx, succs):
    if idx in state.finished:
        for successor in succs[idx]:
            state.indeg[successor] += 1
            state.ready.discard(successor)
        state.finished.discard(idx)
    state.start[idx] = None
    state.end[idx] = None

def tail_instructions_already_run(state, circuit, sessions, session_of, session_id):
    """Called when an epr times out. Returns the indexes of instructions that have already run in the sessions tail"""
    consumer = first_instruction_using_the_pair(circuit, sessions, session_of, session_id)
    tail = sessions[session_id][1:]
    return [idx for idx in tail[:tail.index(consumer)] if state.start[idx] is not None]


def epr_timed_out(state, epr, circuit, sessions, session_of, succs, sessions_opening_from_instruction, reset_time):
    """Called when an epr times out event is popped from the state heap."""
    if epr_used(state, epr, circuit, sessions, session_of, sessions_opening_from_instruction):
        return

    for session_id in sessions_opening_from_instruction[epr]:
        for idx in reversed(tail_instructions_already_run(state, circuit, sessions, session_of, session_id)):
            undo_instruction(state, idx, succs)
        if session_id in state.moved:
            # session was buffered
            slot = state.host[session_id]
            del state.held[slot]
            state.occupy(slot, state.clock + reset_time)
            del state.moved[session_id]
            del state.host[session_id]

    undo_instruction(state, epr, succs)
    state.ready.add(epr)
    state.discards += 1


def finish_instructions_at_clock(state, circuit, sessions, session_of, succs, sessions_closing_from_instruction,
                                 sessions_opening_from_instruction, meta, spare, epr_of, swap_time, reset_time):
    while state.running and state.next_finish() == state.clock:
        time, idx_or_kind, *rest = heapq.heappop(state.running)
        if isinstance(idx_or_kind, BufferEvent):
            continue
        if idx_or_kind == EPREvent.TIMED_OUT:
            epr_timed_out(state, rest[0], circuit, sessions, session_of, succs, sessions_opening_from_instruction, reset_time)
            continue
        if state.end[idx_or_kind] != time or idx_or_kind in state.finished:
            continue
        state.finished.add(idx_or_kind)
        instruction_finished(state, idx_or_kind, succs, sessions_closing_from_instruction, sessions_opening_from_instruction,
                             meta, spare, epr_of, swap_time, reset_time)
        if idx_or_kind in sessions_opening_from_instruction:
            epr_generated(state, idx_or_kind)


def run(circuit, sessions, session_of, meta, preds, succs, priority, durations, pool, choose_host, spare=None,
        mean=100, p=1.0, cutoff=None, trial=0):
    circuit_wires = [wires_of(instruction) for instruction in circuit.data]
    sessions_opening_from_instruction = sessions_opening_at(sessions)
    sessions_closing_from_instruction = sessions_ending_at(sessions)
    epr_of = [instructions[0] for instructions in sessions]
    spare = spare or {}
    swap_time, reset_time = duration_of("swap", durations), duration_of("reset", durations)
    state = StochasticScheduleState(len(circuit.data), preds, mean, p, cutoff, trial)

    while state.ready or state.running:
        started_anything = True
        while started_anything:
            started_anything = False
            for idx in sorted(state.ready, key=lambda i: -priority[i]):
                claims = claims_to_start(state, idx, circuit_wires, session_of, meta, pool, choose_host)
                if claims is None:
                    continue

                start_instruction(state, idx, claims, circuit, circuit_wires, session_of,
                                  durations_for_instruction(state, idx, durations, sessions_opening_from_instruction))
                started_anything = True

        if state.running:
            state.clock = state.next_finish()
            finish_instructions_at_clock(state, circuit, sessions, session_of, succs, sessions_closing_from_instruction,
                                         sessions_opening_from_instruction, meta, spare, epr_of, swap_time, reset_time)
        elif state.ready:
            raise RuntimeError("stalled")

    return state


def simulate(circuit, durations, mean=100, p=1.0, cutoff=None, trial=0, network=None, commuting=True, buffer=False):
    """returns (makespan, discards, state)"""
    renamed_circuit, sessions, sessions_of, meta, preds, succs, pool, priority = prepare(circuit, durations, network, True, commuting)
    choose_host = host_chooser(claim_order_by_priority(sessions, meta, priority), meta)
    spare = spare_slots_by_qpu(renamed_circuit, network) if buffer else None
    state = run(renamed_circuit, sessions, sessions_of, meta, preds, succs, priority, durations, pool, choose_host, spare,
                mean, p, cutoff, trial)
    return max(state.end), state.discards, state


def simulate_many(circuit, durations, trials=100, **options):
    makespans, discards = [], []
    for trial in range(trials):
        makespan, discarded, _ = simulate(circuit, durations, trial=trial, **options)
        makespans.append(makespan)
        discards.append(discarded)
    makespans.sort()
    return {"mean": sum(makespans) / trials, "p90": makespans[int(0.9 * (trials - 1))], "discards": sum(discards) / trials}
