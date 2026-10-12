from qiskit import ClassicalRegister, QuantumCircuit
import re
from disqco.scheduling.evaluator import duration_of, wires_of
from disqco.scheduling.commutation import split_instruction_indicies_to_commuting_groups

DATA_REG = re.compile(r"^Q(\d+)_q$")
COMM_REG = re.compile(r"^C(\d+)_(\d+)$")

def wire_key(index, wire, session_of):
    return (wire, session_of.get((index, wire)))

def predecessors_and_successors(circuit: QuantumCircuit, session_of, include_clbits=True, commuting=False):
    """Returns predecessors and successors of each singular instruction in the circuit"""
    n = len(circuit.data)
    chains = {}    # wire key -> [instruction index in order]
    for index, instruction in enumerate(circuit.data):
        for wire in wires_of(instruction, include_clbits):
            chains.setdefault(wire_key(index, wire, session_of), []).append(index)

    edges = set()
    for indices in chains.values():
        if commuting:
            groups = split_instruction_indicies_to_commuting_groups(circuit, indices)
        else:
            groups = [[index] for index in indices]

        for earlier, later in zip(groups, groups[1:]):
            for a in earlier:
                for b in later:
                    edges.add((a, b))

    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    for a, b in sorted(edges):
        preds[b].append(a)
        succs[a].append(b)

    return preds, succs

def priority_of_operation(circuit: QuantumCircuit, succs, durations):
    """Returns the longest duraation of each operation to the end of the circuit"""
    priority = [0] * len(circuit.data)
    for idx in reversed(range(len(circuit.data))):
        duration = duration_of(circuit.data[idx].operation.name, durations)
        priority[idx] = duration + max((priority[s] for s in succs[idx]), default=0)
    return priority

def critical_path(circuit: QuantumCircuit, succs, durations):
    return max(priority_of_operation(circuit, succs, durations), default=0)

def comm_qubits_of_qpu(circuit: QuantumCircuit):
    """Returns map qpu to [comm qubits] in that qpu"""
    qpu_to_comm_qubits = {}
    for register in circuit.qregs:
        match = COMM_REG.match(register.name)
        if match:
            qpu = int(match.group(1))
            if qpu not in qpu_to_comm_qubits:
                qpu_to_comm_qubits[qpu] = []
            for qubit in register:
                qpu_to_comm_qubits[qpu].append(qubit)

    return qpu_to_comm_qubits

def ancestor_eprs(circuit, preds, sessions_opening_from_instruction):
    """{instruction index: the EPR instructions among its ancestors}. circuit.data is in topological order."""
    ancestors = [frozenset()] * len(circuit.data)
    for idx in range(len(circuit.data)):
        found = set()
        for pred in preds[idx]:
            found |= ancestors[pred]
            if pred in sessions_opening_from_instruction:
                found.add(pred)
        ancestors[idx] = frozenset(found)
    return ancestors

def instructions_to_start_before_first_instruction_on_data(circuit, session_of, preds):
    """{first instruction on a data qubit: predecessors on other wires of the first instruction on that qubit that uses a session}
    A data qubit waits for these before its first instruction, so it does not start dephasing before its pair is on the way."""
    instructions_on_data_qubit = {}
    for idx, instruction in enumerate(circuit.data):
        for qubit in instruction.qubits:
            if session_of.get((idx, qubit)) is None:
                instructions_on_data_qubit.setdefault(qubit, []).append(idx)

    instructions_to_start_before = {}
    for qubit, instructions in instructions_on_data_qubit.items():
        first_instruction_using_a_session = next((idx for idx in instructions
                                                  if any((idx, wire) in session_of for wire in circuit.data[idx].qubits)), None)
        if first_instruction_using_a_session is None or first_instruction_using_a_session == instructions[0]:
            continue
        instructions_to_start_before[instructions[0]] = [pred for pred in preds[first_instruction_using_a_session] if pred not in instructions]
    return instructions_to_start_before
