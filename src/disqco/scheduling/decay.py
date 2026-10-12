import math

from disqco.scheduling.errors import swap_error_cost


def use_by_time_of_half(state, session_id, epr):
    """Time when the epr half decoheres given the cutoff"""
    herald = state.epr_generated_at[epr]
    T2_comm = state.error_model.T2_comm
    T2_data = state.error_model.T2_data
    # epr half was not moved
    if session_id not in state.moved:
        return herald + state.cutoff
    #epr half was moved
    moved_at = state.moved_at[session_id]
    time_on_comm = moved_at - herald
    budget_spent_on_comm = time_on_comm / T2_comm
    budget_left_for_slot = max(0, state.coherence_budget - budget_spent_on_comm)
    time_left_on_slot = budget_left_for_slot * T2_data
    return herald + time_on_comm + time_left_on_slot

def use_by_time_of_pair(state, epr, sessions_opening_from_instruction):
    """The pair dies when its weaker half diies"""
    return min(use_by_time_of_half(state, session_id, epr) for session_id in sessions_opening_from_instruction[epr])