"""Convert gateway TraceRecord to training-compatible Step, plus shared metrics."""

from rllm_model_gateway.models import TraceRecord

from rllm.agents.agent import Step, Trajectory
from rllm.experimental.rollout import ModelOutput


def trace_record_to_step(trace: TraceRecord) -> Step:
    """Convert a gateway TraceRecord to a training Step.

    TraceRecord has clean top-level fields from vLLM:
    - prompt_token_ids
    - completion_token_ids
    - logprobs (per-token)
    """
    content = trace.response_message.get("content", "") or ""
    reasoning = trace.response_message.get("reasoning", "") or ""

    model_output = ModelOutput(
        content=content,
        reasoning=reasoning,
        prompt_ids=trace.prompt_token_ids,
        completion_ids=trace.completion_token_ids,
        logprobs=trace.logprobs or [],
        prompt_length=len(trace.prompt_token_ids),
        completion_length=len(trace.completion_token_ids),
        finish_reason=trace.finish_reason,
    )

    # Build chat_completions: input messages + assistant response
    chat_completions = list(trace.messages)
    chat_completions.append(trace.response_message)

    return Step(
        id=trace.trace_id,
        chat_completions=chat_completions,
        model_output=model_output,
        model_response=content,
        thought=reasoning,
        metadata=trace.metadata,
    )


def compute_step_metrics(trajectories: list[Trajectory]) -> dict:
    """Standard training metrics from trajectories (shared by local and remote engines)."""
    all_response_lens = [len(s.response_ids) for t in trajectories for s in t.steps]
    all_prompt_lens = [len(s.prompt_ids) for t in trajectories for s in t.steps]
    return {
        "num_trajectories": len(trajectories),
        "steps_used": sum(len(t.steps) for t in trajectories),
        "mean_response_len": (sum(all_response_lens) / len(all_response_lens) if all_response_lens else 0),
        "max_response_len": max(all_response_lens, default=0),
        "min_response_len": min(all_response_lens, default=0),
        "max_prompt_len": max(all_prompt_lens, default=0),
        "min_prompt_len": min(all_prompt_lens, default=0),
    }
