from nova_voice.offline_optimizers import (
    OfflineOptimizerPool,
    OptimizerEvaluation,
    OptimizerKind,
    OptimizerProposal,
    compare_frontier_candidate,
)


def _proposal(**updates) -> OptimizerProposal:
    values = {
        "id": "proposal-1",
        "kind": OptimizerKind.MEMORY,
        "input_revision": f"sha256:{'d' * 64}",
        "objective": "find duplicate memory candidates",
        "payload": {"candidates": ["a", "a"]},
    }
    values.update(updates)
    return OptimizerProposal(**values)


async def test_optimizer_workers_emit_recommendations_without_apply_access() -> None:
    pool = OfflineOptimizerPool()
    pool.start()
    try:
        proposal = _proposal()
        await pool.submit(proposal)
        await pool.queues[proposal.kind].join()

        result = pool.results[proposal.id]
        assert result.accepted_for_review
        assert result.applied_changes == 0
        assert pool.health()["foregroundHooks"] == 0
        assert pool.health()["providerAccess"] is False
        assert pool.health()["storeWriteAccess"] is False
    finally:
        await pool.close()


async def test_optimizer_sanitizes_evaluator_claim_of_applied_changes() -> None:
    async def unsafe(proposal: OptimizerProposal) -> OptimizerEvaluation:
        return OptimizerEvaluation(
            proposal_id=proposal.id,
            kind=proposal.kind,
            input_revision=proposal.input_revision,
            accepted_for_review=True,
            score=1,
            applied_changes=3,
        )

    pool = OfflineOptimizerPool(unsafe)
    pool.start()
    try:
        proposal = _proposal()
        await pool.submit(proposal)
        await pool.queues[proposal.kind].join()

        result = pool.results[proposal.id]
        assert not result.accepted_for_review
        assert result.applied_changes == 0
        assert "optimizer_applied_changes" in result.issues
    finally:
        await pool.close()


def test_frontier_candidate_remains_evaluation_only_and_cascade_stays_production() -> None:
    comparison = compare_frontier_candidate(
        cascade_metrics={"latencyMs": 500, "quality": 0.8},
        candidate_metrics={"latencyMs": 300, "quality": 0.9},
    )

    assert comparison["productionRuntime"] == "cascade"
    assert comparison["candidateRole"] == "evaluation_only"
    assert comparison["candidateSelectedForProduction"] is False
    assert comparison["deltas"]["quality"] > 0
