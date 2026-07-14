"""Dependency injection and namespace isolation tests for MemoryAgent."""

from __future__ import annotations

from pathlib import Path
import inspect

from memory_agent.ports import EvolutionPlannerPort, MemoryStorePort, RetrievalPort


def test_dependency_protocols_declare_namespace_operations() -> None:
    retrieval_params = inspect.signature(RetrievalPort.retrieve).parameters
    store_params = inspect.signature(MemoryStorePort.add_memory).parameters
    assert "namespace" in retrieval_params
    assert "namespace" in store_params
    assert "commit" in retrieval_params
    assert "record_access" in retrieval_params
    for method_name in ("get_embedding", "get_memories_by_type", "archive_memory", "record_access"):
        method = getattr(MemoryStorePort, method_name)
        assert "namespace" in inspect.signature(method).parameters
    assert "conn" not in getattr(MemoryStorePort, "__annotations__", {})
    assert hasattr(MemoryStorePort, "commit")


def test_evolution_planner_port_exposes_deterministic_proposal_contract() -> None:
    parameters = inspect.signature(EvolutionPlannerPort.propose).parameters
    assert tuple(parameters)[:4] == ("self", "candidate", "neighbors", "context")


from memory_agent.agent.orchestrator import MemoryAgent
from memory_agent.models import MemoryRecord, RetrievalEvidence, SearchResult


class FakeConnection:
    def commit(self) -> None:
        pass


class FakeStore:
    """Small in-memory store implementing the orchestrator's store contract."""

    def __init__(self) -> None:
        self.conn = FakeConnection()
        self.commit_calls = 0
        self.memories: list[MemoryRecord] = []
        self.sessions: dict[int, str | None] = {}
        self.record_access_commits: list[bool] = []
        self.calls: list[tuple[str, str | None]] = []
        self.ended_sessions: list[tuple[int, str | None]] = []
        self.session_labels: list[tuple[int, str, str | None]] = []
        self.links: list[tuple[int, int, int | None, str | None]] = []
        self.saved_embedding_ids: list[int] = []
        self._next_memory_id = 1
        self._next_session_id = 1

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def commit(self) -> None:
        self.commit_calls += 1

    def add_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        self.calls.append(("add_memory", namespace))
        if namespace is not None:
            memory.namespace = namespace
        memory.id = self._next_memory_id
        self._next_memory_id += 1
        self.memories.append(memory)
        return memory.id

    def get_memory(self, memory_id: int, *, namespace: str | None = None) -> MemoryRecord | None:
        self.calls.append(("get_memory", namespace))
        return next(
            (
                memory
                for memory in self.memories
                if memory.id == memory_id
                and (namespace is None or memory.namespace == namespace)
            ),
            None,
        )

    def update_memory(
        self,
        memory: MemoryRecord,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        self.calls.append(("update_memory", namespace))

    def get_all_active_memories(self, *, namespace: str | None = None) -> list[MemoryRecord]:
        self.calls.append(("get_all_active_memories", namespace))
        return [
            memory
            for memory in self.memories
            if memory.is_active and (namespace is None or memory.namespace == namespace)
        ]

    def get_memories_by_type(
        self, memory_type: str, *, namespace: str | None = None
    ) -> list[MemoryRecord]:
        self.calls.append(("get_memories_by_type", namespace))
        return [
            memory
            for memory in self.get_all_active_memories(namespace=namespace)
            if memory.memory_type == memory_type
        ]

    def count_memories(
        self, *, active_only: bool = True, namespace: str | None = None
    ) -> int:
        self.calls.append(("count_memories", namespace))
        return sum(
            1
            for memory in self.memories
            if (not active_only or memory.is_active)
            and (namespace is None or memory.namespace == namespace)
        )

    def create_session(
        self, label: str = "", *, namespace: str | None = None, commit: bool = True
    ) -> int:
        self.calls.append(("create_session", namespace))
        session_id = self._next_session_id
        self._next_session_id += 1
        self.sessions[session_id] = namespace
        self.session_labels.append((session_id, label, namespace))
        return session_id

    def end_session(
        self, session_id: int, *, namespace: str | None = None, commit: bool = True
    ) -> None:
        self.calls.append(("end_session", namespace))
        self.ended_sessions.append((session_id, namespace))

    def link_memory_to_session(
        self,
        session_id: int,
        memory_id: int,
        turn_index: int | None = None,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        self.calls.append(("link_memory_to_session", namespace))
        self.links.append((session_id, memory_id, turn_index, namespace))

    def save_embedding(
        self,
        memory_id: int,
        embedding: bytes,
        model_name: str,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        self.calls.append(("save_embedding", namespace))
        self.saved_embedding_ids.append(memory_id)

    def update_strengths(
        self,
        updates: list[tuple[float, int]],
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        self.calls.append(("update_strengths", namespace))

    def archive_below_threshold(
        self,
        threshold: float,
        *,
        namespace: str | None = None,
        commit: bool = True,
    ) -> int:
        self.calls.append(("archive_below_threshold", namespace))
        return 0

    def get_embedding(
        self, memory_id: int, *, namespace: str | None = None
    ) -> tuple[bytes, str] | None:
        return None

    def record_access(
        self,
        accesses: list[tuple[int, int]],
        *,
        namespace: str | None = None,
        accessed_at: str | None = None,
        commit: bool = True,
    ) -> None:
        self.calls.append(("record_access", namespace))
        self.record_access_commits.append(commit)

    def archive_memory(
        self,
        memory_id: int,
        *,
        reason: str,
        metadata: dict | None = None,
        namespace: str | None = None,
        commit: bool = True,
    ) -> None:
        self.calls.append(("archive_memory", namespace))

    def get_embedding_count(self, *, namespace: str | None = None) -> int:
        return sum(
            1
            for memory in self.memories
            if namespace is None or memory.namespace == namespace
        )


class FakeEmbedder:
    model_name = "fake"

    def encode(self, text: str) -> bytes:
        return text.encode()

    def decode_vector(self, blob: bytes) -> str:
        return blob.decode()

    def cosine_similarity(self, left: str, right: str) -> float:
        return 1.0 if left == right else 0.0


class FakeRetrieval:
    def __init__(self, results: dict[str | None, list[SearchResult]]) -> None:
        self.results = results
        self.calls: list[str | None] = []

    def retrieve(self, query: str, **kwargs: object) -> list[SearchResult]:
        namespace = kwargs.get("namespace")
        self.calls.append(namespace if isinstance(namespace, str) else None)
        return list(self.results.get(namespace, []))


class FakeTrustPolicy:
    def __init__(self) -> None:
        self.calls: list[int | None] = []

    def evaluate(self, memory: MemoryRecord) -> RetrievalEvidence:
        self.calls.append(memory.id)
        return RetrievalEvidence(
            score=0.8,
            matched_by=("policy",),
            trust="trusted",
            reason="fake trust policy accepted memory",
        )


def _result(content: str, memory_id: int) -> SearchResult:
    memory = MemoryRecord(
        id=memory_id,
        content=content,
        importance=0.8,
        namespace="tenant-a" if memory_id == 1 else "tenant-b",
    )
    evidence = RetrievalEvidence(
        score=0.7,
        matched_by=("semantic",),
        trust="trusted",
        reason="fake retrieval",
    )
    return SearchResult(memory=memory, score=0.7, evidence=evidence)


def test_constructor_injects_store_embedder_retrieval_and_trust_policy() -> None:
    store = FakeStore()
    embedder = FakeEmbedder()
    retrieval = FakeRetrieval({})
    trust_policy = FakeTrustPolicy()

    agent = MemoryAgent(
        store=store,
        embedder=embedder,
        retrieval=retrieval,
        trust_policy=trust_policy,
    )

    assert agent.store is store
    assert agent.embeddings is embedder
    assert agent.retrieval is retrieval
    assert agent.trust_policy is trust_policy
    agent.close()


def test_perceive_threads_namespace_and_user_id_and_returns_lifecycle_evidence() -> None:
    store = FakeStore()
    retrieval = FakeRetrieval({"tenant-a": [_result("A only", 1)]})
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=retrieval,
        trust_policy=FakeTrustPolicy(),
    )

    result = agent.perceive(
        "I like Python",
        namespace="tenant-a",
        user_id="user-a",
    )

    assert result["recollections"]
    assert all(r.memory.namespace == "tenant-a" for r in result["recollections"])
    assert {"recollections", "recollection_text", "new_memories", "turn_count", "total_memories", "archived", "recall_packet", "consolidation_decisions"} <= result.keys()
    assert "evidence" in result
    assert "lifecycle" in result
    assert all(memory.metadata.get("user_id") == "user-a" for memory in result["new_memories"])
    assert all(memory.namespace == "tenant-a" for memory in result["new_memories"])
    assert all(namespace == "tenant-a" for _, namespace in store.calls if namespace is not None)
    assert retrieval.calls == ["tenant-a"]
    agent.close()


def test_perceive_does_not_cross_namespaces() -> None:
    store = FakeStore()
    retrieval = FakeRetrieval(
        {
            "tenant-a": [_result("A secret", 1)],
            "tenant-b": [_result("B secret", 2)],
        }
    )
    for result in retrieval.results["tenant-a"] + retrieval.results["tenant-b"]:
        store.add_memory(result.memory, namespace=result.memory.namespace)
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=retrieval,
        trust_policy=FakeTrustPolicy(),
    )

    first = agent.perceive("What do you know?", namespace="tenant-a", user_id="alice")
    second = agent.perceive("What do you know?", namespace="tenant-b", user_id="bob")

    assert [r.memory.content for r in first["recollections"]] == ["A secret"]
    assert [r.memory.content for r in second["recollections"]] == ["B secret"]
    assert retrieval.calls == ["tenant-a", "tenant-b"]
    agent.close()
def test_consolidation_does_not_match_other_namespace() -> None:
    from memory_agent.core.config import ConsolidationConfig
    from memory_agent.core.consolidation import ConsolidationAction, MemoryConsolidator

    store = FakeStore()
    existing = MemoryRecord(
        content="same private fact",
        memory_type="semantic",
        importance=0.8,
        namespace="tenant-a",
    )
    store.add_memory(existing, namespace="tenant-a")
    consolidator = MemoryConsolidator(
        store,
        type("AlwaysSimilar", (), {"similarity": lambda self, left, right: 1.0})(),
        ConsolidationConfig(),
    )

    candidate = MemoryRecord(
        content="same private fact",
        memory_type="semantic",
        importance=0.8,
        namespace="tenant-b",
    )
    decision = consolidator.consolidate(candidate, namespace="tenant-b")

    assert decision.action is ConsolidationAction.CREATE
    assert existing.strength == 1.0
    assert candidate.namespace == "tenant-b"
    assert [m.namespace for m in store.memories] == ["tenant-a", "tenant-b"]




def test_stats_embedding_count_is_namespace_scoped() -> None:
    store = FakeStore()
    store.add_memory(MemoryRecord(content="A", namespace="tenant-a"), namespace="tenant-a")
    store.add_memory(MemoryRecord(content="B", namespace="tenant-b"), namespace="tenant-b")
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )
    agent.init_session(namespace="tenant-a", user_id="alice")

    stats = agent.get_stats()

    assert stats["total_active"] == 1
    assert stats["embedding_count"] == 1
    agent.close()


def test_session_namespace_and_user_id_are_inherited_by_perceive() -> None:
    store = FakeStore()
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )
    agent.init_session(user_id="user-session")

    result = agent.perceive("I like Python")

    assert result["lifecycle"]["namespace"] == "user-session"
    assert result["lifecycle"]["user_id"] == "user-session"
    assert result["new_memories"]
    assert all(memory.namespace == "user-session" for memory in result["new_memories"])
    assert all(
        memory.metadata.get("user_id") == "user-session"
        for memory in result["new_memories"]
    )
    agent.close()


def test_namespace_override_preserves_session_user_id() -> None:
    store = FakeStore()
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )
    agent.init_session(label="persisted-label", user_id="session-user")
    agent.perceive("I like Python")
    old_session_id = agent.state.session_id
    agent.state.current_context = [MemoryRecord(content="old context")]

    result = agent.perceive("What do you know?", namespace="tenant-override")

    assert old_session_id is not None
    assert result["lifecycle"]["namespace"] == "tenant-override"
    assert result["lifecycle"]["user_id"] == "session-user"
    assert ("create_session", "tenant-override") in store.calls
    assert (old_session_id, "session-user") in store.ended_sessions
    assert agent.state.session_id != old_session_id
    assert store.session_labels[-1][1:] == ("persisted-label", "tenant-override")
    assert agent.state.turn_count == 1
    assert agent.state.current_context == []
    assert store.links[-1][2:] == (1, "tenant-override")
    agent.close()
def test_perceive_commits_store_contract_without_conn() -> None:
    store = FakeStore()
    del store.conn
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )

    agent.perceive("hello", namespace="tenant-a")

    assert store.commit_calls == 1
    agent.close()



def test_default_retrieval_works_with_store_only_dependency() -> None:
    from memory_agent.core.config import RetrievalConfig
    from memory_agent.core.retrieval import RetrievalEngine

    store = FakeStore()
    del store.conn
    store.add_memory(
        MemoryRecord(content="private fact", namespace="tenant-a"),
        namespace="tenant-a",
    )
    retrieval = RetrievalEngine(
        store,
        FakeEmbedder(),
        RetrievalConfig(top_k=1, candidate_k=1, min_score=0.0),
    )

    results = retrieval.retrieve(
        "private", namespace="tenant-a", use_mmr=False, commit=False
    )

    assert results
    assert ("record_access", "tenant-a") in store.calls
    assert store.record_access_commits == [False]
def test_store_memory_assigns_returned_id_before_indexing() -> None:
    class NonMutatingStore(FakeStore):
        def add_memory(
            self,
            memory: MemoryRecord,
            *,
            namespace: str | None = None,
            commit: bool = True,
        ) -> int:
            self.calls.append(("add_memory", namespace))
            if namespace is not None:
                memory.namespace = namespace
            memory_id = self._next_memory_id
            self._next_memory_id += 1
            self.memories.append(memory)
            return memory_id

    store = NonMutatingStore()
    del store.conn
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )
    memory = MemoryRecord(content="indexed", namespace="tenant-a")

    memory_id = agent.store_memory(memory)

    assert memory_id == 1
    assert memory.id == 1
    agent.close()




def test_extracted_memory_assigns_consolidation_id_before_indexing() -> None:
    class NonMutatingStore(FakeStore):
        def add_memory(
            self,
            memory: MemoryRecord,
            *,
            namespace: str | None = None,
            commit: bool = True,
        ) -> int:
            self.calls.append(("add_memory", namespace))
            if namespace is not None:
                memory.namespace = namespace
            memory_id = self._next_memory_id
            self._next_memory_id += 1
            self.memories.append(memory)
            return memory_id

    store = NonMutatingStore()
    del store.conn
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )

    result = agent.perceive("I like Python", namespace="tenant-a")

    assert result["new_memories"]
    extracted = result["new_memories"][0]
    assert extracted.id == 1
    assert 1 in store.saved_embedding_ids
    agent.close()


def test_trust_policy_fuses_without_dropping_retrieval_evidence() -> None:
    result = _result("trusted fact", 1)
    agent = MemoryAgent(
        store=FakeStore(),
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )

    agent._apply_trust_policy([result])

    assert result.evidence is not None
    assert result.evidence.score == 0.7
    assert result.evidence.matched_by == ("semantic",)
    assert result.evidence.trust == "trusted"
    assert result.evidence.reason == "fake trust policy accepted memory"
    agent.close()
def test_trust_override_without_reason_generates_coherent_reason() -> None:
    class TrustedWithoutReason:
        def evaluate(self, memory: MemoryRecord) -> RetrievalEvidence:
            return RetrievalEvidence(trust="trusted")

    result = SearchResult(
        memory=MemoryRecord(id=9, content="trust override"),
        score=0.8,
        evidence=RetrievalEvidence(
            score=0.8,
            trust="untrusted",
            reason="filtered: trust=untrusted",
        ),
    )
    agent = MemoryAgent(
        store=FakeStore(),
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=TrustedWithoutReason(),
    )

    agent._apply_trust_policy([result])

    assert result.evidence is not None
    assert result.evidence.trust == "trusted"
    assert result.evidence.reason
    assert "untrusted" not in result.evidence.reason
    assert "trusted" in result.evidence.reason
    agent.close()




def test_trust_policy_adds_evidence_from_search_result_scores() -> None:
    memory = MemoryRecord(id=4, content="partial evidence")
    result = SearchResult(
        memory=memory,
        score=0.73,
        semantic_score=0.61,
        recency_score=0.52,
        importance_score=0.43,
        strength_score=0.34,
    )
    agent = MemoryAgent(
        store=FakeStore(),
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )

    agent._apply_trust_policy([result])

    assert result.evidence is not None
    assert result.evidence.score == 0.73
    assert result.evidence.semantic_score == 0.61
    assert result.evidence.recency_score == 0.52
    assert result.evidence.importance_score == 0.43
    assert result.evidence.strength_score == 0.34
    assert result.evidence.trust == "trusted"
    agent.close()


def test_untrusted_recollection_is_not_reinforced_or_accessed() -> None:
    class UnsafePolicy:
        def evaluate(self, memory: MemoryRecord) -> RetrievalEvidence:
            return RetrievalEvidence(trust="untrusted", reason="unsafe fixture")

    store = FakeStore()
    del store.conn
    unsafe = MemoryRecord(
        content="unsafe imported claim",
        namespace="tenant-a",
        confidence=0.1,
    )
    store.add_memory(unsafe, namespace="tenant-a")
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        trust_policy=UnsafePolicy(),
    )

    result = agent.perceive("unsafe imported claim", namespace="tenant-a")

    assert result["recollections"] == []
    assert unsafe.access_count == 0
    assert unsafe.strength == 1.0
    assert store.record_access_commits == []
    agent.close()


def test_mmr_diversifies_when_pool_exceeds_top_k() -> None:
    from unittest.mock import Mock

    from memory_agent.core.config import RetrievalConfig
    from memory_agent.core.retrieval import RetrievalEngine

    store = FakeStore()
    for index in range(11):
        store.add_memory(
            MemoryRecord(content=f"fact {index}", namespace="tenant-a"),
            namespace="tenant-a",
        )
    retrieval = RetrievalEngine(
        store,
        FakeEmbedder(),
        RetrievalConfig(top_k=10, candidate_k=20, min_score=0.0),
    )
    mmr = Mock(side_effect=lambda results, *args, **kwargs: results)
    retrieval._mmr_diversify = mmr

    results = retrieval.retrieve(
        "fact", namespace="tenant-a", candidate_k=20, commit=False
    )

    assert mmr.called
    assert len(results) == 10


def test_retrieval_never_returns_more_than_top_k() -> None:
    from memory_agent.core.config import RetrievalConfig
    from memory_agent.core.retrieval import RetrievalEngine

    store = FakeStore()
    store.add_memory(MemoryRecord(content="fact one", namespace="tenant-a"), namespace="tenant-a")
    store.add_memory(MemoryRecord(content="fact two", namespace="tenant-a"), namespace="tenant-a")
    retrieval = RetrievalEngine(
        store,
        FakeEmbedder(),
        RetrievalConfig(top_k=1, candidate_k=2, min_score=0.0),
    )
    results = retrieval.retrieve(
        "fact",
        namespace="tenant-a",
        use_mmr=False,
        candidate_k=2,
        commit=False,
    )

    assert len(results) == 1


def test_direct_namespace_switch_resets_context_without_session() -> None:
    store = FakeStore()
    agent = MemoryAgent(
        store=store,
        embedder=FakeEmbedder(),
        retrieval=FakeRetrieval({}),
        trust_policy=FakeTrustPolicy(),
    )
    agent.perceive("I like Python", namespace="tenant-a", user_id="alice")
    agent.state.current_context = [MemoryRecord(content="old context")]

    result = agent.perceive("hello", namespace="tenant-b")

    assert result["lifecycle"]["namespace"] == "tenant-b"
    assert result["turn_count"] == 1
    assert agent.state.current_context == []
    agent.close()



def test_db_path_constructor_remains_legacy_compatible(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    agent = MemoryAgent(db_path=db_path)
    try:
        assert agent.db_path == db_path.resolve()
    finally:
        agent.close()
