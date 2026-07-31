"""Structured per-agent working, episodic, and trust memory."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from game.security import contains_prompt_injection

_BOTTLE = r"(red|blue|green|yellow)"


@dataclass
class ClaimMemory:
    memory_id: str
    source: str
    round: int
    game_type: str
    raw_text: str
    fact_key: str | None
    value: str | None
    confidence: float
    verified: bool = False
    truth: bool | None = None
    actual_value: str | None = None
    injection_flagged: bool = False


@dataclass
class EpisodeMemory:
    round: int
    game_type: str
    results: dict[str, Any]


class AgentMemory:
    def __init__(
        self,
        owner_id: str,
        *,
        initial_trust: float = 0.5,
        truth_reward: float = 0.1,
        lie_penalty: float = 0.15,
    ) -> None:
        if not 0 <= initial_trust <= 1:
            raise ValueError("initial_trust must be between 0 and 1")
        if truth_reward < 0 or lie_penalty < 0:
            raise ValueError("trust adjustments cannot be negative")
        self.owner_id = owner_id
        self.initial_trust = initial_trust
        self.truth_reward = truth_reward
        self.lie_penalty = lie_penalty
        self.working: dict[str, Any] = {}
        self.claims: list[ClaimMemory] = []
        self.episodes: list[EpisodeMemory] = []
        self.trust: dict[str, float] = {}

    def trust_for(self, source: str) -> float:
        return self.trust.get(source, self.initial_trust)

    def set_working(self, **values: Any) -> None:
        self.working.update(values)

    def clear_working(self) -> None:
        self.working.clear()

    def ingest_message(
        self,
        *,
        source: str,
        text: str,
        round: int,
        game_type: str,
    ) -> ClaimMemory:
        lowered = text.lower()
        fact_key = None
        value = None
        match = re.search(
            rf"\b(?:picked|chose|took)\s+{_BOTTLE}\b",
            lowered,
        )
        if match:
            fact_key = f"bottle_choice:{source}"
            value = match.group(1).title()
        if fact_key is None:
            match = re.search(
                rf"\bpoison(?:ed)?(?:\s+bottle)?"
                rf"(?:\s+is|\s+was|:)?\s+{_BOTTLE}\b",
                lowered,
            )
            if match:
                fact_key = "poisoned_bottle"
                value = match.group(1).title()
        if fact_key is None:
            match = re.search(
                rf"\b{_BOTTLE}\s+(?:is|was)\s+poison(?:ed)?\b",
                lowered,
            )
            if match:
                fact_key = "poisoned_bottle"
                value = match.group(1).title()
        if fact_key is None:
            match = re.search(
                r"\b(?:word|answer)\s+(?:is|was|:)\s*"
                r"([a-z][a-z'-]*)\b",
                lowered,
            )
            if match:
                fact_key = "secret_word"
                value = match.group(1).lower()

        injection = contains_prompt_injection(text)
        base_confidence = self.trust_for(source)
        claim = ClaimMemory(
            memory_id=uuid.uuid4().hex,
            source=source,
            round=round,
            game_type=game_type,
            raw_text=text,
            fact_key=fact_key,
            value=value,
            confidence=min(base_confidence, 0.2) if injection else base_confidence,
            injection_flagged=injection,
        )
        self.claims.append(claim)
        self.claims = self.claims[-100:]
        return claim

    def add_episode(
        self,
        *,
        round: int,
        game_type: str,
        results: dict[str, Any],
    ) -> None:
        self.episodes.append(EpisodeMemory(round, game_type, results))
        self.episodes = self.episodes[-20:]

    def verify_round(
        self,
        *,
        round: int,
        game_type: str,
        results: dict[str, Any],
    ) -> list[ClaimMemory]:
        verified = []
        for claim in self.claims:
            if claim.verified or claim.round != round or not claim.fact_key:
                continue
            actual = None
            if (
                game_type == "poison_bottle"
                and claim.fact_key.startswith("bottle_choice:")
            ):
                player = claim.fact_key.split(":", 1)[1]
                actual = (results.get("choices") or {}).get(player)
            elif (
                game_type == "poison_bottle"
                and claim.fact_key == "poisoned_bottle"
            ):
                actual = results.get("poisoned_bottle")
            elif (
                game_type == "guess_the_word"
                and claim.fact_key == "secret_word"
            ):
                actual = results.get("word")
            if actual is None:
                continue
            claim.verified = True
            claim.actual_value = str(actual)
            claim.truth = str(claim.value).lower() == str(actual).lower()
            current = self.trust_for(claim.source)
            change = self.truth_reward if claim.truth else -self.lie_penalty
            self.trust[claim.source] = round_score(current + change)
            verified.append(claim)
        return verified

    def retrieve(
        self,
        *,
        game_type: str | None = None,
        source: str | None = None,
        limit: int = 8,
    ) -> list[ClaimMemory]:
        candidates = [
            claim
            for claim in self.claims
            if (game_type is None or claim.game_type == game_type)
            and (source is None or claim.source == source)
        ]
        candidates.sort(
            key=lambda claim: (
                claim.verified,
                claim.round,
                claim.confidence,
            ),
            reverse=True,
        )
        return candidates[:limit]

    def prompt_summary(self, game_type: str | None = None) -> str:
        lines = []
        if self.trust:
            trust = ", ".join(
                f"{source}={score:.2f}"
                for source, score in sorted(self.trust.items())
            )
            lines.append(f"Trust estimates: {trust}")
        for claim in self.retrieve(game_type=game_type, limit=6):
            status = (
                f"verified {'true' if claim.truth else 'false'}"
                if claim.verified
                else "unverified"
            )
            flag = " [possible prompt injection]" if claim.injection_flagged else ""
            lines.append(
                f"R{claim.round} {claim.source} claimed: "
                f"{claim.raw_text!r} ({status}, confidence "
                f"{claim.confidence:.2f}){flag}"
            )
        return "\n".join(lines) if lines else "(no relevant long-term memory)"

    def snapshot(self) -> dict:
        return {
            "owner_id": self.owner_id,
            "trust_profile": {
                "initial": self.initial_trust,
                "truth_reward": self.truth_reward,
                "lie_penalty": self.lie_penalty,
            },
            "working": dict(self.working),
            "claims": [asdict(claim) for claim in self.claims],
            "episodes": [asdict(episode) for episode in self.episodes],
            "trust": dict(self.trust),
        }


def round_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)
