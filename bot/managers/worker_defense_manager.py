# Purpose: Pull the closest mining workers to defend against ARES-detected ling rushes
# Key Decisions: Dual gate — get_enemy_ling_rushed latch AND live near-base threat.
#   select_worker(force_close=True) picks miners closest to the incoming attack.
#   Melee chain: AOE dodge (avoidance grid) -> attack in-range target -> advance.
#   Massed melee drones beat lings; kiting loses to faster lings. DEFENDING role
#   + 5s grace prevents Mining recapture and GATHERING<->DEFENDING oscillation.
# Limitations: Ling rushes only (no roach/worker-rush pulls). Gas workers never
#   pulled (select_worker limitation). Memory-ghost threats hold workers idle.

import math

import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import AMove, KeepUnitSafe, ShootTargetInRange
from ares.consts import UnitRole
from cython_extensions import cy_center, cy_closest_to, cy_in_attack_range
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

# ── Constants ────────────────────────────────────────────────────────────────
# Live ground supply near a townhall required to pull workers
WORKER_DEFENCE_SUPPLY_THRESHOLD: float = 3.0
# 2 workers per point of threat supply (1 ling = 0.5 supply -> 2 workers per ling)
WORKERS_PER_THREAT_SUPPLY: float = 2.0
# Hard cap on pulled workers (16 fully saturates one base's mineral line)
MAX_WORKER_DEFENDERS: int = 12
# Never leave fewer than this many workers on the mineral line
MIN_GATHERING_RESERVE: int = 4
# Grace period (game seconds) after threats clear before workers resume mining
WORKER_DEFENCE_GRACE_PERIOD: float = 5.0


def worker_defender_count(threat_supply: float, gathering_count: int) -> int:
    """How many workers to pull: 2 per threat supply point, capped, reserve-guarded."""
    desired: int = math.ceil(threat_supply * WORKERS_PER_THREAT_SUPPLY)
    return max(
        0,
        min(
            MAX_WORKER_DEFENDERS,
            desired,
            gathering_count - MIN_GATHERING_RESERVE,
        ),
    )


class WorkerDefenseManager:
    """Pulls the closest mining workers to fight when a ling rush hits a base.

    QueenManager defends with queens; this manager adds worker mass only when
    the ARES ling-rush detector has latched AND enemies are actually near one
    of our townhalls. Workers return to mining 5s after threats clear.
    """

    def __init__(self, ai: AresBot) -> None:
        self.ai: AresBot = ai
        # Track when each defending worker last saw a threat (for grace period)
        self._defender_last_threat_time: dict[int, float] = {}

    def update(self) -> None:
        """Run every frame: pull, control, or release defending workers."""
        threats: Units = self.ai.mediator.get_main_ground_threats_near_townhall
        has_threat: bool = bool(threats) and (
            self.ai.get_total_supply(threats) >= WORKER_DEFENCE_SUPPLY_THRESHOLD
        )
        rush_latched: bool = self.ai.mediator.get_enemy_ling_rushed

        if rush_latched and has_threat:
            defending: Units = self._get_defenders()
            for worker in defending:
                self._defender_last_threat_time[worker.tag] = self.ai.time
            self._pull_workers(threats, len(defending))
            # Re-fetch: freshly pulled workers fight this same frame
            defending = self._get_defenders()
            self._control_defenders(defending, threats)
        else:
            defending = self._get_defenders()
            self._release_workers(defending)

        self._clean_stale(defending)

    def _get_defenders(self) -> Units:
        """Own workers currently in DEFENDING role."""
        return self.ai.mediator.get_units_from_role(
            role=UnitRole.DEFENDING, unit_type=self.ai.worker_type
        )

    def _pull_workers(self, threats: Units, current_defenders: int) -> None:
        """Pull the miners closest to the incoming attack."""
        gathering: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.GATHERING, unit_type=self.ai.worker_type
        )
        num_to_pull: int = (
            worker_defender_count(self.ai.get_total_supply(threats), gathering.amount)
            - current_defenders
        )
        if num_to_pull <= 0:
            return

        # select_worker(force_close=True) returns the GATHERING worker whose
        # patch is nearest to this position — exactly "closest to the attack".
        # Perf note: O(num_to_pull * workers), num_to_pull <= 12, rush-gated.
        threat_center: Point2 = Point2(cy_center(threats))
        for _ in range(num_to_pull):
            worker: Unit | None = self.ai.mediator.select_worker(
                target_position=threat_center, force_close=True
            )
            if worker is None:
                break
            self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.DEFENDING)
            self._defender_last_threat_time[worker.tag] = self.ai.time

    def _control_defenders(self, defending: Units, threats: Units) -> None:
        """Melee micro chain: AOE dodge → attack in-range target → advance."""
        if not defending:
            return
        # Memory ghosts aren't real targets — never issue attacks on them
        targets: Units = threats.filter(lambda u: not u.is_memory)
        if not targets:
            return

        avoid_grid: np.ndarray = self.ai.mediator.get_ground_avoidance_grid
        grid: np.ndarray = self.ai.mediator.get_ground_grid

        # Perf note: per-worker loop, <=12 defenders at rush scale — trivial.
        for worker in defending:
            maneuver: CombatManeuver = CombatManeuver()
            # AOE dodge (biles, storms) — first in every chain
            maneuver.add(KeepUnitSafe(unit=worker, grid=avoid_grid))
            if in_range := cy_in_attack_range(worker, targets):
                maneuver.add(ShootTargetInRange(unit=worker, targets=in_range))
            else:
                closest: Unit = cy_closest_to(worker.position, targets)
                maneuver.add(AMove(unit=worker, target=closest.position))
            self.ai.register_behavior(maneuver)

    def _release_workers(self, defending: Units) -> None:
        """Return defenders to mining once threats have been clear for 5s."""
        for worker in defending:
            last_threat: float = self._defender_last_threat_time.get(worker.tag, 0.0)
            if self.ai.time - last_threat > WORKER_DEFENCE_GRACE_PERIOD:
                self.ai.mediator.assign_role(tag=worker.tag, role=UnitRole.GATHERING)
                self._defender_last_threat_time.pop(worker.tag, None)

    def _clean_stale(self, defending: Units) -> None:
        """Drop threat timestamps for workers that died or changed role."""
        defender_tags: set[int] = {w.tag for w in defending}
        stale_tags: list[int] = [
            t for t in self._defender_last_threat_time if t not in defender_tags
        ]
        for tag in stale_tags:
            del self._defender_last_threat_time[tag]
