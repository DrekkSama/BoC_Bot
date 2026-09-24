# Purpose: Core army control — squad dispatch, engagement decisions, micro entry point
# Key Decisions: ARES squad system, can_win_fight for engagement, hysteresis for stability.
#   Group behaviors (PathGroupToTarget, AMoveGroup) for map movement to keep army together.
#   Individual behaviors for combat micro when enemies are near.
# Limitations: No neural parasite, no flying squad separation yet

from itertools import cycle
from typing import Optional

import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.group import AMoveGroup, PathGroupToTarget
from ares.consts import (
    ALL_STRUCTURES,
    LOSS_MARGINAL_OR_WORSE,
    VICTORY_MARGINAL_OR_BETTER,
    UnitRole,
    UnitTreeQueryType,
)
from ares.managers.squad_manager import UnitSquad
from cython_extensions import cy_center, cy_closest_to, cy_distance_to
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from bot.combat.unit_micro import UnitMicro

# ── Constants ────────────────────────────────────────────────────────────────
ATTACKING_SQUAD_RADIUS: float = 9.0
# Distance (squared) to consider enemies "near" a squad for group vs individual
ENEMY_NEAR_SQUAD_DISTANCE_SQ: float = 225.0  # 15.0^2

# Unit types to ignore when counting enemies
COMMON_UNIT_IGNORE_TYPES: set[UnitID] = {
    UnitID.EGG,
    UnitID.LARVA,
    UnitID.CREEPTUMOR,
    UnitID.CREEPTUMORQUEEN,
    UnitID.CREEPTUMORBURROWED,
}


class CombatManager:
    """Orchestrates squad-based micro dispatch and engagement tracking."""

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai
        self._micro: UnitMicro = UnitMicro(ai)
        # Per-squad engagement tracking: squad_id -> currently_engaged
        self._squad_engaged: dict[str, bool] = {}
        # Aggressive mode: True = push toward enemy, False = rally at home
        self.aggressive: bool = False
        # Expansion cycling for late-game attack target
        self._expansions_generator: Optional[cycle] = None
        self._current_base_target: Point2 = ai.enemy_start_locations[0]

    @property
    def attack_target(self) -> Point2:
        """Calculate the best attack target for the army."""
        ai = self._ai
        # Priority: closest enemy structure to enemy natural
        enemy_structures: Units = ai.enemy_structures.filter(
            lambda s: s.type_id
            not in {
                UnitID.CREEPTUMOR,
                UnitID.CREEPTUMORQUEEN,
                UnitID.CREEPTUMORBURROWED,
                UnitID.NYDUSCANAL,
            }
        )
        if enemy_structures:
            return cy_closest_to(ai.mediator.get_enemy_nat, enemy_structures).position

        # Early game: head to enemy spawn
        if ai.time < 240.0:
            return ai.enemy_start_locations[0]

        # Late game: search expansion locations
        if self._expansions_generator is None:
            self._expansions_generator = cycle(ai.expansion_locations_list)

        if ai.is_visible(self._current_base_target):
            self._current_base_target = next(self._expansions_generator)
        return self._current_base_target

    @property
    def rally_point(self) -> Point2:
        """Rally point for the army when not aggressive."""
        ai = self._ai
        # Under attack: rally at the tracked threat position (ground or air)
        defense_mgr = ai._defense_mgr
        if defense_mgr is not None and defense_mgr.under_attack:
            if defense_mgr.threat_position is not None:
                return defense_mgr.threat_position

        # Otherwise: ground threat near a townhall takes priority
        if threats := ai.mediator.get_main_ground_threats_near_townhall:
            return Point2(cy_center(threats))

        # Default: natural toward map center
        if len(ai.townhalls.ready) >= 2:
            return ai.mediator.get_own_nat.towards(ai.game_info.map_center, 5.0)
        return ai.main_base_ramp.top_center.towards(ai.start_location, 3.0)

    def step(self, forces: Units) -> None:
        """Main micro dispatch using ARES squad system.

        Uses group behaviors (PathGroupToTarget, AMoveGroup) when the squad
        is moving across the map with no enemies nearby — this keeps the
        army together instead of stringing out. Falls through to per-unit
        micro when enemies are close enough to engage.
        """
        ai = self._ai

        # ── Check if we should be aggressive ─────────────────────────────
        self._check_aggressive(forces)

        # ── Get squads ───────────────────────────────────────────────────
        squads: list[UnitSquad] = ai.mediator.get_squads(
            role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS
        )
        if not squads:
            return

        target: Point2 = self.attack_target if self.aggressive else self.rally_point

        for squad in squads:
            squad_units: list[Unit] = squad.squad_units
            if not squad_units:
                continue

            squad_tags: set[int] = squad.tags
            squad_position: Point2 = squad.squad_position

            # ── Get nearby enemies for this squad ─────────────────────────
            all_close_enemy: Units = ai.mediator.get_units_in_range(
                start_points=[squad_position],
                distances=18.5,
                query_tree=UnitTreeQueryType.AllEnemy,
            )[0].filter(lambda u: u.type_id not in COMMON_UNIT_IGNORE_TYPES)

            # ── Group movement when no enemies nearby ─────────────────────
            # If no enemies are close to the squad, move as a group to keep
            # the army together. This prevents units from stringing out
            # across the map since group behaviors issue one action for all.
            if not all_close_enemy:
                group_maneuver: CombatManeuver = CombatManeuver()
                grid: np.ndarray = ai.mediator.get_ground_grid
                group_maneuver.add(
                    PathGroupToTarget(
                        start=squad_position,
                        group=squad_units,
                        group_tags=squad_tags,
                        grid=grid,
                        target=target,
                    )
                )
                group_maneuver.add(
                    AMoveGroup(
                        group=squad_units,
                        group_tags=squad_tags,
                        target=target,
                    )
                )
                ai.register_behavior(group_maneuver)
                continue

            # ── Engagement tracking ───────────────────────────────────────
            self._track_squad_engagement(squad, all_close_enemy, forces)
            can_engage: bool = self._squad_engaged.get(squad.squad_id, False)

            # ── Per-unit distance queries ─────────────────────────────────
            near_enemy: dict[int, Units] = ai.mediator.get_units_in_range(
                start_points=squad_units,
                distances=15,
                query_tree=UnitTreeQueryType.EnemyGround,
                return_as_dict=True,
            )

            # ── Dispatch micro per unit type ──────────────────────────────
            self._micro.execute(
                squad_units=squad_units,
                all_close_enemy=all_close_enemy,
                near_enemy=near_enemy,
                can_engage=can_engage,
                aggressive=self.aggressive,
                target=target,
                attack_target=self.attack_target,
            )

    def _check_aggressive(self, forces: Units) -> None:
        """Determine if the army should be aggressive or defensive."""
        ai = self._ai

        # Under attack: always defend — cancel any ongoing attack
        defense_mgr = ai._defense_mgr
        if defense_mgr is not None and defense_mgr.under_attack:
            if self.aggressive:
                self.aggressive = False
                self._emit_transition("attack", "defend")
            return

        # If already aggressive, stay aggressive unless losing badly
        if self.aggressive:
            if ai.supply_army < 10:
                self.aggressive = False
                self._emit_transition("attack", "defend")
            return

        # Don't go aggressive too early
        if ai.time < 300.0:
            return

        # Go aggressive if we have enough army supply
        if ai.supply_army >= 40:
            self.aggressive = True
            # Reset engagement trackers when switching to aggressive
            self._squad_engaged = {k: False for k in self._squad_engaged}
            self._emit_transition("defend", "attack")

    def _combat_sim_snapshot(self) -> dict:
        """Snapshot of squad engagement state from can_win_fight() sim results."""
        engaged_ids = [sid for sid, engaged in self._squad_engaged.items() if engaged]
        return {
            "any_squad_engaged": len(engaged_ids) > 0,
            "engaged_squad_ids": engaged_ids,
            "engaged_count": len(engaged_ids),
        }

    def _emit_transition(self, from_state: str, to_state: str) -> None:
        """Emit a state-transition telemetry event on aggressive flip."""
        ai = self._ai
        if ai._telemetry is not None:
            ai._telemetry.record_state_transition(
                from_state=from_state,
                to_state=to_state,
                game_time=ai.time,
                army_supply=ai.supply_army,
                combat_sim_state=self._combat_sim_snapshot(),
            )

    def _track_squad_engagement(
        self,
        squad: UnitSquad,
        close_enemy: Units,
        all_attackers: Units,
    ) -> None:
        """Track whether a squad should engage based on combat simulation."""
        ai = self._ai
        squad_id: str = squad.squad_id

        if squad_id not in self._squad_engaged:
            self._squad_engaged[squad_id] = False

        # No enemy nearby: don't engage
        if not close_enemy:
            self._squad_engaged[squad_id] = False
            return

        # If defending and enemy is near our base, always engage
        # (defense_mgr flag covers ground + flying threats)
        if not self.aggressive:
            defense_mgr = ai._defense_mgr
            under_attack = (
                defense_mgr is not None and defense_mgr.under_attack
            ) or bool(ai.mediator.get_main_ground_threats_near_townhall)
            if under_attack:
                self._squad_engaged[squad_id] = True
                return

        # Use combat sim to decide
        enemy_units_only: list[Unit] = [
            u for u in close_enemy if u.type_id not in ALL_STRUCTURES
        ]
        if not enemy_units_only:
            self._squad_engaged[squad_id] = True
            return

        fight_result = ai.mediator.can_win_fight(
            own_units=all_attackers, enemy_units=enemy_units_only
        )

        # Currently engaged: check if we should disengage
        if self._squad_engaged[squad_id]:
            if fight_result in LOSS_MARGINAL_OR_WORSE:
                self._squad_engaged[squad_id] = False
        # Not engaged: check if we can engage
        else:
            if fight_result in VICTORY_MARGINAL_OR_BETTER:
                self._squad_engaged[squad_id] = True
