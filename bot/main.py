# Purpose: Main AresBot subclass wiring all modules together
# Key Decisions: MacroManager owns economy/tech/upgrades/responses,
#   CombatManager owns army micro, QueenManager owns queen roles.
#   main.py is a thin orchestrator — no game logic lives here.
# Limitations: No ML-based engagement decisions, no neural parasite yet

from typing import Optional

from loguru import logger
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole

from sc2.data import Result

from bot.combat import CombatManager
from bot.managers.macro_manager import MacroManager
from bot.managers.queen_manager import QueenManager
from bot.managers.worker_defense_manager import WorkerDefenseManager
from bot.utilities.game_report import TelemetryRecorder


# ── Patch 5.0.16 compatibility shim ─────────────────────────────────────────
# New AIE maps emit units unknown to the installed python-sc2 enum
# (e.g. XelNagaTowerRangeIndicatorDummy == 2046). Unit.type_id does a strict
# UnitTypeId(value) lookup and raises, crashing _prepare_units before on_start.
# Fall back to NOTAUNIT for any unknown id so these dummy props are skipped
# instead of killing the bot.
_enum_lookup_cache: dict[int, UnitID] = {}


def _safe_type_id(self: Unit) -> UnitID:
    unit_type: int = self._proto.unit_type
    if unit_type in UnitID._value2member_map_:
        return UnitID(unit_type)
    if unit_type not in _enum_lookup_cache:
        logger.warning(
            f"Unknown unit type id {unit_type} "
            f"(patch-added dummy?), treating as NOTAUNIT"
        )
        _enum_lookup_cache[unit_type] = UnitID.NOTAUNIT
    return _enum_lookup_cache[unit_type]


Unit.type_id = property(_safe_type_id)

# ── Constants ────────────────────────────────────────────────────────────────
BEGIN_ATTACK_SUPPLY: float = 6.0

# Unit types that shouldn't be assigned ATTACKING role
# Queens get their own role system via QueenManager
IGNORE_ROLE_TYPES: set[UnitID] = {
    UnitID.EGG,
    UnitID.LARVA,
    UnitID.CREEPTUMORBURROWED,
    UnitID.CREEPTUMORQUEEN,
    UnitID.CREEPTUMOR,
    UnitID.MULE,
    UnitID.OVERLORD,
    UnitID.OVERSEER,
    UnitID.DRONE,
    UnitID.QUEEN,  # Queens managed by QueenManager with QUEEN_* roles
    UnitID.RAVAGERCOCOON,   # Morphing — will get ATTACKING when Ravager emerges
    UnitID.BROODLORDCOCOON, # Morphing — will get ATTACKING when Broodlord emerges
}


class Holdfast(AresBot):
    """Zerg B2GM Roach Ravager bot using ARES framework."""

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)
        self._combat_mgr: Optional[CombatManager] = None
        self._queen_mgr: Optional[QueenManager] = None
        self._macro_mgr: Optional[MacroManager] = None
        self._telemetry: Optional[TelemetryRecorder] = None
        self._worker_defense_mgr: Optional[WorkerDefenseManager] = None

    @property
    def attack_target(self) -> Point2:
        """Delegate to MacroManager for attack target calculation."""
        if self._macro_mgr is not None:
            return self._macro_mgr.attack_target
        return self.enemy_start_locations[0]

    async def on_start(self) -> None:
        """Called once at the start of the game."""
        await super().on_start()
        self._combat_mgr = CombatManager(self)
        self._queen_mgr = QueenManager(self)
        self._macro_mgr = MacroManager(self)
        self._telemetry = TelemetryRecorder(self)
        self._worker_defense_mgr = WorkerDefenseManager(self)

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)
        if not self.all_own_units:
            return

        # ── Macro (economy, production, tech, upgrades, responses) ──────────
        if self._macro_mgr is not None:
            self._macro_mgr.update()

        # ── Combat (army micro) ─────────────────────────────────────────────
        forces: Units = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)

        if self._macro_mgr is not None and not self._macro_mgr.commenced_attack:
            if self.get_total_supply(forces) >= BEGIN_ATTACK_SUPPLY:
                self._macro_mgr.start_attack()

        # Always run combat micro — even during build order, units need
        # to rally and defend. commenced_attack controls aggression,
        # not whether units get controlled at all.
        if forces and self._combat_mgr is not None:
            self._combat_mgr.step(forces)

        # ── Queen management (always run) ───────────────────────────────────
        if self._queen_mgr is not None:
            self._queen_mgr.update()

        # ── Worker defense (pull closest miners vs ling rush) ─────────────
        if self._worker_defense_mgr is not None:
            self._worker_defense_mgr.update()

    async def on_end(self, game_result: Result) -> None:
        """Write game-summary telemetry, then defer to ARES."""
        await super().on_end(game_result)
        if self._telemetry is not None:
            winner: str = "tie"
            if game_result == Result.Victory:
                winner = "self"
            elif game_result == Result.Defeat:
                winner = "opponent"
            self._telemetry.record_game_summary(
                result=game_result,
                game_length=self.time,
                winner=winner,
                map_name=self.game_info.map_name,
            )
            self._telemetry.flush()

    async def on_unit_created(self, unit: Unit) -> None:
        """Assign combat units to ATTACKING role, Queens to QueenManager."""
        await super().on_unit_created(unit)

        if unit.type_id in ALL_STRUCTURES:
            return
        if unit.type_id in WORKER_TYPES:
            return

        # Queens get their own role system via QueenManager
        if unit.type_id == UnitID.QUEEN:
            if self._queen_mgr is not None:
                self._queen_mgr.assign_new_queen(unit)
            return

        if unit.type_id in IGNORE_ROLE_TYPES:
            return

        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    async def on_unit_type_changed(self, unit: Unit, previous_type: UnitID) -> None:
        """Re-assign role when a unit morphs to a new type.

        Morphed units keep the same tag, so their role persists in ARES's
        tag-based role system. However, if the previous type was in
        IGNORE_ROLE_TYPES (e.g. Overlord → Overseer) and the new type is
        a combat unit, we need to assign ATTACKING role. Conversely, if a
        combat unit morphs into a non-combat form, we leave the role as-is
        since it will be cleaned up on death.
        """
        await super().on_unit_type_changed(unit, previous_type)

        # Combat morphs: ensure ATTACKING role for the new type
        COMBAT_MORPH_TYPES: set[UnitID] = {
            UnitID.RAVAGER,
            UnitID.BROODLORD,
            UnitID.LURKERMP,
        }
        if unit.type_id in COMBAT_MORPH_TYPES:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)

    # ── Micro (delegated to managers) ────────────────────────────────────────
    # CombatManager: bot/combat/combat.py
    # QueenManager:  bot/managers/queen_manager.py
    # MacroManager:  bot/managers/macro_manager.py


# Alias for run.py compatibility
MicroBot = Holdfast
