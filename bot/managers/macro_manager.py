# Purpose: Centralized macro management — workers, supply, gas, queens,
#   spawning, tech, upgrades, expansions, and threat responses.
# Key Decisions: MacroPlan runs ONE behavior per frame (short-circuits on
#   first True), so plan order encodes priority:
#   supply -> workers -> gas -> tech -> army -> upgrades -> expansions.
#   Tech precedes army so a missing Roach Warren is built before
#   SpawnController starves on it.
#   Rush response is a three-state machine (see RushState): SECURING holds
#   all non-queen spends until 2 defense queens are queued; DEFENDING runs
#   a ling-only profile (RUSH_DEFENSE_COMP, surplus-bank drones, frozen
#   gas/expansions, held tech/upgrades, pool rebuilt if lost); clearing
#   requires RUSH_CLEAR_GRACE seconds with no enemy combat near bases and
#   latches permanently (the ARES flag never un-fires).
#   Zerg-specific gates: drone ceiling is infrastructure-derived
#   (bases x 22, capped 80) — no worker-priority or mineral-bank gates.
#   The MacroPlan ladder runs army ABOVE workers: SpawnController
#   (affordable-first via prioritize_affordable_units) claims larvae
#   whenever the comp is affordable and tech-ready, and BuildWorkers gets
#   the frames the army declines. Halting expansions (rush states, enemies
#   near bases) freezes the drone ceiling automatically — drones
#   self-constrain with zero drone-specific logic.
#   Reactive tech (Hydralisk Den) only on air detection; gated upgrades
#   only when their prerequisite building exists.
# Limitations: No nydus support, no dynamic composition beyond air/rush.

from enum import Enum, auto

from ares import AresBot
from ares.behaviors.macro import (
    AutoSupply,
    BuildWorkers,
    ExpansionController,
    GasBuildingController,
    MacroPlan,
    Mining,
    SpawnController,
    TechUp,
    UpgradeController,
)
from cython_extensions import cy_closest_to, cy_unit_pending
from loguru import logger
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.ids.upgrade_id import UpgradeId as UpgradeID
from sc2.position import Point2
from sc2.units import Units

from bot.compositions import (
    get_army_comp,
    prioritize_affordable_units,
    should_morph,
    strip_morph_units,
)
from bot.managers.queen_manager import HARMLESS_THREAT_TYPES

# ── Constants ────────────────────────────────────────────────────────────────
BEGIN_ATTACK_SUPPLY: float = 6.0
MID_GAME_TIME: float = 360.0
NATURAL_TIMING_THRESHOLD: float = 210.0  # 3:30

# 16 mineral patches + 6 gas (2 geysers x 3) = 22 ideal per base;
# 20 is the practical early-game saturation threshold.
DRONES_PER_SATURATED_BASE: int = 20
DRONES_PER_FULLY_SATURATED_BASE: int = 22

# Rush defense floor: queens back the defense with transfuse + DPS
RUSH_DEFENSE_QUEENS: int = 2

# Seconds with no enemy combat units near our bases before the rush
# response clears. Grace window prevents oscillation from ling pokes.
RUSH_CLEAR_GRACE: float = 30.0


class RushState(Enum):
    """Rush response lifecycle.

    SECURING: rush latched, defense queens not yet queued. Everything
        except queens, supply, and threat spines is held.
    DEFENDING: queens secured. Ling-only defense profile; the 2-queen
        floor and all defense holds stay until the rush clears.
    NONE: no rush response. Terminal after a cleared rush — _rush_over
        latches because the ARES flag is permanent.
    """

    NONE = auto()
    SECURING = auto()
    DEFENDING = auto()


# Saturation: full mineral+gas saturation per base. This is the ONLY
# drone ceiling — infrastructure-derived, recomputed each frame. Halting
# expansions (rush SECURING, enemies near bases) freezes it automatically:
# drones self-constrain without any worker-priority gate.
DRONES_PER_FULLY_SATURATED_BASE: int = 22
DRONE_HARD_CAP: int = 80

# Fraction of full saturation required before taking another base
EXPANSION_SATURATION_THRESHOLD: float = 0.75

# Drone count at which SpawnController switches to free-flow spending
# (economy can sustain production without proportion management)
FREEFLOW_DRONE_THRESHOLD: int = 60

# Gas scaling: (min_drones, gas_per_base, max_pending_geysers)
GAS_PHASES: list[tuple[int, float, int]] = [
    (0, 1.0, 1),  # 1 geyser per base: ling speed + Lair
    (36, 1.5, 2),  # 1.5 per base: upgrades + ravagers
    (56, 2.0, 2),  # 2 per base: full hive-tech saturation
]

# Upgrade priority order (lower = higher priority)
UPGRADE_PRIORITY: list[UpgradeID] = [
    UpgradeID.ZERGLINGMOVEMENTSPEED,
    UpgradeID.ZERGMELEEWEAPONSLEVEL1,
    UpgradeID.ZERGMISSILEWEAPONSLEVEL1,
    UpgradeID.EVOLVEGROOVEDSPINES,
    UpgradeID.ZERGMELEEWEAPONSLEVEL2,
    UpgradeID.ZERGMISSILEWEAPONSLEVEL2,
    UpgradeID.ZERGMELEEWEAPONSLEVEL3,
    UpgradeID.ZERGMISSILEWEAPONSLEVEL3,
    UpgradeID.CENTRIFICALHOOKS,
    UpgradeID.OVERLORDSPEED,
]

# Enemy structures that signal air tech
AIR_STRUCTURES: set[UnitID] = {
    UnitID.FUSIONCORE,
    UnitID.STARGATE,
    UnitID.STARPORTTECHLAB,
    UnitID.FLEETBEACON,
}

# Combat air units that warrant Hydralisk production
# (non-combat air excluded: Overlord, Overseer, Observer, WarpPrism)
AIR_UNIT_TYPES: set[UnitID] = {
    # Protoss
    UnitID.VOIDRAY,
    UnitID.CARRIER,
    UnitID.ORACLE,
    UnitID.PHOENIX,
    UnitID.TEMPEST,
    UnitID.MOTHERSHIP,
    # Terran
    UnitID.MEDIVAC,
    UnitID.VIKINGFIGHTER,
    UnitID.VIKINGASSAULT,
    UnitID.BANSHEE,
    UnitID.RAVEN,
    UnitID.BATTLECRUISER,
    UnitID.LIBERATOR,
    # Zerg
    UnitID.MUTALISK,
    UnitID.CORRUPTOR,
    UnitID.BROODLORD,
}

# Light units counted for mass-light threat detection
LIGHT_UNIT_TYPES: set[UnitID] = {
    UnitID.ZERGLING,
    UnitID.ZEALOT,
    UnitID.ADEPT,
    UnitID.MARINE,
}

# Threat response counts
SAFETY_ROACH_COUNT: int = 5
EMERGENCY_SPINE_COUNT: int = 2
MINERAL_LINE_SPINE_COUNT: int = 1


class MacroManager:
    """Owns all macro decisions: economy, production, tech, upgrades, responses."""

    def __init__(self, ai: AresBot) -> None:
        self._ai: AresBot = ai
        self._commenced_attack: bool = False
        self._air_signs_detected: bool = False  # Latches True once air threat seen
        # Rush response state machine (see RushState docstring)
        self._rush_state: RushState = RushState.NONE
        # Latches True once the rush reaction fired (build order aborted)
        self._rush_reacted: bool = False
        # Latches True once a rush has been cleared — prevents re-entry,
        # since ARES's rush flag is permanent
        self._rush_over: bool = False
        # Last time enemy combat units were seen near our bases
        self._last_threat_near_base_time: float = -1.0
        # RECOVERY state (derived each frame in _do_macro_plan): economy
        # unsaturated and no active threat — drone rebuild, no investment
        self._recovering: bool = False
        self._threats: dict[str, bool] = {
            "no_natural": False,
            "timing_push": False,
            "air_signs": False,
            "proxy_signs": False,
            "mass_light": False,
            "cannon_rush": False,
            "rush_detected": False,
        }

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def attack_target(self) -> Point2:
        """Determine where the army should attack."""
        if self._ai.enemy_structures:
            return cy_closest_to(
                self._ai.start_location, self._ai.enemy_structures
            ).position
        if self._ai.time < 240.0:
            return self._ai.enemy_start_locations[0]
        # Late game: search expansions
        for expand_pos in self._ai.expansion_locations_list:
            if not self._ai.is_visible(expand_pos):
                return expand_pos
        return self._ai.enemy_start_locations[0]

    @property
    def commenced_attack(self) -> bool:
        return self._commenced_attack

    def start_attack(self) -> None:
        """One-way gate: mark that the army should begin attacking."""
        self._commenced_attack = True

    @property
    def threats(self) -> dict[str, bool]:
        return self._threats

    def update(self) -> None:
        """Run every frame: economy, production, tech, upgrades, responses."""
        self._ai.register_behavior(Mining(mineral_boost=False))
        self._morph_units_standalone()

        # Latch the rush reaction once: abort the opening build so dynamic
        # macro (defense queens, spines, army) can take over. The ARES
        # rush flag is permanent, so this must fire exactly once.
        if (
            not self._rush_reacted
            and self._ai.mediator.get_did_enemy_rush
            and not self._ai.build_order_runner.build_completed
        ):
            self._rush_reacted = True
            self._threats["rush_detected"] = True
            self._ai.build_order_runner.set_build_completed()
            logger.info(
                f"{self._ai.time_formatted}: Rush detected — aborting "
                f"opening build order, switching to rush defense"
            )

        self._update_rush_state()

        # Stalled build with a big bank: force-complete so dynamic macro
        # can take over
        if not self._ai.build_order_runner.build_completed:
            if self._ai.minerals >= 1500:
                self._ai.build_order_runner.set_build_completed()
            else:
                return

        self._assess_threats()
        self._do_macro_plan()
        self._respond_to_threats()

    # ── Rush State Machine ──────────────────────────────────────────────────

    def _update_rush_state(self) -> None:
        """Advance the rush response state machine. Call once per frame.

        Transitions (one-way):
            NONE -> SECURING : rush detected (_rush_reacted latched)
            SECURING -> DEFENDING : 2-queen floor met (incl. pending)
            DEFENDING -> NONE : RUSH_CLEAR_GRACE seconds with no enemy
                combat units near our bases; latches _rush_over
        """
        ai = self._ai

        # Cleared rushes never re-trigger — the ARES flag is permanent
        if self._rush_over:
            return

        if self._enemy_combat_near_bases():
            self._last_threat_near_base_time = ai.time

        if self._rush_state is RushState.NONE:
            if self._rush_reacted:
                self._rush_state = RushState.SECURING

        elif self._rush_state is RushState.SECURING:
            if self._defense_queens_secured():
                self._rush_state = RushState.DEFENDING
                logger.info(
                    f"{ai.time_formatted}: Defense queens secured — "
                    f"rush defense profile active (lings + surplus drones)"
                )

        elif self._rush_state is RushState.DEFENDING:
            if (
                ai.time - self._last_threat_near_base_time >= RUSH_CLEAR_GRACE
                and not self._enemy_combat_near_bases()
            ):
                self._rush_state = RushState.NONE
                self._rush_over = True
                logger.info(
                    f"{ai.time_formatted}: Rush cleared — returning to normal macro"
                )

    def _defense_queens_secured(self) -> bool:
        """True when the 2-queen defense floor is met (incl. pending)."""
        return (
            len(self._ai.mediator.get_own_army_dict[UnitID.QUEEN])
            + cy_unit_pending(self._ai, UnitID.QUEEN)
            >= RUSH_DEFENSE_QUEENS
        )

    def _enemy_combat_near_bases(self) -> bool:
        """True if enemy combat units are near any of our townhalls.

        Uses ARES near-base tracking (ground + flying) with harmless types
        (scouts, observers) filtered out.

        Perf: O(tracked) set union + one tags_in lookup.
        """
        ai = self._ai
        ground: dict[int, set[int]] = ai.mediator.get_ground_enemy_near_bases
        flying: dict[int, set[int]] = ai.mediator.get_flying_enemy_near_bases
        all_tags: set[int] = set()
        for enemy_tags in ground.values():
            all_tags.update(enemy_tags)
        for enemy_tags in flying.values():
            all_tags.update(enemy_tags)
        if not all_tags:
            return False
        threats: Units = ai.enemy_units.tags_in(all_tags)
        return bool(threats.filter(lambda u: u.type_id not in HARMLESS_THREAT_TYPES))

    def _rush_state_securing(self) -> bool:
        """Consumer read: SECURING state — hold all non-queen spends."""
        return self._rush_state is RushState.SECURING

    def _rush_active(self) -> bool:
        """Consumer read: rush defense profile active (SECURING or DEFENDING)."""
        return self._rush_state is not RushState.NONE

    # ── Macro Plan ──────────────────────────────────────────────────────────

    def _do_macro_plan(self) -> None:
        """Build the main MacroPlan: supply, tech, army, workers, upgrades.

        MacroPlan executes ONE behavior per frame (short-circuits on first
        True), so add order is priority order:
        supply -> tech -> ARMY -> workers -> upgrades -> expansions.

        Ladder design (gate-free):
        - The army runs ABOVE workers. SpawnController (affordable-first)
          claims the frame whenever the comp is affordable and tech-ready,
          and declines those frames when it is not. Those declined frames
          fall through to BuildWorkers — "army alongside drones" emerges
          from affordability, not from a tie-breaker gate.
        - The drone ceiling is infrastructure-derived: min(80, bases x 22).
          Halting expansions (rush states, enemies near bases) freezes
          the ceiling — drones self-constrain with zero drone logic.
        - Pre-pool or pool-sniped: no tech-ready units exist, so every
          frame falls through to drones — a pure drone phase with no gate.

        RECOVERY state (post-threat economy rebuild):
        - Entry (all derived, no new thresholds): bases unsaturated
          (workers < EXPANSION_SATURATION_THRESHOLD x ceiling — the same
          concept that gates expansion) AND not under attack
          (DefenseManager flag) AND no rush response active.
        - Effect: the workers rung moves ABOVE army — every larva
          rebuilds the economy — and ALL investment halts (tech,
          upgrades, gas, expansions, proactive + reactive tech). The
          queen floor raises to 4 (2 x RUSH_DEFENSE_QUEENS, also the
          QueenManager inject-activation threshold): a larva-free
          standing defense that absorbs follow-ups until the
          under_attack flip restores army-first order.
        - Exit: saturation reached OR under_attack flips — no latches
          that could contradict the rush states.

        Rush holds:
        - SECURING: everything except queens/supply/spines is held so the
          150m queen bank is never sniped (a queen costs 150m; drones,
          warren, gas, and expansion all cost less and would drain it).
        - DEFENDING (rush_active): ling-only profile — rush comp, frozen
          gas/expansions, held tech and upgrades. Lings are always
          affordable so the ladder gives defense priority emergently. The
          Spawning Pool is rebuilt first-class (the Warren TechUp that
          would normally chain it is itself held).
        """
        macro_plan: MacroPlan = MacroPlan()
        structure_dict: dict = self._ai.mediator.get_own_structures_dict

        macro_plan.add(AutoSupply(base_location=self._ai.start_location))

        rush_securing: bool = self._rush_state_securing()
        rush_active: bool = self._rush_active()

        # RECOVERY state — derived only from existing concepts: the
        # saturation fraction that gates expansion, DefenseManager's
        # hysteresis'd under_attack flag (same read combat.py uses), and
        # the rush states. Stored so _queen_target / _expansion_targets /
        # _respond_to_threats read the same-frame value.
        defense_mgr = getattr(self._ai, "_defense_mgr", None)
        under_attack: bool = defense_mgr is not None and defense_mgr.under_attack
        total_bases: int = len(self._ai.townhalls.ready) + self._ai.structure_pending(
            self._ai.base_townhall_type
        )
        worker_ceiling: int = min(
            DRONE_HARD_CAP, total_bases * DRONES_PER_FULLY_SATURATED_BASE
        )
        self._recovering: bool = (
            not rush_active
            and not under_attack
            and self._ai.supply_workers
            < EXPANSION_SATURATION_THRESHOLD * worker_ceiling
        )
        # Investment (tech/upgrades/gas/expansion) only on a standing
        # economy: rush defense and recovery both claim the whole bank
        # for units and drones instead.
        invest_allowed: bool = not rush_active and not self._recovering

        # Gas (frozen during rush — see _gas_targets; held during recovery)
        if invest_allowed:
            target_gas, max_pending_gas = self._gas_targets()
            macro_plan.add(
                GasBuildingController(
                    to_count=target_gas,
                    max_pending=max_pending_gas,
                )
            )

        # Roach Warren — must precede army so SpawnController never starves
        # on missing tech. Held during rush and recovery (investment):
        # the rush comp needs no warren, and its 150m retries would drain
        # the defense or the drone rebuild.
        if invest_allowed:
            macro_plan.add(
                TechUp(
                    desired_tech=UnitID.ROACHWARREN,
                    base_location=self._ai.start_location,
                )
            )

        # Rush DEFENDING: rebuild the pool first-class if lost — without it
        # there is no ling production for the rest of the rush
        if (
            rush_active
            and not rush_securing
            and not self._ai.structures(UnitID.SPAWNINGPOOL).exists
            and not self._ai.already_pending(UnitID.SPAWNINGPOOL)
        ):
            macro_plan.add(
                TechUp(
                    desired_tech=UnitID.SPAWNINGPOOL,
                    base_location=self._ai.start_location,
                )
            )

        # Lair / Hive — held during rush and recovery (investment)
        lair_tech: bool = (
            len(structure_dict[UnitID.LAIR]) > 0 or len(structure_dict[UnitID.HIVE]) > 0
        )
        if (
            invest_allowed
            and self._ai.vespene >= 100
            and not lair_tech
            and len(self._ai.mediator.get_own_army_dict[UnitID.QUEEN]) >= 4
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.LAIR, base_location=self._ai.start_location)
            )

        if (
            invest_allowed
            and self._ai.supply_used > 170.0
            and len(structure_dict.get(UnitID.HIVE, [])) == 0
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.HIVE, base_location=self._ai.start_location)
            )

        # ARMY + WORKERS — one claimant per frame; the order flips with
        # the economy state. Morph units stripped from the army comp
        # (SpawnController can't morph combat units;
        # _morph_units_standalone handles those). SECURING holds both so
        # minerals reach the queens first.
        if not rush_securing:
            full_army_comp: dict[UnitID, dict] = get_army_comp(
                self._ai.time,
                air_threat=self._threats.get("air_signs", False),
                drone_count=self._ai.supply_workers,
                rush_active=rush_active,
            )
            army_comp: dict[UnitID, dict] = strip_morph_units(full_army_comp)
            freeflow: bool = self._ai.supply_workers >= FREEFLOW_DRONE_THRESHOLD
            if not freeflow:
                # SpawnController hard-breaks on an unaffordable priority-1
                # unit; lead with an affordable one to avoid total stalls
                army_comp = prioritize_affordable_units(
                    army_comp, self._ai.minerals, self._ai.vespene
                )
            army_behavior: SpawnController = SpawnController(
                army_comp, freeflow_mode=freeflow
            )
            worker_behavior: BuildWorkers = BuildWorkers(to_count=worker_ceiling)
            if self._recovering:
                # RECOVERY: every larva rebuilds the economy; the army
                # gets only the frames workers decline. Defense during
                # the window rides the raised queen floor + spines, and
                # an under_attack flip restores army-first instantly.
                macro_plan.add(worker_behavior)
                macro_plan.add(army_behavior)
            else:
                # Normal: army above workers — drones take only the
                # frames the army declines (gate-free arbitration).
                macro_plan.add(army_behavior)
                macro_plan.add(worker_behavior)

        if invest_allowed:
            self._build_proactive_tech()

        # Queens — train() directly, not plan-ordered: they must claim the
        # bank every frame regardless of what the plan short-circuits on
        self._produce_queens()

        # Upgrades — held during rush and recovery (UpgradeController
        # auto-builds required tech, e.g. Evolution Chamber, which would
        # drain the defense or the drone rebuild)
        if self._upgrades_enabled and invest_allowed:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=self._required_upgrades,
                    base_location=self._ai.start_location,
                )
            )

        # Expansions (gated in _expansion_targets)
        target_bases, max_pending = self._expansion_targets()
        macro_plan.add(
            ExpansionController(to_count=target_bases, max_pending=max_pending)
        )

        self._ai.register_behavior(macro_plan)

    # ── Gas Logic ───────────────────────────────────────────────────────────

    def _gas_targets(self) -> tuple[int, int]:
        """Determine target gas count and max pending geysers.

        Scales with drone count via GAS_PHASES. During the rush response,
        gas is frozen at its current count — the defense (lings, drones,
        queens) needs none. Returns:
            (target_gas, max_pending) tuple for GasBuildingController.
        """
        ai = self._ai
        drone_count: int = ai.supply_workers
        base_count: int = len(ai.townhalls.ready)

        if self._rush_active():
            return (len(ai.gas_buildings), 0)

        # Walk through gas phases, pick the highest one we qualify for
        gas_per_base: float = GAS_PHASES[0][1]
        max_pending: int = GAS_PHASES[0][2]
        for min_drones, gpb, pending in GAS_PHASES:
            if drone_count >= min_drones:
                gas_per_base = gpb
                max_pending = pending

        target_gas: int = int(base_count * gas_per_base + 0.999)  # ceil

        # If we have Lair tech, ensure at least 4 gas for upgrades
        if (
            ai.structures(UnitID.LAIR).ready.exists
            or ai.structures(UnitID.HIVE).ready.exists
        ) and target_gas < 4:
            target_gas = 4

        return (target_gas, max_pending)

    # ── Queen Production Logic ─────────────────────────────────────────────

    def _queen_target(self) -> int:
        """Target queen count (incl. pending).

        Normal: ready bases + 1. During the rush response: never fewer
        than RUSH_DEFENSE_QUEENS — two queens are the defense backbone
        (transfuse + DPS), even with a single base.
        """
        ai = self._ai
        base_count: int = len(ai.townhalls.ready)

        if base_count == 0:
            return 0

        if self._rush_state is not RushState.NONE:
            return max(RUSH_DEFENSE_QUEENS, base_count)

        # RECOVERY: 4 queens — 2 x RUSH_DEFENSE_QUEENS, and also the
        # QueenManager inject-activation threshold. A larva-free standing
        # defense for the rebuild window PLUS restored inject throughput;
        # recovery defense comes from queens, never pulled drones.
        if self._recovering:
            return max(2 * RUSH_DEFENSE_QUEENS, base_count)

        return base_count + 1

    def _produce_queens(self) -> None:
        """Train one queen per frame from an idle townhall if below target.

        Direct train() rather than SpawnController: queens need a
        count-based target and train from the townhall build queue,
        not larvae.

        Perf: O(townhalls), typically 2-4.
        """
        ai = self._ai

        current_queens: int = len(ai.mediator.get_own_army_dict[UnitID.QUEEN])
        pending_queens: int = cy_unit_pending(ai, UnitID.QUEEN)
        target: int = self._queen_target()

        if current_queens + pending_queens >= target or target == 0:
            return

        if not ai.can_afford(UnitID.QUEEN):
            return

        for th in ai.townhalls.ready:
            if th.is_idle:
                th.train(UnitID.QUEEN)
                return

    # ── Morph Units ──────────────────────────────────────────────────────────

    def _morph_units_standalone(self) -> None:
        """Morph entry point — runs every frame, including during the
        opening build order (SpawnController can't morph combat units).

        Held during SECURING: morphs would drain the queen bank.
        """
        ai = self._ai
        if self._rush_state_securing():
            return
        army_dict: dict[UnitID, Units] = ai.mediator.get_own_army_dict
        army_counts: dict[UnitID, int] = {
            uid: len(units) for uid, units in army_dict.items()
        }
        full_comp: dict[UnitID, dict] = get_army_comp(
            ai.time,
            air_threat=self._threats.get("air_signs", False),
            drone_count=ai.supply_workers,
            rush_active=self._rush_active(),
        )
        self._morph_units(full_comp, army_counts)

    def _morph_units(
        self,
        full_comp: dict[UnitID, dict],
        army_counts: dict[UnitID, int],
    ) -> None:
        """Manually morph Zerglings→Banelings and Roaches→Ravagers.

        SpawnController can't morph combat units because it requires idle
        build structures, and Zerglings/Roaches are never idle during combat.
        This method directly issues morph commands based on composition
        thresholds from compositions.py.

        Only morphs when:
        1. The base unit population meets the threshold (e.g. >=15% Roaches
           for Ravagers, >=10% Zerglings for Banelings)
        2. We're below the target proportion of morph units
        3. We can afford the morph cost
        4. The prerequisite building exists (Baneling Nest, Lair/Hive)

        Perf: O(base_units) per morph type, typically <30 units.
        """
        ai = self._ai
        structure_dict: dict = ai.mediator.get_own_structures_dict

        # ── Ravagers: Roach → Ravager ───────────────────────────────────
        if should_morph(UnitID.RAVAGER, army_counts, full_comp):
            has_lair: bool = (
                len(structure_dict.get(UnitID.LAIR, [])) > 0
                or len(structure_dict.get(UnitID.HIVE, [])) > 0
            )
            if has_lair and ai.can_afford(UnitID.RAVAGER):
                pending_ravagers: int = cy_unit_pending(ai, UnitID.RAVAGER)
                current_ravagers: int = army_counts.get(UnitID.RAVAGER, 0)
                total_ravagers: int = current_ravagers + pending_ravagers

                comp_types: set[UnitID] = set(full_comp.keys())
                total_comp: int = sum(army_counts.get(uid, 0) for uid in comp_types)
                target_prop: float = full_comp[UnitID.RAVAGER]["proportion"]
                target_count: int = int(total_comp * target_prop)
                needed: int = max(0, target_count - total_ravagers)

                if needed > 0:
                    roaches: Units = ai.units(UnitID.ROACH)
                    for roach in roaches:
                        if needed <= 0:
                            break
                        roach(AbilityId.MORPHTORAVAGER_RAVAGER)
                        needed -= 1

        # ── Banelings: Zergling → Baneling ──────────────────────────────
        if should_morph(UnitID.BANELING, army_counts, full_comp):
            has_bane_nest: bool = len(structure_dict.get(UnitID.BANELINGNEST, [])) > 0
            if has_bane_nest and ai.can_afford(UnitID.BANELING):
                pending_banelings: int = cy_unit_pending(ai, UnitID.BANELING)
                current_banelings: int = army_counts.get(UnitID.BANELING, 0)
                total_banelings: int = current_banelings + pending_banelings

                comp_types: set[UnitID] = set(full_comp.keys())
                total_comp: int = sum(army_counts.get(uid, 0) for uid in comp_types)
                target_prop: float = full_comp[UnitID.BANELING]["proportion"]
                target_count: int = int(total_comp * target_prop)
                needed: int = max(0, target_count - total_banelings)

                if needed > 0:
                    zerglings: Units = ai.units(UnitID.ZERGLING)
                    for zergling in zerglings:
                        if needed <= 0:
                            break
                        zergling(AbilityId.MORPHTOBANELING_BANELING)
                        needed -= 1

    # ── Expansion Logic ─────────────────────────────────────────────────────

    def _bases_saturated(
        self, threshold: float = EXPANSION_SATURATION_THRESHOLD
    ) -> bool:
        """True if every ready base has at least `threshold` fraction of
        full saturation (drones per base >= threshold * DRONES_PER_SATURATED_BASE).
        Pending bases are excluded — existing bases must be saturated
        before taking more.

        Args:
            threshold: Fraction of full saturation required (0.0-1.0).
        """
        ai = self._ai
        ready_bases: int = len(ai.townhalls.ready)
        if ready_bases == 0:
            return False
        drones_per_base: float = ai.supply_workers / ready_bases
        return drones_per_base >= DRONES_PER_SATURATED_BASE * threshold

    def _expansion_targets(self) -> tuple[int, int]:
        """Determine target base count and max pending expansions.

        Gates, in order: rush response (no expansion spend), enemies near
        bases, one-at-a-time pending cap, existing-base saturation. When
        all gates pass, expansions are unlimited — saturation and the
        pending cap are the self-limiters.

        Returns:
            (target_bases, max_pending) tuple for ExpansionController.
        """
        ai = self._ai
        ready_bases: int = len(ai.townhalls.ready)
        pending_bases: int = ai.structure_pending(ai.base_townhall_type)

        if self._rush_active():
            return (ready_bases, 0)

        # RECOVERY: no expansion spend — the bank goes to the drone rebuild
        if self._recovering:
            return (ready_bases, 0)

        if self._enemy_combat_near_bases():
            return (ready_bases, 0)

        if pending_bases >= 1:
            return (ready_bases + pending_bases, 1)

        if not self._bases_saturated():
            return (ready_bases, 0)

        return (99, 1)

    # ── Upgrades ────────────────────────────────────────────────────────────

    @property
    def _required_upgrades(self) -> list[UpgradeID]:
        """Upgrade list for UpgradeController.

        Reactive-tech upgrades are only included when their prerequisite
        building exists, preventing UpgradeController from auto-teching
        to buildings we don't want yet:
        - Grooved Spines: only when Hydralisk Den exists (air reaction)
        - Centrifugal Hooks: only when Baneling Nest exists (built proactively)
        """
        # Upgrades that require a reactive/proactive building — exclude by default
        gated_upgrades: dict[UpgradeID, UnitID] = {
            UpgradeID.EVOLVEGROOVEDSPINES: UnitID.HYDRALISKDEN,
            UpgradeID.CENTRIFICALHOOKS: UnitID.BANELINGNEST,
        }

        upgrades: list[UpgradeID] = []
        for u in UPGRADE_PRIORITY:
            if u in gated_upgrades:
                required_building: UnitID = gated_upgrades[u]
                if self._ai.structures(required_building).ready.exists:
                    upgrades.append(u)
            else:
                upgrades.append(u)
        return upgrades

    @property
    def _upgrades_enabled(self) -> bool:
        """Only research upgrades when we have gas to spare."""
        if self._ai.supply_workers < 36:
            return False
        return (self._ai.vespene > 95) or (
            self._ai.minerals > 500 and self._ai.vespene > 350
        )

    # ── Proactive Tech Buildings ────────────────────────────────────────────

    def _build_proactive_tech(self) -> None:
        """Build tech buildings when the economy can support them.

        Baneling Nest: built once we have 24+ drones (early game economy
        stable enough to afford banelings without starving roach production).
        Infestation Pit: built once we have 36+ drones (mid-game economy
        ready for gas-heavy caster support).
        These are proactive, not reactive — they're built because the
        composition needs them, not because of a threat.
        """
        ai = self._ai

        # Baneling Nest: needed for banelings in early comp
        if (
            ai.supply_workers >= 24
            and not ai.structures(UnitID.BANELINGNEST).exists
            and not ai.already_pending(UnitID.BANELINGNEST)
            and ai.can_afford(UnitID.BANELINGNEST)
        ):
            ai.build(UnitID.BANELINGNEST, near=ai.start_location)

        # Infestation Pit: needed for infestors in mid comp
        # Only build when economy is ready (36+ drones) and Lair is up
        if (
            ai.supply_workers >= 36
            and (
                ai.structures(UnitID.LAIR).ready.exists
                or ai.structures(UnitID.HIVE).ready.exists
            )
            and not ai.structures(UnitID.INFESTATIONPIT).exists
            and not ai.already_pending(UnitID.INFESTATIONPIT)
            and ai.can_afford(UnitID.INFESTATIONPIT)
        ):
            ai.build(UnitID.INFESTATIONPIT, near=ai.start_location)

    # ── Threat Assessment ──────────────────────────────────────────────────

    def _assess_threats(self) -> None:
        """Detect common threats and update the threat dict (rebuilt per frame)."""
        self._threats = {
            "no_natural": False,
            "timing_push": False,
            "air_signs": False,
            "proxy_signs": False,
            "mass_light": False,
            "cannon_rush": False,
            "rush_detected": False,
        }

        ai = self._ai

        # Mirror the state machine, not the raw ARES flag — that one is
        # permanent and would resurrect the threat after a clear
        if self._rush_state is not RushState.NONE:
            self._threats["rush_detected"] = True

        # No enemy natural by 3:30
        if ai.time > NATURAL_TIMING_THRESHOLD:
            enemy_naturals: list = [
                th
                for th in ai.enemy_structures
                if th.type_id in {UnitID.HATCHERY, UnitID.COMMANDCENTER, UnitID.NEXUS}
                and 50 < th.distance_to(ai.enemy_start_locations[0]) < 200
            ]
            if not enemy_naturals:
                self._threats["no_natural"] = True

        # Air signs: latching — air tech once seen is never un-seen.
        # Memory units count: vision loss doesn't dismiss a Void Ray.
        if not self._air_signs_detected:
            for structure in ai.enemy_structures:
                if structure.type_id in AIR_STRUCTURES:
                    self._air_signs_detected = True
                    break
        if not self._air_signs_detected:
            for unit in ai.enemy_units:
                if unit.type_id in AIR_UNIT_TYPES:
                    self._air_signs_detected = True
                    break
        self._threats["air_signs"] = self._air_signs_detected

        # Enemy structures close to our start
        for structure in ai.enemy_structures:
            if (
                structure.distance_to(ai.start_location) < 80
                and structure.type_id != UnitID.XELNAGATOWER
            ):
                self._threats["proxy_signs"] = True
                break

        # Mass light units
        light_count: int = sum(
            1
            for u in ai.enemy_units
            if u.type_id in LIGHT_UNIT_TYPES and not u.is_memory
        )
        if light_count >= 10:
            self._threats["mass_light"] = True

        # Forge / Photon Cannon close to our start
        for structure in ai.enemy_structures:
            if structure.distance_to(ai.start_location) < 60:
                if structure.type_id in {UnitID.FORGE, UnitID.PHOTONCANNON}:
                    self._threats["cannon_rush"] = True
                    break

    # ── Threat Responses ────────────────────────────────────────────────────

    def _respond_to_threats(self) -> None:
        """Execute responses for active threats.

        Reactive tech competes with the rush defense for the mineral bank
        and is held while SECURING. Spines are never held — they ARE the
        threat response.
        """
        if not any(self._threats.values()):
            return

        ai = self._ai
        # Reactive tech is investment too: held while SECURING and during
        # RECOVERY (the rebuild owns the bank). Spines are never held.
        tech_held: bool = self._rush_state_securing() or self._recovering

        if self._threats["no_natural"]:
            self._build_emergency_spines(count=EMERGENCY_SPINE_COUNT)

        if self._threats["air_signs"]:
            if (
                not tech_held
                and not ai.structures(UnitID.HYDRALISKDEN).exists
                and not ai.already_pending(UnitID.HYDRALISKDEN)
                and ai.can_afford(UnitID.HYDRALISKDEN)
            ):
                ai.build(UnitID.HYDRALISKDEN, near=ai.start_location)
            self._build_mineral_line_spines(count=MINERAL_LINE_SPINE_COUNT)

        if self._threats["proxy_signs"]:
            self._build_emergency_spines(count=1)

        if self._threats["mass_light"] and not tech_held:
            if (
                not ai.structures(UnitID.BANELINGNEST).exists
                and not ai.already_pending(UnitID.BANELINGNEST)
                and ai.can_afford(UnitID.BANELINGNEST)
            ):
                ai.build(UnitID.BANELINGNEST, near=ai.start_location)

        if self._threats["cannon_rush"]:
            self._build_emergency_spines(count=1)

    # ── Building Helpers ────────────────────────────────────────────────────

    def _build_emergency_spines(self, count: int = 2) -> None:
        """Build emergency spine crawlers near our bases."""
        ai = self._ai
        existing: int = len(ai.structures(UnitID.SPINECRAWLER))
        pending: int = ai.already_pending(UnitID.SPINECRAWLER)
        needed: int = max(0, count - existing - pending)

        for _ in range(needed):
            if ai.can_afford(UnitID.SPINECRAWLER):
                for th in ai.townhalls:
                    ai.build(
                        UnitID.SPINECRAWLER,
                        near=th.position.towards(ai.game_info.map_center, 3),
                    )
                    break

    def _build_mineral_line_spines(self, count: int = 1) -> None:
        """Build spine crawlers in mineral lines for air defense."""
        ai = self._ai
        existing: int = len(ai.structures(UnitID.SPINECRAWLER))
        pending: int = ai.already_pending(UnitID.SPINECRAWLER)
        needed: int = max(0, count + len(ai.townhalls) - existing - pending)

        for _ in range(needed):
            if ai.can_afford(UnitID.SPINECRAWLER):
                for th in ai.townhalls:
                    mfs = ai.mineral_field.closer_than(10, th)
                    if mfs:
                        ai.build(
                            UnitID.SPINECRAWLER,
                            near=th.position.towards(mfs.center, 2),
                        )
                    break
