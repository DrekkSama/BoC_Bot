# Purpose: Centralized macro management — workers, supply, gas, queens,
#   spawning, tech, upgrades, expansions, and reactive building.
# Key Decisions: All MacroPlan behaviors live here. Research and response
#   logic folded in from their old standalone modules. The attack_target
#   property is also owned here since it's a macro-level decision.
#   Queen production: target = ready bases + 1, produced via SpawnController.
#   Proactive tech (Baneling Nest, Infestation Pit) built on economy thresholds.
#   Roach Warren ensured via TechUp placed BEFORE SpawnController in the
#   plan — MacroPlan short-circuits on first True, so tech must unlock
#   before production can stall on it (covers aborted builds + snipes).
#   Rush response is a three-state machine (RushState, below), following
#   the ReactionManager lifecycle pattern from PiGBot:
#   SECURING (rush latched, queens < 2): abort the opening build and hold
#   ALL tech + unit + expansion spends so queens come first — queens train
#   from townhalls (no larva contest), and every other spend would snipe
#   the 150m queen bank. DEFENDING (queens up, rush not cleared): rush
#   defense profile — pure zerglings (RUSH_DEFENSE_COMP), drones capped
#   at RUSH_WORKER_CAP, gas/expansions frozen, all tech/upgrades held;
#   the Spawning Pool is rebuilt first-class if the rush killed it, and
#   the 2-queen defense floor stays. NONE again after RUSH_CLEAR_GRACE
#   seconds with no enemy combat units near our bases ("no enemies
#   around" → back to the plan as usual). ARES rush detection latches
#   permanently, so clearing also latches (_rush_over) and the raw
#   mediator flag is never treated as a live condition.
#   Reactive tech (Hydralisk Den) built only on air threat detection.
#   Gated upgrades (Grooved Spines, Centrifugal Hooks) only included when
#   their prerequisite building exists, preventing auto-tech-up.
#   Expansion gated on base saturation: won't take a new base until existing
#   ones are near-full. Worker target scales with total (ready+pending) bases.
#   Worker production above WORKER_PRIORITY_THRESHOLD is gated on mineral
#   banking (NOT idle townhalls — Zerg hatcheries are idle almost every
#   frame, so that gate never blocked anything and drones ate all larvae).
#   Upgrades flow exclusively through the gated UpgradeController (which
#   auto-builds required tech) — the old ungated _research_upgrades drained
#   the first 100/100 bank the moment it appeared.
# Limitations: No nydus network support yet, no dynamic composition
#   switching beyond air detection.

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

# Saturation: drones per base for full mineral+gas saturation
# 16 mineral patches + 6 gas (2 geysers × 3) = 22 ideal, but 20 is
# a practical threshold — bases rarely have 2 full geysers early on.
DRONES_PER_SATURATED_BASE: int = 20
DRONES_PER_FULLY_SATURATED_BASE: int = 22

# Rush defense: minimum queens to target when a rush is detected
# (two queens back the defense with transfuse + DPS)
RUSH_DEFENSE_QUEENS: int = 2

# Rush defense: max workers while the rush response is active. Larvae go
# to zerglings — a drone wave costs 50m apiece and robs the ling army.
RUSH_WORKER_CAP: int = 30

# Rush response clear condition: seconds with no enemy combat units near
# our bases before the rush is considered over and normal macro resumes.
# Mirrors PiGBot's CHEESE_THREAT_CLEAR_GRACE — a grace window prevents
# oscillation from intermittent ling pokes during the transition.
RUSH_CLEAR_GRACE: float = 30.0


class RushState(Enum):
    """Rush response lifecycle — see header for the full design.

    SECURING: rush latched, defense queens not yet up. Everything except
        queens, supply, and threat spines is held.
    DEFENDING: queens secured. Rush defense profile: pure lings + capped
        drones, tech/gas/expansion/upgrades held, pool rebuilt if dead.
        The 2-queen floor stays until the rush clears.
    NONE: no rush response active (also the terminal state after a
        cleared rush — _rush_over latches so the permanent ARES flag
        can't re-trigger).
    """

    NONE = auto()
    SECURING = auto()
    DEFENDING = auto()


# Worker priority: below this drone count, always produce workers
# even if larvae are contested. Above this, drones are gated on
# mineral banking (see DRONE_BANK_MINERALS) so army gets larvae
# priority — idle townhalls was NOT a valid Zerg gate since
# hatcheries are idle almost every frame.
WORKER_PRIORITY_THRESHOLD: int = 30

# Above WORKER_PRIORITY_THRESHOLD, only resume droning when the
# mineral bank is high enough to not compete with army larva spends
# (a drone wave is 50m each; army units run 50-150m each).
DRONE_BANK_MINERALS: int = 400

# Saturation threshold to allow taking another base (0.75 of a
# saturated base = 15/20 drones per base). Was 0.85, which lagged
# 4th/5th bases behind the drone-count phases by ~7 drones.
EXPANSION_SATURATION_THRESHOLD: float = 0.75

# Free-flow spawning: once we have this many drones, the economy is
# strong enough that SpawnController should ignore proportions and
# spend freely — just produce whatever we can afford.
FREEFLOW_DRONE_THRESHOLD: int = 60

# Gas phases: (min_drone_count, gas_per_base, max_pending_geysers)
# Phase 1: Post-build — 1 gas per base (2 total), enough for ling speed + Lair
# Phase 2: Mid-game — 1.5 gas per base (round up), supports upgrades + ravagers
# Phase 3: Late-game — 2 gas per base, full saturation for hive tech
GAS_PHASES: list[tuple[int, float, int]] = [
    (0, 1.0, 1),  # Phase 1: 1 geyser per base
    (36, 1.5, 2),  # Phase 2: 1.5 geysers per base (rounds up)
    (56, 2.0, 2),  # Phase 3: 2 geysers per base
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

# Air-detection structure types
AIR_STRUCTURES: set[UnitID] = {
    UnitID.FUSIONCORE,
    UnitID.STARGATE,
    UnitID.STARPORTTECHLAB,
    UnitID.FLEETBEACON,
}

# Air unit types that warrant Hydralisk production
# Excludes non-combat air (Overlord, Overseer, Observer, WarpPrism)
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

# Light unit types for mass detection
LIGHT_UNIT_TYPES: set[UnitID] = {
    UnitID.ZERGLING,
    UnitID.ZEALOT,
    UnitID.ADEPT,
    UnitID.MARINE,
}

# Response constants
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
        # Always mine (mineral boosting / speedmining disabled)
        self._ai.register_behavior(Mining(mineral_boost=False))

        # Morph units every frame, even during build order — SpawnController
        # can't morph combat units (they're never idle), so we do it manually.
        # This must run before the build-runner return so morphing continues
        # during the opening build order.
        self._morph_units_standalone()

        # Rush flag: latches once ARES detects an enemy rush (ARES's flag is
        # itself permanent). Aborts the opening build order so dynamic macro
        # (defense queens, spines, army) takes over.
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

        # Failsafe: if build is still active but minerals are piling up,
        # force-complete the build so dynamic macro can take over
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
        """Advance the rush response lifecycle. Call once per frame.

        Transitions (one-way, matching PiGBot's ReactionManager):
          NONE → SECURING : ARES detects a rush (latched by _rush_reacted)
          SECURING → DEFENDING : 2-queen defense floor met (incl. pending)
          DEFENDING → NONE : RUSH_CLEAR_GRACE seconds without enemy
                             combat units near our bases (latches
                             _rush_over — ARES's flag is permanent)
        """
        ai = self._ai

        # Terminal: cleared rushes never re-trigger from the raw flag
        if self._rush_over:
            return

        # Track near-base threats for the clear condition (also feeds
        # DefenseManager-style presence checks without a second scan)
        if self._enemy_combat_near_bases():
            self._last_threat_near_base_time = ai.time

        # NONE → SECURING: rush just detected
        if self._rush_state is RushState.NONE:
            if self._rush_reacted:
                self._rush_state = RushState.SECURING

        # SECURING → DEFENDING: defense queens secured
        elif self._rush_state is RushState.SECURING:
            if self._defense_queens_secured():
                self._rush_state = RushState.DEFENDING
                logger.info(
                    f"{ai.time_formatted}: Defense queens secured — "
                    f"rush defense profile active (lings + capped drones)"
                )

        # DEFENDING → NONE: grace window with no enemies near bases
        elif self._rush_state is RushState.DEFENDING:
            if (
                ai.time - self._last_threat_near_base_time >= RUSH_CLEAR_GRACE
                and not self._enemy_combat_near_bases()
            ):
                self._rush_state = RushState.NONE
                self._rush_over = True
                logger.info(
                    f"{ai.time_formatted}: Rush cleared — returning to " f"normal macro"
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

        The rush-clear condition: reuses ARES's near-base enemy tracking
        (ground + flying, per-townhall tag sets) exactly like
        DefenseManager._collect_threats, dropping harmless types so
        overlords/scouts don't count as "enemies around".

        Perf note: O(tracked tags) set-union + one tags_in lookup.
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
        """Consumer read: hold all tech/unit/expansion spends this frame."""
        return self._rush_state is RushState.SECURING

    def _rush_active(self) -> bool:
        """Consumer read: the rush defense profile is active this frame.

        True during SECURING and DEFENDING — the entire rush response.
        Gates the rush comp, worker cap, and defense-tech holds.
        """
        return self._rush_state is not RushState.NONE

    # ── Macro Plan ──────────────────────────────────────────────────────────

    def _do_macro_plan(self) -> None:
        """Build the main MacroPlan: supply, workers, gas, tech, spawning, expand.

        MacroPlan.execute() short-circuits on the first behavior returning
        True — only ONE macro action runs per frame. Ordering therefore
        encodes priority: supply → workers → gas → TECH → army → upgrades
        → expansions. Tech (Roach Warren) must precede SpawnController so
        it can never be starved by constant ling/roach spending.
        """
        macro_plan: MacroPlan = MacroPlan()
        structure_dict: dict = self._ai.mediator.get_own_structures_dict

        # Supply
        macro_plan.add(AutoSupply(base_location=self._ai.start_location))

        # Rush SECURING: hold ALL other macro spends — every one of them
        # (drones 50m, army units, warren 150m, gas 75m, expansion 300m)
        # would snipe the 150m queen bank before the defense queens can
        # train. Queens go first; AutoSupply above keeps supply flowing so
        # the queen is never blocked. Spines stay active via
        # _respond_to_threats (run in update() regardless).
        rush_securing: bool = self._rush_state_securing()
        # Rush DEFENDING (rush_active incl. SECURING): rush defense profile —
        # lings + capped drones only. No tech, no gas expansion, no upgrades,
        # no new bases until the rush clears.
        rush_active: bool = self._rush_active()

        # Workers — produce enough to saturate all existing + pending bases
        # Below WORKER_PRIORITY_THRESHOLD: always drone (standard opening).
        # Above it: drone only when the mineral bank is high enough that
        # workers don't compete with army larvae. The old `idle_townhalls`
        # gate was useless for Zerg — hatcheries are idle almost every
        # frame (only Queen training / morphs occupy them), so drones
        # monopolized larvae up to the 66+ target while army starved.
        total_bases: int = len(self._ai.townhalls.ready) + self._ai.structure_pending(
            self._ai.base_townhall_type
        )
        max_workers: int = min(80, total_bases * DRONES_PER_FULLY_SATURATED_BASE)
        if rush_active:
            max_workers = min(max_workers, RUSH_WORKER_CAP)
        need_workers: bool = (
            self._ai.supply_workers < WORKER_PRIORITY_THRESHOLD
            or self._ai.minerals >= DRONE_BANK_MINERALS
        ) and not rush_securing
        if need_workers:
            macro_plan.add(BuildWorkers(to_count=max_workers))

        # Gas — phased based on drone count and base count
        target_gas, max_pending_gas = self._gas_targets()
        macro_plan.add(
            GasBuildingController(
                to_count=target_gas,
                max_pending=max_pending_gas,
            )
        )

        # Tech — Roach Warren BEFORE army spawning.
        # Roach Warren: roaches are priority-1 in every composition, and the
        # safety-roach rush defense depends on them. Covers games where the
        # opening build aborted before the scripted `18 roachwarren` step,
        # plus mid-game warren snipes. Must precede SpawnController in the
        # plan: MacroPlan short-circuits, and with the Warren missing the
        # SpawnController over-produces lings (its only tech-ready unit)
        # every single frame — TechUp never got a look in. TechUp
        # self-guards: skips if the warren is present/pending, chains a
        # Spawning Pool if missing, returns True only on the queue frame.
        # Held while rush_active — the rush comp is pure lings; the Warren
        # (150m) would drain the ling army every frame it retries.
        if not rush_active:
            macro_plan.add(
                TechUp(
                    desired_tech=UnitID.ROACHWARREN,
                    base_location=self._ai.start_location,
                )
            )

        # Rush DEFENDING: Spawning Pool is the linchpin of the defense
        # (lings). Rebuild it first-class if the rush killed it — the
        # warren TechUp's chained pool rebuild is held above, so without
        # this a dead pool means no ling production for the rest of the
        # rush. Held during SECURING (queens still go first).
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

        # Lair — held while rush_active (defense spends nothing on tech)
        lair_tech: bool = (
            len(structure_dict[UnitID.LAIR]) > 0 or len(structure_dict[UnitID.HIVE]) > 0
        )
        if (
            not rush_active
            and self._ai.vespene >= 100
            and not lair_tech
            and len(self._ai.mediator.get_own_army_dict[UnitID.QUEEN]) >= 4
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.LAIR, base_location=self._ai.start_location)
            )

        # Hive at high supply — held while rush_active (same reason)
        if (
            not rush_active
            and self._ai.supply_used > 170.0
            and len(structure_dict.get(UnitID.HIVE, [])) == 0
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.HIVE, base_location=self._ai.start_location)
            )

        # Spawning — use composition from compositions.py
        # Strip morph units (Ravager, Baneling) from SpawnController because
        # it can't morph combat units (they're never idle). Morphing is
        # handled separately in _morph_units_standalone() which runs every
        # frame, even during the build order.
        # Rush active: rush comp (pure lings). Held during SECURING —
        # army larvae and minerals go to queens first.
        if not rush_securing:
            full_army_comp: dict[UnitID, dict] = get_army_comp(
                self._ai.time,
                air_threat=self._threats.get("air_signs", False),
                drone_count=self._ai.supply_workers,
                rush_active=rush_active,
            )
            army_comp: dict[UnitID, dict] = strip_morph_units(full_army_comp)
            freeflow: bool = self._ai.supply_workers >= FREEFLOW_DRONE_THRESHOLD
            # Non-freeflow SpawnController hard-breaks on its highest
            # priority unit being unaffordable. Reorder so an affordable
            # unit leads — prevents total army stall when Roach gas runs
            # dry (Roach 75/25, Ling 50/0).
            if not freeflow:
                army_comp = prioritize_affordable_units(
                    army_comp, self._ai.minerals, self._ai.vespene
                )
            macro_plan.add(SpawnController(army_comp, freeflow_mode=freeflow))

        # Proactive tech buildings — built when economy supports them.
        # Held while rush_active (Baneling Nest costs 100m — would drain
        # the ling army).
        if not rush_active:
            self._build_proactive_tech()

        # Queen production — target = ready bases + 1
        # Uses direct train() instead of SpawnController because queens
        # need a count-based target (not proportion-based), and they're
        # trained from Hatch/Lair/Hive (not larvae).
        self._produce_queens()

        # Upgrades — only when we have gas to spare. Held while rush_active:
        # UpgradeController auto-builds required tech (e.g. Evolution
        # Chamber), which must not drain the ling army.
        if self._upgrades_enabled and not rush_active:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=self._required_upgrades,
                    base_location=self._ai.start_location,
                )
            )

        # Expansions — gated on saturation + rush state
        target_bases, max_pending = self._expansion_targets()
        macro_plan.add(
            ExpansionController(to_count=target_bases, max_pending=max_pending)
        )

        self._ai.register_behavior(macro_plan)

    # ── Gas Logic ───────────────────────────────────────────────────────────

    def _gas_targets(self) -> tuple[int, int]:
        """Determine target gas count and max pending geysers.

        Scales gas with base count and economy maturity. Under heavy
        rush pressure, delay extra gas to prioritize army units.

        Returns:
            (target_gas, max_pending) tuple for GasBuildingController.
        """
        ai = self._ai
        drone_count: int = ai.supply_workers
        base_count: int = len(ai.townhalls.ready)

        # Rush active (SECURING + DEFENDING): freeze EXTRA gas at its current
        # count — the rush defense (lings, drones, queens) needs no gas.
        # (Gas the opening build already started is kept, never canceled.)
        # Once the rush clears, normal gas phases resume so roaches flow.
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
        """Determine target queen count: ready bases + 1 extra.

        Under rush pressure the queen floor is RUSH_DEFENSE_QUEENS:
        two queens are the backbone of the defense (transfuse + DPS),
        so never target fewer — even with a single base.
        Without a rush, target is bases + 1.

        Returns:
            Target number of queens we want to have (including pending).
        """
        ai = self._ai
        base_count: int = len(ai.townhalls.ready)

        # No bases = no queens
        if base_count == 0:
            return 0

        # Rush response active (SECURING or DEFENDING): defense queen
        # floor — never target fewer than 2, even with a single base
        if self._rush_state is not RushState.NONE:
            return max(RUSH_DEFENSE_QUEENS, base_count)

        return base_count + 1

    def _produce_queens(self) -> None:
        """Train a queen from an idle Hatch/Lair/Hive if we need more.

        Queens are trained from townhalls (not larvae), so we use
        direct train() instead of SpawnController. Only trains one
        queen per frame to avoid blocking larvae production on
        other townhalls.

        Perf note: O(townhalls) to find idle one, typically 2-4.
        """
        ai = self._ai

        # Check if we need more queens
        current_queens: int = len(ai.mediator.get_own_army_dict[UnitID.QUEEN])
        pending_queens: int = cy_unit_pending(ai, UnitID.QUEEN)
        total_queens: int = current_queens + pending_queens
        target: int = self._queen_target()

        if total_queens >= target or target == 0:
            return

        # Find an idle Hatch/Lair/Hive to train from
        # Queens cost 150 minerals, 0 gas, 2 supply
        if not ai.can_afford(UnitID.QUEEN):
            return

        for th in ai.townhalls.ready:
            if th.is_idle:
                th.train(UnitID.QUEEN)
                return  # Only one per frame

    # ── Morph Units ──────────────────────────────────────────────────────────

    def _morph_units_standalone(self) -> None:
        """Entry point for morph logic — runs every frame, even during build order.

        Computes army counts and composition, then delegates to _morph_units.
        This must run before the build-runner return so morphing continues
        during the opening build order. Held during rush until the queen
        floor is met — morphs drain minerals/gas the queens need.
        """
        ai = self._ai
        # Rush SECURING: hold morphs — they'd drain the queen bank.
        # DEFENDING: morphs flow (banes are prime anti-ling defense).
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

        Perf note: O(base_units) per morph type, typically <30 units.
        """
        ai = self._ai
        structure_dict: dict = ai.mediator.get_own_structures_dict

        # ── Ravagers: Roach → Ravager ───────────────────────────────────
        if should_morph(UnitID.RAVAGER, army_counts, full_comp):
            # Need Lair or Hive tech
            has_lair: bool = (
                len(structure_dict.get(UnitID.LAIR, [])) > 0
                or len(structure_dict.get(UnitID.HIVE, [])) > 0
            )
            if has_lair and ai.can_afford(UnitID.RAVAGER):
                # Count morphing/pending ravagers to avoid over-morphing
                pending_ravagers: int = cy_unit_pending(ai, UnitID.RAVAGER)
                current_ravagers: int = army_counts.get(UnitID.RAVAGER, 0)
                total_ravagers: int = current_ravagers + pending_ravagers

                # Calculate how many more we need
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
            # Need Baneling Nest
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
        """Check if all existing (ready) bases are saturated enough to expand.

        A base is considered saturated at `threshold * DRONES_PER_SATURATED_BASE`
        drones. Default 0.85 means ~17/20 drones per base. Pending bases are
        excluded — we want existing bases saturated before taking more.

        Args:
            threshold: Fraction of full saturation required (0.0–1.0).

        Returns:
            True if every ready base has enough drones.
        """
        ai = self._ai
        ready_bases: int = len(ai.townhalls.ready)
        if ready_bases == 0:
            return False
        drones_per_base: float = ai.supply_workers / ready_bases
        return drones_per_base >= DRONES_PER_SATURATED_BASE * threshold

    def _expansion_targets(self) -> tuple[int, int]:
        """Determine target base count and max pending expansions.

        Follows the examples' pattern (Clicadinha / PiGBot): no hardcoded
        phase table — ExpansionController targets all bases and the real
        gates are saturation and pending caps, which self-limit to one
        expansion at a time. Held entirely during SECURING (300m must
        never snipe the queen bank) and while enemies are near bases.

        Returns:
            (target_bases, max_pending) tuple for ExpansionController.
        """
        ai = self._ai
        ready_bases: int = len(ai.townhalls.ready)
        pending_bases: int = ai.structure_pending(ai.base_townhall_type)

        # Rush active: no expansion spend — the bank goes to the defense
        if self._rush_active():
            return (ready_bases, 0)

        # Enemies currently near our bases: don't start a new base
        if self._enemy_combat_near_bases():
            return (ready_bases, 0)

        # One expansion at a time, always
        if pending_bases >= 1:
            return (ready_bases + pending_bases, 1)

        # Existing bases must be saturated before taking more
        if not self._bases_saturated():
            return (ready_bases, 0)

        # Expand freely — saturation + pending cap are the only gates
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
        """Detect common threats and update internal threat dict."""
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

        # Rush: mirror the state machine (NOT the raw ARES flag — that one
        # latches permanently and would resurrect the threat after clear)
        if self._rush_state is not RushState.NONE:
            self._threats["rush_detected"] = True

        # No natural at 3:30
        if ai.time > NATURAL_TIMING_THRESHOLD:
            enemy_naturals: list = [
                th
                for th in ai.enemy_structures
                if th.type_id in {UnitID.HATCHERY, UnitID.COMMANDCENTER, UnitID.NEXUS}
                and 50 < th.distance_to(ai.enemy_start_locations[0]) < 200
            ]
            if not enemy_naturals:
                self._threats["no_natural"] = True

        # Air signs: enemy air tech structures OR any air unit we've seen
        # Include memory units — if we ever saw a Void Ray, that threat
        # persists even if we lose vision of it.
        # This is a latching flag: once detected, stays True for the rest
        # of the game. You don't un-see air tech.
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

        # Proxy signs
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

        # Cannon rush
        for structure in ai.enemy_structures:
            if structure.distance_to(ai.start_location) < 60:
                if structure.type_id in {UnitID.FORGE, UnitID.PHOTONCANNON}:
                    self._threats["cannon_rush"] = True
                    break

    # ── Threat Responses ────────────────────────────────────────────────────

    def _respond_to_threats(self) -> None:
        """Execute minimal responses for active threats.

        Reactive tech (Baneling Nest, Hydra Den) is held during SECURING —
        it competes with queens for the mineral bank. Spines are
        intentionally never held: they are the threat response itself.
        """
        if not any(self._threats.values()):
            return

        ai = self._ai
        tech_held: bool = self._rush_state_securing()

        # No natural: spines + safety roaches
        if self._threats["no_natural"]:
            self._build_emergency_spines(count=EMERGENCY_SPINE_COUNT)

        # Air signs: hydra den + mineral line spines
        if self._threats["air_signs"]:
            if (
                not tech_held
                and not ai.structures(UnitID.HYDRALISKDEN).exists
                and not ai.already_pending(UnitID.HYDRALISKDEN)
                and ai.can_afford(UnitID.HYDRALISKDEN)
            ):
                ai.build(UnitID.HYDRALISKDEN, near=ai.start_location)
            self._build_mineral_line_spines(count=MINERAL_LINE_SPINE_COUNT)

        # Proxy signs: spine + safety roaches
        if self._threats["proxy_signs"]:
            self._build_emergency_spines(count=1)

        # Mass light: bane nest
        if self._threats["mass_light"] and not tech_held:
            if (
                not ai.structures(UnitID.BANELINGNEST).exists
                and not ai.already_pending(UnitID.BANELINGNEST)
                and ai.can_afford(UnitID.BANELINGNEST)
            ):
                ai.build(UnitID.BANELINGNEST, near=ai.start_location)

        # Cannon rush: spine
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
