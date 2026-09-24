# Purpose: Centralized macro management — workers, supply, gas, queens,
#   spawning, tech, upgrades, expansions, and reactive building.
# Key Decisions: All MacroPlan behaviors live here. Research and response
#   logic folded in from their old standalone modules. The attack_target
#   property is also owned here since it's a macro-level decision.
#   Queen production: target = ready bases + 1, produced via SpawnController.
#   Proactive tech (Baneling Nest, Infestation Pit) built on economy thresholds.
#   Reactive tech (Hydralisk Den) built only on air threat detection.
#   Gated upgrades (Grooved Spines, Centrifugal Hooks) only included when
#   their prerequisite building exists, preventing auto-tech-up.
#   Expansion gated on base saturation: won't take a new base until existing
#   ones are near-full. Worker target scales with total (ready+pending) bases.
# Limitations: No nydus network support yet, no dynamic composition
#   switching beyond air detection.

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

from bot.compositions import get_army_comp, should_morph, strip_morph_units

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

# Expansion phases: (min_drone_count, target_base_count)
# max_pending is always 1 — never build two bases at once.
# Phase 1: Opening (post-build) — 2 bases from build order, hold at 2
# Phase 2: Early mid-game — 3 bases once we have 30+ drones
# Phase 3: Mid-game — 4 bases once we have 44+ drones
# Phase 4: Late-game — expand aggressively when economy is saturated
EXPANSION_PHASES: list[tuple[int, int]] = [
    (0, 2),  # Phase 1: hold at 2 bases (natural from build order)
    (30, 3),  # Phase 2: take 3rd when 30+ drones
    (44, 4),  # Phase 3: take 4th when 44+ drones
    (60, 99),  # Phase 4: expand freely when 60+ drones
]

# Worker priority: below this drone count, always produce workers
# even if townhalls are busy with army. Above this, only build
# workers when townhalls are idle (army gets larvae priority).
WORKER_PRIORITY_THRESHOLD: int = 30

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
        # Latches True once the rush reaction fired (build order aborted)
        self._rush_reacted: bool = False
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

        # Rush flag: latches once ARES detects an enemy rush. Mirrors the
        # cheese reaction in PiGBot — abort the opening build order so
        # dynamic macro (emergency queens, spines, army) takes over.
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

        # Failsafe: if build is still active but minerals are piling up,
        # force-complete the build so dynamic macro can take over
        if not self._ai.build_order_runner.build_completed:
            if self._ai.minerals >= 1500:
                self._ai.build_order_runner.set_build_completed()
            else:
                return

        self._assess_threats()
        self._do_macro_plan()
        self._research_upgrades()
        self._respond_to_threats()

    # ── Macro Plan ──────────────────────────────────────────────────────────

    def _do_macro_plan(self) -> None:
        """Build the main MacroPlan: supply, workers, gas, spawning, tech, expand."""
        macro_plan: MacroPlan = MacroPlan()
        structure_dict: dict = self._ai.mediator.get_own_structures_dict

        # Supply
        macro_plan.add(AutoSupply(base_location=self._ai.start_location))

        # Workers — produce enough to saturate all existing + pending bases
        # Priority: below WORKER_PRIORITY_THRESHOLD, always build workers.
        # Above it, only build workers when townhalls are idle so army
        # gets larvae priority. MacroPlan stops after first action, so
        # BuildWorkers before SpawnController = workers first, but we
        # skip workers when army needs larvae more.
        total_bases: int = len(self._ai.townhalls.ready) + self._ai.structure_pending(
            self._ai.base_townhall_type
        )
        max_workers: int = min(80, total_bases * DRONES_PER_FULLY_SATURATED_BASE)
        if self._threats.get("rush_detected", False) and self._ai.supply_army < 16:
            max_workers = min(max_workers, 30)
        idle_townhalls: bool = bool(self._ai.townhalls.idle)
        need_workers: bool = (
            self._ai.supply_workers < WORKER_PRIORITY_THRESHOLD or idle_townhalls
        )
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

        # Spawning — use composition from compositions.py
        # Strip morph units (Ravager, Baneling) from SpawnController because
        # it can't morph combat units (they're never idle). Morphing is
        # handled separately in _morph_units_standalone() which runs every
        # frame, even during the build order.
        full_army_comp: dict[UnitID, dict] = get_army_comp(
            self._ai.time,
            air_threat=self._threats.get("air_signs", False),
            drone_count=self._ai.supply_workers,
        )
        army_comp: dict[UnitID, dict] = strip_morph_units(full_army_comp)
        freeflow: bool = self._ai.supply_workers >= FREEFLOW_DRONE_THRESHOLD
        macro_plan.add(SpawnController(army_comp, freeflow_mode=freeflow))

        # Proactive tech buildings — built when economy supports them
        self._build_proactive_tech()

        # Queen production — target = ready bases + 1
        # Uses direct train() instead of SpawnController because queens
        # need a count-based target (not proportion-based), and they're
        # trained from Hatch/Lair/Hive (not larvae).
        self._produce_queens()

        # Tech — Lair when we have enough queens and gas
        lair_tech: bool = (
            len(structure_dict[UnitID.LAIR]) > 0 or len(structure_dict[UnitID.HIVE]) > 0
        )
        if (
            self._ai.vespene >= 100
            and not lair_tech
            and len(self._ai.mediator.get_own_army_dict[UnitID.QUEEN]) >= 4
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.LAIR, base_location=self._ai.start_location)
            )

        # Hive at high supply
        if (
            self._ai.supply_used > 170.0
            and len(structure_dict.get(UnitID.HIVE, [])) == 0
        ):
            macro_plan.add(
                TechUp(desired_tech=UnitID.HIVE, base_location=self._ai.start_location)
            )

        # Upgrades — only when we have gas to spare
        if self._upgrades_enabled:
            macro_plan.add(
                UpgradeController(
                    upgrade_list=self._required_upgrades,
                    base_location=self._ai.start_location,
                )
            )

        # Expansions — phased based on drone count and threat state
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

        # Under rush pressure: don't build more gas than we already have
        if self._threats.get("rush_detected", False) and ai.supply_army < 16:
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

        # Rush: ensure at least two queens exist for defense
        if self._threats.get("rush_detected", False):
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
        during the opening build order.
        """
        ai = self._ai
        army_dict: dict[UnitID, Units] = ai.mediator.get_own_army_dict
        army_counts: dict[UnitID, int] = {
            uid: len(units) for uid, units in army_dict.items()
        }
        full_comp: dict[UnitID, dict] = get_army_comp(
            ai.time,
            air_threat=self._threats.get("air_signs", False),
            drone_count=ai.supply_workers,
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

    def _bases_saturated(self, threshold: float = 0.85) -> bool:
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

        Gates expansion on base saturation: existing bases must be near-full
        before we take more. Always caps max_pending at 1 — never build
        two bases at once. Under rush pressure, block expansions.

        Returns:
            (target_bases, max_pending) tuple for ExpansionController.
        """
        ai = self._ai
        drone_count: int = ai.supply_workers
        ready_bases: int = len(ai.townhalls.ready)
        pending_bases: int = ai.structure_pending(ai.base_townhall_type)

        # Under rush pressure with small army: block all expansion
        if self._threats.get("rush_detected", False) and ai.supply_army < 16:
            return (ready_bases, 0)

        # If a base is already pending, don't start another one
        if pending_bases >= 1:
            return (ready_bases + pending_bases, 1)

        # Walk through phases, pick the highest one we qualify for
        target_bases: int = EXPANSION_PHASES[0][1]
        for min_drones, bases in EXPANSION_PHASES:
            if drone_count >= min_drones:
                target_bases = bases

        # Saturation gate: don't expand unless existing bases are saturated
        if target_bases > ready_bases and not self._bases_saturated(threshold=0.85):
            target_bases = ready_bases

        return (target_bases, 1)

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

    def _research_upgrades(self) -> None:
        """Research the next available upgrade in priority order.

        Only one upgrade per frame to avoid starving production.
        Uses `already_pending_upgrade` to avoid duplicate research.
        """
        for upgrade_id in UPGRADE_PRIORITY:
            if self._ai.already_pending_upgrade(upgrade_id) > 0:
                continue
            if not self._ai.can_afford(upgrade_id):
                continue
            self._ai.research(upgrade_id)
            break  # Only one upgrade per frame

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

        # Rush detection (from ARES mediator)
        if ai.mediator.get_did_enemy_rush:
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
        """Execute minimal responses for active threats."""
        if not any(self._threats.values()):
            return

        ai = self._ai

        # No natural: spines + safety roaches
        if self._threats["no_natural"]:
            self._build_emergency_spines(count=EMERGENCY_SPINE_COUNT)

        # Air signs: hydra den + mineral line spines
        if self._threats["air_signs"]:
            if (
                not ai.structures(UnitID.HYDRALISKDEN).exists
                and not ai.already_pending(UnitID.HYDRALISKDEN)
                and ai.can_afford(UnitID.HYDRALISKDEN)
            ):
                ai.build(UnitID.HYDRALISKDEN, near=ai.start_location)
            self._build_mineral_line_spines(count=MINERAL_LINE_SPINE_COUNT)

        # Proxy signs: spine + safety roaches
        if self._threats["proxy_signs"]:
            self._build_emergency_spines(count=1)

        # Mass light: bane nest
        if self._threats["mass_light"]:
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
