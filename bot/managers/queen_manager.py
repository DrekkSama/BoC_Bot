# Purpose: Queen role assignment and control using ARES Queen-specific roles
# Key Decisions: No permanent QUEEN_DEFENCE role — eliminated to prevent oscillation.
#   New queens default to QUEEN_CREEP. Inject queens assigned per-townhall.
#   Threat-response defense temporarily steals creep queens to DEFENDING role.
#   When threats clear, queens return to QUEEN_CREEP after a grace period.
#   Energy priority: inject > transfuse > creep tumor. Queens reserve 25 energy
#   for inject if near a TH that needs it; inject queens spread creep with surplus.
#   Uses property_cache_once_per_frame for role requirement calculations.
# Limitations: No nydus queen support, no offensive queen mode yet

import numpy as np
from ares import AresBot
from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    AMove,
    KeepUnitSafe,
    PathUnitToTarget,
    QueenSpreadCreep,
    ShootTargetInRange,
    StutterUnitBack,
    TumorSpreadCreep,
    UseTransfuse,
)
from ares.cache import property_cache_once_per_frame
from ares.consts import UnitRole
from cython_extensions import cy_closest_to, cy_distance_to, cy_in_attack_range
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId as UnitID
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

# ── Constants ────────────────────────────────────────────────────────────────
QUEEN_INJECT_ENERGY: float = 25.0
CREEP_TUMOR_ENERGY: float = 25.0
TRANSFUSE_ENERGY: float = 50.0
# Energy threshold for inject queens to also spread creep
# Must have enough for inject (25) + tumor (25) = 50 surplus after reserving inject
INJECT_SURPLUS_FOR_CREEP: float = QUEEN_INJECT_ENERGY + CREEP_TUMOR_ENERGY
# How close a queen must be to a townhall for inject
INJECT_RANGE: float = 10.0
# How close a creep queen must be to a TH to consider injecting
CREEP_QUEEN_INJECT_RANGE: float = 12.0
# Max creep spreaders when not under pressure
MAX_CREEP_SPREADERS: int = 5
# Min queens before we start assigning inject/creep roles
MIN_QUEENS_FOR_SPECIALIZATION: int = 2
# Creep coverage threshold to stop spreading
CREEP_COVERAGE_STOP: float = 85.0
# Max tumors before we stop spreading
MAX_TUMORS: int = 25
# Distance from townhall to detect threats for defense
DEFENCE_THREAT_RANGE: float = 15.0
# Supply threshold for ground threats to trigger queen defense
DEFENCE_GROUND_SUPPLY_THRESHOLD: float = 3.0
# Supply threshold for air threats to trigger queen defense
DEFENCE_AIR_SUPPLY_THRESHOLD: float = 4.0
# Max queens to pull for defense (don't pull all creep queens)
MAX_DEFENDERS: int = 3
# Grace period (game seconds) before returning a defender to creep duty
DEFENCE_GRACE_PERIOD: float = 5.0


class QueenManager:
    """Manages Queen roles and behavior using ARES role system.

    Role hierarchy:
    - New queens default to QUEEN_CREEP (spread creep as primary job)
    - Inject queens are assigned per-townhall (QUEEN_INJECT)
    - No permanent QUEEN_DEFENCE role — eliminated to prevent oscillation
    - When threats appear near bases, creep queens are temporarily
      reassigned to DEFENDING role. When threats clear, they return
      to QUEEN_CREEP after a grace period.
    """

    def __init__(self, ai: AresBot) -> None:
        self.ai: AresBot = ai
        # Track which queen tag is assigned to which townhall tag for inject
        self._inject_queen_to_th: dict[int, int] = {}
        # Track when each defending queen last saw a threat (for grace period)
        self._defender_last_threat_time: dict[int, float] = {}

    def assign_new_queen(self, queen: Unit) -> None:
        """Assign a newly created queen to QUEEN_CREEP role.

        Called from main.py on_unit_created for Queen units.
        The role controller will dynamically reassign as needed.
        """
        self.ai.mediator.assign_role(tag=queen.tag, role=UnitRole.QUEEN_CREEP)

    def update(self) -> None:
        """Run every frame: adjust roles, then execute queen behaviors."""
        # Get queens by current role
        # (DEFENDING filtered to queens: WorkerDefenseManager also assigns
        #  drones to DEFENDING, which must not leak into queen logic)
        inject_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_INJECT
        )
        creep_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.QUEEN_CREEP
        )
        defending_queens: Units = self.ai.mediator.get_units_from_role(
            role=UnitRole.DEFENDING, unit_type=UnitID.QUEEN
        )

        # Adjust role assignments
        self._manage_inject_role(creep_queens, inject_queens)
        self._manage_creep_role(creep_queens, inject_queens)

        # Threat-response defense: steal creep queens when threats appear
        self._assign_base_defenders(creep_queens, defending_queens)

        # Re-fetch after role adjustments (roles may have changed)
        inject_queens = self.ai.mediator.get_units_from_role(role=UnitRole.QUEEN_INJECT)
        creep_queens = self.ai.mediator.get_units_from_role(role=UnitRole.QUEEN_CREEP)
        defending_queens = self.ai.mediator.get_units_from_role(
            role=UnitRole.DEFENDING, unit_type=UnitID.QUEEN
        )

        # Execute behaviors per role
        self._control_inject_queens(inject_queens)
        self._control_creep_queens(creep_queens)
        self._control_defending_queens(defending_queens)

        # Tumor spread (always run for existing tumors)
        self._spread_tumors()

    # ── Role Requirement Properties (cached per frame) ──────────────────────

    @property_cache_once_per_frame
    def _required_injectors(self) -> int:
        """How many inject queens we need.

        From the example: skip injects during rush or with very few queens.
        Otherwise, 1 per townhall that doesn't already have an inject queen.
        """
        num_queens: int = len(self.ai.mediator.get_own_army_dict[UnitID.QUEEN])

        # Don't inject during rush or with very few queens
        if self.ai.mediator.get_did_enemy_rush:
            return 0
        if num_queens < MIN_QUEENS_FOR_SPECIALIZATION:
            return 0
        # Too many bases to manage injects effectively
        if len(self.ai.townhalls) >= 5:
            return 0

        return len(self.ai.townhalls)

    @property_cache_once_per_frame
    def _required_creep_spreaders(self) -> int:
        """How many creep queens we need.

        From the example: scale with queen count, cap at MAX_CREEP_SPREADERS.
        Stop if creep coverage is high or too many tumors.
        Don't spread when under significant pressure.
        """
        num_queens: int = len(self.ai.mediator.get_own_army_dict[UnitID.QUEEN])

        if num_queens < MIN_QUEENS_FOR_SPECIALIZATION:
            return 0

        # Stop spreading if coverage is high or too many tumors
        coverage: float = self.ai.mediator.get_creep_coverage
        if coverage > CREEP_COVERAGE_STOP:
            return 0
        num_tumors: int = len(
            self.ai.mediator.get_own_structures_dict[UnitID.CREEPTUMORBURROWED]
        )
        if num_tumors > MAX_TUMORS:
            return 0

        # Don't spread when under significant pressure
        ground_threats: Units = self.ai.mediator.get_main_ground_threats_near_townhall
        if ground_threats and self.ai.get_total_supply(ground_threats) >= 4.0:
            return 0
        air_threats: Units = self.ai.mediator.get_main_air_threats_near_townhall
        if air_threats and self.ai.get_total_supply(air_threats) >= 6.0:
            return 0

        # Scale with queen count, cap at MAX_CREEP_SPREADERS
        return min(MAX_CREEP_SPREADERS, max(1, num_queens - 3))

    # ── Role Management ─────────────────────────────────────────────────────

    def _manage_inject_role(self, creep_queens: Units, inject_queens: Units) -> None:
        """Assign/unassign inject queens based on game state.

        Steal from creep queens if we need more injectors.
        Unassign extras back to creep.
        """
        num_required: int = self._required_injectors

        # Assign: steal from creep if we need more injectors
        if num_required and len(inject_queens) < num_required and creep_queens:
            # Find a townhall that doesn't have an inject queen yet
            th_tags_taken: set[int] = set(self._inject_queen_to_th.values())
            available_ths: list[Unit] = [
                th
                for th in self.ai.townhalls
                if th.build_progress > 0.95 and th.tag not in th_tags_taken
            ]
            if available_ths:
                queen: Unit = creep_queens[0]
                # Pick closest townhall to this queen
                closest_th: Unit = cy_closest_to(queen.position, available_ths)
                self.ai.mediator.assign_role(tag=queen.tag, role=UnitRole.QUEEN_INJECT)
                self._inject_queen_to_th[queen.tag] = closest_th.tag

        # Unassign: too many injectors → return to creep
        if len(inject_queens) > num_required:
            num_to_unassign: int = len(inject_queens) - num_required
            for i in range(num_to_unassign):
                tag: int = inject_queens[i].tag
                self.ai.mediator.assign_role(tag=tag, role=UnitRole.QUEEN_CREEP)
                self._inject_queen_to_th.pop(tag, None)

        # Clean up stale mappings (queen died or role changed)
        active_tags: set[int] = {q.tag for q in inject_queens}
        stale: list[int] = [t for t in self._inject_queen_to_th if t not in active_tags]
        for tag in stale:
            del self._inject_queen_to_th[tag]

    def _manage_creep_role(self, creep_queens: Units, inject_queens: Units) -> None:
        """Assign/unassign creep spreaders based on game state.

        Only steal from inject pool if we have excess injectors.
        Unassign extras back to inject (if needed) or just leave as creep.
        """
        num_required: int = self._required_creep_spreaders

        # Unassign: too many creep queens → return to inject if needed
        if len(creep_queens) > num_required:
            num_to_unassign: int = len(creep_queens) - num_required
            num_inject_needed: int = self._required_injectors - len(inject_queens)
            for i in range(num_to_unassign):
                tag: int = creep_queens[i].tag
                if num_inject_needed > 0:
                    self.ai.mediator.assign_role(tag=tag, role=UnitRole.QUEEN_INJECT)
                    num_inject_needed -= 1
                # Otherwise leave as creep — excess creep queens are fine

    # ── Queen Behaviors ─────────────────────────────────────────────────────

    def _control_inject_queens(self, inject_queens: Units) -> None:
        """Inject queens: inject assigned TH, spread creep with surplus energy.

        Energy priority: inject > transfuse > creep tumor.
        - Always inject if TH needs it and queen has 25 energy.
        - If TH is already injected, spread creep with surplus energy
          (energy >= 50, i.e. enough for inject + tumor reserve).
        - If not enough surplus, just move toward TH.
        """
        grid: np.ndarray = self.ai.mediator.get_ground_avoidance_grid

        for queen in inject_queens:
            th_tag: int = self._inject_queen_to_th.get(queen.tag, -1)
            assigned_th: Unit | None = None
            for th in self.ai.townhalls:
                if th.tag == th_tag:
                    assigned_th = th
                    break

            # If assigned TH no longer exists, reassign
            if assigned_th is None:
                th_tags_taken: set[int] = set(self._inject_queen_to_th.values())
                available_ths: list[Unit] = [
                    th
                    for th in self.ai.townhalls
                    if th.build_progress > 0.95 and th.tag not in th_tags_taken
                ]
                if available_ths:
                    new_th: Unit = cy_closest_to(queen.position, available_ths)
                    self._inject_queen_to_th[queen.tag] = new_th.tag
                    assigned_th = new_th
                else:
                    # No available TH, move to closest existing TH
                    if self.ai.townhalls:
                        assigned_th = cy_closest_to(queen.position, self.ai.townhalls)

            if assigned_th is None:
                continue

            # Priority 1: Inject if TH needs it and we have energy
            th_needs_inject: bool = not assigned_th.has_buff(
                BuffId.QUEENSPAWNLARVATIMER
            )
            if queen.energy >= QUEEN_INJECT_ENERGY and th_needs_inject:
                queen(AbilityId.EFFECT_INJECTLARVA, assigned_th)
                continue

            # Priority 2: Transfuse injured friendlies
            if queen.energy >= TRANSFUSE_ENERGY:
                maneuver: CombatManeuver = CombatManeuver()
                maneuver.add(UseTransfuse(unit=queen, targets=self.ai.all_own_units))
                # Still move toward TH after transfuse
                dist: float = cy_distance_to(queen.position, assigned_th.position)
                if dist > INJECT_RANGE:
                    maneuver.add(
                        PathUnitToTarget(
                            unit=queen, grid=grid, target=assigned_th.position
                        )
                    )
                self.ai.register_behavior(maneuver)
                continue

            # Priority 3: Spread creep with surplus energy
            # Only if TH is already injected (or we don't have inject energy)
            # and we have enough surplus for tumor while reserving inject
            has_surplus: bool = queen.energy >= INJECT_SURPLUS_FOR_CREEP
            if has_surplus and not th_needs_inject:
                maneuver = CombatManeuver()
                maneuver.add(QueenSpreadCreep(unit=queen, cancel_if_close_enemy=True))
                self.ai.register_behavior(maneuver)
                continue

            # Priority 4: Move toward assigned TH using ARES pathing
            dist = cy_distance_to(queen.position, assigned_th.position)
            if dist > INJECT_RANGE:
                maneuver = CombatManeuver()
                maneuver.add(
                    PathUnitToTarget(unit=queen, grid=grid, target=assigned_th.position)
                )
                self.ai.register_behavior(maneuver)

    def _control_creep_queens(self, creep_queens: Units) -> None:
        """Creep queens: inject nearby THs first, then spread tumors.

        Energy priority: inject > transfuse > creep tumor.
        - If near an un-injected TH and have 25 energy, inject it.
        - Reserve 25 energy for inject if near a TH that will need it soon
          (i.e. only spread creep if energy >= 50 when near an injectable TH).
        - Otherwise spread creep normally.

        Each queen gets exactly ONE CombatManeuver per frame.
        QueenSpreadCreep handles movement to creep edge internally and
        has built-in safety via cancel_if_close_enemy.
        """
        for queen in creep_queens:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Inject nearby un-injected townhall
            # Creep queens near a TH that needs inject should do it first
            nearby_th_needing_inject: Unit | None = self._find_nearby_th_needing_inject(
                queen
            )
            if nearby_th_needing_inject and queen.energy >= QUEEN_INJECT_ENERGY:
                queen(AbilityId.EFFECT_INJECTLARVA, nearby_th_needing_inject)
                continue

            # Priority 2: Transfuse injured friendlies
            if queen.energy >= TRANSFUSE_ENERGY:
                maneuver.add(UseTransfuse(unit=queen, targets=self.ai.all_own_units))

            # Priority 3: Spread creep (with energy reservation for inject)
            # If near a TH that will need inject soon, reserve 25 energy
            near_injectable_th: bool = nearby_th_needing_inject is not None
            energy_floor: float = QUEEN_INJECT_ENERGY if near_injectable_th else 0.0
            has_energy: bool = queen.energy >= CREEP_TUMOR_ENERGY + energy_floor

            if has_energy:
                maneuver.add(QueenSpreadCreep(unit=queen, cancel_if_close_enemy=True))
            else:
                # Not enough energy: pre-move toward creep edge
                maneuver.add(
                    QueenSpreadCreep(
                        unit=queen,
                        cancel_if_close_enemy=True,
                        pre_move_queen_to_tumor=True,
                    )
                )

            self.ai.register_behavior(maneuver)

    # ── Threat-Response Defense ─────────────────────────────────────────────

    def _assign_base_defenders(
        self, creep_queens: Units, defending_queens: Units
    ) -> None:
        """Temporarily steal creep queens to defend bases when threats appear.

        This replaces the old permanent QUEEN_DEFENCE role. Queens are only
        pulled from creep duty when there are actual threats near bases.
        When threats clear, queens return to QUEEN_CREEP after a grace period.

        Inspired by combat_manager_sample._assign_base_defenders but adapted
        for Zerg queens: uses supply-based threat assessment and ARES
        CombatManeuver behaviors instead of raw attack commands.
        """
        ground_threats: Units = self.ai.mediator.get_main_ground_threats_near_townhall
        air_threats: Units = self.ai.mediator.get_main_air_threats_near_townhall

        has_ground_threat: bool = (
            ground_threats
            and self.ai.get_total_supply(ground_threats)
            >= DEFENCE_GROUND_SUPPLY_THRESHOLD
        )
        has_air_threat: bool = (
            air_threats
            and self.ai.get_total_supply(air_threats) >= DEFENCE_AIR_SUPPLY_THRESHOLD
        )
        has_threat: bool = has_ground_threat or has_air_threat

        # ── Update threat timestamps for current defenders ────────────
        if has_threat:
            for queen in defending_queens:
                self._defender_last_threat_time[queen.tag] = self.ai.time

        # ── Pull creep queens into defense if needed ──────────────────
        if has_threat:
            num_defenders: int = len(defending_queens)
            if num_defenders < MAX_DEFENDERS and creep_queens:
                # Steal the closest creep queen to the threat
                num_to_steal: int = min(
                    MAX_DEFENDERS - num_defenders,
                    len(creep_queens),
                )
                # Pick queens closest to the threatened base
                for _ in range(num_to_steal):
                    if not creep_queens:
                        break
                    # Find the closest creep queen to any threatened townhall
                    threatened_ths: list[Unit] = self._threatened_townhalls(
                        ground_threats, air_threats
                    )
                    if not threatened_ths:
                        break
                    # Use first threatened TH as reference point
                    ref_pos: Point2 = threatened_ths[0].position
                    closest_creep: Unit = cy_closest_to(ref_pos, creep_queens)
                    self.ai.mediator.assign_role(
                        tag=closest_creep.tag, role=UnitRole.DEFENDING
                    )
                    self._defender_last_threat_time[closest_creep.tag] = self.ai.time
                    # Remove from local list so we don't pick the same queen twice
                    creep_queens = creep_queens.filter(
                        lambda q: q.tag != closest_creep.tag
                    )

        # ── Return defenders to creep duty when threats clear ─────────
        else:
            queens_to_return: list[int] = []
            for queen in defending_queens:
                last_threat: float = self._defender_last_threat_time.get(queen.tag, 0.0)
                time_since_threat: float = self.ai.time - last_threat
                if time_since_threat > DEFENCE_GRACE_PERIOD:
                    queens_to_return.append(queen.tag)

            for tag in queens_to_return:
                self.ai.mediator.assign_role(tag=tag, role=UnitRole.QUEEN_CREEP)
                self._defender_last_threat_time.pop(tag, None)

        # ── Clean up stale threat timestamps ──────────────────────────
        defender_tags: set[int] = {q.tag for q in defending_queens}
        stale_tags: list[int] = [
            t for t in self._defender_last_threat_time if t not in defender_tags
        ]
        for tag in stale_tags:
            del self._defender_last_threat_time[tag]

    def _threatened_townhalls(
        self, ground_threats: Units, air_threats: Units
    ) -> list[Unit]:
        """Return townhalls that have threats nearby.

        Perf note: O(threats * townhalls) but typically <5 of each.
        """
        threatened: list[Unit] = []
        for th in self.ai.townhalls:
            for enemy in ground_threats:
                if cy_distance_to(th.position, enemy.position) < DEFENCE_THREAT_RANGE:
                    threatened.append(th)
                    break
            else:
                for enemy in air_threats:
                    if (
                        cy_distance_to(th.position, enemy.position)
                        < DEFENCE_THREAT_RANGE
                    ):
                        threatened.append(th)
                        break
        return threatened

    # ── Queen Behaviors ─────────────────────────────────────────────────────

    def _control_defending_queens(self, defending_queens: Units) -> None:
        """Defending queens: fight threats near bases, transfuse allies.

        Behavior chain: transfuse → AOE dodge (avoidance grid) →
        weapon_ready branch: shoot in-range targets or advance when ready,
        stutter back from the closest enemy on cooldown. StutterUnitBack
        requires the ground (influence) grid — enemy unit positions add
        retreat cost there; the avoidance grid only tracks spell effects
        and would never trigger a retreat.

        When threats clear, _assign_base_defenders returns them to
        QUEEN_CREEP after a grace period.
        """
        if not defending_queens:
            return

        avoid_grid: np.ndarray = self.ai.mediator.get_ground_avoidance_grid
        grid: np.ndarray = self.ai.mediator.get_ground_grid
        enemies_near_base: Units = self._enemies_near_bases()

        for queen in defending_queens:
            maneuver: CombatManeuver = CombatManeuver()

            # Priority 1: Transfuse injured friendlies
            if queen.energy >= TRANSFUSE_ENERGY:
                maneuver.add(UseTransfuse(unit=queen, targets=self.ai.all_own_units))

            # Priority 2: Dodge AOE effects (storms, biles, disruptors)
            maneuver.add(KeepUnitSafe(unit=queen, grid=avoid_grid))

            # Priority 3: Stutter-step fight when enemies are near a base
            if enemies_near_base:
                closest_enemy: Unit = cy_closest_to(queen.position, enemies_near_base)
                if queen.weapon_ready:
                    # Weapon ready: shoot best target in range, else advance
                    if in_range := cy_in_attack_range(queen, enemies_near_base):
                        maneuver.add(ShootTargetInRange(unit=queen, targets=in_range))
                    else:
                        maneuver.add(AMove(unit=queen, target=closest_enemy.position))
                else:
                    # Cooldown: stutter back from the closest enemy
                    maneuver.add(
                        StutterUnitBack(unit=queen, target=closest_enemy, grid=grid)
                    )
            else:
                # No immediate threats: move toward closest TH using ARES pathing
                # (grace period in _assign_base_defenders handles return to creep)
                if self.ai.townhalls:
                    closest_th: Unit = cy_closest_to(queen.position, self.ai.townhalls)
                    dist_to_th: float = cy_distance_to(
                        queen.position, closest_th.position
                    )
                    if dist_to_th > INJECT_RANGE:
                        maneuver.add(
                            PathUnitToTarget(
                                unit=queen, grid=grid, target=closest_th.position
                            )
                        )

            self.ai.register_behavior(maneuver)

    def _spread_tumors(self) -> None:
        """Spread existing creep tumors toward enemy."""
        target: Point2 = self.ai.attack_target
        tumors: Units = self.ai.structures(
            {UnitID.CREEPTUMORBURROWED, UnitID.CREEPTUMORQUEEN}
        )
        for tumor in tumors:
            self.ai.register_behavior(TumorSpreadCreep(unit=tumor, target=target))

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _find_nearby_th_needing_inject(self, queen: Unit) -> Unit | None:
        """Find a nearby townhall that needs inject and isn't covered by an inject queen.

        Used by creep queens to opportunistically inject when they're near
        an un-injected TH. Only considers THs within CREEP_QUEEN_INJECT_RANGE
        that don't already have an inject queen assigned.

        Perf note: O(townhalls) per queen, typically 1-4 THs.
        """
        inject_th_tags: set[int] = set(self._inject_queen_to_th.values())
        for th in self.ai.townhalls:
            if th.build_progress < 0.95:
                continue
            if th.has_buff(BuffId.QUEENSPAWNLARVATIMER):
                continue
            # Skip THs that already have a dedicated inject queen
            if th.tag in inject_th_tags:
                continue
            dist: float = cy_distance_to(queen.position, th.position)
            if dist <= CREEP_QUEEN_INJECT_RANGE:
                return th
        return None

    def _enemies_near_bases(self) -> Units:
        """Get visible enemy units near any of our townhalls.

        Memory ghosts (last-seen fog positions) are excluded so queens
        never stutter toward empty space.

        Perf note: O(bases * enemies_near_base) but typically 1-3 bases
        and <50 visible enemies, so fine.
        """
        result: list[Unit] = []
        seen_tags: set[int] = set()
        for th in self.ai.townhalls:
            near_th: Units = self.ai.enemy_units.closer_than(
                DEFENCE_THREAT_RANGE, th.position
            )
            for unit in near_th:
                if unit.is_memory or unit.tag in seen_tags:
                    continue
                result.append(unit)
                seen_tags.add(unit.tag)
        return Units(result, self.ai)
