# Purpose: Army composition definitions for SpawnController / ProductionController
# Key Decisions: Early game roach/ling/bane/ravager, mid-game adds infestor,
#   hydra only when air threats detected. Composition switches on economy
#   (drone count) with time as fallback, not time alone.
#   SpawnController (non-freeflow) hard-breaks when its HIGHEST priority
#   unit is unaffordable — see prioritize_affordable_units, which reorders
#   priorities so an affordable unit leads and production never stalls.
# Limitations: No dynamic composition switching beyond air/economy detection yet

from ares.dicts.cost_dict import COST_DICT
from sc2.game_data import Cost
from sc2.ids.unit_typeid import UnitTypeId as UnitID

# Early game (< 6 min or < 36 drones): roach/ling/bane/ravager
# Queens excluded — managed separately via inject/creep/defense
# Banelings provide AOE for early mass-light encounters
EARLY_COMP: dict[UnitID, dict] = {
    UnitID.ROACH: {"proportion": 0.48, "priority": 1},
    UnitID.ZERGLING: {"proportion": 0.25, "priority": 2},
    UnitID.BANELING: {"proportion": 0.15, "priority": 3},
    UnitID.RAVAGER: {"proportion": 0.12, "priority": 4},
}

# Mid game (6+ min): roach/ravager/ling/infestor — no hydra unless air detected
MID_COMP: dict[UnitID, dict] = {
    UnitID.ROACH: {"proportion": 0.45, "priority": 1},
    UnitID.RAVAGER: {"proportion": 0.20, "priority": 2},
    UnitID.ZERGLING: {"proportion": 0.25, "priority": 3},
    UnitID.INFESTOR: {"proportion": 0.10, "priority": 4},
}

# Anti-air variant: heavy hydra for air-heavy opponents
# Hydralisk Den is built reactively by response_manager when air_signs detected
ANTI_AIR_COMP: dict[UnitID, dict] = {
    UnitID.ROACH: {"proportion": 0.30, "priority": 1},
    UnitID.HYDRALISK: {"proportion": 0.35, "priority": 2},
    UnitID.RAVAGER: {"proportion": 0.10, "priority": 3},
    UnitID.ZERGLING: {"proportion": 0.20, "priority": 4},
    UnitID.INFESTOR: {"proportion": 0.05, "priority": 5},
}

# Economy thresholds for composition switching
# 36 drones = gas phase 2, enough economy to support infestors
MID_GAME_DRONES: int = 36
# Time fallback: switch at 6 min even if drone count is low (e.g. heavy pressure)
MID_GAME_TIME: float = 360.0

# Morph thresholds: minimum proportion of base unit in army before morphing
# Prevents morphing away all base units (e.g. turning every Roach into a Ravager)
# These are checked against the *current* army composition each frame.
RAVAGER_MORPH_THRESHOLD: float = 0.15  # Roaches must be >= 15% of army
BANELING_MORPH_THRESHOLD: float = 0.40  # Zerglings must be >= 40% of army

# Maps morph unit -> (base unit, threshold)
MORPH_GATES: dict[UnitID, tuple[UnitID, float]] = {
    UnitID.RAVAGER: (UnitID.ROACH, RAVAGER_MORPH_THRESHOLD),
    UnitID.BANELING: (UnitID.ZERGLING, BANELING_MORPH_THRESHOLD),
}


def get_army_comp(
    time: float, air_threat: bool = False, drone_count: int = 0
) -> dict[UnitID, dict]:
    """Return the appropriate army composition based on game state.

    Switches to mid-game comp when either the economy is ready (36+ drones)
    or enough time has passed (6 min). Air threat overrides to anti-air comp.

    Args:
        time: Current game time in seconds.
        air_threat: True if air signs detected (stargate, starport techlab,
            fusion core, fleet beacon, or visible air units).
        drone_count: Current number of workers (supply_workers).
    """
    if air_threat:
        return ANTI_AIR_COMP

    if time < MID_GAME_TIME and drone_count < MID_GAME_DRONES:
        return EARLY_COMP

    return MID_COMP


def strip_morph_units(
    base_comp: dict[UnitID, dict],
) -> dict[UnitID, dict]:
    """Remove morph units from the composition for SpawnController.

    SpawnController cannot morph combat units (Zergling→Baneling, Roach→Ravager)
    because it requires idle build structures, and combat units are never idle.
    Morphing is handled separately by MacroManager._morph_units().

    Proportions are renormalized to sum to 1.0 so SpawnController's
    assertion check passes.

    Args:
        base_comp: The full army composition dict.
    """
    stripped_comp: dict[UnitID, dict] = {
        uid: dict(info) for uid, info in base_comp.items()
    }
    removed_proportion: float = 0.0
    for morph_unit in MORPH_GATES:
        if morph_unit in stripped_comp:
            removed_proportion += stripped_comp[morph_unit]["proportion"]
            del stripped_comp[morph_unit]

    # Renormalize proportions so they sum to 1.0 (SpawnController asserts this)
    if removed_proportion > 0.0 and stripped_comp:
        current_sum: float = sum(v["proportion"] for v in stripped_comp.values())
        if current_sum > 0.0:
            scale: float = 1.0 / current_sum
            for uid in stripped_comp:
                stripped_comp[uid] = dict(stripped_comp[uid])
                stripped_comp[uid]["proportion"] = (
                    stripped_comp[uid]["proportion"] * scale
                )

    return stripped_comp


def should_morph(
    morph_unit: UnitID,
    army_counts: dict[UnitID, int],
    comp: dict[UnitID, dict],
) -> bool:
    """Check if a morph unit type should be produced this frame.

    Returns True if the base unit population meets the threshold AND
    we don't already have enough of the morph unit (including pending).

    Args:
        morph_unit: The morph unit type (e.g. UnitID.BANELING, UnitID.RAVAGER).
        army_counts: Current count of each unit type (from get_own_army_dict).
        comp: The FULL army composition (before stripping morph units).
    """
    if morph_unit not in MORPH_GATES:
        return False

    # If this morph unit isn't in the current composition, skip it
    if morph_unit not in comp:
        return False

    base_unit, threshold = MORPH_GATES[morph_unit]

    # Only count comp-relevant units for total
    comp_unit_types: set[UnitID] = set(comp.keys())
    total_comp_units: int = sum(army_counts.get(uid, 0) for uid in comp_unit_types)
    if total_comp_units <= 0:
        return False

    # Check base unit threshold
    base_count: int = army_counts.get(base_unit, 0)
    base_proportion: float = base_count / total_comp_units
    if base_proportion < threshold:
        return False

    # Check if we already have enough morph units (including pending cocoons)
    morph_count: int = army_counts.get(morph_unit, 0)
    target_proportion: float = comp[morph_unit]["proportion"]
    current_proportion: float = morph_count / total_comp_units
    # Allow morphing if we're below target proportion (with small buffer)
    if current_proportion >= target_proportion:
        return False

    return True


def prioritize_affordable_units(
    comp: dict[UnitID, dict],
    minerals: float,
    vespene: float,
) -> dict[UnitID, dict]:
    """Reorder priorities so affordable units lead; unaffordable go last.

    SpawnController (non-freeflow) iterates by priority and hard-BREAKS on
    the first unaffordable unit, producing nothing that frame. If our
    priority-1 unit (Roach, 75m/25g) is gas-starved, Zerglings (50m/0g)
    behind it are never produced — the exact "lings stall when gas runs
    out, then the bank rots" footgun. This mirrors PiGBot's
    `reorder_priorities_by_resources`: push unaffordable types behind
    affordable ones so the break only bites after spendable options.

    Proportions are untouched; only "priority" values change
    (lower = higher priority in SpawnController). Returns a new dict.

    Perf note: O(k log k) sort, k = unit types in comp (~4). Negligible.

    Args:
        comp: Army composition dict (already morph-stripped).
        minerals: Current mineral bank.
        vespene: Current vespene bank.
    """
    affordable: list[UnitID] = []
    unaffordable: list[UnitID] = []
    for unit_type in comp:
        cost: Cost = COST_DICT.get(unit_type, Cost(0, 0))
        if minerals >= cost.minerals and vespene >= cost.vespene:
            affordable.append(unit_type)
        else:
            unaffordable.append(unit_type)

    # Nothing to reorder — everything affordable or nothing at all
    if not unaffordable or not affordable:
        return comp

    # Stable: preserves existing priority order within each group
    affordable.sort(key=lambda ut: comp[ut]["priority"])
    unaffordable.sort(key=lambda ut: comp[ut]["priority"])

    reordered: dict[UnitID, dict] = {}
    for new_priority, unit_type in enumerate(affordable + unaffordable):
        info: dict = dict(comp[unit_type])
        info["priority"] = new_priority
        reordered[unit_type] = info
    return reordered
