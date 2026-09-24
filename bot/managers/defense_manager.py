# Purpose: Detect threats near our bases and redirect the main army to defend
# Key Decisions: Hysteresis on the under_attack flag (trigger at 4.0 supply,
#   clear below 2.0 after a 5s grace) prevents army flip-flopping on threat
#   flicker. Combines ground + flying near-base tracking so air harass
#   (banshees, mutas) also pulls the army home. ARES EnemyToBaseManager
#   already applies distance hysteresis (tracks within 15, clears at 24).
#   Queens (3 supply threshold) and rush workers handle sub-army threats.
# Limitations: Single threat blob (largest); simultaneous multi-base attacks
#   rely on queens, workers, and emergency spines elsewhere.

from ares import AresBot
from cython_extensions import cy_center
from sc2.position import Point2
from sc2.units import Units

from bot.managers.queen_manager import HARMLESS_THREAT_TYPES

# ── Constants ────────────────────────────────────────────────────────────────
# Near-base threat supply that flips the army to defensive
UNDER_ATTACK_TRIGGER_SUPPLY: float = 4.0
# Threat supply must drop below this before the flag clears
UNDER_ATTACK_CLEAR_SUPPLY: float = 2.0
# Seconds the threat must stay small before the army resumes attacking
UNDER_ATTACK_GRACE_PERIOD: float = 5.0


class DefenseManager:
    """Tracks enemy forces near our bases and exposes an under_attack flag.

    The main army redirects to the threat when the flag flips True
    (CombatManager reads threat_position for its rally point). Hysteresis
    keeps the flag stable: high threshold to trigger, low threshold plus
    a grace window to clear.
    """

    def __init__(self, ai: AresBot) -> None:
        self.ai: AresBot = ai
        self._under_attack: bool = False
        self._threat_position: Point2 | None = None
        self._last_high_threat_time: float = 0.0

    @property
    def under_attack(self) -> bool:
        """Whether a significant enemy force is near one of our bases."""
        return self._under_attack

    @property
    def threat_position(self) -> Point2 | None:
        """Center of the largest enemy blob near our bases, or None."""
        return self._threat_position

    def update(self) -> None:
        """Refresh near-base threat state with hysteresis. Call once per frame."""
        threats: Units = self._collect_threats()

        if threats:
            self._threat_position = Point2(cy_center(threats))
        # else keep last known position while the grace window runs

        # Perf note: single supply sum over near-base enemies — trivial.
        supply: float = self.ai.get_total_supply(threats)

        if supply >= UNDER_ATTACK_TRIGGER_SUPPLY:
            self._under_attack = True
            self._last_high_threat_time = self.ai.time
        elif self._under_attack:
            # Hysteresis: only clear once the threat has been small for a
            # sustained window (and is actually below the clear threshold)
            if (
                supply < UNDER_ATTACK_CLEAR_SUPPLY
                and self.ai.time - self._last_high_threat_time
                >= UNDER_ATTACK_GRACE_PERIOD
            ):
                self._under_attack = False
                self._threat_position = None

    def _collect_threats(self) -> Units:
        """All enemy units ARES tracks near our townhalls (ground + air).

        Harmless units (overlords, observers, single scouts) are dropped so
        the army only responds to real threats.

        Perf note: tag-set union + one tags_in lookup — O(tracked).
        """
        ground: dict[int, set[int]] = self.ai.mediator.get_ground_enemy_near_bases
        flying: dict[int, set[int]] = self.ai.mediator.get_flying_enemy_near_bases

        all_tags: set[int] = set()
        for enemy_tags in ground.values():
            all_tags.update(enemy_tags)
        for enemy_tags in flying.values():
            all_tags.update(enemy_tags)

        if not all_tags:
            return Units([], self.ai)
        threats: Units = self.ai.enemy_units.tags_in(all_tags)
        return threats.filter(lambda u: u.type_id not in HARMLESS_THREAT_TYPES)
