"""Zerg Test Bot — Modular Zerg opponent for testing threat detection.
Purpose: Controlled Zerg opponent with selectable behavior profiles.
         Each profile tests a specific threat type that PiGBot needs to handle.
Key Decisions: Minimal economy, profiles are toggled via class attributes.
Limitations: No defense, no upgrades, no injects. The notes assume combat losses
             free up supply, so Overlords are morphed whenever supply-blocked.

Profiles:
  - "12_pool_zerg_rush" (pre-5.0.16, 12 starting workers): attacks at 16 Zerglings
  - "8_pool_zerg_rush" (patch 5.0.16, 8 starting workers): same rush shifted
    4 supply down (Pool at 8, Overlord/drones at 10, Hatchery at 12, Queen at 14)
  - "fungal_test" (enable_rush=False, enable_fungal=True): Infestor Fungal module
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

# ── Rush profiles (pre-5.0.16 starts with 12 workers, 5.0.16 with 8) ────────
# All-in: lings attack-move the moment they pop from the pool. No wave gating.
# 12_pool_zerg_rush:
#   12  Spawning Pool
#   14  Overlord / drones to 14
#   14  Zergling x3
#   16  Hatchery (natural)
#   18  Queen
#   20  Zergling x2
# 8_pool_zerg_rush (same build shifted 4 supply down):
#   8   Spawning Pool
#   10  Overlord / drones to 10
#   10  Zergling x3
#   12  Hatchery (natural)
#   14  Queen
#   16  Zergling x2
RUSH_PROFILES: dict = {
    "12_pool_zerg_rush": {
        "drone_cap": 14,          # drone target once the pool has started
        "pool_min_workers": 11,   # pool ordered at game start (12 supply)
        "hatchery_supply": 16,    # expand to the natural (total supply used)
    },
    "8_pool_zerg_rush": {
        "drone_cap": 10,
        "pool_min_workers": 8,
        "hatchery_supply": 12,
    },
}

# config.yml `Patch` key -> rush profile
PATCH_RUSH_PROFILES: dict = {
    "Current": "12_pool_zerg_rush",
    "5.0.16": "8_pool_zerg_rush",
}
DEFAULT_RUSH_PROFILE: str = "12_pool_zerg_rush"


class ZergTestBot(BotAI):
    """Zerg test bot with modular threat behaviors.

    Profiles are selected via class attributes:
      - rush_profile: name into RUSH_PROFILES ("12_pool_zerg_rush" default)
      - enable_rush (default True): run the selected rush profile
      - enable_fungal (default False): Infestor Fungal Growth module
    """

    rush_profile: str = DEFAULT_RUSH_PROFILE
    enable_rush: bool = True
    enable_fungal: bool = False

    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.gather(self.mineral_field.closest_to(worker))

        if not self.townhalls:
            return
        cc = self.townhalls.first

        # --- Shared economy ---
        await self._run_economy(cc)

        # --- Behavior profiles ---
        if self.enable_rush:
            await self._run_rush(cc)

        if self.enable_fungal:
            await self._run_fungal(cc)

    def _rush_settings(self) -> dict:
        """Tunables for the active rush profile (falls back to the default)."""
        return RUSH_PROFILES.get(self.rush_profile, RUSH_PROFILES[DEFAULT_RUSH_PROFILE])

    async def _run_economy(self, cc) -> None:
        """Profile-aware economy.

        rush: drones to the profile cap, Overlords only when supply-blocked, no gas.
        fungal_test: original freeform economy (Drones, Overlords, Extractors).
        """
        larvae = self.larva
        if not larvae:
            return

        if self.enable_rush:
            drone_cap: int = self._rush_settings()["drone_cap"]
            # Drones only until the cap, then larvae are reserved for lings
            if self.workers.amount < drone_cap and self.supply_left > 0:
                if self.can_afford(UnitTypeId.DRONE):
                    larvae.random.train(UnitTypeId.DRONE)
            # Rush notes assume Overlords only when supply-blocked
            if self.supply_left <= 0 and not self.already_pending(UnitTypeId.OVERLORD):
                if self.can_afford(UnitTypeId.OVERLORD):
                    larvae.random.train(UnitTypeId.OVERLORD)
        else:
            if self.can_afford(UnitTypeId.DRONE) and self.supply_left > 0:
                larvae.random.train(UnitTypeId.DRONE)

            if self.supply_left < 4 and not self.already_pending(UnitTypeId.OVERLORD):
                if self.can_afford(UnitTypeId.OVERLORD):
                    larvae.random.train(UnitTypeId.OVERLORD)

            if self.structures.of_type(UnitTypeId.EXTRACTOR).amount < 2:
                if self.can_afford(UnitTypeId.EXTRACTOR):
                    vgs = self.vespene_geyser.closer_than(15, cc)
                    if vgs:
                        for vg in vgs:
                            if not self.structures.of_type(UnitTypeId.EXTRACTOR).closer_than(1, vg):
                                await self.build(UnitTypeId.EXTRACTOR, near=vg)
                                break

        for drone in self.workers.idle:
            if not self.enable_rush and self.structures.of_type(UnitTypeId.EXTRACTOR).ready:
                ref = self.structures.of_type(UnitTypeId.EXTRACTOR).ready.first
                if ref.surplus_harvesters < 0:
                    drone.gather(ref)
                    continue
            drone.gather(self.mineral_field.closest_to(drone))

    async def _run_rush(self, cc) -> None:
        """Rush profile (see RUSH_PROFILES): pool first thing, lings from the
        pool, natural Hatchery, Queen, then all-in attack at the wave size."""
        settings: dict = self._rush_settings()

        # Spawning Pool at game start (supply = workers + 1 overlord)
        if (
            not self.structures(UnitTypeId.SPAWNINGPOOL)
            and not self.already_pending(UnitTypeId.SPAWNINGPOOL)
            and self.can_afford(UnitTypeId.SPAWNINGPOOL)
            and self.workers.amount >= settings["pool_min_workers"]
        ):
            await self.build(
                UnitTypeId.SPAWNINGPOOL,
                near=cc.position.towards(self.game_info.map_center, 8),
            )

        # Hatchery at the natural (profile's total-supply mark)
        if (
            self.supply_used >= settings["hatchery_supply"]
            and len(self.townhalls) < 2
            and not self.already_pending(UnitTypeId.HATCHERY)
            and self.can_afford(UnitTypeId.HATCHERY)
        ):
            await self.expand_now()

        # Queen from the natural once it completes
        if (
            len(self.townhalls.ready) >= 2
            and self.units(UnitTypeId.QUEEN).amount < 1
            and cc.is_idle
            and not self.already_pending(UnitTypeId.QUEEN)
            and self.can_afford(UnitTypeId.QUEEN)
        ):
            cc.train(UnitTypeId.QUEEN)

        # Zerglings: all remaining larvae go to lings while the pool is up
        if self.structures(UnitTypeId.SPAWNINGPOOL).ready:
            if self.can_afford(UnitTypeId.ZERGLING) and self.supply_left >= 2:
                larvae = self.larva
                if larvae:
                    larvae.random.train(UnitTypeId.ZERGLING)

        # All-in: every idle zergling attack-moves the moment it pops.
        # Fresh lings stream to the front as soon as they hatch.
        zerglings = self.units(UnitTypeId.ZERGLING)
        if zerglings:
            target = self._rush_target()
            for ling in zerglings.idle:
                ling.attack(target)

    def _rush_target(self):
        """Prefer known enemy structures, else the enemy start location."""
        if self.enemy_structures:
            return self.enemy_structures.closest_to(self.start_location)
        if self.enemy_start_locations:
            return self.enemy_start_locations[0]
        return self.game_info.map_center

    async def _run_fungal(self, cc) -> None:
        """Fungal Growth module: build Infestors and cast on enemy clumps."""
        # Build Spawning Pool (prerequisite for Lair)
        if not self.structures.of_type(UnitTypeId.SPAWNINGPOOL) and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            await self.build(UnitTypeId.SPAWNINGPOOL, near=cc.position.towards(self.game_info.map_center, 8))

        # Upgrade to Lair (prerequisite for Infestation Pit)
        if (
            self.structures.of_type(UnitTypeId.SPAWNINGPOOL).ready
            and not self.structures.of_type(UnitTypeId.LAIR)
            and not self.already_pending(UnitTypeId.LAIR)
            and self.can_afford(UnitTypeId.LAIR)
        ):
            cc.build(UnitTypeId.LAIR)

        # Build Infestation Pit (prerequisite for Infestor)
        if (
            self.structures.of_type(UnitTypeId.LAIR).ready
            and not self.structures.of_type(UnitTypeId.INFESTATIONPIT)
            and self.can_afford(UnitTypeId.INFESTATIONPIT)
        ):
            await self.build(UnitTypeId.INFESTATIONPIT, near=cc.position.towards(self.game_info.map_center, 10))

        # Produce Infestors from Larva
        if self.structures.of_type(UnitTypeId.INFESTATIONPIT).ready:
            if self.can_afford(UnitTypeId.INFESTOR) and self.supply_left > 0:
                larvae = self.larva
                if larvae:
                    larvae.random.train(UnitTypeId.INFESTOR)

        # Infestor micro: move toward enemy and cast Fungal
        infestors = self.units(UnitTypeId.INFESTOR)
        if not infestors:
            return

        enemy_units = self.enemy_units
        if not enemy_units:
            if self.enemy_start_locations:
                target = self.enemy_start_locations[0]
                for inf in infestors:
                    inf.move(target)
            return

        for inf in infestors:
            if inf.energy >= 75:
                # Find best fungal target: largest clump of enemies
                best_target = None
                best_count = 0
                for enemy in enemy_units:
                    count = len(enemy_units.closer_than(2.0, enemy))
                    if count > best_count:
                        best_count = count
                        best_target = enemy.position

                if best_target is not None and best_count >= 2:
                    inf(AbilityId.FUNGALGROWTH_FUNGALGROWTH, best_target)
                else:
                    closest_enemy = enemy_units.closest_to(inf)
                    inf.move(closest_enemy.position.towards(inf.position, -5))
            else:
                # Not enough energy — move toward enemy
                closest_enemy = enemy_units.closest_to(inf)
                dist = inf.distance_to(closest_enemy)
                if dist > 10:
                    inf.move(closest_enemy.position.towards(inf.position, -5))