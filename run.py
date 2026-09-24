import random
import sys
from os import path
from pathlib import Path
import platform
from typing import List
from loguru import logger

from sc2 import maps
from sc2.data import AIBuild, Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

sys.path.append("ares-sc2/src/ares")
sys.path.append("ares-sc2/src")
sys.path.append("ares-sc2")

import yaml

from bot.main import Holdfast as MyBot
from competitors.Zerg_Test_Bot import PATCH_RUSH_PROFILES, ZergTestBot
from ladder import run_ladder_game

plt = platform.system()
# change if non default setup / linux
# if having issues with this, modify `map_list` below manually
if plt == "Windows":
    MAPS_PATH: str = "C:\\Program Files (x86)\\StarCraft II\\Maps"
elif plt == "Darwin":
    MAPS_PATH: str = "/Applications/StarCraft II/Maps"
elif plt == "Linux":
    # path would look a bit like this on linux after installing
    # SC2 via lutris
    MAPS_PATH: str = (
        "~/<username>/Games/battlenet/drive_c/Program Files (x86)/StarCraft II/Maps"
    )
else:
    logger.error(f"{plt} not supported")
    sys.exit()

CONFIG_FILE: str = "config.yml"
MAP_FILE_EXT: str = "SC2Map"
MY_BOT_NAME: str = "MyBotName"
MY_BOT_RACE: str = "MyBotRace"
OPPONENT_BOT: str = "OpponentBot"
MAPS_KEY: str = "Maps"
PATCH_KEY: str = "Patch"

# Selectable local opponents: name in config.yml -> (Bot factory, Race)
LOCAL_OPPONENTS: dict = {
    "ZergTestBot": (ZergTestBot, Race.Zerg),
}


def get_local_opponent(config: dict):
    """Return a Bot instance from LOCAL_OPPONENTS via the OpponentBot config key,
    or None to fall back to a random Computer. Sets the rush profile from Patch."""
    name: str = config.get(OPPONENT_BOT, "")
    if name not in LOCAL_OPPONENTS:
        if name:
            logger.warning(f"Unknown OpponentBot '{name}', using Computer opponent")
        return None
    bot_cls, race = LOCAL_OPPONENTS[name]
    bot = bot_cls()
    if bot_cls is ZergTestBot:
        patch: str = config.get(PATCH_KEY, "Current")
        profile: str = PATCH_RUSH_PROFILES.get(patch, "")
        if profile:
            bot.rush_profile = profile
        elif patch in PATCH_RUSH_PROFILES.values():
            # patch key already holds a profile name directly
            bot.rush_profile = patch
        else:
            logger.warning(
                f"Unknown Patch '{patch}', using {bot.rush_profile}. "
                f"Valid: {sorted(PATCH_RUSH_PROFILES)}"
            )
    return Bot(race, bot, name)


def get_map_list(config: dict) -> List[str]:
    """Maps from the Maps config key, else auto-discovered .SC2Map files."""
    configured: List[str] = config.get(MAPS_KEY) or []
    if configured:
        return configured
    return [
        p.name.replace(f".{MAP_FILE_EXT}", "")
        for p in Path(MAPS_PATH).glob(f"*.{MAP_FILE_EXT}")
        if p.is_file()
    ]


def main():
    bot_name: str = "MyBot"
    race: Race = Race.Random
    config: dict = {}

    __user_config_location__: str = path.abspath(".")
    user_config_path: str = path.join(__user_config_location__, CONFIG_FILE)
    # attempt to get race and bot name from config file if they exist
    if path.isfile(user_config_path):
        with open(user_config_path) as config_file:
            config: dict = yaml.safe_load(config_file) or {}
        if MY_BOT_NAME in config:
            bot_name = config[MY_BOT_NAME]
        if MY_BOT_RACE in config:
            race = Race[config[MY_BOT_RACE].title()]

    bot1 = Bot(race, MyBot(), bot_name)

    if "--LadderServer" in sys.argv:
        # Ladder game started by LadderManager
        print("Starting ladder game...")
        result, opponentid = run_ladder_game(bot1)
        print(result, " against opponent ", opponentid)
    else:
        # Local game
        map_list: List[str] = get_map_list(config)
        if len(map_list) == 0:
            logger.error(f"Can't find maps, please check `MAPS_PATH` in `run.py'")
            logger.info("Trying back up option")
            logger.info(
                f"\nLooking for maps in {MAPS_PATH} but didn't find anything. \n"
                f"If this path is correct please ensure maps are present. \n"
                f"If this path is incorrect please edit the `MAPS_PATH` in `run.py` \n"
                f"Tip: If you're using linux, MAPS_PATH will definitely need updating\n"
            )

        # Local game: configured opponent bot if set, else random Computer
        opponent = get_local_opponent(config)
        if opponent is None:
            opponent = Computer(
                random.choice([Race.Terran, Race.Zerg, Race.Protoss]),
                Difficulty.CheatVision,
                ai_build=AIBuild.Macro,
            )

        print("Starting local game...")
        run_game(
            maps.get(random.choice(map_list)),
            [bot1, opponent],
            realtime=False,
        )


# Start game
if __name__ == "__main__":
    main()
