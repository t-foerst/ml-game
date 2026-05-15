"""Sound-Manager für ml-game Client."""

from pathlib import Path
import pygame

_ASSETS = Path(__file__).parent / "assets"


class SoundManager:
    def __init__(self) -> None:
        self.enabled = True
        self._thrust_playing  = False
        pygame.mixer.set_num_channels(8)
        self._shoot_channel     = pygame.mixer.Channel(0)
        self._thrust_channel    = pygame.mixer.Channel(1)
        self._thrustend_channel = pygame.mixer.Channel(2)
        self._explosion_channel = pygame.mixer.Channel(3)

        self._shoot     = self._load_sound("shoot.mp3",     volume=0.5)
        self._explosion = self._load_sound("explosion.mp3", volume=1.0)
        self._thrust    = self._load_sound("thrust.mp3",    volume=0.02)
        self._thrustend = self._load_sound("thrustend.mp3", volume=0.02)

        try:
            pygame.mixer.music.load(str(_ASSETS / "background-music.mp3"))
            pygame.mixer.music.set_volume(0.24)
        except Exception as e:
            print(f"Musik-Ladefehler: {e}")

    def _load_sound(self, filename: str, volume: float = 1.0) -> pygame.mixer.Sound | None:
        try:
            s = pygame.mixer.Sound(str(_ASSETS / filename))
            s.set_volume(volume)
            return s
        except Exception as e:
            print(f"Sound-Ladefehler {filename}: {e}")
            return None

    # ── Musik ─────────────────────────────────────────────────────────────────

    def start_music(self) -> None:
        if self.enabled:
            pygame.mixer.music.play(-1)

    def stop_music(self) -> None:
        pygame.mixer.music.stop()

    # ── Effekte ───────────────────────────────────────────────────────────────

    def shoot(self) -> None:
        if not self.enabled or self._shoot is None:
            return
        self._shoot_channel.stop()
        self._shoot_channel.play(self._shoot)

    def explosion(self) -> None:
        if not self.enabled or self._explosion is None:
            return
        self._explosion_channel.play(self._explosion)

    def update_thrust(self, thrusting: bool) -> None:
        if not self.enabled:
            if self._thrust_playing:
                self._thrust_channel.stop()
                self._thrust_playing = False
            return

        if thrusting and not self._thrust_playing:
            if self._thrust:
                self._thrust_channel.play(self._thrust, loops=-1)
            self._thrust_playing = True
        elif not thrusting and self._thrust_playing:
            self._thrust_channel.stop()
            if self._thrustend:
                self._thrustend_channel.play(self._thrustend)
            self._thrust_playing = False

    def stop_all(self) -> None:
        self._thrust_channel.stop()
        self._shoot_channel.stop()
        self._thrust_playing = False
