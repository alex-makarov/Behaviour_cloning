"""Utilities of visualising an environment."""

# This is a modified version of the original code from OpenAI's Gym
# It is modified to work with the CarRacing-v3 environment and use the steering wheel controller
# to generate the continuous actions. It will not work for other environments without modification.

from __future__ import annotations

from typing import Callable, List

import numpy as np

import gymnasium as gym
from gymnasium import Env, logger
from gymnasium.core import ActType


try:
    import pygame
    from pygame import Surface
    from pygame.event import Event
    from pygame import joystick
    pygame.joystick.init()
except ImportError as e:
    raise gym.error.DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
    ) from e


class MissingControllerToAction(Exception):
    """Raised when the environment does not have a default ``controller_to_action`` mapping."""


class ControllerPlayableGame:
    """Wraps an environment allowing controller inputs to interact with the environment."""

    def __init__(
        self,
        env: Env,
        controller_to_action: dict[tuple[int, float, bool], tuple[float, float]] | None = None,
        zoom: float | None = None,
    ):
        """Wraps an environment with a dictionary of controller configuration to action and if to zoom in on the environment.

        Args:
            env: The environment to play
            controller_to_action: The dictionary of controller axis and action value
            zoom: If to zoom in on the environment render
        """
        if env.render_mode not in {"rgb_array", "rgb_array_list"}:
            raise ValueError(
                "PlayableGame wrapper works only with rgb_array and rgb_array_list render modes, "
                f"but your environment render_mode = {env.render_mode}."
            )

        self.env = env
        self.relevant_axis = self._get_relevant_axis(controller_to_action)
        self.controller_to_action = controller_to_action
        # self.video_size is the size of the video that is being displayed.
        # The window size may be larger, in that case we will add black bars
        self.video_size = self._get_video_size(zoom)
        self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.joystick = joystick.Joystick(0)
        self.joystick_values = [None] * self.joystick.get_numaxes()
        self.running = True

    def _get_relevant_axis(
        self, controller_to_action: dict[tuple[int, float, bool], tuple[float, float]] | None = None
    ) -> set:
        if controller_to_action is None:
            if self.env.has_wrapper_attr("get_controller_to_action"):
                controller_to_action = self.env.get_wrapper_attr("get_controller_to_action")()
            else:
                assert self.env.spec is not None
                raise MissingControllerToAction(
                    f"{self.env.spec.id} does not have explicit controller to action mapping, "
                    "please specify one manually, `play(env, controller_to_action=...)`"
                )
        assert isinstance(controller_to_action, dict)
        relevant_axis = set(sum((list(k) for k in controller_to_action.keys()), []))
        return relevant_axis

    def joystick_to_action(self) -> ActType:
        i = 0
        action = [0.0, 0.0, 0.0]
        for axis_config, mapped_action in self.controller_to_action.items():
            if self.joystick_values[axis_config[0]] is not None: # Handle inital None values from joystick
                raw_value = self.joystick_values[axis_config[0]]
                # Invert if needed
                if axis_config[2]:
                    raw_value = -raw_value
                # Scale and clip
                raw_value = np.clip(raw_value*axis_config[1], -1.0, 1.0)

                # Map python's joystick default [-1.0, 1.0] range to the action space
                min_raw, max_raw = -1.0, 1.0
                min_mapped, max_mapped = mapped_action[0], mapped_action[1]
                mapped_value = (raw_value - min_raw) / (max_raw - min_raw) * (max_mapped - min_mapped) + min_mapped
                action[i] = mapped_value
            i += 1

        return np.array(action, dtype=np.float32)

    def _get_video_size(self, zoom: float | None = None) -> tuple[int, int]:
        rendered = self.env.render()
        if isinstance(rendered, List):
            rendered = rendered[-1]
        assert rendered is not None and isinstance(rendered, np.ndarray)
        video_size = (rendered.shape[1], rendered.shape[0])

        if zoom is not None:
            video_size = (int(video_size[0] * zoom), int(video_size[1] * zoom))

        return video_size

    def process_event(self, event: Event):
        """Processes a PyGame event.

        In particular, this function is used to keep track of which buttons are currently pressed
        and to exit the :func:`play` function when the PyGame window is closed.

        Args:
            event: The event to process
        """
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.WINDOWRESIZED:
            # Compute the maximum video size that fits into the new window
            scale_width = event.x / self.video_size[0]
            scale_height = event.y / self.video_size[1]
            scale = min(scale_height, scale_width)
            self.video_size = (scale * self.video_size[0], scale * self.video_size[1])
        elif event.type == pygame.JOYAXISMOTION:
            if event.axis in self.relevant_axis:
                self.joystick_values[event.axis] = event.value


def display_arr(
    screen: Surface, arr: np.ndarray, video_size: tuple[int, int], transpose: bool
):
    """Displays a numpy array on screen.

    Args:
        screen: The screen to show the array on
        arr: The array to show
        video_size: The video size of the screen
        transpose: If to transpose the array on the screen
    """
    assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    # We might have to add black bars if surface_size is larger than video_size
    surface_size = screen.get_size()
    width_offset = (surface_size[0] - video_size[0]) / 2
    height_offset = (surface_size[1] - video_size[1]) / 2
    screen.fill((0, 0, 0))
    screen.blit(pyg_img, (width_offset, height_offset))


def play(
    env: Env,
    transpose: bool | None = True,
    fps: int | None = None,
    zoom: float | None = None,
    callback: Callable | None = None,
    controller_to_action: dict[tuple[int, float, float], tuple[float, float]] | None = None,
    seed: int | None = None,
    noop: ActType = 0
):
    """Allows the user to play the environment using a controller.

    Args:
        env: Environment to use for playing.
        transpose: If this is ``True``, the output of observation is transposed. Defaults to ``True``.
        fps: Maximum number of steps of the environment executed every second. If ``None`` (the default),
            ``env.metadata["render_fps""]`` (or 30, if the environment does not specify "render_fps") is used.
        zoom: Zoom the observation in, ``zoom`` amount, should be positive float
        callback: If a callback is provided, it will be executed after every step. It takes the following input:

            * obs_t: observation before performing action
            * obs_tp1: observation after performing action
            * action: action that was executed
            * rew: reward that was received
            * terminated: whether the environment is terminated or not
            * truncated: whether the environment is truncated or not
            * info: debug info
        controller_to_action:  Mapping from axis config to action performed.
            For example:

            controller_to_action={
                (0, 2.0, False): (-1.0, 1.0),
                (2, 1.0, True): (0.0, 1.0),
                (1, 1.0, True): (0.0, 1.0)
            }

            Axis 0 controls the steering, axis 2 controls the throttle, and axis 1 controls the brake. 
            Throttle and brake are inverted, and the steering is scaled by 2.0 to increase sensitivity.

        seed: Random seed used when resetting the environment. If None, no seed is used.
        noop: The action used when no key input has been entered, or the entered key combination is unknown.

    Example:
        >>> import gymnasium as gym
        >>> import numpy as np
        >>> from gymnasium.utils.play import play
        >>> play(gym.make("CarRacing-v3", render_mode="rgb_array"),  # doctest: +SKIP
        ...     controller_to_action={
        ...         (0, 2.0, False): (-1.0, 1.0),
        ...         (2, 1.0, True): (0.0, 1.0),
        ...         (1, 1.0, True): (0.0, 1.0)
        ...     },
        ...     noop=np.array([0, 0, 0], dtype=np.float32)
        ... )

        Above code works also if the environment is wrapped, so it's particularly useful in
        verifying that the frame-level preprocessing does not render the game
        unplayable.

        If you wish to plot real time statistics as you play, you can use
        :class:`PlayPlot` from the original Gym utilities.
    """
    env.reset(seed=seed)

    if controller_to_action is None:
        if env.has_wrapper_attr("get_controller_to_action"):
            controller_to_action = env.get_wrapper_attr("get_controller_to_action")()
        else:
            assert env.spec is not None
            raise MissingControllerToAction(
                f"{env.spec.id} does not have explicit controller to action mapping, "
                "please specify one manually"
            )

    assert controller_to_action is not None

    assert isinstance(controller_to_action, dict)
    for axis, action in controller_to_action.items():
        assert len(axis) == 3
        assert all(isinstance(a, (int, float, bool)) for a in axis)
        assert len(action) == 2
        assert all(isinstance(a, float) for a in action)


    game = ControllerPlayableGame(env, controller_to_action, zoom)

    if fps is None:
        fps = env.metadata.get("render_fps", 30)

    done, obs = True, None
    clock = pygame.time.Clock()

    while game.running:
        if done:
            done = False
            obs = env.reset(seed=seed)
        else:
            action = game.joystick_to_action()
            print (action)
            prev_obs = obs
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if callback is not None:
                callback(prev_obs, obs, action, rew, terminated, truncated, info)
        if obs is not None:
            rendered = env.render()
            if isinstance(rendered, List):
                rendered = rendered[-1]
            assert rendered is not None and isinstance(rendered, np.ndarray)
            display_arr(
                game.screen, rendered, transpose=transpose, video_size=game.video_size
            )

        # process pygame events
        for event in pygame.event.get():
            game.process_event(event)

        pygame.display.flip()
        clock.tick(fps)
    pygame.quit()
