"""Map editing strength onto an actual stored inversion level."""
from dataclasses import dataclass
from decimal import Decimal
import math
from numbers import Real

from diffusion_editor.diffusion.schedule import DDIMSchedule


@dataclass(frozen=True)
class StrengthSelection:
    strength: float
    reverse_steps: int
    start_timestep: int
    alpha_cumprod: float

    @property
    def noise_std(self) -> float:
        return math.sqrt(1 - self.alpha_cumprod)


def select_edit_start(schedule: DDIMSchedule, strength: float) -> StrengthSelection:
    """Use k=ceil(strength*N), select inversion level k (0 is clean).

    Strength controls a discrete number of reverse steps, not a linear timestep
    or linear noise standard deviation. Every nonzero strength selects >=1 step.
    """
    if (isinstance(strength, bool) or not isinstance(strength, Real)
            or not math.isfinite(strength) or not 0 <= strength <= 1):
        raise ValueError("strength must be finite and in [0,1]")
    # Decimal text avoids e.g. 0.14 * 50 becoming 7.000000000000001 and
    # spuriously selecting step 8. The public strength is still a real scalar.
    count = math.ceil(Decimal(str(float(strength))) * len(schedule.sampling_timesteps))
    timestep = schedule.inversion_timesteps[count]
    return StrengthSelection(float(strength), count, timestep, schedule.alpha(timestep).item())
