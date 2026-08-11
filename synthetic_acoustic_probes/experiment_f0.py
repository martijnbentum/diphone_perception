'''Experiment entry points for the pure-tone F0 probe.'''

from pathlib import Path

from phraser import Store

import locations

from .phraser_store import add_stimuli
from .stimuli import pure_tone_stimuli


def create_auditory_stimuli(output_root=locations.f0_pure_tone_stimuli,
    overwrite=False):
    '''Generate and save the complete pure-tone F0 stimulus grid.

    output_root:  Destination package directory.
    overwrite:  Replace an existing stimulus package when true.
    '''

    return pure_tone_stimuli(
        save=True,
        output_root=output_root,
        overwrite=overwrite,
    )


def create_f0_pure_tone_phraser_store(
    stimulus_package=locations.f0_pure_tone_stimuli,
    store_path=locations.f0_pure_tone_phraser_store,
):
    '''Create and fill the experiment-specific Phraser store.

    stimulus_package:  Package created by ``create_auditory_stimuli``.
    store_path:  Destination of the dedicated Phraser store.
    '''

    store = Store(store_path)
    try: add_stimuli(stimulus_package, store)
    except Exception:
        store.close()
        raise
    return store


def load_f0_pure_tone_phraser_store(
    store_path=locations.f0_pure_tone_phraser_store,
):
    '''Open and return the existing experiment-specific Phraser store.

    store_path:  Path created by ``create_f0_pure_tone_phraser_store``.
    '''

    store_path = Path(store_path)
    if not store_path.is_dir():
        raise FileNotFoundError(f'F0 Phraser store not found: {store_path}')
    return Store(store_path)
