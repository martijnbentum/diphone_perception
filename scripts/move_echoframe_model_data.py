'''Move one model's hidden-state data into a dedicated Echoframe store.'''

from pathlib import Path

from echoframe import Store
from echoframe.transfer import move_hidden_states_for_model


def move_echoframe_model_data(model_name, source_path, destination_path):
    '''Move all hidden states for model_name into a new Echoframe store.'''
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError('model_name must be a non-empty string')

    source_path = Path(source_path).expanduser()
    destination_path = Path(destination_path).expanduser()
    if not source_path.is_dir():
        raise FileNotFoundError(f'source store does not exist: {source_path}')
    if destination_path.exists() or destination_path.is_symlink():
        raise FileExistsError(
            f'destination store already exists: {destination_path}')

    print(f'Moving hidden states for {model_name!r}', flush=True)
    print(f'  Source:      {source_path}', flush=True)
    print(f'  Destination: {destination_path}', flush=True)

    source = None
    destination = None
    try:
        print('Opening the source store...', flush=True)
        source = Store(source_path)
        model_metadata = source.load_model_metadata(model_name)
        if model_metadata is None:
            raise ValueError(
                f'model is not registered in the source store: {model_name!r}')

        print('Creating the empty destination store...', flush=True)
        destination = Store(
            destination_path,
            max_shard_size_bytes=100_000_000,
        )

        print(
            'Copying data and verifying the destination; source data will '
            'only be deleted after verification...',
            flush=True,
        )
        result = move_hidden_states_for_model(
            source,
            destination,
            model_name=model_name,
            batch_size=100,
        )

        print('Move completed successfully.', flush=True)
        print(
            f"  Copied records:       {result['copied_count']:,}",
            flush=True,
        )
        print(
            f"  Destination shards:  {result['destination_shard_count']:,}",
            flush=True,
        )
        print(
            f"  Deleted records:      {result['deleted_count']:,}",
            flush=True,
        )
        print(
            f"  Deleted source shards: {result['deleted_shard_count']:,}",
            flush=True,
        )
        return result
    except Exception as error:
        print(f'Move failed: {error}', flush=True)
        raise
    finally:
        print('Closing stores...', flush=True)
        try:
            if destination is not None:
                destination.close()
        finally:
            if source is not None:
                source.close()
        print('Stores closed.', flush=True)
