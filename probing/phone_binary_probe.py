'''Public API and command-line interface for binary phone probes.'''

import argparse
import sys
import traceback
from pathlib import Path

from probing import metadata
from probing.extract_embeddings import (
    default_model_stores_root,
    default_phraser_source_id,
)
from probing.phone_probe_metadata import (
    _default_metadata_batch_size,
    check_phone_binary_probe_metadata,
)
from probing.phone_probe_report import build_phone_binary_probe_report
from probing.phone_probe_sweep import (
    PhoneBinaryProbeSweepInterrupted,
    _default_sweep_jobs,
    run_phone_binary_probe_sweep,
)
from probing.phone_probe_worker import train_phone_binary_probe
from probing.probe_utils import default_probe_save_dir, default_results_dir


__all__ = (
    'PhoneBinaryProbeSweepInterrupted',
    'build_argument_parser',
    'build_phone_binary_probe_report',
    'check_phone_binary_probe_metadata',
    'main',
    'run_phone_binary_probe_sweep',
    'train_phone_binary_probe',
)


def _positive_integer(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return parsed


def _add_inventory_arguments(parser, allow_disabled_replacements):
    parser.add_argument(
        '--metadata-path', type=Path, default=metadata.metadata_file)
    parser.add_argument(
        '--sentence-path', type=Path, default=metadata.sentence_file)
    parser.add_argument(
        '--phraser-key-path', type=Path, default=metadata.phraser_key_file)
    if allow_disabled_replacements:
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            '--duplicate-replacement-phraser-key-path',
            type=Path,
            dest='duplicate_replacement_phraser_key_path',
        )
        group.add_argument(
            '--no-duplicate-replacement-phraser-key',
            dest='duplicate_replacement_phraser_key_path',
            action='store_const',
            const=None,
        )
        parser.set_defaults(duplicate_replacement_phraser_key_path=(
            metadata.duplicate_replacement_phraser_key_file))
    else:
        parser.add_argument(
            '--duplicate-replacement-phraser-key-path',
            type=Path,
            default=metadata.duplicate_replacement_phraser_key_file,
        )


def _add_probe_arguments(parser):
    parser.add_argument('--collar', type=int, default=2000)
    parser.add_argument('--n-embeds', type=_positive_integer)
    parser.add_argument('--n-splits', type=_positive_integer, default=5)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument(
        '--standardize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--save-probes', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        '--probe-save-dir', type=Path, default=default_probe_save_dir)
    parser.add_argument(
        '--save-predictions',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        '--results-dir', type=Path, default=default_results_dir)


def _add_verbose_argument(parser):
    parser.add_argument(
        '--verbose', action=argparse.BooleanOptionalAction, default=True)


def _add_train_arguments(parser):
    parser.add_argument('--phone', required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--layer', required=True, type=_positive_integer)
    _add_inventory_arguments(parser, allow_disabled_replacements=True)
    parser.add_argument('--model-store-path', type=Path)
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    _add_probe_arguments(parser)
    parser.add_argument(
        '--overwrite', action=argparse.BooleanOptionalAction, default=False)
    _add_verbose_argument(parser)
    parser.add_argument('--task-status-path', type=Path)


def _add_check_metadata_arguments(parser):
    _add_inventory_arguments(parser, allow_disabled_replacements=False)
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    parser.add_argument('--collar', type=int, default=2000)
    parser.add_argument(
        '--batch-size', type=_positive_integer,
        default=_default_metadata_batch_size)
    parser.add_argument('--force-metadata-check', action='store_true')
    _add_verbose_argument(parser)


def _add_sweep_arguments(parser):
    _add_inventory_arguments(parser, allow_disabled_replacements=True)
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    _add_probe_arguments(parser)
    parser.add_argument(
        '--overwrite', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        '--jobs', type=_positive_integer, default=_default_sweep_jobs)
    parser.add_argument(
        '--metadata-batch-size',
        type=_positive_integer,
        default=_default_metadata_batch_size,
    )
    parser.add_argument('--force-metadata-check', action='store_true')
    _add_verbose_argument(parser)


def _add_report_arguments(parser):
    _add_inventory_arguments(parser, allow_disabled_replacements=True)
    parser.add_argument(
        '--model-stores-root', type=Path, default=default_model_stores_root)
    _add_probe_arguments(parser)
    parser.add_argument(
        '--verify-checksums',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_verbose_argument(parser)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='Train and inspect path-based binary phone probes.')
    commands = parser.add_subparsers(dest='command', required=True)

    train_parser = commands.add_parser(
        'train', help='train every fold for one phone/model/layer task')
    _add_train_arguments(train_parser)
    train_parser.set_defaults(command_handler=_run_train_command)

    metadata_parser = commands.add_parser(
        'check-metadata',
        help='check and cache all checkpoint embedding inventories',
    )
    _add_check_metadata_arguments(metadata_parser)
    metadata_parser.set_defaults(command_handler=_run_check_metadata_command)

    sweep_parser = commands.add_parser(
        'sweep', help='train all complete checkpoint probes in subprocesses')
    _add_sweep_arguments(sweep_parser)
    sweep_parser.set_defaults(command_handler=_run_sweep_command)

    report_parser = commands.add_parser(
        'report', help='rebuild the report from persisted probe artifacts')
    _add_report_arguments(report_parser)
    report_parser.set_defaults(command_handler=_run_report_command)
    return parser


def _common_probe_arguments(arguments):
    return {
        'metadata_path': arguments.metadata_path,
        'sentence_path': arguments.sentence_path,
        'phraser_key_path': arguments.phraser_key_path,
        'duplicate_replacement_phraser_key_path': (
            arguments.duplicate_replacement_phraser_key_path),
        'model_stores_root': arguments.model_stores_root,
        'collar': arguments.collar,
        'n_embeds': arguments.n_embeds,
        'n_splits': arguments.n_splits,
        'random_state': arguments.random_state,
        'standardize': arguments.standardize,
        'save_probes': arguments.save_probes,
        'probe_save_dir': arguments.probe_save_dir,
        'save_predictions': arguments.save_predictions,
        'results_dir': arguments.results_dir,
        'verbose': arguments.verbose,
    }


def _run_train_command(arguments):
    options = _common_probe_arguments(arguments)
    options.update({
        'model_store_path': arguments.model_store_path,
        'overwrite': arguments.overwrite,
        'task_status_path': arguments.task_status_path,
    })
    train_phone_binary_probe(
        arguments.phone, arguments.model_name, arguments.layer, **options)
    return 0


def _run_check_metadata_command(arguments):
    report = check_phone_binary_probe_metadata(
        metadata_path=arguments.metadata_path,
        sentence_path=arguments.sentence_path,
        phraser_key_path=arguments.phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            arguments.duplicate_replacement_phraser_key_path),
        model_stores_root=arguments.model_stores_root,
        collar=arguments.collar,
        batch_size=arguments.batch_size,
        force_metadata_check=arguments.force_metadata_check,
        verbose=arguments.verbose,
    )
    return 0 if report['status'] == 'complete' else 1


def _run_sweep_command(arguments):
    report = run_phone_binary_probe_sweep(
        **_common_probe_arguments(arguments),
        overwrite=arguments.overwrite,
        jobs=arguments.jobs,
        metadata_batch_size=arguments.metadata_batch_size,
        force_metadata_check=arguments.force_metadata_check,
    )
    return 0 if report['status'] == 'complete' else 1


def _run_report_command(arguments):
    report = build_phone_binary_probe_report(
        **_common_probe_arguments(arguments),
        verify_checksums=arguments.verify_checksums,
    )
    return 0 if report['status'] == 'complete' else 1


def main(argv=None):
    arguments = build_argument_parser().parse_args(argv)
    try:
        return arguments.command_handler(arguments)
    except KeyboardInterrupt:
        return 130
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
