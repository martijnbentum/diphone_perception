'''Bounded subprocess orchestration for binary phone probes.'''

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from probing import metadata, probe_utils
from probing.extract_embeddings import default_model_stores_root
from probing.phone_probe_common import (
    _close_phones_store,
    _compact_error,
    _sweep_counts,
    _task_identity,
    _utc_timestamp,
    _validate_cached_phone_labels,
    _validated_phone_label_inventory,
)
from probing.phone_probe_metadata import (
    _default_metadata_batch_size,
    _validate_metadata_preflight_arguments,
    check_phone_binary_probe_metadata,
)
from probing.phone_probe_report import build_phone_binary_probe_report
from probing.probe_utils import default_probe_save_dir, default_results_dir


_default_sweep_jobs = 31
_default_sweep_poll_interval = 0.2
_failure_log_tail_lines = 40


class PhoneBinaryProbeSweepInterrupted(KeyboardInterrupt):
    '''Raised after an interrupted sweep has stopped and recorded its work.'''

    def __init__(self, report):
        super().__init__('phone binary-probe sweep interrupted')
        self.report = report


def _validate_sweep_arguments(jobs, n_embeds, n_splits, standardize):
    if isinstance(jobs, bool) or not isinstance(jobs, int):
        raise TypeError('jobs must be a positive integer')
    if jobs <= 0:
        raise ValueError('jobs must be a positive integer')
    if n_embeds is not None:
        if isinstance(n_embeds, bool) or not isinstance(n_embeds, int):
            raise TypeError('n_embeds must be a positive integer or None')
        if n_embeds <= 0:
            raise ValueError('n_embeds must be a positive integer or None')
    probe_utils.validate_probe_arguments(n_splits, standardize)


def _load_sweep_phone_labels(
    metadata_path,
    sentence_path,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    n_embeds,
):
    phones = metadata.Phones(
        path=metadata_path,
        sentence_path=sentence_path,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
    )
    try:
        labels, _ = _validated_phone_label_inventory(
            phones, n_embeds=n_embeds)
    finally:
        _close_phones_store(phones)
    return labels


def _sweep_phone_labels_from_preflight(
    metadata_report,
    *,
    metadata_path,
    sentence_path,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    n_embeds,
):
    labels = metadata_report.get('phone_labels')
    if labels is not None:
        return _validate_cached_phone_labels(
            labels, metadata_report.get('phones_per_label'), n_embeds)
    if metadata_report.get('phraser_store_opened'):
        raise ValueError(
            'metadata preflight opened the Phraser store but could not '
            'produce a valid phone-label inventory; refusing to open a '
            'second Phraser store in this process')
    return _load_sweep_phone_labels(
        metadata_path,
        sentence_path,
        phraser_key_path,
        duplicate_replacement_phraser_key_path,
        n_embeds,
    )


def _build_sweep_task_lists(metadata_report, phone_labels):
    tasks = []
    skipped = []
    for model in metadata_report['models']:
        model_name = model['model_name']
        store_path = model['store_path']
        for layer in model['layers']:
            if layer['status'] == 'complete' and layer['complete']:
                tasks.extend({
                    'phone': phone,
                    'model_name': model_name,
                    'layer': layer['layer'],
                    'model_store_path': store_path,
                } for phone in phone_labels)
                continue
            skipped.append({
                'model_name': model_name,
                'model_store_path': store_path,
                'layer': layer['layer'],
                'metadata_status': layer['status'],
                'n_total': layer.get('n_total'),
                'n_available': layer.get('n_available'),
                'n_missing': layer.get('n_missing'),
                'n_phone_tasks': len(phone_labels),
                'reason': 'embedding metadata inventory is not complete',
                'error': layer.get('error'),
            })
    return tasks, skipped


def _append_boolean_cli_argument(command, name, value):
    command.append(f'--{name}' if value else f'--no-{name}')


def _build_train_subprocess_command(
    task,
    *,
    metadata_path,
    sentence_path,
    phraser_key_path,
    duplicate_replacement_phraser_key_path,
    model_stores_root,
    collar,
    n_embeds,
    n_splits,
    random_state,
    standardize,
    save_probes,
    probe_save_dir,
    save_predictions,
    results_dir,
    overwrite,
    verbose,
    task_status_path,
):
    command = [
        sys.executable,
        '-u',
        '-m',
        'probing.phone_binary_probe',
        'train',
        '--phone',
        task['phone'],
        '--model-name',
        task['model_name'],
        '--layer',
        str(task['layer']),
        '--metadata-path',
        str(metadata_path),
        '--sentence-path',
        str(sentence_path),
        '--phraser-key-path',
        str(phraser_key_path),
    ]
    if duplicate_replacement_phraser_key_path is None:
        command.append('--no-duplicate-replacement-phraser-key')
    else:
        command.extend([
            '--duplicate-replacement-phraser-key-path',
            str(duplicate_replacement_phraser_key_path),
        ])
    command.extend([
        '--model-store-path',
        str(task['model_store_path']),
        '--model-stores-root',
        str(model_stores_root),
        '--collar',
        str(collar),
    ])
    if n_embeds is not None:
        command.extend(['--n-embeds', str(n_embeds)])
    command.extend([
        '--n-splits',
        str(n_splits),
        '--random-state',
        str(random_state),
    ])
    _append_boolean_cli_argument(command, 'standardize', standardize)
    _append_boolean_cli_argument(command, 'save-probes', save_probes)
    command.extend(['--probe-save-dir', str(probe_save_dir)])
    _append_boolean_cli_argument(
        command, 'save-predictions', save_predictions)
    command.extend(['--results-dir', str(results_dir)])
    _append_boolean_cli_argument(command, 'overwrite', overwrite)
    _append_boolean_cli_argument(command, 'verbose', verbose)
    command.extend(['--task-status-path', str(task_status_path)])
    return command


def _read_task_status(path):
    try:
        with Path(path).open(encoding='utf-8') as file:
            value = json.load(file)
        if not isinstance(value, dict):
            raise TypeError('task status must be a JSON object')
        return value, None
    except (OSError, ValueError, TypeError) as error:
        return None, _compact_error(error)


def _read_log_tail(path, n_lines=_failure_log_tail_lines):
    try:
        with Path(path).open('rb') as file:
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(max(0, size - 64 * 1024))
            text = file.read().decode('utf-8', errors='replace')
    except OSError as error:
        return f'Could not read worker log: {type(error).__name__}: {error}'
    return '\n'.join(text.splitlines()[-n_lines:])


def _worker_outcome(worker):
    worker['log_file'].close()
    status, status_error = _read_task_status(worker['status_path'])
    expected_identity = _task_identity(
        worker['task']['phone'],
        worker['task']['model_name'],
        worker['task']['layer'],
    )
    valid_status = (
        status is not None
        and status.get('task') == expected_identity
        and status.get('status') in {'completed', 'already_complete'}
    )
    succeeded = worker['process'].returncode == 0 and valid_status
    outcome = {
        'task_index': worker['task_index'],
        'task': expected_identity,
        'model_store_path': str(worker['task']['model_store_path']),
        'status': status['status'] if succeeded else 'failed',
        'returncode': worker['process'].returncode,
        'elapsed_seconds': round(
            time.perf_counter() - worker['started'], 6),
        'worker_status': status,
        'command': worker['command'],
    }
    if not succeeded:
        if status is not None and status.get('error') is not None:
            outcome['error'] = status['error']
        elif status_error is not None:
            outcome['error'] = status_error
        else:
            outcome['error'] = {
                'type': 'WorkerProcessError',
                'message': (
                    f'worker exited with status '
                    f'{worker["process"].returncode} without a valid '
                    'completed task status'),
            }
        outcome['log_tail'] = _read_log_tail(worker['log_path'])
    return outcome


def _failed_launch_outcome(task, task_index, command, log_path, error):
    return {
        'task_index': task_index,
        'task': _task_identity(
            task['phone'], task['model_name'], task['layer']),
        'model_store_path': str(task['model_store_path']),
        'status': 'failed',
        'returncode': None,
        'elapsed_seconds': 0.0,
        'worker_status': None,
        'command': command,
        'error': _compact_error(error),
        'log_tail': _read_log_tail(log_path),
    }


def _interrupted_worker_outcome(worker):
    worker['log_file'].close()
    status, _ = _read_task_status(worker['status_path'])
    return {
        'task_index': worker['task_index'],
        'task': _task_identity(
            worker['task']['phone'],
            worker['task']['model_name'],
            worker['task']['layer'],
        ),
        'model_store_path': str(worker['task']['model_store_path']),
        'status': 'interrupted',
        'returncode': worker['process'].returncode,
        'elapsed_seconds': round(
            time.perf_counter() - worker['started'], 6),
        'worker_status': status,
        'command': worker['command'],
        'log_tail': _read_log_tail(worker['log_path']),
    }


def _format_clock(seconds):
    if seconds is None:
        return '--:--:--'
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class _SweepProgress:
    def __init__(self, total, started, verbose):
        self.total = total
        self.started = started
        self.verbose = verbose
        self.is_terminal = sys.stdout.isatty()
        self.last_length = 0
        self.last_finished = None
        self.last_printed = 0.0

    def update(self, active, outcomes, force=False):
        if not self.verbose:
            return
        counts = _sweep_counts(outcomes)
        finished = sum(counts.values())
        now = time.perf_counter()
        interval = 1.0 if self.is_terminal else 30.0
        changed = finished != self.last_finished
        if not force and not changed and now - self.last_printed < interval:
            return
        elapsed = now - self.started
        eta = (
            elapsed / finished * (self.total - finished)
            if finished else None
        )
        line = (
            f'Phone probes: {finished}/{self.total} finished | '
            f'{active} active | {counts["completed"]} trained | '
            f'{counts["already_complete"]} already complete | '
            f'{counts["failed"]} failed | '
            f'elapsed {_format_clock(elapsed)} | ETA {_format_clock(eta)}'
        )
        if self.is_terminal:
            print(
                '\r' + line.ljust(self.last_length),
                end='',
                flush=True,
            )
            self.last_length = max(self.last_length, len(line))
        else:
            print(line, flush=True)
        self.last_finished = finished
        self.last_printed = now

    def finish(self, active, outcomes):
        self.update(active, outcomes, force=True)
        if self.verbose and self.is_terminal:
            print(flush=True)


def _print_worker_failure(outcome):
    task = outcome['task']
    print(
        f'Phone probe worker failed: {task["model_name"]} layer '
        f'{task["layer"]} phone {task["phone"]!r}',
        file=sys.stderr,
        flush=True,
    )
    log_tail = outcome.get('log_tail')
    if log_tail:
        print('--- worker log tail ---', file=sys.stderr)
        print(log_tail, file=sys.stderr)
        print('--- end worker log tail ---', file=sys.stderr, flush=True)


def _stop_active_workers(active):
    for worker in active:
        if worker['process'].poll() is None:
            try:
                worker['process'].terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    while any(worker['process'].poll() is None for worker in active):
        if time.monotonic() >= deadline:
            break
        try:
            time.sleep(0.05)
        except KeyboardInterrupt:
            break
    for worker in active:
        if worker['process'].poll() is None:
            try:
                worker['process'].kill()
            except ProcessLookupError:
                pass
    for worker in active:
        try:
            worker['process'].wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass


def _run_sweep_subprocesses(
    tasks,
    *,
    jobs,
    temporary_directory,
    command_options,
    poll_interval,
    verbose,
):
    tasks = list(tasks)
    active = []
    outcomes = []
    next_task_index = 0
    started = time.perf_counter()
    progress = _SweepProgress(len(tasks), started, verbose)
    interrupted = False
    repository_root = Path(__file__).resolve().parent.parent

    try:
        while next_task_index < len(tasks) or active:
            while next_task_index < len(tasks) and len(active) < jobs:
                task_index = next_task_index
                task = tasks[task_index]
                next_task_index += 1
                status_path = (
                    Path(temporary_directory)
                    / f'task-{task_index:06d}.status.json')
                log_path = (
                    Path(temporary_directory)
                    / f'task-{task_index:06d}.log')
                command = _build_train_subprocess_command(
                    task,
                    task_status_path=status_path,
                    **command_options,
                )
                log_file = log_path.open('wb')
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=repository_root,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                    )
                except Exception as error:
                    log_file.close()
                    outcome = _failed_launch_outcome(
                        task, task_index, command, log_path, error)
                    outcomes.append(outcome)
                    _print_worker_failure(outcome)
                    continue
                active.append({
                    'task_index': task_index,
                    'task': task,
                    'command': command,
                    'process': process,
                    'status_path': status_path,
                    'log_path': log_path,
                    'log_file': log_file,
                    'started': time.perf_counter(),
                })

            completed_workers = [
                worker
                for worker in active
                if worker['process'].poll() is not None
            ]
            for worker in completed_workers:
                active.remove(worker)
                outcome = _worker_outcome(worker)
                outcomes.append(outcome)
                if outcome['status'] == 'failed':
                    _print_worker_failure(outcome)
            progress.update(len(active), outcomes)
            if active and not completed_workers:
                time.sleep(poll_interval)
    except KeyboardInterrupt:
        interrupted = True
        _stop_active_workers(active)
        outcomes.extend(
            _interrupted_worker_outcome(worker) for worker in active)
        active.clear()
    except BaseException:
        _stop_active_workers(active)
        for worker in active:
            worker['log_file'].close()
        raise
    finally:
        progress.finish(len(active), outcomes)

    outcomes.sort(key=lambda outcome: outcome['task_index'])
    return {
        'outcomes': outcomes,
        'interrupted': interrupted,
        'n_not_started': len(tasks) - next_task_index,
        'elapsed_seconds': round(time.perf_counter() - started, 6),
    }


def _sweep_status(metadata_report, counts, skipped, n_not_started):
    if counts['interrupted'] or n_not_started:
        return 'interrupted'
    has_metadata_issues = bool(skipped) or metadata_report['status'] != 'complete'
    if counts['failed'] or has_metadata_issues:
        return 'completed_with_issues'
    return 'complete'


def run_phone_binary_probe_sweep(
    *,
    metadata_path=metadata.metadata_file,
    sentence_path=metadata.sentence_file,
    phraser_key_path=metadata.phraser_key_file,
    duplicate_replacement_phraser_key_path=(
        metadata.duplicate_replacement_phraser_key_file),
    model_stores_root=default_model_stores_root,
    collar=2000,
    n_embeds=None,
    n_splits=5,
    random_state=42,
    standardize=False,
    save_probes=True,
    probe_save_dir=default_probe_save_dir,
    save_predictions=True,
    results_dir=default_results_dir,
    overwrite=False,
    jobs=_default_sweep_jobs,
    metadata_batch_size=_default_metadata_batch_size,
    force_metadata_check=False,
    verbose=True,
    poll_interval=_default_sweep_poll_interval,
):
    '''Run all complete checkpoint phone probes in bounded subprocesses.

    The metadata preflight always finishes before task construction. Each
    child runs this module's ``train`` command for one phone/model/layer and
    handles every fold for that task. Worker output and atomic status files
    live only in a unique temporary directory and are incorporated into the
    returned structure before that directory is removed.
    '''
    _validate_sweep_arguments(jobs, n_embeds, n_splits, standardize)
    _validate_metadata_preflight_arguments(collar, metadata_batch_size)
    if isinstance(poll_interval, bool) or not isinstance(
        poll_interval, (int, float),
    ):
        raise TypeError('poll_interval must be a positive number')
    if poll_interval <= 0:
        raise ValueError('poll_interval must be a positive number')

    sweep_started = time.perf_counter()
    started_at = _utc_timestamp()
    metadata_report = check_phone_binary_probe_metadata(
        metadata_path=metadata_path,
        sentence_path=sentence_path,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
        model_stores_root=model_stores_root,
        collar=collar,
        batch_size=metadata_batch_size,
        force_metadata_check=force_metadata_check,
        verbose=verbose,
    )
    phone_labels = _sweep_phone_labels_from_preflight(
        metadata_report,
        metadata_path=metadata_path,
        sentence_path=sentence_path,
        phraser_key_path=phraser_key_path,
        duplicate_replacement_phraser_key_path=(
            duplicate_replacement_phraser_key_path),
        n_embeds=n_embeds,
    )
    tasks, metadata_skipped = _build_sweep_task_lists(
        metadata_report, phone_labels)
    command_options = {
        'metadata_path': metadata_path,
        'sentence_path': sentence_path,
        'phraser_key_path': phraser_key_path,
        'duplicate_replacement_phraser_key_path': (
            duplicate_replacement_phraser_key_path),
        'model_stores_root': model_stores_root,
        'collar': collar,
        'n_embeds': n_embeds,
        'n_splits': n_splits,
        'random_state': random_state,
        'standardize': standardize,
        'save_probes': save_probes,
        'probe_save_dir': probe_save_dir,
        'save_predictions': save_predictions,
        'results_dir': results_dir,
        'overwrite': overwrite,
        'verbose': verbose,
    }

    with tempfile.TemporaryDirectory(
        prefix='diphone-phone-probes-', dir='/tmp',
    ) as temporary_directory:
        run_id = Path(temporary_directory).name.removeprefix(
            'diphone-phone-probes-')
        subprocess_result = _run_sweep_subprocesses(
            tasks,
            jobs=jobs,
            temporary_directory=temporary_directory,
            command_options=command_options,
            poll_interval=float(poll_interval),
            verbose=verbose,
        )
        outcomes = subprocess_result['outcomes']
        counts = _sweep_counts(outcomes)
        status = _sweep_status(
            metadata_report,
            counts,
            metadata_skipped,
            subprocess_result['n_not_started'],
        )
        finished_at = _utc_timestamp()
        elapsed_seconds = round(time.perf_counter() - sweep_started, 6)
        report_settings = {
            'metadata_path': str(metadata_path),
            'sentence_path': str(sentence_path),
            'phraser_key_path': str(phraser_key_path),
            'duplicate_replacement_phraser_key_path': (
                None
                if duplicate_replacement_phraser_key_path is None
                else str(duplicate_replacement_phraser_key_path)
            ),
            'model_stores_root': str(model_stores_root),
            'collar': collar,
            'n_embeds': n_embeds,
            'n_splits': n_splits,
            'random_state': random_state,
            'standardize': standardize,
            'save_probes': save_probes,
            'probe_save_dir': str(probe_save_dir),
            'save_predictions': save_predictions,
            'results_dir': str(results_dir),
            'overwrite': overwrite,
            'verbose': verbose,
            'jobs': jobs,
            'metadata_batch_size': metadata_batch_size,
            'force_metadata_check': bool(force_metadata_check),
        }
        current_sweep = {
            'run_id': run_id,
            'status': status,
            'started_at': started_at,
            'finished_at': finished_at,
            'elapsed_seconds': elapsed_seconds,
            'settings': report_settings,
            'interrupted': subprocess_result['interrupted'],
            'n_not_started': subprocess_result['n_not_started'],
        }
        report = build_phone_binary_probe_report(
            metadata_path=metadata_path,
            sentence_path=sentence_path,
            phraser_key_path=phraser_key_path,
            duplicate_replacement_phraser_key_path=(
                duplicate_replacement_phraser_key_path),
            model_stores_root=model_stores_root,
            collar=collar,
            n_embeds=n_embeds,
            n_splits=n_splits,
            random_state=random_state,
            standardize=standardize,
            save_probes=save_probes,
            probe_save_dir=probe_save_dir,
            save_predictions=save_predictions,
            results_dir=results_dir,
            phone_labels=phone_labels,
            current_worker_outcomes=outcomes,
            current_sweep=current_sweep,
            current_metadata_report=metadata_report,
            verbose=False,
        )

    if verbose:
        summary = report['summary']
        completion = (
            'interrupted'
            if report['status'] == 'interrupted'
            else 'complete'
        )
        print(
            f'Phone probe sweep {completion}: '
            f'{summary["n_trained"]} trained, '
            f'{summary["n_already_complete"]} already complete, '
            f'{summary["n_current_failed"]} failed, '
            f'{summary["n_metadata_skipped_tasks"]} metadata-skipped',
            flush=True,
        )
    if subprocess_result['interrupted']:
        raise PhoneBinaryProbeSweepInterrupted(report)
    return report
